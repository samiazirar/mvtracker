#!/usr/bin/env python3
"""
Generates high-quality, sparse, high-resolution depth maps from low-resolution
RGB-D data via independent per-camera geometric reprojection.

This script processes RH20T task folders by taking the depth information from a
low-resolution camera view, converting it to a 3D point cloud, and then
projecting that same point cloud back into the corresponding high-resolution
camera view. This method avoids multi-view fusion, preventing artifacts like
color bleeding ("flow") at object boundaries.

Usage Example:
-------------
python create_clean_sparse_depth.py \\
  --task-folder /path/to/low_res_data/task_0065... \\
  --high-res-folder /path/to/high_res_data/task_0065... \\
  --out-dir ./output/sparse_high_res \\
  --max-frames 100
"""

# Standard library imports
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Third-party imports
import cv2
import numpy as np
import open3d as o3d
import rerun as rr
import torch
from PIL import Image
from tqdm import tqdm

# Updated Project-specific imports
from rh20t_api.rh20t_api.configurations import load_conf
from rh20t_api.rh20t_api.scene import RH20TScene
from mvtracker.utils.visualizer_rerun import log_pointclouds_to_rerun
from create_sparse_depth_map import get_synchronized_timestamps

# --- File I/O & Utility Functions ---

NUM_RE = re.compile(r"(\d+)")

def _num_in_name(p: Path) -> Optional[int]:
    """Extracts the first integer from a filename's stem."""
    m = NUM_RE.search(p.stem)
    return int(m.group(1)) if m else None

def read_rgb(path: Path) -> np.ndarray:
    """Reads an image file into a NumPy array (H, W, 3) in RGB format."""
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)

def read_depth(path: Path, is_l515: bool, min_d: float = 0.3, max_d: float = 0.8) -> np.ndarray:
    """Reads a 16-bit depth image, converts it to meters, and clamps the range."""
    arr = np.asarray(Image.open(path)).astype(np.uint16)
    scale = 4000.0 if is_l515 else 1000.0
    depth_m = arr.astype(np.float32) / scale
    depth_m[(depth_m < min_d) | (depth_m > max_d)] = 0.0
    return depth_m

def list_frames(folder: Path) -> Dict[int, Path]:
    """Lists all image files in a folder, indexed by their timestamp."""
    if not folder.exists(): return {}
    return dict(sorted({
        t: p for ext in ("*.jpg", "*.jpeg", "*.png")
        for p in folder.glob(ext)
        if (t := _num_in_name(p)) is not None
    }.items()))

def scale_intrinsics_matrix(raw_K: np.ndarray, width: int, height: int, base_width: int, base_height: int) -> np.ndarray:
    """Rescales a 3x3 intrinsic matrix to match an image resolution."""
    if base_width <= 0 or base_height <= 0:
        raise ValueError("Base calibration resolution must be positive.")
    K = raw_K[:, :3].astype(np.float32, copy=True)
    scale_x = width / float(base_width)
    scale_y = height / float(base_height)
    K[0, 0] *= scale_x
    K[0, 2] *= scale_x
    K[1, 1] *= scale_y
    K[1, 2] *= scale_y
    return K

def infer_calibration_resolution(scene: RH20TScene, camera_id: str) -> Optional[Tuple[int, int]]:
    """Reads the calibration image resolution for a given camera if available."""
    try:
        calib_dir = Path(scene.calib_folder) / "imgs"
    except AttributeError:
        return None
    calib_path = calib_dir / f"cam_{camera_id}_c.png"
    if not calib_path.exists():
        return None
    with Image.open(calib_path) as img:
        return img.size

def get_distortion_coefficients(scene: RH20TScene, camera_id: str) -> Optional[np.ndarray]:
    """
    Attempts to load distortion coefficients for a camera.
    
    RH20T doesn't currently expose distortion coefficients in the standard API,
    but this function provides infrastructure for when they become available.
    """
    try:
        calib_dir = Path(scene.calib_folder)
        
        # Try to load from a hypothetical distortion.npy file
        distortion_file = calib_dir / "distortion.npy"
        if distortion_file.exists():
            distortion_dict = np.load(distortion_file, allow_pickle=True).item()
            if camera_id in distortion_dict:
                return distortion_dict[camera_id]
        
        # Try to load from individual camera distortion files
        cam_distortion_file = calib_dir / f"distortion_{camera_id}.npy"
        if cam_distortion_file.exists():
            return np.load(cam_distortion_file)
            
    except (AttributeError, FileNotFoundError, KeyError):
        pass
    
    # Default: no distortion (4 coefficients for OpenCV)
    return np.zeros(4, dtype=np.float32)

# --- 3D Geometry & Reprojection Functions ---

def apply_edge_aware_mask(depth: np.ndarray, rgb: np.ndarray, edge_dilation: int = 3) -> np.ndarray:
    """Apply edge-aware mask to remove mixed pixels near edges."""
    # Convert RGB to grayscale for edge detection
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    
    # Detect edges in both RGB and depth
    rgb_edges = cv2.Canny(gray, 100, 200)
    
    # Compute depth gradients
    depth_grad_x = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
    depth_grad_y = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)
    depth_grad_mag = np.sqrt(depth_grad_x**2 + depth_grad_y**2)
    
    # Find depth discontinuities (large gradients)
    valid_depth_mask = depth > 0
    if np.sum(valid_depth_mask) > 100:  # Only compute if we have enough valid depth points
        depth_grad_median = np.median(depth_grad_mag[valid_depth_mask])
        depth_edges = (depth_grad_mag > 5 * depth_grad_median).astype(np.uint8) * 255
    else:
        depth_edges = np.zeros_like(depth, dtype=np.uint8)
    
    # Combine edge maps
    combined_edges = cv2.bitwise_or(rgb_edges, depth_edges)
    
    # Dilate edges to create exclusion zones
    kernel = np.ones((edge_dilation, edge_dilation), np.uint8)
    edge_mask = cv2.dilate(combined_edges, kernel, iterations=1).astype(bool)
    
    # Apply mask to depth
    masked_depth = depth.copy()
    masked_depth[edge_mask] = 0.0
    
    return masked_depth

def unproject_depth_only(depth: np.ndarray, K: np.ndarray, E_inv: np.ndarray) -> o3d.geometry.PointCloud:
    """Unprojection using depth intrinsics only (no colors yet)."""
    H, W = depth.shape
    
    # Create point cloud from depth only
    o3d_depth = o3d.geometry.Image(depth)
    intrinsics = o3d.camera.PinholeCameraIntrinsic(W, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    
    # Create point cloud without color
    pcd_cam = o3d.geometry.PointCloud.create_from_depth_image(
        o3d_depth, intrinsics, depth_scale=1.0, depth_trunc=10.0
    )
    
    # Transform to world coordinates
    return pcd_cam.transform(E_inv)

def sample_colors_bilinear(rgb: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Sample colors from RGB image using bilinear interpolation."""
    H, W = rgb.shape[:2]
    
    # Clamp coordinates to image bounds
    u_clamped = np.clip(u, 0, W - 1)
    v_clamped = np.clip(v, 0, H - 1)
    
    # Bilinear sampling using cv2.remap
    map_x = u_clamped.astype(np.float32)
    map_y = v_clamped.astype(np.float32)
    
    # Sample each channel separately
    colors = np.zeros((len(u), 3), dtype=np.uint8)
    for c in range(3):
        colors[:, c] = cv2.remap(
            rgb[:, :, c], map_x, map_y, 
            interpolation=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_REFLECT_101
        ).astype(np.uint8)
    
    return colors

def project_world_points(points_world: np.ndarray, K: np.ndarray, E: np.ndarray, dist_coeffs: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points to camera using proper distortion handling."""
    R, t = E[:3, :3], E[:3, 3]
    rvec, _ = cv2.Rodrigues(R)
    tvec = t.reshape(3, 1)
    
    # Use distortion coefficients if provided
    if dist_coeffs is None:
        dist_coeffs = np.zeros(4)  # No distortion
    
    projected_pts, _ = cv2.projectPoints(
        points_world, rvec, tvec, K, distCoeffs=dist_coeffs
    )
    projected_pts = projected_pts.squeeze(1)  # Shape: (N, 2)
    
    # Compute depths in camera coordinates
    pts_cam = (R @ points_world.T + tvec).T
    depths = pts_cam[:, 2]
    
    # Check which points are in front of camera
    in_front = depths > 1e-6
    
    return projected_pts[:, 0], projected_pts[:, 1], in_front

def unproject_to_world_improved(
    depth: np.ndarray, 
    rgb: np.ndarray, 
    K_depth: np.ndarray, 
    E_depth_inv: np.ndarray,
    K_color: np.ndarray,
    E_color: np.ndarray,
    dist_coeffs: Optional[np.ndarray] = None,
    apply_edge_mask: bool = True
) -> o3d.geometry.PointCloud:
    """
    Improved unprojection that handles depth/color intrinsics separately.
    
    Since RH20T doesn't expose separate depth/color intrinsics, this function
    provides the infrastructure for when they become available, but currently
    uses the same intrinsics for both.
    """
    # Apply edge-aware masking to remove mixed pixels
    if apply_edge_mask:
        depth_clean = apply_edge_aware_mask(depth, rgb)
    else:
        depth_clean = depth
    
    # 1. Unproject using depth intrinsics (no color yet)
    pcd_world = unproject_depth_only(depth_clean, K_depth, E_depth_inv)
    
    if not pcd_world.has_points():
        return pcd_world
    
    # 2. Color each 3D point by projecting to color camera
    points_world = np.asarray(pcd_world.points)
    
    # Project to color camera coordinates  
    u_color, v_color, in_front = project_world_points(
        points_world, K_color, E_color, dist_coeffs
    )
    
    # Sample colors using bilinear interpolation
    H_color, W_color = rgb.shape[:2]
    in_bounds = (u_color >= 0) & (u_color < W_color) & (v_color >= 0) & (v_color < H_color)
    valid_mask = in_front & in_bounds
    
    if np.any(valid_mask):
        colors = sample_colors_bilinear(rgb, u_color[valid_mask], v_color[valid_mask])
        
        # Create colored point cloud
        all_colors = np.zeros((len(points_world), 3), dtype=np.uint8)
        all_colors[valid_mask] = colors
        
        # Filter to only keep points with valid colors
        pcd_colored = o3d.geometry.PointCloud()
        pcd_colored.points = o3d.utility.Vector3dVector(points_world[valid_mask])
        pcd_colored.colors = o3d.utility.Vector3dVector(all_colors[valid_mask] / 255.0)
        
        return pcd_colored
    
    return o3d.geometry.PointCloud()

def unproject_to_world_o3d(depth: np.ndarray, rgb: np.ndarray, K: np.ndarray, E_inv: np.ndarray) -> o3d.geometry.PointCloud:
    """Legacy function - creates a colored point cloud in world coordinates from a SINGLE view."""
    # For now, use the same intrinsics for depth and color since RH20T doesn't separate them
    # Apply edge masking to reduce mixed pixel artifacts
    return unproject_to_world_improved(
        depth, rgb, K, E_inv, K, np.linalg.inv(E_inv), 
        dist_coeffs=None, apply_edge_mask=True
    )

def reproject_to_sparse_depth_improved(
    pcd: o3d.geometry.PointCloud, 
    high_res_shape: Tuple[int, int], 
    K: np.ndarray, 
    E: np.ndarray,
    dist_coeffs: Optional[np.ndarray] = None,
    supersample_factor: int = 1,
    depth_epsilon: float = 0.005
) -> np.ndarray:
    """
    Projects a point cloud to a sparse depth map with improved sub-pixel handling.
    
    Args:
        pcd: Point cloud to project
        high_res_shape: Target image shape (H, W)
        K: Camera intrinsics matrix
        E: Camera extrinsics matrix (world to camera)
        dist_coeffs: Distortion coefficients (if None, assumes no distortion)
        supersample_factor: Render at higher resolution then downsample (1 = no supersampling)
        depth_epsilon: Minimum depth difference for z-buffer updates (meters)
    """
    H, W = high_res_shape
    if supersample_factor > 1:
        H_render, W_render = H * supersample_factor, W * supersample_factor
        K_render = K.copy()
        K_render[0, 0] *= supersample_factor  # fx
        K_render[1, 1] *= supersample_factor  # fy  
        K_render[0, 2] *= supersample_factor  # cx
        K_render[1, 2] *= supersample_factor  # cy
    else:
        H_render, W_render = H, W
        K_render = K
        
    sparse_depth = np.zeros((H_render, W_render), dtype=np.float32)
    
    if not pcd.has_points():
        if supersample_factor > 1:
            return cv2.resize(sparse_depth, (W, H), interpolation=cv2.INTER_AREA)
        return sparse_depth

    pts_world = np.asarray(pcd.points)
    
    # Project points with proper distortion handling
    u, v, in_front = project_world_points(pts_world, K_render, E, dist_coeffs)
    
    # Keep sub-pixel coordinates for better accuracy
    depths = (E[:3, :3] @ pts_world.T + E[:3, 3:4]).T[:, 2]
    
    # Filter for points in front of camera
    valid_front = in_front & (depths > 1e-6)
    if not np.any(valid_front):
        if supersample_factor > 1:
            return cv2.resize(sparse_depth, (W, H), interpolation=cv2.INTER_AREA)
        return sparse_depth
    
    u_valid = u[valid_front]
    v_valid = v[valid_front] 
    depths_valid = depths[valid_front]
    
    # Check bounds with sub-pixel precision
    in_bounds = (u_valid >= 0) & (u_valid <= W_render - 1) & (v_valid >= 0) & (v_valid <= H_render - 1)
    
    if not np.any(in_bounds):
        if supersample_factor > 1:
            return cv2.resize(sparse_depth, (W, H), interpolation=cv2.INTER_AREA)
        return sparse_depth
    
    u_final = u_valid[in_bounds]
    v_final = v_valid[in_bounds]
    depths_final = depths_valid[in_bounds]
    
    # Convert to integer pixel coordinates using floor (more stable than round)
    u_int = np.floor(u_final).astype(np.int32)
    v_int = np.floor(v_final).astype(np.int32)
    
    # Ensure we're still in bounds after flooring
    pixel_bounds = (u_int >= 0) & (u_int < W_render) & (v_int >= 0) & (v_int < H_render)
    u_int = u_int[pixel_bounds]
    v_int = v_int[pixel_bounds]
    depths_final = depths_final[pixel_bounds]
    
    if len(depths_final) == 0:
        if supersample_factor > 1:
            return cv2.resize(sparse_depth, (W, H), interpolation=cv2.INTER_AREA)
        return sparse_depth
    
    # Improved Z-buffering with epsilon-based depth comparison
    for i in range(len(depths_final)):
        y, x, d = v_int[i], u_int[i], depths_final[i]
        current_depth = sparse_depth[y, x]
        
        # Update if pixel is empty or new depth is significantly closer
        if current_depth == 0 or d < (current_depth - depth_epsilon):
            sparse_depth[y, x] = d
    
    # Downsample if we rendered at higher resolution
    if supersample_factor > 1:
        # Use min pooling to preserve closest depths when downsampling
        sparse_depth_downsampled = np.zeros((H, W), dtype=np.float32)
        for y in range(H):
            for x in range(W):
                y_start, y_end = y * supersample_factor, (y + 1) * supersample_factor
                x_start, x_end = x * supersample_factor, (x + 1) * supersample_factor
                patch = sparse_depth[y_start:y_end, x_start:x_end]
                nonzero_patch = patch[patch > 0]
                if len(nonzero_patch) > 0:
                    sparse_depth_downsampled[y, x] = np.min(nonzero_patch)
        return sparse_depth_downsampled
    
    return sparse_depth

def reproject_to_sparse_depth_cv2(pcd: o3d.geometry.PointCloud, high_res_shape: Tuple[int, int], K: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Legacy function - projects a point cloud to a sparse depth map using OpenCV's Z-buffering."""
    return reproject_to_sparse_depth_improved(
        pcd, high_res_shape, K, E, dist_coeffs=None, supersample_factor=1, depth_epsilon=0.005
    )

# --- Main Workflow & Orchestration ---

def process_frames(
    scene_low: RH20TScene,
    scene_high: RH20TScene,
    final_cam_ids: List[str],
    cam_dirs_low: List[Path],
    cam_dirs_high: List[Path],
    timeline: np.ndarray,
    per_cam_low_ts: List[np.ndarray],
    per_cam_high_ts: List[np.ndarray],
):
    """Iterates through timestamps to process frames via independent per-camera reprojection."""
    C, T = len(final_cam_ids), len(timeline)
    is_l515_flags = [cid.startswith('f') for cid in final_cam_ids]

    color_lookup_low = [list_frames(cdir / 'color') for cdir in cam_dirs_low]
    depth_lookup_low = [list_frames(cdir / 'depth') for cdir in cam_dirs_low]
    color_lookup_high = [list_frames(cdir / 'color') for cdir in cam_dirs_high]

    def _resolve_frame(frame_map: Dict[int, Path], ts: int, cam_name: str, label: str) -> Path:
        path = frame_map.get(ts)
        if path is not None:
            return path

        if not frame_map:
            raise KeyError(f"No frames available for {cam_name} ({label}).")

        closest_ts = min(frame_map.keys(), key=lambda k: abs(k - ts))
        delta = abs(closest_ts - ts)
        print(
            f"[WARN] Timestamp {ts} not found for {cam_name} ({label}); using closest {closest_ts} (|Δ|={delta} ms)."
        )
        return frame_map[closest_ts]

    # Pre-calculate scaled intrinsics, distortion coefficients, and output shapes
    scaled_low_intrinsics, scaled_high_intrinsics = [], []
    low_distortion_coeffs, high_distortion_coeffs = [], []
    H_out, W_out = 0, 0
    
    for ci, cid in enumerate(final_cam_ids):
        # Low-res intrinsics and distortion
        first_low_ts = int(per_cam_low_ts[ci][0])
        low_img = read_rgb(_resolve_frame(color_lookup_low[ci], first_low_ts, cid, "low-res color"))
        h_low, w_low = low_img.shape[:2]
        base_res_low = infer_calibration_resolution(scene_low, cid) or (w_low, h_low)
        scaled_low_intrinsics.append(scale_intrinsics_matrix(scene_low.intrinsics[cid], w_low, h_low, *base_res_low))
        low_distortion_coeffs.append(get_distortion_coefficients(scene_low, cid))

        # High-res intrinsics, distortion, and output shape
        first_high_ts = int(per_cam_high_ts[ci][0])
        high_img = read_rgb(_resolve_frame(color_lookup_high[ci], first_high_ts, cid, "high-res color"))
        h_high, w_high = high_img.shape[:2]
        base_res_high = infer_calibration_resolution(scene_high, cid) or base_res_low
        scaled_high_intrinsics.append(scale_intrinsics_matrix(scene_high.intrinsics[cid], w_high, h_high, *base_res_high))
        high_distortion_coeffs.append(get_distortion_coefficients(scene_high, cid))
        
        if ci == 0: H_out, W_out = h_high, w_high

    print(f"[INFO] Outputting sparse depth at {H_out}x{W_out} resolution.")
    rgbs_out = np.zeros((C, T, H_out, W_out, 3), dtype=np.uint8)
    depths_out = np.zeros((C, T, H_out, W_out), dtype=np.float32)
    intrs_out = np.zeros((C, T, 3, 3), dtype=np.float32)
    extrs_out = np.zeros((C, T, 3, 4), dtype=np.float32)

    for ti in tqdm(range(T), desc="Processing Frames"):
        for ci, cid in enumerate(final_cam_ids):
            # --- Independent Per-Camera Reprojection Logic with Improved Edge Handling ---
            
            # 1. Create a sparse point cloud from this camera's LOW-RES view
            t_low = int(per_cam_low_ts[ci][ti])
            depth_low = read_depth(_resolve_frame(depth_lookup_low[ci], t_low, cid, "low-res depth"), is_l515_flags[ci])
            rgb_low = read_rgb(_resolve_frame(color_lookup_low[ci], t_low, cid, "low-res color"))
            
            # Get camera parameters for low-res view
            K_low = scaled_low_intrinsics[ci]
            E_world_to_cam_low = scene_low.extrinsics_base_aligned[cid]
            E_inv_low = np.linalg.inv(E_world_to_cam_low)
            distortion_low = low_distortion_coeffs[ci]

            # Create improved point cloud with edge-aware masking
            # Since RH20T doesn't separate depth/color intrinsics, use same K for both
            pcd_this_view = unproject_to_world_improved(
                depth_low, rgb_low, 
                K_depth=K_low, E_depth_inv=E_inv_low,
                K_color=K_low, E_color=E_world_to_cam_low,
                dist_coeffs=distortion_low,
                apply_edge_mask=True
            )

            # 2. Load the corresponding HIGH-RES view and parameters
            t_high = int(per_cam_high_ts[ci][ti])
            high_res_rgb = read_rgb(_resolve_frame(color_lookup_high[ci], t_high, cid, "high-res color"))
            K_high = scaled_high_intrinsics[ci]
            E_world_to_cam_high = scene_high.extrinsics_base_aligned[cid]
            distortion_high = high_distortion_coeffs[ci]

            # 3. Reproject the single-view point cloud into the high-res frame with improved z-buffering
            sparse_depth_high_res = reproject_to_sparse_depth_improved(
                pcd_this_view, 
                high_res_rgb.shape[:2], 
                K_high, 
                E_world_to_cam_high,
                dist_coeffs=distortion_high,
                supersample_factor=2,  # Render at 2x resolution for better edge quality
                depth_epsilon=0.005    # 5mm depth comparison tolerance
            )

            # 4. Store the results
            rgbs_out[ci, ti] = high_res_rgb
            depths_out[ci, ti] = sparse_depth_high_res
            intrs_out[ci, ti] = K_high
            extrs_out[ci, ti] = E_world_to_cam_high[:3, :4]

    return rgbs_out, depths_out, intrs_out, extrs_out

def save_and_visualize(args, rgbs, depths, intrs, extrs, final_cam_ids, timestamps):
    """Saves the processed data and generates a Rerun visualization."""
    out_path_npz = args.out_dir / f"{args.task_folder.name}_sparse_reprojected.npz"
    np.savez_compressed(
        out_path_npz,
        rgbs=np.moveaxis(rgbs, -1, 2),  # HWC to CHW
        depths=depths[:, :, None, :, :],
        intrs=intrs,
        extrs=extrs,
        timestamps=timestamps,
        camera_ids=np.array(final_cam_ids, dtype=object),
    )
    print(f"✅ [OK] Wrote sparse depth NPZ file to: {out_path_npz}")

    # Generate Rerun Visualization
    print("[INFO] Logging data to Rerun...")
    log_pointclouds_to_rerun(
        dataset_name="rh20t_sparse_reprojection",
        datapoint_idx=0,
        rgbs=torch.from_numpy(np.moveaxis(rgbs, -1, 2)).float().unsqueeze(0),
        depths=torch.from_numpy(depths[:, :, None, :, :]).float().unsqueeze(0),
        intrs=torch.from_numpy(intrs).float().unsqueeze(0),
        extrs=torch.from_numpy(extrs).float().unsqueeze(0),
        camera_ids=final_cam_ids,
        log_rgb_pointcloud=True,
        log_camera_frustrum=True,
    )

def main():
    """Parses arguments and orchestrates the data processing workflow."""
    parser = argparse.ArgumentParser(
        description="Generate clean, sparse, high-resolution depth maps via per-camera reprojection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--task-folder", required=True, type=Path, help="Path to the primary (low-res) RH20T task folder.")
    parser.add_argument("--high-res-folder", required=True, type=Path, help="Path to the high-resolution task folder for reprojection.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for .npz and .rrd files.")
    parser.add_argument("--config", default="rh20t_api/configs/configs.json", type=Path, help="Path to RH20T robot configs JSON.")
    parser.add_argument("--max-frames", type=int, default=50, help="Limit frames to process (0 for all).")
    args = parser.parse_args()

    if not args.config.exists():
        raise FileNotFoundError(f"RH20T config file not found at: {args.config}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    robot_configs = load_conf(str(args.config))

    # Step 1: Load Scenes and Filter for valid, common cameras
    scene_low = RH20TScene(str(args.task_folder), robot_configs)
    scene_high = RH20TScene(str(args.high_res_folder), robot_configs)
    
    in_hand_serials = set(scene_low.configuration.in_hand_serial)
    
    id_to_dir_low, id_to_dir_high = {}, {}
    for cdir in sorted(p for p in args.task_folder.glob("cam_*") if p.is_dir()):
        cid = cdir.name.replace("cam_", "")
        if cid in scene_low.intrinsics and cid in scene_low.extrinsics_base_aligned and cid not in in_hand_serials:
            id_to_dir_low[cid] = cdir
            
    for cdir in sorted(p for p in args.high_res_folder.glob("cam_*") if p.is_dir()):
        cid = cdir.name.replace("cam_", "")
        if cid in scene_high.intrinsics and cid in scene_high.extrinsics_base_aligned and cid not in in_hand_serials:
            id_to_dir_high[cid] = cdir

    final_cam_ids = sorted(set(id_to_dir_low.keys()) & set(id_to_dir_high.keys()))
    if not final_cam_ids:
        print("[ERROR] No common calibrated cameras found between the two folders.")
        return
    
    if len(final_cam_ids) < 2:
        print("[ERROR] Fewer than 2 common cameras between low and high resolution data.")
        return
    
    final_cam_dirs_low = [id_to_dir_low[cid] for cid in final_cam_ids]
    final_cam_dirs_high = [id_to_dir_high[cid] for cid in final_cam_ids]
    print(f"[INFO] Found {len(final_cam_ids)} common cameras: {final_cam_ids}")

    # Step 2: Synchronize Timestamps
    sync_low = get_synchronized_timestamps(final_cam_dirs_low)
    sync_high = get_synchronized_timestamps(final_cam_dirs_high, require_depth=False)
    
    if sync_low.timeline.size == 0:
        print("[ERROR] Low-resolution synchronization returned no frames.")
        return
    
    if sync_high.timeline.size == 0:
        print("[ERROR] High-resolution synchronization returned no frames.")
        return

    # Filter cameras that actually passed synchronization
    valid_low_indices = set(sync_low.camera_indices)
    valid_high_indices = set(sync_high.camera_indices)
    keep_indices = sorted(valid_low_indices & valid_high_indices)
    
    if len(keep_indices) < 1:
        print("[ERROR] No cameras passed synchronization in both low and high resolution data.")
        return
    
    # Map camera indices to synchronized timestamps
    index_map_low = {idx: arr for idx, arr in zip(sync_low.camera_indices, sync_low.per_camera_timestamps)}
    index_map_high = {idx: arr for idx, arr in zip(sync_high.camera_indices, sync_high.per_camera_timestamps)}
    
    # Update camera lists to only include cameras that passed synchronization
    final_cam_ids = [final_cam_ids[i] for i in keep_indices]
    final_cam_dirs_low = [final_cam_dirs_low[i] for i in keep_indices]
    final_cam_dirs_high = [final_cam_dirs_high[i] for i in keep_indices]
    
    print(f"[INFO] {len(final_cam_ids)} cameras passed synchronization: {final_cam_ids}")

    timeline_common = np.intersect1d(sync_low.timeline, sync_high.timeline)
    if timeline_common.size == 0:
        print("[ERROR] No overlapping synchronized timeline found.")
        return
    
    timeline_common = np.asarray(timeline_common, dtype=np.int64)
    idx_map_low = {int(t): idx for idx, t in enumerate(sync_low.timeline)}
    idx_map_high = {int(t): idx for idx, t in enumerate(sync_high.timeline)}
    indices_low = [idx_map_low[int(t)] for t in timeline_common]
    indices_high = [idx_map_high[int(t)] for t in timeline_common]

    per_cam_low = [index_map_low[i][indices_low] for i in keep_indices]
    per_cam_high = [index_map_high[i][indices_high] for i in keep_indices]

    # Step 3: Select Frames
    num_frames = len(timeline_common)
    if args.max_frames and 0 < args.max_frames < num_frames:
        start_idx = (num_frames - args.max_frames) // 2
        end_idx = start_idx + args.max_frames
        timestamps = timeline_common[start_idx:end_idx]
        per_cam_low_sel = [ts[start_idx:end_idx] for ts in per_cam_low]
        per_cam_high_sel = [ts[start_idx:end_idx] for ts in per_cam_high]
        print(f"[INFO] Selected middle {len(timestamps)} frames for processing.")
    else:
        timestamps = timeline_common
        per_cam_low_sel = per_cam_low
        per_cam_high_sel = per_cam_high

    if timestamps.size == 0:
        print("[ERROR] No frames remaining after selection.")
        return
    
    # Validate that we have enough data for processing
    if len(per_cam_low_sel) != len(final_cam_ids) or len(per_cam_high_sel) != len(final_cam_ids):
        print("[ERROR] Mismatch between number of cameras and synchronized timestamp arrays.")
        return
    
    print(f"[INFO] Processing {len(timestamps)} frames across {len(final_cam_ids)} cameras.")

    # Step 4: Process Frames with the clean, independent method
    rgbs, depths, intrs, extrs = process_frames(
        scene_low, scene_high, final_cam_ids, final_cam_dirs_low, final_cam_dirs_high,
        timestamps, per_cam_low_sel, per_cam_high_sel
    )

    # Step 5: Save and Visualize
    rr.init("RH20T_Clean_Reprojection", spawn=False)
    save_and_visualize(args, rgbs, depths, intrs, extrs, final_cam_ids, timestamps)

    rrd_path = args.out_dir / f"{args.task_folder.name}_reprojected_clean.rrd"
    rr.save(str(rrd_path))
    print(f"✅ [OK] Saved Rerun visualization to: {rrd_path}")

if __name__ == "__main__":
    main()