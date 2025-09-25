#!/usr/bin/env python3
"""
Processes RH20T task folders to generate sparse, high-resolution depth maps by
reprojecting a point cloud from low-resolution depth onto high-resolution views.

This script uses the Open3D and OpenCV libraries for robust 3D data handling and
includes an optional color-based alignment check to ensure high-quality results.

Usage Examples:
-----------------

# 1. Basic Processing (Low-Resolution Data Only)
# Creates a standard NPZ and point cloud from a single low-res folder.
# Useful for quickly visualizing the source data.
python create_sparse_depth_map.py \\
  --task-folder /path/to/low_res_data/task_0010_user_0011_scene_0010_cfg_0003 \\
  --out-dir ./output/low_res_processed

# 2. Reprojection Workflow (Low-Res Depth to High-Res RGB)
# This is the primary use case. It generates a sparse, high-resolution depth map
# by reprojecting the geometry from the low-res data onto the high-res views.
# It also limits the processing to the middle 100 frames of the sequence.
python create_sparse_depth_map.py \\
  --task-folder /path/to/low_res_data/task_0010_... \\
  --high-res-folder /path/to/high_res_data/task_0010_... \\
  --out-dir ./output/high_res_reprojected \\
  --max-frames 100

# 3. Advanced Reprojection with Color Alignment Check
# This produces the cleanest, most reliable sparse depth map by filtering out
# points where the original color (from low-res) doesn't match the target
# color (in high-res), indicating a potential calibration misalignment.

python create_sparse_depth_map.py   --task-folder /data/rh20t_api/data/test_data_full_rgb_upscaled_depth/uncompressed_low_res_data/task_0065_user_0010_scene_0009_cfg_0004   --high-res-folder /data/rh20t_api/data/test_data_full_rgb_upscaled_depth/rgb_data/RH20T_cfg4/task_0065_user_0010_scene_0009_cfg_0004  --out-dir ./data/high_res_filtered   --max-frames 100   --color-alignment-check   --color-threshold 35
"""


# Standard library imports
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Third-party imports
import numpy as np
import torch
from PIL import Image
import cv2
import open3d as o3d
import rerun as rr

from tqdm import tqdm

# Project-specific imports
from mvtracker.utils.visualizer_rerun import log_pointclouds_to_rerun
from rh20t_api.rh20t_api.configurations import load_conf
from rh20t_api.rh20t_api.scene import RH20TScene

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
    # L515 cameras have a different depth scale factor
    scale = 4000.0 if is_l515 else 1000.0
    depth_m = arr.astype(np.float32) / scale
    # Invalidate depth values outside the specified operational range
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

# --- Data Loading & Synchronization ---

def load_scene_data(task_path: Path, robot_configs: List) -> Tuple[Optional[RH20TScene], List[str], List[Path]]:
    """
    Loads an RH20T scene and filters for calibrated, external (non-hand) cameras.

    Args:
        task_path: Path to the RH20T task folder.
        robot_configs: Loaded robot configurations from the RH20T API.

    Returns:
        A tuple containing the scene object, a list of valid camera IDs, and
        a list of their corresponding directory paths.
    """
    scene = RH20TScene(str(task_path), robot_configs)
    # CHANGED: Use the public configuration property exposed by RH20TScene
    # instead of accessing the non-existent legacy `.conf` attribute.
    # TODO: use one image of each timw window, and remove if not there son that no time lagging 
    in_hand_serials = set(scene.configuration.in_hand_serial)
    
    valid_cam_ids, valid_cam_dirs = [], []
    all_cam_dirs = sorted(p for p in task_path.glob("cam_*") if p.is_dir())
    for cdir in all_cam_dirs:
        cid = cdir.name.replace("cam_", "")
        if cid in scene.intrinsics and cid in scene.extrinsics_base_aligned and cid not in in_hand_serials:
            valid_cam_ids.append(cid)
            valid_cam_dirs.append(cdir)
    
    print(f"[INFO] Found {len(valid_cam_ids)} valid external cameras in {task_path.name}.")
    return scene, valid_cam_ids, valid_cam_dirs

def get_synchronized_timestamps(cam_dirs: List[Path]) -> np.ndarray:
    """Finds timestamps for which all cameras have both a color and depth image."""
    raise NotImplementedError("implement the time stamp alignment")
    if not cam_dirs: return np.array([], dtype=np.int64)
    
    # Get the set of valid timestamps for each camera
    ts_sets = [
        set(list_frames(cdir / "color").keys()).intersection(list_frames(cdir / "depth").keys())
        for cdir in cam_dirs
    ]
    
    # Find the intersection of all sets to get universally available timestamps
    common_ts = set.intersection(*ts_sets) if ts_sets else set()
    return np.array(sorted(list(common_ts)), dtype=np.int64)

def select_frames(timestamps: np.ndarray, max_frames: Optional[int], selection_method: str) -> np.ndarray:
    """Selects a subset of frames from a list of timestamps."""
    if not max_frames or max_frames <= 0 or max_frames >= len(timestamps):
        return timestamps

    total_frames = len(timestamps)
    if selection_method == "first":
        selected_ts = timestamps[:max_frames]
    elif selection_method == "last":
        selected_ts = timestamps[-max_frames:]
    else:  # "middle"
        start_idx = (total_frames - max_frames) // 2
        selected_ts = timestamps[start_idx : start_idx + max_frames]
    
    print(f"[INFO] Selected {len(selected_ts)} frames using '{selection_method}' method.")
    return selected_ts

# --- 3D Geometry & Reprojection Functions ---

def unproject_to_world_o3d(depth: np.ndarray, rgb: np.ndarray, K: np.ndarray, E_inv: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Creates a colored point cloud in world coordinates from a single view using Open3D.

    Args:
        depth: The depth map (H, W).
        rgb: The color image (H, W, 3).
        K: The 3x3 intrinsic camera matrix.
        E_inv: The 4x4 inverse extrinsic matrix (camera-to-world transformation).

    Returns:
        An Open3D PointCloud object in world coordinates.
    """
    o3d_depth = o3d.geometry.Image(depth)
    o3d_rgb = o3d.geometry.Image(rgb)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_rgb, o3d_depth, depth_scale=1.0, depth_trunc=10.0, convert_rgb_to_intensity=False
    )
    
    H, W = depth.shape
    intrinsics = o3d.camera.PinholeCameraIntrinsic(W, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    
    pcd_cam = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
    
    # The point cloud is created in the camera's local coordinate system.
    # Transform it to the global world coordinate system.
    return pcd_cam.transform(E_inv)

def reproject_to_sparse_depth_cv2(
    pcd: o3d.geometry.PointCloud, high_res_rgb: np.ndarray, K: np.ndarray, E: np.ndarray, color_threshold: float
) -> np.ndarray:
    """
    Projects an Open3D point cloud to a sparse depth map using OpenCV.

    Includes a color alignment check to filter out misaligned points.

    Args:
        pcd: The Open3D PointCloud in world coordinates.
        high_res_rgb: The target high-resolution color image.
        K: The 3x3 intrinsic matrix of the target camera.
        E: The 3x4 extrinsic matrix of the target camera (world-to-camera).
        color_threshold: The maximum allowed color difference (0-255) for a point to be kept.

    Returns:
        A sparse depth map of the same resolution as the high_res_rgb.
    """
    H, W, _ = high_res_rgb.shape
    sparse_depth = np.zeros((H, W), dtype=np.float32)
    
    if not pcd.has_points():
        return sparse_depth

    pts_world = np.asarray(pcd.points)
    orig_colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)

    # Decompose extrinsics into rotation and translation vectors for OpenCV
    R, t = E[:3, :3], E[:3, 3]
    rvec, _ = cv2.Rodrigues(R)
    tvec = t.reshape(3, 1)

    # Project all 3D points into the 2D image plane of the target camera
    projected_pts, _ = cv2.projectPoints(pts_world, rvec, tvec, K, distCoeffs=None)
    projected_pts = projected_pts.squeeze(1)
    
    # Calculate the depth of each point relative to the new camera
    pts_cam = (R @ pts_world.T + tvec).T
    depths = pts_cam[:, 2]

    # --- Filtering Stage ---
    # 1. Filter points behind the camera
    in_front_mask = depths > 1e-6
    # 2. Filter points outside the image boundaries
    u, v = projected_pts[in_front_mask, 0], projected_pts[in_front_mask, 1]
    bounds_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    
    # Apply combined filters
    valid_mask = np.where(in_front_mask)[0][bounds_mask]
    u_idx = np.round(u[bounds_mask]).astype(int)
    v_idx = np.round(v[bounds_mask]).astype(int)
    depth_final = depths[valid_mask]
    orig_colors_final = orig_colors[valid_mask]

    # 3. Color Alignment Check
    target_colors = high_res_rgb[v_idx, u_idx]
    color_diff = np.mean(np.abs(orig_colors_final.astype(float) - target_colors.astype(float)), axis=1)
    color_match_mask = color_diff < color_threshold
    
    u_final, v_final = u_idx[color_match_mask], v_idx[color_match_mask]
    depth_final = depth_final[color_match_mask]
    
    # 4. Z-Buffering: Handle occlusions by keeping only the closest point for each pixel
    # Sort points by depth in reverse order, so closer points are processed last and overwrite farther ones.
    sorted_indices = np.argsort(depth_final)[::-1]
    sparse_depth[v_final[sorted_indices], u_final[sorted_indices]] = depth_final[sorted_indices]
    
    return sparse_depth

# --- Main Workflow & Orchestration ---

def process_frames(args, scene_low, scene_high, final_cam_ids, cam_dirs_low, cam_dirs_high, timestamps):
    """
    Iterates through timestamps to process frames, generate point clouds,
    and create the final data arrays (RGB, depth, intrinsics, extrinsics).
    """
    C, T = len(final_cam_ids), len(timestamps)
    is_l515_flags = [cid.startswith('f') for cid in final_cam_ids]

    # Determine output resolution and initialize data containers
    if args.high_res_folder:
        H_out, W_out, _ = read_rgb(list_frames(cam_dirs_high[0] / 'color')[timestamps[0]]).shape
        print(f"[INFO] Outputting high-resolution data ({H_out}x{W_out}) with reprojected depth.")
    else:
        H_out, W_out, _ = read_rgb(list_frames(cam_dirs_low[0] / 'color')[timestamps[0]]).shape
        print(f"[INFO] Outputting low-resolution data ({H_out}x{W_out}).")

    rgbs_out = np.zeros((C, T, H_out, W_out, 3), dtype=np.uint8)
    depths_out = np.zeros((C, T, H_out, W_out), dtype=np.float32)
    intrs_out = np.zeros((C, T, 3, 3), dtype=np.float32)
    extrs_out = np.zeros((C, T, 3, 4), dtype=np.float32)

    for ti in tqdm(range(T), desc="Processing Frames"):
        t = timestamps[ti]
        
        # Step 1: Create a combined point cloud for the current frame from all low-res views
        combined_pcd = o3d.geometry.PointCloud()
        if args.high_res_folder:
            pcds_per_cam = []
            for ci in range(C):
                cid = final_cam_ids[ci]
                # Load low-res data for point cloud generation
                depth_low = read_depth(list_frames(cam_dirs_low[ci] / "depth")[t], is_l515_flags[ci])
                rgb_low = read_rgb(list_frames(cam_dirs_low[ci] / "color")[t])
                K_low = scene_low.intrinsics[cid][:, :3]
                E_inv = np.linalg.inv(np.vstack([scene_low.extrinsics_base_aligned[cid], [0, 0, 0, 1]]))
                
                # Create and add the point cloud for this view
                pcds_per_cam.append(unproject_to_world_o3d(depth_low, rgb_low, K_low, E_inv))
            
            # Merge all individual point clouds into a single scene representation
            for pcd in pcds_per_cam:
                combined_pcd += pcd

        # Step 2: Generate the final output data for each camera view
        for ci in range(C):
            cid = final_cam_ids[ci]
            E_world_to_cam = scene_low.extrinsics_base_aligned[cid]
            extrs_out[ci, ti] = E_world_to_cam[:3, :4]
            
            if args.high_res_folder:
                # Reprojection workflow
                high_res_rgb = read_rgb(list_frames(cam_dirs_high[ci] / "color")[t])
                K_high = scene_high.intrinsics[cid][:, :3]
                rgbs_out[ci, ti] = high_res_rgb
                intrs_out[ci, ti] = K_high
                
                # If color check is disabled, use a threshold that allows all points to pass
                threshold = args.color_threshold if args.color_alignment_check else 256.0
                depths_out[ci, ti] = reproject_to_sparse_depth_cv2(
                    combined_pcd, high_res_rgb, K_high, E_world_to_cam, threshold
                )
            else:
                # Standard low-resolution workflow
                rgbs_out[ci, ti] = read_rgb(list_frames(cam_dirs_low[ci] / "color")[t])
                depths_out[ci, ti] = read_depth(list_frames(cam_dirs_low[ci] / "depth")[t], is_l515_flags[ci])
                intrs_out[ci, ti] = scene_low.intrinsics[cid][:, :3]

    return rgbs_out, depths_out, intrs_out, extrs_out

def save_and_visualize(args, rgbs, depths, intrs, extrs, final_cam_ids, timestamps):
    """Saves the processed data to an NPZ file and generates a Rerun visualization."""
    # Convert to channels-first format for NPZ
    rgbs_final = np.moveaxis(rgbs, -1, 2)
    depths_final = depths[:, :, None, :, :]
    
    npz_payload = {
        'rgbs': rgbs_final,
        'depths': depths_final,
        'intrs': intrs,
        'extrs': extrs,
        'timestamps': timestamps,
        'camera_ids': np.array(final_cam_ids, dtype=object),
    }

    out_path_npz = args.out_dir / f"{args.task_folder.name}_processed.npz"
    np.savez_compressed(out_path_npz, **npz_payload)
    print(f"✅ [OK] Wrote NPZ file to: {out_path_npz}")

    # Generate Rerun Visualization
    if not args.no_pointcloud:
        print("[INFO] Logging data to Rerun...")
        log_pointclouds_to_rerun(
            dataset_name="rh20t_reprojection",
            datapoint_idx=0,
            rgbs=torch.from_numpy(rgbs_final).float(),
            depths=torch.from_numpy(depths_final).float(),
            intrs=torch.from_numpy(intrs).float(),
            extrs=torch.from_numpy(extrs).float(),
            camera_ids=final_cam_ids,
            log_rgb_pointcloud=True,
            log_camera_frustrum=True,
        )

def main():
    """Parses arguments and orchestrates the entire data processing workflow."""
    parser = argparse.ArgumentParser(
        description="Process RH20T data with optional color-checked reprojection using Open3D and OpenCV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--task-folder", required=True, type=Path, help="Path to the primary (low-res) RH20T task folder.")
    parser.add_argument("--high-res-folder", type=Path, default=None, help="Optional: Path to the high-resolution task folder for reprojection.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for .npz and .rrd files.")
    parser.add_argument("--config", default="rh20t_api/configs/configs.json", type=Path, help="Path to RH20T robot configs JSON.")
    parser.add_argument("--max-frames", type=int, default=50, help="Limit frames to process (0 for all).")
    parser.add_argument("--frame-selection", choices=["first", "last", "middle"], default="middle", help="Method for selecting frames.")
    parser.add_argument("--color-alignment-check", action="store_true", help="Enable color-based filtering of reprojected points.")
    parser.add_argument("--color-threshold", type=float, default=40.0, help="Max average color difference (0-255) for a point to be aligned.")
    parser.add_argument("--no-pointcloud", action="store_true", help="Only generate the .npz file, skip visualization.")
    args = parser.parse_args()

    if not args.config.exists():
        print(f"[ERROR] RH20T config file not found at: {args.config}")
        return
    
    args.out_dir.mkdir(parents=True, exist_ok=True)
    robot_configs = load_conf(str(args.config))
    
    # --- Step 1: Load and Synchronize Data ---
    scene_low, cam_ids_low, cam_dirs_low = load_scene_data(args.task_folder, robot_configs)
    if not scene_low or not cam_ids_low: return

    scene_high, cam_ids_high, cam_dirs_high = (load_scene_data(args.high_res_folder, robot_configs) if args.high_res_folder else (None, None, None))
    
    if args.high_res_folder:
        final_cam_ids = sorted(list(set(cam_ids_low) & set(cam_ids_high)))
        cam_dirs_low = [d for d, i in zip(cam_dirs_low, cam_ids_low) if i in final_cam_ids]
        cam_dirs_high = [d for d, i in zip(cam_dirs_high, cam_ids_high) if i in final_cam_ids]
        ts = np.intersect1d(get_synchronized_timestamps(cam_dirs_low), get_synchronized_timestamps(cam_dirs_high))
    else:
        final_cam_ids = cam_ids_low
        ts = get_synchronized_timestamps(cam_dirs_low)

    if len(final_cam_ids) < 2: print("[ERROR] Fewer than 2 synchronized cameras found."); return
    if len(ts) == 0: print("[ERROR] No common timestamps found."); return
    
    timestamps = select_frames(ts, args.max_frames, args.frame_selection)
    if len(timestamps) == 0: print("[ERROR] No frames remaining after selection."); return

    # --- Step 2: Process Frames ---
    rgbs, depths, intrs, extrs = process_frames(args, scene_low, scene_high, final_cam_ids, cam_dirs_low, cam_dirs_high, timestamps)

    # --- Step 3: Save and Visualize ---
    rr.init("RH20T_Reprojection_Frameworks", spawn=False)
    save_and_visualize(args, rgbs, depths, intrs, extrs, final_cam_ids, timestamps)

    if not args.no_pointcloud:
        rrd_path = args.out_dir / f"{args.task_folder.name}_reprojected.rrd"
        rr.save(str(rrd_path))
        print(f"✅ [OK] Saved Rerun visualization to: {rrd_path}")

if __name__ == "__main__":
    main()
