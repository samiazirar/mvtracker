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

# --- 3D Geometry & Reprojection Functions ---

def unproject_to_world_o3d(depth: np.ndarray, rgb: np.ndarray, K: np.ndarray, E_inv: np.ndarray) -> o3d.geometry.PointCloud:
    """Creates a colored point cloud in world coordinates from a SINGLE view."""
    o3d_depth = o3d.geometry.Image(depth)
    o3d_rgb = o3d.geometry.Image(rgb)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_rgb, o3d_depth, depth_scale=1.0, depth_trunc=10.0, convert_rgb_to_intensity=False
    )
    H, W = depth.shape
    intrinsics = o3d.camera.PinholeCameraIntrinsic(W, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    pcd_cam = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
    return pcd_cam.transform(E_inv)

def reproject_to_sparse_depth_cv2(pcd: o3d.geometry.PointCloud, high_res_shape: Tuple[int, int], K: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Projects a point cloud to a sparse depth map using OpenCV's Z-buffering."""
    H, W = high_res_shape
    sparse_depth = np.zeros((H, W), dtype=np.float32)
    if not pcd.has_points():
        return sparse_depth

    pts_world = np.asarray(pcd.points)
    R, t = E[:3, :3], E[:3, 3]
    rvec, _ = cv2.Rodrigues(R)
    tvec = t.reshape(3, 1)

    projected_pts, _ = cv2.projectPoints(pts_world, rvec, tvec, K, distCoeffs=None)
    projected_pts = projected_pts.squeeze(1)
    
    pts_cam = (R @ pts_world.T + tvec).T
    depths = pts_cam[:, 2]

    in_front_mask = depths > 1e-6
    u, v = projected_pts[in_front_mask, 0], projected_pts[in_front_mask, 1]
    bounds_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    
    valid_mask = np.where(in_front_mask)[0][bounds_mask]
    u_idx = np.round(u[bounds_mask]).astype(int)
    v_idx = np.round(v[bounds_mask]).astype(int)
    depth_final = depths[valid_mask]

    # Z-Buffering: Handle occlusions by keeping only the closest point for each pixel.
    sorted_indices = np.argsort(depth_final)[::-1]
    sparse_depth[v_idx[sorted_indices], u_idx[sorted_indices]] = depth_final[sorted_indices]
    
    return sparse_depth

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

    def _resolve_frame(frame_map: Dict[int, Path], ts: int) -> Path:
        return frame_map.get(ts) or min(frame_map.keys(), key=lambda k: abs(k - ts))

    # Pre-calculate scaled intrinsics and output shapes
    scaled_low_intrinsics, scaled_high_intrinsics = [], []
    H_out, W_out = 0, 0
    for ci, cid in enumerate(final_cam_ids):
        # Low-res intrinsics
        first_low_ts = int(per_cam_low_ts[ci][0])
        low_img = read_rgb(_resolve_frame(color_lookup_low[ci], first_low_ts))
        h_low, w_low = low_img.shape[:2]
        base_res_low = infer_calibration_resolution(scene_low, cid) or (w_low, h_low)
        scaled_low_intrinsics.append(scale_intrinsics_matrix(scene_low.intrinsics[cid], w_low, h_low, *base_res_low))

        # High-res intrinsics and output shape
        first_high_ts = int(per_cam_high_ts[ci][0])
        high_img = read_rgb(_resolve_frame(color_lookup_high[ci], first_high_ts))
        h_high, w_high = high_img.shape[:2]
        base_res_high = infer_calibration_resolution(scene_high, cid) or base_res_low
        scaled_high_intrinsics.append(scale_intrinsics_matrix(scene_high.intrinsics[cid], w_high, h_high, *base_res_high))
        
        if ci == 0: H_out, W_out = h_high, w_high

    print(f"[INFO] Outputting sparse depth at {H_out}x{W_out} resolution.")
    rgbs_out = np.zeros((C, T, H_out, W_out, 3), dtype=np.uint8)
    depths_out = np.zeros((C, T, H_out, W_out), dtype=np.float32)
    intrs_out = np.zeros((C, T, 3, 3), dtype=np.float32)
    extrs_out = np.zeros((C, T, 3, 4), dtype=np.float32)

    for ti in tqdm(range(T), desc="Processing Frames"):
        for ci, cid in enumerate(final_cam_ids):
            # --- Independent Per-Camera Reprojection Logic ---
            
            # 1. Create a sparse point cloud from this camera's LOW-RES view
            t_low = int(per_cam_low_ts[ci][ti])
            depth_low = read_depth(_resolve_frame(depth_lookup_low[ci], t_low), is_l515_flags[ci])
            rgb_low = read_rgb(_resolve_frame(color_lookup_low[ci], t_low))
            K_low = scaled_low_intrinsics[ci]
            E_world_to_cam_low = scene_low.extrinsics_base_aligned[cid]
            E_inv_low = np.linalg.inv(E_world_to_cam_low)

            pcd_this_view = unproject_to_world_o3d(depth_low, rgb_low, K_low, E_inv_low)

            # 2. Load the corresponding HIGH-RES view and parameters
            t_high = int(per_cam_high_ts[ci][ti])
            high_res_rgb = read_rgb(_resolve_frame(color_lookup_high[ci], t_high))
            K_high = scaled_high_intrinsics[ci]
            E_world_to_cam_high = scene_high.extrinsics_base_aligned[cid]

            # 3. Reproject the single-view point cloud into the high-res frame
            sparse_depth_high_res = reproject_to_sparse_depth_cv2(
                pcd_this_view, high_res_rgb.shape[:2], K_high, E_world_to_cam_high
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
    
    final_cam_dirs_low = [id_to_dir_low[cid] for cid in final_cam_ids]
    final_cam_dirs_high = [id_to_dir_high[cid] for cid in final_cam_ids]
    print(f"[INFO] Found {len(final_cam_ids)} common cameras: {final_cam_ids}")

    # Step 2: Synchronize Timestamps
    sync_low = get_synchronized_timestamps(final_cam_dirs_low)
    sync_high = get_synchronized_timestamps(final_cam_dirs_high, require_depth=False)

    timeline_common = np.intersect1d(sync_low.timeline, sync_high.timeline)
    if timeline_common.size == 0:
        print("[ERROR] No overlapping synchronized timeline found.")
        return
    
    idx_map_low = {t: i for i, t in enumerate(sync_low.timeline)}
    idx_map_high = {t: i for i, t in enumerate(sync_high.timeline)}
    indices_low = [idx_map_low[t] for t in timeline_common]
    indices_high = [idx_map_high[t] for t in timeline_common]

    per_cam_low = [ts[indices_low] for ts in sync_low.per_camera_timestamps]
    per_cam_high = [ts[indices_high] for ts in sync_high.per_camera_timestamps]

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