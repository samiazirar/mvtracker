#!/usr/bin/env python3
"""
Processes RH20T task folders to generate sparse, high-resolution depth maps by
reprojecting a point cloud from low-resolution depth onto high-resolution views.

This script uses a robust multi-view fusion approach:
1. A dense point cloud is generated from EACH low-resolution camera view using
   a method identical to the `demo.py` script (direct unprojection).
2. These individual clouds are merged into a single, dense "before" cloud.
3. The merged cloud is cleaned with a radius outlier filter.
4. This final, high-quality cloud is reprojected onto high-resolution views
   using a method with an optional color alignment check.
5. A sparse "after" point cloud is generated from the result for comparison.
"""

#TODO: Redo from scratch

# Standard library imports
import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Third-party imports
import numpy as np
import torch
import torch.nn.functional as F
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

@dataclass
class SyncResult:
    """Container for synchronized timeline information."""
    timeline: np.ndarray
    per_camera_timestamps: List[np.ndarray]
    achieved_fps: float
    target_fps: float
    dropped_ratio: float
    warnings: List[str]
    camera_indices: List[int]

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
    if base_width <= 0 or base_height <= 0: raise ValueError("Base resolution must be positive.")
    K = raw_K[:, :3].astype(np.float32, copy=True)
    K[0, 0] *= width / float(base_width)
    K[0, 2] *= width / float(base_width)
    K[1, 1] *= height / float(base_height)
    K[1, 2] *= height / float(base_height)
    return K

def infer_calibration_resolution(scene: RH20TScene, camera_id: str) -> Optional[Tuple[int, int]]:
    """Reads the calibration image resolution for a given camera if available."""
    calib_dir = Path(scene.calib_folder) if hasattr(scene, 'calib_folder') else None
    if calib_dir and (calib_path := calib_dir / "imgs" / f"cam_{camera_id}_c.png").exists():
        with Image.open(calib_path) as img:
            return img.size
    return None

def _resolve_frame(frame_map: Dict[int, Path], ts: int, cam_name: str, label: str) -> Path:
    """Finds the closest frame to a given timestamp if an exact match isn't found."""
    path = frame_map.get(ts)
    if path is not None:
        return path

    if not frame_map:
        raise KeyError(f"No frames available for {cam_name} ({label}).")

    closest_ts = min(frame_map.keys(), key=lambda k: abs(k - ts))
    delta = abs(closest_ts - ts)
    if delta > 100: # Warn if the closest frame is more than 100ms away
        print(f"[WARN] Timestamp {ts} not found for {cam_name} ({label}); using closest {closest_ts} (|Δ|={delta} ms).")
    return frame_map[closest_ts]


# --- Data Loading & Synchronization ---

def load_scene_data(task_path: Path, robot_configs: List) -> Tuple[Optional[RH20TScene], List[str], List[Path]]:
    """Loads an RH20T scene and filters for calibrated, external (non-hand) cameras."""
    try:
        scene = RH20TScene(str(task_path), robot_configs)
    except Exception as e:
        print(f"[ERROR] Failed to load scene from {task_path}: {e}")
        return None, [], []

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

def get_synchronized_timestamps(
    cam_dirs: List[Path],
    frame_rate_hz: float = 10.0,
    jitter_tolerance_ms: Optional[float] = None,
    require_depth: bool = True,
) -> SyncResult:
    """Synchronize camera streams onto a uniform timeline with tolerance-based matching."""
    if not cam_dirs:
        return SyncResult([], [], 0.0, frame_rate_hz, 1.0, ["No camera directories provided."], [])

    potentially_good_cameras = []
    for idx, cdir in enumerate(cam_dirs):
        color_map = list_frames(cdir / "color")
        if not color_map:
            continue
        
        valid_ts = sorted(color_map.keys())
        if require_depth:
            depth_map = list_frames(cdir / "depth")
            valid_ts = sorted(set(valid_ts).intersection(depth_map.keys()))
        
        if valid_ts:
            potentially_good_cameras.append({"dir": cdir, "timestamps": np.asarray(valid_ts, dtype=np.int64), "idx": idx})

    if len(potentially_good_cameras) < 1:
        return SyncResult([], [], 0.0, frame_rate_hz, 1.0, ["No cameras with valid frames found."], [])

    ts_lists = [cam["timestamps"] for cam in potentially_good_cameras]
    consensus_start = int(max(ts[0] for ts in ts_lists if len(ts) > 0))
    consensus_end = int(min(ts[-1] for ts in ts_lists if len(ts) > 0))

    if consensus_start >= consensus_end:
        return SyncResult([], [], 0.0, frame_rate_hz, 1.0, ["No overlapping recording time found."], [])

    step_ms = max(int(round(1000.0 / frame_rate_hz)), 1)
    tolerance_ms = int(jitter_tolerance_ms) if jitter_tolerance_ms is not None else max(int(step_ms * 0.5), 1)
    
    grid = np.arange(consensus_start, consensus_end + step_ms, step_ms, dtype=np.int64)
    
    aligned = [[] for _ in ts_lists]
    accepted_grid = []
    
    for g in grid:
        slot_matches = []
        for ci, arr in enumerate(ts_lists):
            # Find closest timestamp in arr to grid time g
            idx = np.searchsorted(arr, g)
            candidates = []
            if idx < len(arr): candidates.append(idx)
            if idx > 0: candidates.append(idx - 1)
            
            if not candidates:
                break # This camera has no more frames
            
            best_idx = min(candidates, key=lambda i: abs(arr[i] - g))
            if abs(arr[best_idx] - g) <= tolerance_ms:
                slot_matches.append(arr[best_idx])
            else:
                break
        
        if len(slot_matches) == len(ts_lists):
            accepted_grid.append(g)
            for i in range(len(ts_lists)):
                aligned[i].append(slot_matches[i])

    timeline = np.asarray(accepted_grid, dtype=np.int64)
    per_cam_ts = [np.asarray(vals, dtype=np.int64) for vals in aligned]
    
    duration_s = (consensus_end - consensus_start) / 1000.0
    achieved_fps = len(timeline) / duration_s if duration_s > 0 else 0
    dropped_ratio = 1.0 - (len(timeline) / len(grid)) if len(grid) > 0 else 0

    print(f"\n[SUCCESS] Synchronized {len(timeline)} frames at ~{achieved_fps:.2f} FPS.")
    
    return SyncResult(
        timeline=timeline,
        per_camera_timestamps=per_cam_ts,
        achieved_fps=achieved_fps,
        target_fps=frame_rate_hz,
        dropped_ratio=dropped_ratio,
        warnings=[],
        camera_indices=[cam['idx'] for cam in potentially_good_cameras]
    )

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

def create_dense_point_cloud_from_view(
    depth: np.ndarray,
    rgb: np.ndarray,
    K: np.ndarray,
    E_inv: np.ndarray
) -> o3d.geometry.PointCloud:
    """
    Replicates demo.py's dense point cloud generation by unprojecting every valid
    depth pixel without any advanced filtering.
    """
    H, W = depth.shape
    valid_mask = depth > 0
    if not np.any(valid_mask):
        return o3d.geometry.PointCloud()

    ys, xs = np.indices((H, W))
    pixel_coords = np.stack([xs[valid_mask], ys[valid_mask], np.ones(np.count_nonzero(valid_mask))], axis=0).astype(np.float32)

    K_inv = np.linalg.inv(K)
    cam_points = (K_inv @ pixel_coords) * depth[valid_mask]
    cam_points_h = np.vstack((cam_points, np.ones(cam_points.shape[1], dtype=np.float32)))
    world_points = (E_inv @ cam_points_h)[:3].T.astype(np.float64)
    colors = rgb[valid_mask].astype(np.float32) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(world_points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def reproject_to_sparse_depth_cv2(
    pcd: o3d.geometry.PointCloud,
    high_res_rgb: np.ndarray,
    K: np.ndarray,
    E: np.ndarray,
    use_color_check: bool,
    color_threshold: float
) -> np.ndarray:
    """
    Projects a point cloud to a sparse depth map with an optional color alignment check.
    """
    H, W, _ = high_res_rgb.shape
    sparse_depth = np.zeros((H, W), dtype=np.float32)
    if not pcd.has_points():
        return sparse_depth

    pts_world = np.asarray(pcd.points)
    R, t = E[:3, :3], E[:3, 3]
    rvec, _ = cv2.Rodrigues(R)
    
    projected_pts, _ = cv2.projectPoints(pts_world, rvec, t, K, distCoeffs=None)
    projected_pts = projected_pts.squeeze(1)

    pts_cam = (R @ pts_world.T + t.reshape(3, 1)).T
    depths = pts_cam[:, 2]

    u, v = projected_pts.T
    mask = (depths > 1e-6) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not np.any(mask):
        return sparse_depth

    u_idx = np.round(u[mask]).astype(int)
    v_idx = np.round(v[mask]).astype(int)
    np.clip(u_idx, 0, W - 1, out=u_idx)
    np.clip(v_idx, 0, H - 1, out=v_idx)
    depth_final = depths[mask]
    
    if use_color_check and pcd.has_colors():
        orig_colors = (np.asarray(pcd.colors)[mask] * 255)
        target_colors = high_res_rgb[v_idx, u_idx]
        color_diff = np.linalg.norm(orig_colors - target_colors, axis=1)
        color_mask = color_diff < color_threshold
        u_idx, v_idx, depth_final = u_idx[color_mask], v_idx[color_mask], depth_final[color_mask]

    # Z-Buffering: Keep only the closest point for each pixel
    sorted_indices = np.argsort(depth_final)
    u_sorted, v_sorted, d_sorted = u_idx[sorted_indices], v_idx[sorted_indices], depth_final[sorted_indices]
    _, unique_indices = np.unique(np.stack([v_sorted, u_sorted], axis=1), axis=0, return_index=True)
    
    sparse_depth[v_sorted[unique_indices], u_sorted[unique_indices]] = d_sorted[unique_indices]
    return sparse_depth

def create_pcd_from_sparse_depth(
    depth_map: np.ndarray,
    rgb: np.ndarray,
    K: np.ndarray,
    E_world_to_cam: np.ndarray
) -> o3d.geometry.PointCloud:
    """Converts a final sparse depth map back into a point cloud for visualization."""
    if E_world_to_cam.shape == (4, 4):
        E = E_world_to_cam.astype(np.float64, copy=True)
    else:
        E = np.eye(4, dtype=np.float64)
        E[:3, :4] = E_world_to_cam
    cam_to_world = np.linalg.inv(E)
    
    depth_image = o3d.geometry.Image(depth_map.astype(np.float32))
    color_image = o3d.geometry.Image(rgb)
    H, W = depth_map.shape
    intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_image, depth_image, depth_scale=1.0, depth_trunc=10.0, convert_rgb_to_intensity=False
    )
    
    pcd_cam = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    return pcd_cam.transform(cam_to_world)


# --- Main Workflow & Orchestration ---
def process_frames(
    args, scene_low, scene_high, final_cam_ids, cam_dirs_low, cam_dirs_high,
    timeline: np.ndarray, per_cam_low_ts: List[np.ndarray], per_cam_high_ts: Optional[List[np.ndarray]]
):
    """Iterates through timestamps, fuses point clouds, and generates final data arrays."""
    C, T = len(final_cam_ids), len(timeline)
    is_l515_flags = [cid.startswith('f') for cid in final_cam_ids]

    color_lookup_low = [list_frames(cdir / 'color') for cdir in cam_dirs_low]
    depth_lookup_low = [list_frames(cdir / 'depth') for cdir in cam_dirs_low]
    color_lookup_high = [list_frames(cdir / 'color') for cdir in cam_dirs_high] if cam_dirs_high else None

    # Pre-calculate scaled intrinsics and output shapes
    def _sample_frame_path(frame_lookup: Dict[int, Path], timestamps: Optional[np.ndarray], cid: str, label: str) -> Path:
        """Return a representative frame path for calibration/scaling checks."""
        if frame_lookup:
            if timestamps is not None and len(timestamps) > 0:
                try:
                    return _resolve_frame(frame_lookup, int(timestamps[0]), cid, label)
                except KeyError:
                    pass
            return next(iter(frame_lookup.values()))
        raise RuntimeError(f"No frames found for camera {cid} ({label}).")

    scaled_low_intrinsics, low_shapes = [], []
    for ci, cid in enumerate(final_cam_ids):
        low_path = _sample_frame_path(color_lookup_low[ci], per_cam_low_ts[ci], cid, "low-res color")
        h, w, _ = read_rgb(low_path).shape; low_shapes.append((h, w))
        base_w, base_h = infer_calibration_resolution(scene_low, cid) or (w, h)
        scaled_low_intrinsics.append(scale_intrinsics_matrix(scene_low.intrinsics[cid], w, h, base_w, base_h))

    scaled_high_intrinsics, high_shapes = [], []
    if args.high_res_folder and color_lookup_high:
        for ci, cid in enumerate(final_cam_ids):
            high_ts = per_cam_high_ts[ci] if per_cam_high_ts else None
            high_path = _sample_frame_path(color_lookup_high[ci], high_ts, cid, "high-res color")
            h, w, _ = read_rgb(high_path).shape; high_shapes.append((h, w))
            base_w, base_h = infer_calibration_resolution(scene_high, cid) or infer_calibration_resolution(scene_low, cid) or (w, h)
            scaled_high_intrinsics.append(scale_intrinsics_matrix(scene_high.intrinsics[cid], w, h, base_w, base_h))

    H_out, W_out = (high_shapes[0] if high_shapes else low_shapes[0])
    rgbs_out = np.zeros((C, T, H_out, W_out, 3), dtype=np.uint8)
    depths_out = np.zeros((C, T, H_out, W_out), dtype=np.float32)
    intrs_out = np.zeros((C, T, 3, 3), dtype=np.float32)
    extrs_out = np.zeros((C, T, 3, 4), dtype=np.float32)

    before_pcd_entries, after_pcd_entries = [], []

    for ti in tqdm(range(T), desc="Processing Frames"):
        pcds_for_frame = []
        for ci in range(C):
            cid = final_cam_ids[ci]
            t_low = int(per_cam_low_ts[ci][ti])
            depth_low = read_depth(_resolve_frame(depth_lookup_low[ci], t_low, cid, "low-res depth"), is_l515_flags[ci])
            rgb_low = read_rgb(_resolve_frame(color_lookup_low[ci], t_low, cid, "low-res color"))
            K_low = scaled_low_intrinsics[ci]
            E_inv = np.linalg.inv(scene_low.extrinsics_base_aligned[cid])
            pcd_view = create_dense_point_cloud_from_view(depth_low, rgb_low, K_low, E_inv)
            if pcd_view.has_points():
                pcds_for_frame.append(pcd_view)

        if not pcds_for_frame: continue

        before_pcd = o3d.geometry.PointCloud()
        for pcd in pcds_for_frame: before_pcd += pcd
        before_pcd_entries.append((ti, int(timeline[ti]), "fused", before_pcd))

        pcd_to_reproject = before_pcd
        if args.clean_pointcloud:
            pcd_to_reproject, _ = before_pcd.remove_radius_outlier(nb_points=args.pc_clean_min_points, radius=args.pc_clean_radius)

        if not args.high_res_folder:
            print("[INFO] No high-res folder provided. Skipping reprojection.")
            continue

        for ci in range(C):
            cid = final_cam_ids[ci]
            E_world_to_cam = scene_high.extrinsics_base_aligned[cid]
            extrs_out[ci, ti] = E_world_to_cam[:3, :4]
            t_high = int(per_cam_high_ts[ci][ti]) if per_cam_high_ts else int(per_cam_low_ts[ci][ti])
            high_res_rgb = read_rgb(_resolve_frame(color_lookup_high[ci], t_high, cid, "high-res color"))
            K_high = scaled_high_intrinsics[ci]
            intrs_out[ci, ti] = K_high
            rgbs_out[ci, ti] = high_res_rgb
            sparse_depth = reproject_to_sparse_depth_cv2(pcd_to_reproject, high_res_rgb, K_high, E_world_to_cam, args.color_alignment_check, args.color_threshold)
            depths_out[ci, ti] = sparse_depth
            after_pcd = create_pcd_from_sparse_depth(sparse_depth, high_res_rgb, K_high, E_world_to_cam)
            if after_pcd.has_points():
                after_pcd_entries.append((ti, int(timeline[ti]), cid, after_pcd))

    return rgbs_out, depths_out, intrs_out, extrs_out, before_pcd_entries, after_pcd_entries

def _log_pointcloud_entries(entries: List[Tuple[int, int, str, o3d.geometry.PointCloud]], root: str):
    """Helper function to log a list of point clouds to Rerun."""
    for frame_idx, timestamp, cid, pcd in entries:
        rr.set_time_sequence("frame_idx", frame_idx)
        rr.set_time_seconds("timestamp", timestamp / 1000.0)
        entity = f"{root}/{cid}"
        points = np.asarray(pcd.points)
        if points.size > 0:
            colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
            rr.log(entity, rr.Points3D(points, colors=colors, radii=0.005))

def save_and_visualize(
    args, rgbs, depths, intrs, extrs, final_cam_ids, timestamps, per_camera_timestamps,
    before_pcd_entries: List[Tuple[int, int, str, o3d.geometry.PointCloud]],
    after_pcd_entries: List[Tuple[int, int, str, o3d.geometry.PointCloud]],
):
    """Saves data to NPZ and generates Rerun visualizations."""
    out_path_npz = args.out_dir / f"{args.task_folder.name}_processed.npz"
    np.savez_compressed(out_path_npz,
        rgbs=np.moveaxis(rgbs, -1, 2),
        depths=depths[:, :, np.newaxis, :, :],
        intrs=intrs,
        extrs=extrs,
        timestamps=timestamps,
        per_camera_timestamps=np.stack(per_camera_timestamps, axis=0),
        camera_ids=np.array(final_cam_ids, dtype=object),
    )
    print(f"✅ [OK] Wrote NPZ file to: {out_path_npz}")

    if not args.no_pointcloud:
        if before_pcd_entries:
            rr.init("rh20t_before_reprojection", spawn=False)
            _log_pointcloud_entries(before_pcd_entries, "dense_point_cloud")
            before_rrd_path = args.out_dir / f"{args.task_folder.name}_before.rrd"
            rr.save(str(before_rrd_path)); print(f"✅ [OK] Saved 'before' visualization to: {before_rrd_path}")
            rr.disconnect()

        rr.init("rh20t_after_reprojection", spawn=False)
        if after_pcd_entries:
            _log_pointcloud_entries(after_pcd_entries, "reprojected_point_clouds")

        for ti, ts_val in enumerate(timestamps):
            rr.set_time_sequence("frame_idx", ti)
            rr.set_time_seconds("timestamp", ts_val / 1000.0)
            for ci, cid in enumerate(final_cam_ids):
                H, W = rgbs.shape[2], rgbs.shape[3]
                E = np.eye(4); E[:3, :4] = extrs[ci, ti]
                cam_to_world = np.linalg.inv(E)
                rr.log(f"cameras/{cid}", rr.Pinhole(image_from_camera=intrs[ci, ti], width=W, height=H))
                rr.log(f"cameras/{cid}", rr.Transform3D(translation=cam_to_world[:3, 3], mat3x3=cam_to_world[:3, :3]))

        after_rrd_path = args.out_dir / f"{args.task_folder.name}_after.rrd"
        rr.save(str(after_rrd_path)); print(f"✅ [OK] Saved 'after' visualization to: {after_rrd_path}")
        rr.disconnect()

def main():
    """Parses arguments and orchestrates the entire data processing workflow."""
    parser = argparse.ArgumentParser(description="Process RH20T data by reprojecting a dense, fused low-res point cloud.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--task-folder", required=True, type=Path, help="Path to the primary (low-res) RH20T task folder.")
    parser.add_argument("--high-res-folder", type=Path, help="Path to the high-resolution task folder for reprojection.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for .npz and .rrd files.")
    parser.add_argument("--config", default="rh20t_api/configs/configs.json", type=Path, help="Path to RH20T robot configs JSON.")
    parser.add_argument("--max-frames", type=int, default=50, help="Limit frames to process (0 for all).")
    parser.add_argument("--frame-selection", choices=["first", "last", "middle"], default="middle", help="Method for selecting frames.")
    
    parser.add_argument("--color-alignment-check", action='store_true', default=True, help="Enable color-based filtering of reprojected points.")
    parser.add_argument("--no-color-alignment-check", action='store_false', dest='color_alignment_check', help="Disable color-based filtering.")
    parser.add_argument("--color-threshold", type=float, default=35.0, help="Max Euclidean distance in RGB space for a point to be aligned.")
    
    parser.add_argument("--clean-pointcloud", action="store_true", help="Apply radius-based cleaning to the fused point cloud before reprojection.")
    parser.add_argument("--pc-clean-radius", type=float, default=0.03, help="Radius (m) for the Open3D radius outlier removal filter.")
    parser.add_argument("--pc-clean-min-points", type=int, default=15, help="Minimum neighbors within radius to keep a point during cleaning.")

    parser.add_argument("--no-pointcloud", action="store_true", help="Only generate the .npz file, skip Rerun visualization.")
    parser.add_argument("--sync-fps", type=float, default=10.0, help="Target FPS for synchronization output timeline.")
    parser.add_argument("--sync-tolerance-ms", type=float, default=50.0, help="Max timestamp deviation (ms) for matching frames.")
    
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    robot_configs = load_conf(str(args.config))
    
    scene_low, cam_ids_low, cam_dirs_low = load_scene_data(args.task_folder, robot_configs)
    if not scene_low or not cam_ids_low: return

    sync_kwargs = dict(frame_rate_hz=args.sync_fps, jitter_tolerance_ms=args.sync_tolerance_ms)
    
    if args.high_res_folder:
        scene_high, cam_ids_high, cam_dirs_high = load_scene_data(args.high_res_folder, robot_configs)
        if not scene_high or not cam_ids_high: return
        
        shared_ids = sorted(list(set(cam_ids_low) & set(cam_ids_high)))
        id_to_dir_low = dict(zip(cam_ids_low, cam_dirs_low))
        id_to_dir_high = dict(zip(cam_ids_high, cam_dirs_high))

        final_cam_ids = shared_ids
        final_cam_dirs_low = [id_to_dir_low[cid] for cid in final_cam_ids]
        final_cam_dirs_high = [id_to_dir_high[cid] for cid in final_cam_ids]

        sync_low = get_synchronized_timestamps(final_cam_dirs_low, **sync_kwargs)
        sync_high = get_synchronized_timestamps(final_cam_dirs_high, require_depth=False, **sync_kwargs)

        low_idx_map = {orig_idx: pos for pos, orig_idx in enumerate(sync_low.camera_indices)}
        high_idx_map = {orig_idx: pos for pos, orig_idx in enumerate(sync_high.camera_indices)}
        active_indices = sorted(set(low_idx_map.keys()) & set(high_idx_map.keys()))
        if not active_indices:
            print("[ERROR] No cameras have synchronized data in both low and high resolution recordings.")
            return

        final_cam_ids = [final_cam_ids[i] for i in active_indices]
        final_cam_dirs_low = [final_cam_dirs_low[i] for i in active_indices]
        final_cam_dirs_high = [final_cam_dirs_high[i] for i in active_indices]

        per_cam_low_full = [sync_low.per_camera_timestamps[low_idx_map[i]] for i in active_indices]
        per_cam_high_full = [sync_high.per_camera_timestamps[high_idx_map[i]] for i in active_indices]

        timeline_common, idx_low, idx_high = np.intersect1d(sync_low.timeline, sync_high.timeline, return_indices=True)
        if timeline_common.size == 0:
            print("[ERROR] No overlapping synchronized timeline between low and high resolution data.")
            return

        per_cam_low = [ts[idx_low] for ts in per_cam_low_full]
        per_cam_high = [ts[idx_high] for ts in per_cam_high_full]
    else:
        scene_high, final_cam_dirs_high, per_cam_high = None, None, None
        final_cam_ids = cam_ids_low
        final_cam_dirs_low = cam_dirs_low
        sync_low = get_synchronized_timestamps(final_cam_dirs_low, **sync_kwargs)

        if not sync_low.camera_indices:
            print("[ERROR] No cameras with synchronized data found in the low resolution recording.")
            return

        low_idx_map = {orig_idx: pos for pos, orig_idx in enumerate(sync_low.camera_indices)}
        active_indices = sorted(low_idx_map.keys())
        final_cam_ids = [final_cam_ids[i] for i in active_indices]
        final_cam_dirs_low = [final_cam_dirs_low[i] for i in active_indices]
        per_cam_low_full = [sync_low.per_camera_timestamps[low_idx_map[i]] for i in active_indices]

        timeline_common = sync_low.timeline
        per_cam_low = per_cam_low_full

    if timeline_common.size == 0:
        print("[ERROR] Synchronization returned no frames.")
        return

    timestamps = select_frames(timeline_common, args.max_frames, args.frame_selection)
    if timestamps.size == 0:
        print("[ERROR] No frames remaining after selection.")
        return

    # Filter the per-camera timestamps to match the selected timeline
    selected_indices = np.where(np.isin(timeline_common, timestamps))[0]
    per_cam_low_sel = [ts[selected_indices] for ts in per_cam_low]
    per_cam_high_sel = [ts[selected_indices] for ts in per_cam_high] if per_cam_high else None

    rgbs, depths, intrs, extrs, before, after = process_frames(
        args, scene_low, scene_high, final_cam_ids, final_cam_dirs_low, final_cam_dirs_high,
        timestamps, per_cam_low_sel, per_cam_high_sel
    )
    
    save_and_visualize(
        args, rgbs, depths, intrs, extrs, final_cam_ids, timestamps, 
        per_cam_high_sel if per_cam_high_sel else per_cam_low_sel,
        before, after
    )

if __name__ == "__main__":
    main()
