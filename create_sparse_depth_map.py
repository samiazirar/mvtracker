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

python create_sparse_depth_map.py   --task-folder /data/rh20t_api/data/test_data_full_rgb_upscaled_depth/uncompressed_low_res_data/task_0065_user_0010_scene_0009_cfg_0004   --high-res-folder /data/rh20t_api/data/test_data_full_rgb_upscaled_depth/rgb_data/RH20T_cfg4/task_0065_user_0010_scene_0009_cfg_0004  --out-dir ./data/high_res_filtered   --max-frames 100   --color-alignment-check   --color-threshold 35 --use-splatting --splat-mode recolor --splat-radius 0.01
"""


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

# PyTorch3D imports for point splatting
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,  # Changed from FoVPerspectiveCameras
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor,
)
from pytorch3d.renderer.points.pulsar.unified import PulsarPointsRenderer
PYTORCH3D_AVAILABLE = True

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


def scale_intrinsics_matrix(raw_K: np.ndarray, width: int, height: int, base_width: int, base_height: int) -> np.ndarray:
    """Rescales a 3x4 intrinsic matrix to match an image resolution."""
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
    calib_dir = Path(scene.calib_folder) if hasattr(scene, 'calib_folder') else None
    if calib_dir is None:
        return None

    calib_path = calib_dir / "imgs" / f"cam_{camera_id}_c.png"
    if not calib_path.exists():
        return None

    with Image.open(calib_path) as img:
        width, height = img.size
    return width, height

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

def get_synchronized_timestamps(
    cam_dirs: List[Path],
    frame_rate_hz: float = 10.0,
    min_density: float = 0.6,
    target_fps: Optional[float] = None,
    max_fps_drift: float = 0.05,
    jitter_tolerance_ms: Optional[float] = None,
    require_depth: bool = True,
) -> SyncResult:
    """Synchronize camera streams onto a uniform timeline with tolerance-based matching."""

    desired_fps = target_fps if target_fps and target_fps > 0 else frame_rate_hz
    warnings_local: List[str] = []

    if len(cam_dirs) < 2:
        msg = "[WARNING] Less than 2 camera directories provided. Synchronization not possible."
        print(msg)
        warnings_local.append(msg)
        return SyncResult(
            timeline=np.array([], dtype=np.int64),
            per_camera_timestamps=[],
            achieved_fps=0.0,
            target_fps=float(desired_fps),
            dropped_ratio=1.0,
            warnings=warnings_local,
            camera_indices=[],
        )

    potentially_good_cameras = []
    for idx, cdir in enumerate(cam_dirs):
        color_map = list_frames(cdir / "color")
        color_ts = list(color_map.keys())

        depth_ts: List[int]
        if require_depth:
            depth_map = list_frames(cdir / "depth")
            depth_ts = list(depth_map.keys())
            valid_ts = sorted(set(color_ts).intersection(depth_ts))
            if not valid_ts:
                print(f"[INFO] Skipping camera {cdir.name}: No valid color/depth frame pairs found.")
                continue
            print(f"[INFO] Camera {cdir.name}: Found {len(valid_ts)} valid color/depth frame pairs.")
        else:
            valid_ts = sorted(color_ts)
            if not valid_ts:
                print(f"[INFO] Skipping camera {cdir.name}: No color frames found.")
                continue
            print(f"[INFO] Camera {cdir.name}: Found {len(valid_ts)} color frames (depth not required).")

        potentially_good_cameras.append({"dir": cdir, "timestamps": np.asarray(valid_ts, dtype=np.int64), "idx": idx})

    if len(potentially_good_cameras) < 2:
        msg = "[WARNING] Fewer than 2 cameras have valid data. Synchronization not possible."
        print(msg)
        warnings_local.append(msg)
        return SyncResult(
            timeline=np.array([], dtype=np.int64),
            per_camera_timestamps=[],
            achieved_fps=0.0,
            target_fps=float(desired_fps),
            dropped_ratio=1.0,
            warnings=warnings_local,
            camera_indices=[],
        )

    ts_lists = [cam["timestamps"] for cam in potentially_good_cameras]
    consensus_start = int(max(ts_list[0] for ts_list in ts_lists))
    consensus_end = int(min(ts_list[-1] for ts_list in ts_lists))

    if consensus_start >= consensus_end:
        msg = "[WARNING] No overlapping recording time found among valid cameras."
        print(msg)
        warnings_local.append(msg)
        return SyncResult(
            timeline=np.array([], dtype=np.int64),
            per_camera_timestamps=[],
            achieved_fps=0.0,
            target_fps=float(desired_fps),
            dropped_ratio=1.0,
            warnings=warnings_local,
            camera_indices=[],
        )

    duration_ms = consensus_end - consensus_start
    duration_s = duration_ms / 1000.0 if duration_ms > 0 else 0.0
    expected_frames = duration_s * frame_rate_hz if duration_s > 0 else 0.0

    good_cameras = []
    for cam in potentially_good_cameras:
        ts = cam["timestamps"]
        mask = (ts >= consensus_start) & (ts <= consensus_end)
        ts_window = ts[mask]
        frames_in_window = int(ts_window.size)
        density = frames_in_window / expected_frames if expected_frames > 0 else 0.0

        if density < min_density:
            print(
                f"[INFO] Skipping camera {cam['dir'].name}: Failed density check "
                f"({density:.2%} < {min_density:.2%})."
            )
            continue

        fps = frames_in_window / duration_s if duration_s > 0 else 0.0
        median_gap = float(np.median(np.diff(ts_window))) if frames_in_window > 1 else float("inf")
        good_cameras.append(
            {
                "dir": cam["dir"],
                "timestamps": ts_window,
                "fps": fps,
                "median_gap": median_gap,
                "idx": cam["idx"],
            }
        )

    if len(good_cameras) < 2:
        msg = "[WARNING] Fewer than 2 cameras passed all checks. No final synchronization is possible."
        print(msg)
        warnings_local.append(msg)
        return SyncResult(
            timeline=np.array([], dtype=np.int64),
            per_camera_timestamps=[],
            achieved_fps=0.0,
            target_fps=float(desired_fps),
            dropped_ratio=1.0,
            warnings=warnings_local,
            camera_indices=[],
        )

    camera_fps_values = [cam["fps"] for cam in good_cameras if cam["fps"] > 0]
    if (not target_fps or target_fps <= 0) and camera_fps_values:
        desired_fps = min(frame_rate_hz, min(camera_fps_values))
    desired_fps = max(desired_fps, 1e-6)

    step_ms = max(int(round(1000.0 / desired_fps)), 1)
    tolerance_ms = int(jitter_tolerance_ms) if jitter_tolerance_ms else max(int(step_ms * 0.5), 1)

    if duration_ms <= 0:
        grid = np.array([consensus_start], dtype=np.int64)
    else:
        grid = np.arange(consensus_start, consensus_end + step_ms, step_ms, dtype=np.int64)

    per_camera_arrays = [cam["timestamps"] for cam in good_cameras]
    aligned = [[] for _ in per_camera_arrays]
    last_indices = [-1 for _ in per_camera_arrays]
    accepted_grid: List[int] = []
    dropped_slots = 0
    exhausted = False

    for g in grid:
        slot_matches = []
        for ci, arr in enumerate(per_camera_arrays):
            start_idx = last_indices[ci] + 1
            if start_idx >= arr.size:
                exhausted = True
                slot_matches = None
                break

            arr_sub = arr[start_idx:]
            idx = int(np.searchsorted(arr_sub, g))

            candidate_indices = []
            if idx < arr_sub.size:
                candidate_indices.append(start_idx + idx)
            if idx > 0:
                candidate_indices.append(start_idx + idx - 1)

            if not candidate_indices:
                slot_matches = None
                break

            best_idx = min(candidate_indices, key=lambda j: abs(int(arr[j]) - int(g)))
            best_val = int(arr[best_idx])

            if abs(best_val - int(g)) > tolerance_ms:
                slot_matches = None
                break

            slot_matches.append((ci, best_idx, best_val))

        if slot_matches is None:
            if exhausted:
                break
            dropped_slots += 1
            continue

        accepted_grid.append(int(g))
        for ci, best_idx, best_val in slot_matches:
            aligned[ci].append(best_val)
            last_indices[ci] = best_idx

    timeline = np.asarray(accepted_grid, dtype=np.int64)
    per_camera_timestamps = [np.asarray(vals, dtype=np.int64) for vals in aligned]

    if timeline.size == 0:
        msg = "[ERROR] No synchronized timestamps found after tolerance-based matching."
        print(msg)
        warnings_local.append(msg)
        return SyncResult(
            timeline=np.array([], dtype=np.int64),
            per_camera_timestamps=[],
            achieved_fps=0.0,
            target_fps=float(desired_fps),
            dropped_ratio=1.0,
            warnings=warnings_local,
            camera_indices=[],
        )

    if any(len(vals) != timeline.size for vals in per_camera_timestamps):
        raise RuntimeError("Internal synchronization error: camera alignment mismatch.")

    achieved_fps = (timeline.size / duration_s) if duration_s > 0 else 0.0
    dropped_ratio = dropped_slots / grid.size if grid.size > 0 else 0.0

    for cam, aligned_vals in zip(good_cameras, per_camera_timestamps):
        print(
            f"[INFO] Camera {cam['dir'].name}: Aligned {aligned_vals.size} frames "
            f"(median gap {cam['median_gap']:.1f} ms)."
        )

    summary = (
        f"\n[SUCCESS] Synchronized {timeline.size} frames at ~{achieved_fps:.2f} FPS "
        f"(target {desired_fps:.2f} FPS, tolerance {tolerance_ms} ms)."
    )
    print(summary)

    if desired_fps > 0 and achieved_fps < desired_fps * (1 - max_fps_drift):
        msg = (
            f"[WARNING] Achieved FPS ({achieved_fps:.2f}) is more than {max_fps_drift:.0%} "
            f"below target ({desired_fps:.2f})."
        )
        print(msg)
        warnings_local.append(msg)

    if dropped_ratio > max_fps_drift:
        msg = (
            f"[WARNING] Dropped {dropped_ratio:.2%} of timeline slots due to missing frames."
        )
        print(msg)
        warnings_local.append(msg)

    return SyncResult(
        timeline=timeline,
        per_camera_timestamps=per_camera_timestamps,
        achieved_fps=achieved_fps,
        target_fps=float(desired_fps),
        dropped_ratio=float(dropped_ratio),
        warnings=warnings_local,
        camera_indices=[cam['idx'] for cam in good_cameras],
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

def splat_point_cloud(
    pcd: o3d.geometry.PointCloud,
    K: np.ndarray,
    E: np.ndarray,
    image_height: int,
    image_width: int,
    splat_mode: str = "mask",
    splat_radius: float = 0.005,
    device: str = "cuda"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splat a 3D point cloud onto a 2D image using PyTorch3D's efficient point rendering.
    
    Args:
        pcd: Open3D PointCloud in world coordinates
        K: 3x3 camera intrinsic matrix
        E: 3x4 camera extrinsic matrix (world-to-camera)
        image_height: Output image height
        image_width: Output image width
        splat_mode: 'recolor' for RGB rendering, 'mask' for binary mask
        splat_radius: Radius of splats in world units (meters)
        device: PyTorch device ('cuda' or 'cpu')
        
    Returns:
        Tuple of:
        - For 'recolor' mode: RGB image (H, W, 3) as uint8
        - For 'mask' mode: Binary mask (H, W) as uint8 (0 or 255)
        - Depth map (H, W) as float32 (0 for background)
    """
    if not PYTORCH3D_AVAILABLE:
        raise ImportError("PyTorch3D is required for point splatting but is not available.")
    
    if not pcd.has_points():
        # Return empty image/mask and depth if no points
        empty_depth = np.zeros((image_height, image_width), dtype=np.float32)
        if splat_mode == "recolor":
            return np.zeros((image_height, image_width, 3), dtype=np.uint8), empty_depth
        else:
            return np.zeros((image_height, image_width), dtype=np.uint8), empty_depth
    
    # Convert Open3D point cloud to PyTorch tensors
    points_world = torch.tensor(np.asarray(pcd.points), dtype=torch.float32, device=device)
    if pcd.has_colors():
        colors = torch.tensor(np.asarray(pcd.colors), dtype=torch.float32, device=device)
    else:
        # Default to white points if no colors
        colors = torch.ones_like(points_world, dtype=torch.float32, device=device)
    
    # Transform points to camera coordinate system
    R = torch.tensor(E[:3, :3], dtype=torch.float32, device=device)
    t = torch.tensor(E[:3, 3], dtype=torch.float32, device=device)
    points_cam = (R @ points_world.T + t.unsqueeze(1)).T
    
    # Filter points behind the camera
    valid_mask = points_cam[:, 2] > 1e-6
    points_cam = points_cam[valid_mask]
    colors = colors[valid_mask]
    
    if points_cam.shape[0] == 0:
        # No valid points
        empty_depth = np.zeros((image_height, image_width), dtype=np.float32)
        if splat_mode == "recolor":
            return np.zeros((image_height, image_width, 3), dtype=np.uint8), empty_depth
        else:
            return np.zeros((image_height, image_width), dtype=np.uint8), empty_depth
    
    # Create PyTorch3D Pointclouds structure
    # Add batch dimension (batch_size=1)
    points_batch = points_cam.unsqueeze(0)  # (1, N, 3)
    colors_batch = colors.unsqueeze(0)      # (1, N, 3)
    point_cloud = Pointclouds(points=points_batch, features=colors_batch)
    
    # Set up camera parameters for PyTorch3D
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Convert to PyTorch3D format - use PerspectiveCameras instead of FoVPerspectiveCameras
    focal_length = torch.tensor([[fx, fy]], dtype=torch.float32, device=device)
    principal_point = torch.tensor([[cx, cy]], dtype=torch.float32, device=device)
    image_size = torch.tensor([[image_height, image_width]], dtype=torch.float32, device=device)
    
    # Create camera object (identity transformation since points are already in camera space)
    cameras = PerspectiveCameras(
        focal_length=focal_length,
        principal_point=principal_point,
        image_size=image_size,
        device=device
    )
    
    # Configure rasterization settings
    # The radius should be converted to screen space for rendering
    # A rough conversion: splat_radius in world units to pixel radius
    pixel_radius = max(1.0, splat_radius * fx / torch.mean(points_cam[:, 2]).item())
    
    raster_settings = PointsRasterizationSettings(
        image_size=(image_height, image_width),
        radius=pixel_radius,
        points_per_pixel=10,  # Allow multiple points per pixel for proper z-buffering
    )
    
    # Create rasterizer
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    
    # Get fragments from rasterizer to extract Z-buffer
    with torch.no_grad():
        fragments = rasterizer(point_cloud, cameras=cameras)
    
    # Extract depth map from Z-buffer
    zbuf = fragments.zbuf[0, ..., 0].cpu().numpy()  # Remove batch dimension and get first layer
    depth_map = zbuf.copy().astype(np.float32)
    depth_map[depth_map < 0] = 0.0  # Set background pixels (marked as -1) to 0
    
    # Choose compositor based on mode
    if splat_mode == "recolor":
        # Use alpha compositor for color blending with proper depth testing
        compositor = AlphaCompositor(background_color=(0.0, 0.0, 0.0))
    else:  # mask mode
        # Use norm-weighted compositor to create coverage mask
        compositor = NormWeightedCompositor(background_color=(0.0, 0.0, 0.0))
    
    # Create renderer
    renderer = PointsRenderer(rasterizer=rasterizer, compositor=compositor)
    
    # Render the point cloud
    with torch.no_grad():
        rendered_image = renderer(point_cloud, cameras=cameras)
    
    # Convert output to numpy
    rendered_np = rendered_image[0].cpu().numpy()  # Remove batch dimension
    
    if splat_mode == "recolor":
        # Convert to uint8 RGB image
        rgb_output = (rendered_np[..., :3] * 255).astype(np.uint8)
        return rgb_output, depth_map
    else:  # mask mode
        # Create binary mask: any pixel with non-zero values becomes 1
        mask = (rendered_np[..., :3].sum(axis=-1) > 1e-6).astype(np.uint8) * 255
        return mask, depth_map

def _confidence_from_depth(depth: np.ndarray) -> np.ndarray:
    """Create confidence mask from depth (like demo.py)."""
    conf = np.ones_like(depth, dtype=np.float32)
    conf[~np.isfinite(depth)] = 0
    conf[depth <= 0] = 0
    return conf

def _smooth_depth_with_weights(depth: np.ndarray, weights: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Smooth depth with confidence weights (like demo.py)."""
    if kernel_size < 1:
        return depth
    
    # Convert to torch tensors for processing
    depth_torch = torch.tensor(depth, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    weights_torch = torch.tensor(weights, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    padding = kernel_size // 2
    kernel = torch.ones(1, 1, kernel_size, kernel_size, dtype=torch.float32)
    
    weighted_depth = F.conv2d(depth_torch * weights_torch, kernel, padding=padding)
    summed_weights = F.conv2d(weights_torch, kernel, padding=padding).clamp_min(1e-6)
    smoothed = (weighted_depth / summed_weights).squeeze().numpy()
    
    return smoothed

def create_confidence_filtered_point_cloud(
    depth: np.ndarray, 
    rgb: np.ndarray, 
    K: np.ndarray, 
    E_inv: np.ndarray,
    confidence_threshold: float = 0.1,
    edge_margin: int = 10,
    gradient_threshold: float = 0.1,
    smooth_kernel: int = 3
) -> o3d.geometry.PointCloud:
    """
    Create point cloud with confidence filtering like demo.py.
    
    This matches the depth preprocessing approach used in demo.py for high-quality
    point cloud generation.
    
    Args:
        depth: The depth map (H, W).
        rgb: The color image (H, W, 3).
        K: The 3x3 intrinsic camera matrix.
        E_inv: The 4x4 inverse extrinsic matrix (camera-to-world transformation).
        confidence_threshold: Minimum confidence to keep a point
        edge_margin: Pixels to remove from image borders
        gradient_threshold: Maximum depth gradient to keep (meters)
        smooth_kernel: Kernel size for depth smoothing
        
    Returns:
        An Open3D PointCloud object in world coordinates with confidence filtering.
    """
    H, W = depth.shape
    
    # Step 1: Create confidence mask (like demo.py)
    confidence = _confidence_from_depth(depth)
    
    # Step 2: Edge filtering - remove boundary pixels (often unreliable)
    if edge_margin > 0:
        confidence[:edge_margin, :] = 0
        confidence[-edge_margin:, :] = 0  
        confidence[:, :edge_margin] = 0
        confidence[:, -edge_margin:] = 0
    
    # Step 3: Gradient-based edge detection (high depth gradients = unreliable)
    depth_grad_x = np.abs(np.gradient(depth, axis=1))
    depth_grad_y = np.abs(np.gradient(depth, axis=0))
    depth_gradient = np.sqrt(depth_grad_x**2 + depth_grad_y**2)
    
    # Remove high-gradient pixels (depth discontinuities)
    confidence[depth_gradient > gradient_threshold] = 0
    
    # Step 4: Apply confidence threshold
    valid_mask = confidence > confidence_threshold
    
    # Step 5: Create filtered depth map
    filtered_depth = depth.copy()
    filtered_depth[~valid_mask] = 0
    
    # Step 6: Smooth depth with confidence weights (like demo.py)
    if smooth_kernel > 1:
        filtered_depth = _smooth_depth_with_weights(filtered_depth, confidence, smooth_kernel)
        # Re-apply the mask after smoothing
        filtered_depth[~valid_mask] = 0
    
    # Step 7: Create point cloud with filtered data
    o3d_depth = o3d.geometry.Image(filtered_depth)
    o3d_rgb = o3d.geometry.Image(rgb)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_rgb, o3d_depth, depth_scale=1.0, depth_trunc=10.0, convert_rgb_to_intensity=False
    )
    
    intrinsics = o3d.camera.PinholeCameraIntrinsic(W, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    pcd_cam = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
    
    # Transform to world coordinates
    return pcd_cam.transform(E_inv)

def unproject_to_world_o3d(depth: np.ndarray, rgb: np.ndarray, K: np.ndarray, E_inv: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Creates a colored point cloud in world coordinates from a single view using Open3D.
    
    This is the legacy function - use create_confidence_filtered_point_cloud for demo.py-style processing.

    Args:
        depth: The depth map (H, W).
        rgb: The color image (H, W, 3).
        K: The 3x3 intrinsic camera matrix.
        E_inv: The 4x4 inverse extrinsic matrix (camera-to-world transformation).

    Returns:
        An Open3D PointCloud object in world coordinates.
    """
    return create_confidence_filtered_point_cloud(depth, rgb, K, E_inv)


def clean_point_cloud_radius(pcd: o3d.geometry.PointCloud, radius: float, min_points: int) -> o3d.geometry.PointCloud:
    """Radius-based outlier removal leveraging Open3D's built-in filter."""
    if not pcd.has_points():
        return pcd

    _, ind = pcd.remove_radius_outlier(nb_points=max(1, int(min_points)), radius=float(radius))

    if len(ind) == 0:
        print("[WARN] Radius-based point cloud cleaning removed all points; keeping original cloud.")
        return pcd

    return pcd.select_by_index(ind)


def reconstruct_mesh_from_pointcloud(pcd: o3d.geometry.PointCloud, depth: int) -> Optional[o3d.geometry.TriangleMesh]:
    """Reconstructs a mesh via Poisson surface reconstruction to sharpen geometry."""
    if not pcd.has_points():
        return None

    pcd_for_mesh = o3d.geometry.PointCloud(pcd)
    pcd_for_mesh.estimate_normals()

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_for_mesh, depth=int(depth))

    densities_arr = np.asarray(densities)
    if densities_arr.size:
        density_thresh = np.quantile(densities_arr, 0.01)
        mask = densities_arr < density_thresh
        mesh.remove_vertices_by_mask(mask)

    if len(mesh.vertices) == 0:
        return None

    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


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
        raise ValueError("Point cloud has no points; cannot reproject to depth map.")
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
    u_idx = np.round(u[bounds_mask]).astype(np.int32)
    v_idx = np.round(v[bounds_mask]).astype(np.int32)
    np.clip(u_idx, 0, W - 1, out=u_idx)
    np.clip(v_idx, 0, H - 1, out=v_idx)
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

def process_frames(
    args,
    scene_low,
    scene_high,
    final_cam_ids,
    cam_dirs_low,
    cam_dirs_high,
    timeline: np.ndarray,
    per_cam_low_ts: List[np.ndarray],
    per_cam_high_ts: Optional[List[np.ndarray]],
):
    """
    Iterates through timestamps to process frames, generate point clouds,
    and create the final data arrays (RGB, depth, intrinsics, extrinsics).
    """
    C, T = len(final_cam_ids), len(timeline)
    is_l515_flags = [cid.startswith('f') for cid in final_cam_ids]

    color_lookup_low = [list_frames(cdir / 'color') for cdir in cam_dirs_low]
    depth_lookup_low = [list_frames(cdir / 'depth') for cdir in cam_dirs_low]

    if args.high_res_folder and cam_dirs_high:
        color_lookup_high = [list_frames(cdir / 'color') for cdir in cam_dirs_high]
    else:
        color_lookup_high = None

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

    scaled_low_intrinsics: List[np.ndarray] = []
    low_shapes: List[Tuple[int, int]] = []
    for ci in range(C):
        cid = final_cam_ids[ci]
        first_low_ts = int(per_cam_low_ts[ci][0])
        low_path = _resolve_frame(color_lookup_low[ci], first_low_ts, cid, "low-res color")
        low_img = read_rgb(low_path)
        h_low, w_low = low_img.shape[:2]
        low_shapes.append((h_low, w_low))
        base_res = infer_calibration_resolution(scene_low, cid)
        if base_res is None:
            base_res = (max(1, int(round(scene_low.intrinsics[cid][0, 2] * 2))),
                        max(1, int(round(scene_low.intrinsics[cid][1, 2] * 2))))
            print(
                f"[WARN] Calibration image for camera {cid} not found; "
                f"estimating base resolution as {base_res[0]}x{base_res[1]} using intrinsics."
            )
        base_w, base_h = base_res
        scaled_low_intrinsics.append(scale_intrinsics_matrix(scene_low.intrinsics[cid], w_low, h_low, base_w, base_h))

    scaled_high_intrinsics: Optional[List[np.ndarray]] = None
    high_shapes: Optional[List[Tuple[int, int]]] = None
    if args.high_res_folder and color_lookup_high:
        scaled_high_intrinsics = []
        high_shapes = []
        for ci in range(C):
            cid = final_cam_ids[ci]
            first_high_ts = int(per_cam_high_ts[ci][0]) if per_cam_high_ts else int(per_cam_low_ts[ci][0])
            high_path = _resolve_frame(color_lookup_high[ci], first_high_ts, cid, "high-res color")
            high_img = read_rgb(high_path)
            h_high, w_high = high_img.shape[:2]
            high_shapes.append((h_high, w_high))
            base_res = infer_calibration_resolution(scene_high, cid) or infer_calibration_resolution(scene_low, cid)
            if base_res is None:
                base_res = (max(1, int(round(scene_high.intrinsics[cid][0, 2] * 2))),
                            max(1, int(round(scene_high.intrinsics[cid][1, 2] * 2))))
                print(
                    f"[WARN] Calibration image for camera {cid} not found in high-res scene; "
                    f"estimating base resolution as {base_res[0]}x{base_res[1]} using intrinsics."
                )
            base_w, base_h = base_res
            scaled_high_intrinsics.append(scale_intrinsics_matrix(scene_high.intrinsics[cid], w_high, h_high, base_w, base_h))

    # Determine output resolution and initialize data containers
    if args.high_res_folder and high_shapes:
        H_out, W_out = high_shapes[0]
        print(f"[INFO] Outputting high-resolution data ({H_out}x{W_out}) with reprojected depth.")
    else:
        H_out, W_out = low_shapes[0]
        print(f"[INFO] Outputting low-resolution data ({H_out}x{W_out}).")

    rgbs_out = np.zeros((C, T, H_out, W_out, 3), dtype=np.uint8)
    depths_out = np.zeros((C, T, H_out, W_out), dtype=np.float32)
    intrs_out = np.zeros((C, T, 3, 3), dtype=np.float32)
    extrs_out = np.zeros((C, T, 3, 4), dtype=np.float32)

    for ti in tqdm(range(T), desc="Processing Frames"):
        # Step 1: Create a high-quality point cloud using demo.py approach (single best camera)
        combined_pcd = o3d.geometry.PointCloud()
        if args.high_res_folder:
            # Find the camera with the most reliable depth for this frame (like demo.py)
            best_ci = 0  # Start with first camera
            best_depth_coverage = 0
            
            for ci in range(C):
                cid = final_cam_ids[ci]
                t_low = int(per_cam_low_ts[ci][ti])
                depth_low = read_depth(_resolve_frame(depth_lookup_low[ci], t_low, cid, "low-res depth"), is_l515_flags[ci])
                
                # Calculate depth coverage (how much valid depth we have)
                valid_depth_ratio = np.sum(depth_low > 0) / depth_low.size
                
                if valid_depth_ratio > best_depth_coverage:
                    best_depth_coverage = valid_depth_ratio
                    best_ci = ci
            
            # Create point cloud from the best camera view only (like demo.py)
            cid = final_cam_ids[best_ci]
            t_low = int(per_cam_low_ts[best_ci][ti])
            depth_low = read_depth(_resolve_frame(depth_lookup_low[best_ci], t_low, cid, "low-res depth"), is_l515_flags[best_ci])
            rgb_low = read_rgb(_resolve_frame(color_lookup_low[best_ci], t_low, cid, "low-res color"))
            K_low = scaled_low_intrinsics[best_ci]
            E_inv = np.linalg.inv(scene_low.extrinsics_base_aligned[cid])
            
            # Use confidence-filtered point cloud creation (like demo.py)
            combined_pcd = create_confidence_filtered_point_cloud(
                depth_low, rgb_low, K_low, E_inv, 
                confidence_threshold=args.confidence_threshold,
                edge_margin=args.edge_margin,
                gradient_threshold=args.gradient_threshold,
                smooth_kernel=args.smooth_kernel
            )
            
            print(f"[INFO] Frame {ti}: Using camera {cid} with {best_depth_coverage:.1%} depth coverage")
            print(f"[INFO] Created confidence-filtered point cloud with {len(combined_pcd.points)} points")

            # Apply additional cleaning using demo.py-style approach
            if combined_pcd.has_points():
                initial_count = len(combined_pcd.points)
                
                # Statistical outlier removal (like demo.py uses)
                if args.clean_pointcloud:
                    print(f"[INFO] Applying statistical outlier removal (demo.py style)...")
                    cleaned_pcd, outlier_indices = combined_pcd.remove_statistical_outlier(
                        nb_neighbors=20, std_ratio=2.0
                    )
                    if len(cleaned_pcd.points) > initial_count * 0.3:  # Keep at least 30%
                        combined_pcd = cleaned_pcd
                        print(f"[INFO] Statistical cleaning: {initial_count} → {len(combined_pcd.points)} points")
                    else:
                        print(f"[WARN] Statistical cleaning would remove too many points, keeping original")
                
                print(f"[INFO] Final high-quality point cloud has {len(combined_pcd.points)} points for reprojection")

        # Step 2: Generate the final output data for each camera view
        for ci in range(C):
            cid = final_cam_ids[ci]

            if args.high_res_folder and scene_high is not None:
                # Use high-res calibration when available; low-res version is only for point-cloud generation.
                E_world_to_cam = scene_high.extrinsics_base_aligned[cid]
            else:
                E_world_to_cam = scene_low.extrinsics_base_aligned[cid]

            extrs_out[ci, ti] = E_world_to_cam[:3, :4]
            
            if args.high_res_folder:
                # Reprojection workflow
                t_high = int(per_cam_high_ts[ci][ti]) if per_cam_high_ts else int(per_cam_low_ts[ci][ti])
                high_res_rgb = read_rgb(_resolve_frame(color_lookup_high[ci], t_high, cid, "high-res color"))
                K_high = scaled_high_intrinsics[ci] if scaled_high_intrinsics is not None else scaled_low_intrinsics[ci]
                intrs_out[ci, ti] = K_high
                
                if args.use_splatting:
                    # Use PyTorch3D point splatting instead of sparse depth reprojection
                    H_out, W_out = high_res_rgb.shape[:2]
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    
                    splatted_result, splatted_depth = splat_point_cloud(
                        combined_pcd,
                        K_high,
                        E_world_to_cam,
                        H_out,
                        W_out,
                        splat_mode=args.splat_mode,
                        splat_radius=args.splat_radius,
                        device=device
                    )
                    
                    # Save the depth map from splatting
                    depths_out[ci, ti] = splatted_depth
                    
                    if args.splat_mode == "recolor":
                        # Use splatted colors as RGB output
                        rgbs_out[ci, ti] = splatted_result
                    else:  # mask mode
                        # Apply mask to original high-res RGB
                        masked_rgb = high_res_rgb.copy()
                        mask_bool = splatted_result > 0  # Convert to boolean mask
                        masked_rgb[~mask_bool] = 0  # Set masked pixels to black
                        rgbs_out[ci, ti] = masked_rgb
                else:
                    # Original sparse depth reprojection method
                    rgbs_out[ci, ti] = high_res_rgb
                    # If color check is disabled, use a threshold that allows all points to pass
                    threshold = args.color_threshold if args.color_alignment_check else 256.0
                    depths_out[ci, ti] = reproject_to_sparse_depth_cv2(
                        combined_pcd, high_res_rgb, K_high, E_world_to_cam, threshold
                    )
            else:
                # Standard low-resolution workflow
                t_low = int(per_cam_low_ts[ci][ti])
                rgbs_out[ci, ti] = read_rgb(_resolve_frame(color_lookup_low[ci], t_low, cid, "low-res color"))
                depths_out[ci, ti] = read_depth(_resolve_frame(depth_lookup_low[ci], t_low, cid, "low-res depth"), is_l515_flags[ci])
                intrs_out[ci, ti] = scaled_low_intrinsics[ci]

    return rgbs_out, depths_out, intrs_out, extrs_out

def save_and_visualize(args, rgbs, depths, intrs, extrs, final_cam_ids, timestamps, per_camera_timestamps):
    """Saves the processed data to an NPZ file and generates a Rerun visualization."""
    # Convert to channels-first format for NPZ
    rgbs_final = np.moveaxis(rgbs, -1, 2)
    depths_final = depths[:, :, None, :, :]
    
    per_cam_ts_arr = np.stack(per_camera_timestamps, axis=0).astype(np.int64)

    npz_payload = {
        'rgbs': rgbs_final,
        'depths': depths_final,
        'intrs': intrs,
        'extrs': extrs,
        'timestamps': timestamps,
        'per_camera_timestamps': per_cam_ts_arr,
        'camera_ids': np.array(final_cam_ids, dtype=object),
    }

    out_path_npz = args.out_dir / f"{args.task_folder.name}_processed.npz"
    np.savez_compressed(out_path_npz, **npz_payload)
    print(f"✅ [OK] Wrote NPZ file to: {out_path_npz}")

    # Generate Rerun Visualization
    if not args.no_pointcloud:
        print("[INFO] Logging data to Rerun...")
        rgbs_tensor = torch.from_numpy(rgbs_final).float().unsqueeze(0)
        depths_tensor = torch.from_numpy(depths_final).float().unsqueeze(0)
        intrs_tensor = torch.from_numpy(intrs).float().unsqueeze(0)
        extrs_tensor = torch.from_numpy(extrs).float().unsqueeze(0)
        log_pointclouds_to_rerun(
            dataset_name="rh20t_reprojection",
            datapoint_idx=0,
            rgbs=rgbs_tensor,
            depths=depths_tensor,
            intrs=intrs_tensor,
            extrs=extrs_tensor,
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
    parser.add_argument(
        "--color-alignment-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable color-based filtering of reprojected points (disable with --no-color-alignment-check)."
    )
    parser.add_argument("--color-threshold", type=float, default=25.0, help="Max average color difference (0-255) for a point to be aligned.")
    parser.add_argument(
        "--sharpen-edges-with-mesh",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Densify geometry via Poisson meshing for sharper edges (disable with --no-sharpen-edges-with-mesh)."
    )
    parser.add_argument(
        "--mesh-depth",
        type=int,
        default=9,
        help="Poisson reconstruction depth controlling mesh resolution."
    )
    parser.add_argument(
        "--clean-pointcloud",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply radius-based cleaning to the fused point cloud before reprojection (disable with --no-clean-pointcloud)."
    )
    parser.add_argument(
        "--pc-clean-radius",
        type=float,
        default=0.02,
        help="Radius (meters) for the Open3D radius outlier removal filter."
    )
    parser.add_argument(
        "--pc-clean-min-points",
        type=int,
        default=10,
        help="Minimum number of neighbors within radius to keep a point during cleaning."
    )
    parser.add_argument("--no-pointcloud", action="store_true", help="Only generate the .npz file, skip visualization.")
    parser.add_argument("--sync-fps", type=float, default=10.0, help="Target FPS for synchronization output timeline.")
    parser.add_argument("--sync-min-density", type=float, default=0.6, help="Minimum density ratio required per camera during synchronization.")
    parser.add_argument("--sync-max-drift", type=float, default=0.05, help="Maximum tolerated fractional FPS shortfall before warning.")
    parser.add_argument("--sync-tolerance-ms", type=float, default=50.0, help="Maximum timestamp deviation (ms) when matching frames; defaults to half frame period.")
    # Point splatting arguments
    parser.add_argument("--use-splatting", action="store_true", help="Enable point splatting using PyTorch3D instead of sparse depth reprojection.")
    parser.add_argument("--splat-mode", choices=["recolor", "mask"], default="mask", help="Splatting mode: 'recolor' renders point colors, 'mask' creates a binary mask.")
    parser.add_argument("--splat-radius", type=float, default=0.005, help="Radius of splats in world units (meters).")
    
    # Confidence filtering arguments (demo.py style)
    parser.add_argument("--confidence-threshold", type=float, default=0.2, help="Minimum confidence threshold for depth pixels (demo.py style).")
    parser.add_argument("--edge-margin", type=int, default=10, help="Pixels to remove from image borders (unreliable edges).")
    parser.add_argument("--gradient-threshold", type=float, default=0.1, help="Maximum depth gradient to keep (meters, for edge detection).")
    parser.add_argument("--smooth-kernel", type=int, default=3, help="Kernel size for confidence-weighted depth smoothing.")
    
    args = parser.parse_args()

    # Validate PyTorch3D availability for splatting
    if args.use_splatting and not PYTORCH3D_AVAILABLE:
        print("[ERROR] Point splatting requires PyTorch3D, but it is not available.")
        print("[ERROR] Please install PyTorch3D or disable splatting with --no-use-splatting.")
        return

    if not args.config.exists():
        print(f"[ERROR] RH20T config file not found at: {args.config}")
        return
    
    args.out_dir.mkdir(parents=True, exist_ok=True)
    robot_configs = load_conf(str(args.config))
    
    # --- Step 1: Load and Synchronize Data ---
    scene_low, cam_ids_low, cam_dirs_low = load_scene_data(args.task_folder, robot_configs)
    if not scene_low or not cam_ids_low: return

    scene_high, cam_ids_high, cam_dirs_high = (load_scene_data(args.high_res_folder, robot_configs) if args.high_res_folder else (None, None, None))

    sync_kwargs = dict(
        frame_rate_hz=args.sync_fps,
        min_density=args.sync_min_density,
        target_fps=args.sync_fps,
        max_fps_drift=args.sync_max_drift,
        jitter_tolerance_ms=args.sync_tolerance_ms,
    )

    if args.high_res_folder:
        shared_ids = sorted(set(cam_ids_low) & set(cam_ids_high))

        id_to_dir_low = {cid: d for cid, d in zip(cam_ids_low, cam_dirs_low)}
        id_to_dir_high = {cid: d for cid, d in zip(cam_ids_high, cam_dirs_high)}

        final_cam_ids = [cid for cid in shared_ids if cid in id_to_dir_low and cid in id_to_dir_high]
        cam_dirs_low = [id_to_dir_low[cid] for cid in final_cam_ids]
        cam_dirs_high = [id_to_dir_high[cid] for cid in final_cam_ids]

        if len(final_cam_ids) < 2:
            print("[ERROR] Fewer than 2 common cameras between low and high resolution data.")
            return

        sync_low = get_synchronized_timestamps(cam_dirs_low, **sync_kwargs)
        sync_high = get_synchronized_timestamps(cam_dirs_high, require_depth=False, **sync_kwargs)

        valid_low = set(sync_low.camera_indices)
        valid_high = set(sync_high.camera_indices)
        keep_indices = sorted(valid_low & valid_high)

        if len(keep_indices) < 2:
            print("[ERROR] Synchronization rejected too many cameras; fewer than 2 remain aligned across resolutions.")
            return

        index_map_low = {idx: arr for idx, arr in zip(sync_low.camera_indices, sync_low.per_camera_timestamps)}
        index_map_high = {idx: arr for idx, arr in zip(sync_high.camera_indices, sync_high.per_camera_timestamps)}

        final_cam_ids = [final_cam_ids[i] for i in keep_indices]
        cam_dirs_low = [cam_dirs_low[i] for i in keep_indices]
        cam_dirs_high = [cam_dirs_high[i] for i in keep_indices]
        per_cam_low_full = [index_map_low[i] for i in keep_indices]
        per_cam_high_full = [index_map_high[i] for i in keep_indices]

        timeline_common = np.intersect1d(sync_low.timeline, sync_high.timeline)
        if timeline_common.size == 0:
            print("[ERROR] No overlapping synchronized timeline between low and high resolution data.")
            return

        timeline_common = np.asarray(timeline_common, dtype=np.int64)
        idx_map_low = {int(t): idx for idx, t in enumerate(sync_low.timeline)}
        idx_map_high = {int(t): idx for idx, t in enumerate(sync_high.timeline)}
        idx_low = [idx_map_low[int(t)] for t in timeline_common]
        idx_high = [idx_map_high[int(t)] for t in timeline_common]
        per_cam_low = [arr[idx_low] for arr in per_cam_low_full]
        per_cam_high = [arr[idx_high] for arr in per_cam_high_full]
    else:
        id_to_dir_low = {cid: d for cid, d in zip(cam_ids_low, cam_dirs_low)}
        final_cam_ids = [cid for cid in sorted(set(cam_ids_low)) if cid in id_to_dir_low]
        cam_dirs_low = [id_to_dir_low[cid] for cid in final_cam_ids]

        sync_low = get_synchronized_timestamps(cam_dirs_low, **sync_kwargs)
        valid_low = sorted(sync_low.camera_indices)
        if len(valid_low) < 2:
            print("[ERROR] Fewer than 2 cameras available for processing.")
            return
        final_cam_ids = [final_cam_ids[i] for i in valid_low]
        cam_dirs_low = [cam_dirs_low[i] for i in valid_low]
        index_map_low = {idx: arr for idx, arr in zip(sync_low.camera_indices, sync_low.per_camera_timestamps)}
        per_cam_low_full = [index_map_low[i] for i in valid_low]
        per_cam_low = per_cam_low_full
        per_cam_high = None
        timeline_common = np.asarray(sync_low.timeline, dtype=np.int64)

    if timeline_common.size == 0:
        print("[ERROR] Synchronization returned no frames.")
        return

    timestamps = select_frames(timeline_common, args.max_frames, args.frame_selection)
    if timestamps.size == 0:
        print("[ERROR] No frames remaining after selection.")
        return

    idx_map_common = {int(t): idx for idx, t in enumerate(timeline_common)}
    selected_idx = [idx_map_common[int(t)] for t in timestamps]
    per_cam_low_sel = [arr[selected_idx] for arr in per_cam_low]
    per_cam_high_sel = [arr[selected_idx] for arr in per_cam_high] if per_cam_high is not None else None

    # --- Step 2: Process Frames ---
    rgbs, depths, intrs, extrs = process_frames(
        args,
        scene_low,
        scene_high,
        final_cam_ids,
        cam_dirs_low,
        cam_dirs_high if args.high_res_folder else None,
        timestamps,
        per_cam_low_sel,
        per_cam_high_sel,
    )

    # --- Step 3: Save and Visualize ---
    rr.init("RH20T_Reprojection_Frameworks", spawn=False)
    per_cam_for_npz = per_cam_high_sel if per_cam_high_sel is not None else per_cam_low_sel
    save_and_visualize(args, rgbs, depths, intrs, extrs, final_cam_ids, timestamps, per_cam_for_npz)

    if not args.no_pointcloud:
        rrd_path = args.out_dir / f"{args.task_folder.name}_reprojected.rrd"
        rr.save(str(rrd_path))
        print(f"✅ [OK] Saved Rerun visualization to: {rrd_path}")

if __name__ == "__main__":
    main()
