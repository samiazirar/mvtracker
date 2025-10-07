#!/usr/bin/env python3
"""
Processes RH20T task folders to generate sparse, high-resolution depth maps by
reprojecting a point cloud from low-resolution depth onto high-resolution views.

generate npz with transform to npz
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

python create_sparse_depth_map.py   --task-folder /data/rh20t_api/data/test_data_full_rgb_upscaled_depth/uncompressed_low_res_data/task_0065_user_0010_scene_0009_cfg_0004   --high-res-folder /data/rh20t_api/data/test_data_full_rgb_upscaled_depth/rgb_data/RH20T_cfg4/task_0065_user_0010_scene_0009_cfg_0004  --out-dir ./data/high_res_filtered   --max-frames 100   --color-alignment-check   --color-threshold 35 --no-sharpen-edges-with-mesh

"""


# Standard library imports
import argparse
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from rh20t_api_safety.rh20t_api.configurations import get_conf_from_dir_name, load_conf
from rh20t_api_safety.rh20t_api.scene import RH20TScene
from rh20t_api_safety.utils.robot import RobotModel
import warnings
# --- File I/O & Utility Functions ---

NUM_RE = re.compile(r"(\d+)")

ROBOT_EE_LINK_MAP = {
    "ur5": "ee_link",
    "flexiv": "link7",
    "kuka": "lbr_iiwa_link_7",
    "franka": "panda_link8",
}

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
    try:
        calib_dir = Path(scene.calib_folder) / "imgs"
    except AttributeError:
        return None

    calib_path = calib_dir / f"cam_{camera_id}_c.png"
    if not calib_path.exists():
        return None

    with Image.open(calib_path) as img:
        width, height = img.size
    return width, height


def _create_robot_model(
    sample_path: Path,
    configs_path: Optional[Path] = None,
    rh20t_root: Optional[Path] = None,
) -> Optional[RobotModel]:
    """Create a RobotModel instance for the current scene."""
    configs_path = configs_path or (Path(__file__).resolve().parent / "rh20t_api" / "configs" / "configs.json")
    rh20t_root = rh20t_root or (Path(__file__).resolve().parent / "rh20t_api")

    try:
        confs = load_conf(str(configs_path))
    except Exception as exc:  # pragma: no cover - best-effort logging
        warnings.warn(f"Failed to load RH20T configurations ({exc}); skipping robot overlay.")
        return None

    try:
        conf = get_conf_from_dir_name(str(sample_path), confs)
    except Exception as exc:  # pragma: no cover - dataset naming mismatches
        warnings.warn(f"Could not infer robot configuration from '{sample_path}' ({exc}); skipping robot overlay.")
        return None

    robot_urdf = (rh20t_root / conf.robot_urdf).resolve()
    robot_mesh_root = (rh20t_root / conf.robot_mesh).resolve()

    if not robot_urdf.exists():
        warnings.warn(f"Robot URDF not found at '{robot_urdf}'; skipping robot overlay.")
        return None
    if not robot_mesh_root.exists():
        warnings.warn(f"Robot mesh directory '{robot_mesh_root}' not found; skipping robot overlay.")
        return None

    try:
        robot_model = RobotModel(
            robot_joint_sequence=conf.robot_joint_sequence,
            robot_urdf=str(robot_urdf),
            robot_mesh=str(robot_mesh_root),
        )
    except Exception as exc:  # pragma: no cover - URDF parsing is best-effort
        warnings.warn(f"Failed to load robot model from URDF ({exc}); skipping robot overlay.")
        return None
    
    # Load MTL colors from mesh directory (graphics subdirectory)
    graphics_dir = robot_mesh_root / "graphics"
    if not graphics_dir.exists():
        graphics_dir = robot_mesh_root  # Fallback to robot_mesh_root itself
    mtl_colors = _load_mtl_colors_for_mesh_dir(graphics_dir)

    return {
        "model": robot_model,
        "mtl_colors": mtl_colors,
    }


def _load_urdf_link_colors(urdf_path: Path) -> Dict[str, np.ndarray]:
    """Parse URDF file to extract material colors for each link.
    
    Returns:
        Dictionary mapping link names to RGB colors (0-1 range)
    """
    import xml.etree.ElementTree as ET
    
    link_colors = {}
    
    try:
        tree = ET.parse(str(urdf_path))
        root = tree.getroot()
        
        # First, collect material definitions
        materials = {}
        for material in root.findall('.//material'):
            name = material.get('name')
            if not name:
                continue
            color_elem = material.find('color')
            if color_elem is not None:
                rgba = color_elem.get('rgba')
                if rgba:
                    rgba_values = [float(x) for x in rgba.split()]
                    materials[name] = np.array(rgba_values[:3])  # RGB only
        
        # Now map links to their visual materials
        for link in root.findall('.//link'):
            link_name = link.get('name')
            if not link_name:
                continue
                
            visuals = link.findall('visual')
            if visuals:
                # Use first visual's material
                material = visuals[0].find('material')
                if material is not None:
                    mat_name = material.get('name')
                    color_elem = material.find('color')
                    
                    if color_elem is not None:
                        # Inline color definition
                        rgba = color_elem.get('rgba')
                        if rgba:
                            rgba_values = [float(x) for x in rgba.split()]
                            link_colors[link_name] = np.array(rgba_values[:3])
                    elif mat_name and mat_name in materials:
                        # Reference to material definition
                        link_colors[link_name] = materials[mat_name]
    
    except Exception as e:
        print(f"[WARN] Could not parse URDF colors: {e}")
    
    return link_colors


def _load_mtl_colors_for_mesh_dir(mesh_dir: Path) -> Dict[str, np.ndarray]:
    """Load material colors from MTL files in the mesh directory.
    
    Args:
        mesh_dir: Directory containing mesh and MTL files
        
    Returns:
        Dictionary mapping mesh filenames (without extension) to RGB colors (0-1 range)
    """
    mesh_colors = {}
    
    try:
        # Find all MTL files in the directory
        for mtl_file in mesh_dir.glob("*.mtl"):
            # Parse the MTL file
            material_color = None
            with open(mtl_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Look for diffuse color (Kd)
                    if line.startswith('Kd '):
                        parts = line.split()
                        if len(parts) >= 4:
                            try:
                                r, g, b = float(parts[1]), float(parts[2]), float(parts[3])
                                material_color = np.array([r, g, b])
                                break
                            except ValueError:
                                continue
            
            if material_color is not None:
                # Map this color to all mesh files with the same base name
                base_name = mtl_file.stem  # e.g., "link0_body"
                mesh_colors[base_name] = material_color
                
    except Exception as e:
        print(f"[WARN] Could not parse MTL colors: {e}")
    
    return mesh_colors


def _robot_model_to_pointcloud(
    robot_model: RobotModel,
    joint_angles: np.ndarray,
    is_first_time: bool = False,
    points_per_mesh: int = 10000,
    debug_mode: bool = False,
    urdf_colors: Optional[Dict[str, np.ndarray]] = None,
    mtl_colors: Optional[Dict[str, np.ndarray]] = None,
) -> o3d.geometry.PointCloud:
    """Convert a robot model at a specific joint state to a point cloud.
    
    Note: This function gets the transformed mesh vertices from the robot model,
    which have already been transformed by robot_model.update() to the correct pose.
    
    Args:
        robot_model: The robot model to convert
        joint_angles: Joint angles for the robot configuration
        is_first_time: Whether this is the first frame (affects mesh loading)
        points_per_mesh: Number of points to sample per mesh
        debug_mode: If True, use colorful debug colors per part. If False, use mesh colors.
        urdf_colors: Dictionary mapping link names to RGB colors from URDF
        mtl_colors: Dictionary mapping mesh base names to RGB colors from MTL files
    """
    # Update the robot model with the current joint angles
    # This transforms the internal meshes to the correct pose
    try:
        robot_model.update(joint_angles, first_time=is_first_time)
    except Exception as e:
        print(f"[ERROR] robot_model.update() failed: {e}")
        return o3d.geometry.PointCloud()

    fk = None
    joint_sequence = getattr(robot_model, "joint_sequence", [])
    if joint_sequence:
        if len(joint_sequence) != len(joint_angles):
            warnings.warn(
                f"Joint angle length ({len(joint_angles)}) does not match robot joint sequence ({len(joint_sequence)})."
            )
        try:
            joint_map = {name: float(joint_angles[i]) for i, name in enumerate(joint_sequence)}
            fk = robot_model.chain.forward_kinematics(joint_map)
        except Exception as exc:
            warnings.warn(f"Failed to compute forward kinematics for robot model ({exc}).")
    
    # Combine all robot meshes into a single point cloud
    robot_pcd = o3d.geometry.PointCloud()
    try:
        geometries = robot_model.geometries_to_add if is_first_time else robot_model.geometries_to_update
    except Exception as e:
        print(f"[ERROR] Failed to get geometries: {e}")
        return robot_pcd
    
    # Define debug colors for different robot parts (only used if debug_mode=True)
    debug_part_colors = [
        [0.2, 0.3, 0.8],   # Blue
        [0.8, 0.3, 0.2],   # Red-orange
        [0.2, 0.8, 0.3],   # Green
        [0.8, 0.8, 0.2],   # Yellow
        [0.8, 0.2, 0.8],   # Magenta
        [0.2, 0.8, 0.8],   # Cyan
        [0.9, 0.5, 0.1],   # Orange
    ]
    
    for mesh_idx, mesh in enumerate(geometries):
        # The mesh vertices are already in the correct world position
        # thanks to robot_model.update() transforming them
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        if vertices.size == 0 or triangles.size == 0:
            continue

        # Create an Open3D mesh from the current transformed state
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(triangles)
        
        # IMPORTANT: Copy vertex colors from original mesh if they exist
        if mesh.has_vertex_colors():
            o3d_mesh.vertex_colors = mesh.vertex_colors
        
        # Sample points from the mesh surface (already in correct position)
        mesh_pcd = o3d_mesh.sample_points_uniformly(number_of_points=points_per_mesh)
        
        # Assign colors based on mode
        if debug_mode:
            # Debug mode: Use bright, distinct colors for each part
            num_points = len(mesh_pcd.points)
            part_color = debug_part_colors[mesh_idx % len(debug_part_colors)]
            mesh_pcd.colors = o3d.utility.Vector3dVector(np.tile(part_color, (num_points, 1)))
        else:
            # Normal mode: Try MTL colors first, then URDF colors, then mesh colors, then default gray
            color_assigned = False
            
            # First try: MTL material colors (most accurate colors from the actual mesh files)
            if mtl_colors:
                mesh_basename = robot_model.get_filename_for_mesh(mesh)
                if mesh_basename in mtl_colors:
                    num_points = len(mesh_pcd.points)
                    mtl_color = mtl_colors[mesh_basename]
                    mesh_pcd.colors = o3d.utility.Vector3dVector(np.tile(mtl_color, (num_points, 1)))
                    color_assigned = True
            
            # Second try: URDF material colors
            if not color_assigned and urdf_colors:
                link_name = robot_model.get_link_for_mesh(mesh)
                if link_name in urdf_colors:
                    num_points = len(mesh_pcd.points)
                    urdf_color = urdf_colors[link_name]
                    mesh_pcd.colors = o3d.utility.Vector3dVector(np.tile(urdf_color, (num_points, 1)))
                    color_assigned = True
            
            # Third try: Original mesh vertex colors
            if not color_assigned and mesh.has_vertex_colors():
                colors = np.asarray(mesh.vertex_colors)
                if colors.size > 0:
                    # Transfer colors from mesh to sampled points using nearest neighbor
                    kdtree = o3d.geometry.KDTreeFlann(o3d_mesh)
                    sampled_colors = np.zeros((len(mesh_pcd.points), 3))
                    for i, point in enumerate(np.asarray(mesh_pcd.points)):
                        _, idx, _ = kdtree.search_knn_vector_3d(point, 1)
                        sampled_colors[i] = colors[idx[0]]
                    mesh_pcd.colors = o3d.utility.Vector3dVector(sampled_colors)
                    color_assigned = True
            
            # Fallback: Use a neutral gray/silver color
            if not color_assigned:
                num_points = len(mesh_pcd.points)
                default_color = [0.7, 0.7, 0.7]  # Light gray
                mesh_pcd.colors = o3d.utility.Vector3dVector(np.tile(default_color, (num_points, 1)))
        
        robot_pcd += mesh_pcd

    return robot_pcd


#TODO: add jounts
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


def clean_point_cloud_radius(pcd: o3d.geometry.PointCloud, radius: float, min_points: int) -> o3d.geometry.PointCloud:
    """Radius-based outlier removal leveraging Open3D's built-in filter."""
    if not pcd.has_points():
        return pcd

    try:
        _, ind = pcd.remove_radius_outlier(nb_points=max(1, int(min_points)), radius=float(radius))
    except Exception as exc:  # pragma: no cover - Open3D failures are best-effort
        print(f"[WARN] Radius-based point cloud cleaning failed ({exc}); skipping filter.")
        return pcd

    if len(ind) == 0:
        print("[WARN] Radius-based point cloud cleaning removed all points; keeping original cloud.")
        return pcd

    return pcd.select_by_index(ind)


def reconstruct_mesh_from_pointcloud(pcd: o3d.geometry.PointCloud, depth: int) -> Optional[o3d.geometry.TriangleMesh]:
    """Reconstructs a mesh via Poisson surface reconstruction to sharpen geometry."""
    if not pcd.has_points():
        return None

    pcd_for_mesh = o3d.geometry.PointCloud(pcd)
    try:
        pcd_for_mesh.estimate_normals()
    except Exception as exc:  # pragma: no cover - best-effort
        print(f"[WARN] Normal estimation failed ({exc}); skipping mesh reconstruction.")
        return None

    try:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_for_mesh, depth=int(depth))
    except Exception as exc:  # pragma: no cover - Poisson may fail on sparse clouds
        print(f"[WARN] Mesh reconstruction failed ({exc}); skipping Poisson meshing.")
        return None

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
    pcd: o3d.geometry.PointCloud, 
    high_res_rgb: np.ndarray, 
    K: np.ndarray, 
    E: np.ndarray, 
    color_threshold: float,
) -> np.ndarray:
    """
    Projects an Open3D point cloud to a sparse depth map using OpenCV.

    Includes color alignment check:
    - If color diff > color_threshold: remove point
    - Otherwise: keep point with original color

    Args:
        pcd: The Open3D PointCloud in world coordinates.
        high_res_rgb: The target high-resolution color image.
        K: The 3x3 intrinsic matrix of the target camera.
        E: The 3x4 extrinsic matrix of the target camera (world-to-camera).
        color_threshold: Maximum color difference to keep a point.

    Returns:
        A sparse depth map of the same resolution as the high_res_rgb.
    """
    H, W, _ = high_res_rgb.shape
    sparse_depth = np.zeros((H, W), dtype=np.float32)
    
    if not pcd.has_points():
        return sparse_depth

    pts_world = np.asarray(pcd.points)
    
    # Handle point clouds with or without colors
    if pcd.has_colors():
        orig_colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    else:
        # If no colors, create a default gray color for all points
        num_points = pts_world.shape[0]
        orig_colors = np.full((num_points, 3), 128, dtype=np.uint8)  # Gray color

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
    np.clip(u_idx, 0, W - 1, out=u_idx)
    np.clip(v_idx, 0, H - 1, out=v_idx)
    depth_final = depths[valid_mask]
    orig_colors_final = orig_colors[valid_mask]

    # 3. Color Alignment Check
    target_colors = high_res_rgb[v_idx, u_idx]
    color_diff = np.mean(np.abs(orig_colors_final.astype(float) - target_colors.astype(float)), axis=1)
    
    # Filter out points with color difference above threshold
    keep_mask = color_diff < color_threshold
    u_final = u_idx[keep_mask]
    v_final = v_idx[keep_mask]
    depth_final = depth_final[keep_mask]
    
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

    # Collect robot points if robot model will be enabled
    # We need this regardless of debug mode so we can log it to Rerun
    robot_debug_points: Optional[List[np.ndarray]] = [] if args.add_robot else None
    robot_debug_colors: Optional[List[np.ndarray]] = [] if robot_debug_points is not None else None

    color_lookup_low = [list_frames(cdir / 'color') for cdir in cam_dirs_low]
    depth_lookup_low = [list_frames(cdir / 'depth') for cdir in cam_dirs_low]

    if args.high_res_folder and cam_dirs_high:
        color_lookup_high = [list_frames(cdir / 'color') for cdir in cam_dirs_high]
    else:
        color_lookup_high = None

    # Create robot model if requested
    robot_model = None
    robot_urdf_colors = None
    robot_mtl_colors = None
    if args.add_robot:
        configs_path = args.config.resolve()
        rh20t_root = configs_path.parent.parent if configs_path.parent.name == "configs" else configs_path.parent
        robot_bundle = _create_robot_model(
            sample_path=args.task_folder,
            configs_path=configs_path,
            rh20t_root=rh20t_root,
        )
        if robot_bundle:
            robot_model = robot_bundle.get("model")
            robot_mtl_colors = robot_bundle.get("mtl_colors")
        if robot_model:
            print("[INFO] Robot model loaded successfully.")
            # Load URDF colors for the robot (if any exist in URDF)
            try:
                confs = load_conf(str(configs_path))
                conf = get_conf_from_dir_name(str(args.task_folder), confs)
                robot_urdf_path = (rh20t_root / conf.robot_urdf).resolve()
                robot_urdf_colors = _load_urdf_link_colors(robot_urdf_path)
                if robot_urdf_colors:
                    print(f"[INFO] Loaded colors for {len(robot_urdf_colors)} robot links from URDF")
            except Exception as e:
                print(f"[WARN] Could not load URDF colors: {e}")
            # Report MTL colors if loaded
            if robot_mtl_colors and len(robot_mtl_colors) > 0:
                print(f"[INFO] Loaded MTL colors for {len(robot_mtl_colors)} robot parts")

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
        debug_pts = None
        debug_cols = None
        # Step 1: Create a combined point cloud for the current frame from all low-res views
        combined_pcd = o3d.geometry.PointCloud()
        if args.high_res_folder:
            pcds_per_cam = []
            for ci in range(C):
                cid = final_cam_ids[ci]
                # Load low-res data for point cloud generation
                t_low = int(per_cam_low_ts[ci][ti])
                depth_low = read_depth(_resolve_frame(depth_lookup_low[ci], t_low, cid, "low-res depth"), is_l515_flags[ci])
                rgb_low = read_rgb(_resolve_frame(color_lookup_low[ci], t_low, cid, "low-res color"))
                K_low = scaled_low_intrinsics[ci]
                E_inv = np.linalg.inv(scene_low.extrinsics_base_aligned[cid])
                
                # Create and add the point cloud for this view
                pcds_per_cam.append(unproject_to_world_o3d(depth_low, rgb_low, K_low, E_inv))
            
            # Merge all individual point clouds into a single scene representation
            for pcd in pcds_per_cam:
                combined_pcd += pcd

            # Add robot point cloud with current joint state if available
            if robot_model is not None:
                try:
                    # Get the timestamp for the current frame
                    t_low = int(per_cam_low_ts[0][ti])
                    # Get joint angles at this timestamp
                    joint_angles = scene_low.get_joint_angles_aligned(t_low)
                    
                    # Ensure we have the correct number of joints
                    # The robot model expects len(conf.robot_joint_sequence) joints
                    expected_joints = len(scene_low.configuration.robot_joint_sequence)
                    gripper_joint_angle = None
                    
                    if len(joint_angles) < expected_joints:
                        # Pad with zeros (neutral gripper position) if missing joints
                        padding = np.zeros(expected_joints - len(joint_angles))
                        joint_angles_padded = np.concatenate([joint_angles, padding])
                        if ti == 0:
                            print(f"[INFO] Padded joint angles from {len(joint_angles)} to {len(joint_angles_padded)}")
                        # Don't use padded zero - interpolate from gripper dictionary instead
                        gripper_joint_angle = 0.0  # Will be replaced below
                        # Use all joints for robot model (including padded gripper)
                        robot_joint_angles = joint_angles_padded
                    elif len(joint_angles) > expected_joints:
                        # Truncate if we have too many joints
                        robot_joint_angles = joint_angles[:expected_joints]
                        gripper_joint_angle = robot_joint_angles[-1]
                        if ti == 0:
                            print(f"[WARN] Truncated joint angles from {len(joint_angles)} to {expected_joints}")
                    else:
                        robot_joint_angles = joint_angles
                        gripper_joint_angle = joint_angles[-1]
                    
                    # Get actual gripper width from scene.gripper dictionary
                    # The gripper dictionary has structure: scene.gripper[camera_id][timestamp] = {'gripper_command': [width, ...]}
                    # We need to pick one camera's gripper data
                    gripper_data_source = None
                    if hasattr(scene_low, 'gripper') and len(scene_low.gripper) > 0:
                        # Find first camera with gripper data
                        for cam_id in sorted(scene_low.gripper.keys()):
                            if len(scene_low.gripper[cam_id]) > 0:
                                gripper_data_source = scene_low.gripper[cam_id]
                                if ti == 0:
                                    print(f"[INFO] Using gripper data from camera {cam_id}")
                                break
                    
                    if gripper_data_source is not None:
                        gripper_timestamps = sorted(gripper_data_source.keys())
                        # Find closest gripper timestamp to current frame
                        closest_idx = min(range(len(gripper_timestamps)), 
                                        key=lambda i: abs(gripper_timestamps[i] - t_low))
                        closest_ts = gripper_timestamps[closest_idx]
                        
                        # Get gripper command (first element is gripper width in mm)
                        gripper_width_mm = gripper_data_source[closest_ts]['gripper_command'][0]
                        
                        # Robotiq 2F-85 gripper: 0mm (closed) to 85mm (fully open)
                        # Pass width directly to _robot_model_to_pointcloud for finger animation
                        gripper_joint_angle = gripper_width_mm  # Width in mm
                        
                        if ti == 0:
                            print(f"[INFO] Gripper width at frame 0: {gripper_width_mm:.2f} mm")
                    
                    robot_pcd = _robot_model_to_pointcloud(
                        robot_model,
                        robot_joint_angles,
                        is_first_time=(ti == 0),
                        points_per_mesh=10000,
                        debug_mode=getattr(args, "debug_mode", False),
                        urdf_colors=robot_urdf_colors,
                        mtl_colors=robot_mtl_colors,
                    )
                    
                    # NOTE: Removed align_mat_base transformation
                    # The robot joint angles from get_joint_angles_aligned() are already in the aligned frame
                    # Applying align_mat_base was causing the robot to face the wrong direction
                    # if robot_pcd.has_points():
                    #     align_mat = scene_low.configuration.align_mat_base
                    #     robot_pcd.transform(align_mat)

                    if robot_pcd.has_points():
                        if robot_debug_points is not None:
                            pts_np = np.asarray(robot_pcd.points, dtype=np.float32)
                            if robot_pcd.has_colors():
                                cols_np = (np.asarray(robot_pcd.colors) * 255).astype(np.uint8)
                            else:
                                cols_np = np.full((pts_np.shape[0], 3), [180, 180, 180], dtype=np.uint8)
                            debug_pts = pts_np
                            debug_cols = cols_np
                        combined_pcd += robot_pcd
                        if ti == 0:
                            print(f"[INFO] Added robot with {len(robot_pcd.points)} points for frame {ti}")
                        elif ti % 5 == 0:
                            print(f"[INFO] Frame {ti}: Robot has {len(robot_pcd.points)} points")
                except Exception as exc:
                    if ti == 0:
                        print(f"[WARN] Failed to add robot to point cloud: {exc}")
            if robot_debug_points is not None and debug_pts is None:
                debug_pts = np.empty((0, 3), dtype=np.float32)
                debug_cols = np.empty((0, 3), dtype=np.uint8)

            if combined_pcd.has_points():
                if args.clean_pointcloud:
                    before_count = len(combined_pcd.points)
                    cleaned = clean_point_cloud_radius(
                        combined_pcd,
                        radius=args.pc_clean_radius,
                        min_points=args.pc_clean_min_points,
                    )
                    after_count = len(cleaned.points)
                    if after_count == 0:
                        print(
                            f"[WARN] Radius filter removed all points at frame index {ti}; "
                            "using raw combined point cloud instead."
                        )
                    else:
                        if ti == 0 or after_count != before_count:
                            print(
                                f"[INFO] Radius filter kept {after_count}/{before_count} points "
                                f"(radius={args.pc_clean_radius}, min_pts={args.pc_clean_min_points})."
                            )
                        combined_pcd = cleaned

                if args.sharpen_edges_with_mesh:
                    mesh = reconstruct_mesh_from_pointcloud(combined_pcd, depth=args.mesh_depth)
                    if mesh is not None and len(mesh.vertices) > 0:
                        mesh_pcd = o3d.geometry.PointCloud()
                        mesh_pcd.points = mesh.vertices
                        if mesh.has_vertex_colors() and len(mesh.vertex_colors) == len(mesh.vertices):
                            mesh_pcd.colors = mesh.vertex_colors
                        elif combined_pcd.has_colors():
                            combined_colors = np.asarray(combined_pcd.colors)
                            if combined_colors.size:
                                kdtree = o3d.geometry.KDTreeFlann(combined_pcd)
                                mesh_vertices = np.asarray(mesh.vertices)
                                remap_colors = np.zeros_like(mesh_vertices)
                                for vidx, vertex in enumerate(mesh_vertices):
                                    _, idx, _ = kdtree.search_knn_vector_3d(vertex, 1)
                                    remap_colors[vidx] = combined_colors[idx[0]]
                                mesh_pcd.colors = o3d.utility.Vector3dVector(remap_colors)
                        combined_pcd = mesh_pcd

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
                rgbs_out[ci, ti] = high_res_rgb
                intrs_out[ci, ti] = K_high
                
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

        if robot_debug_points is not None:
            if debug_pts is None:
                debug_pts = np.empty((0, 3), dtype=np.float32)
                debug_cols = np.empty((0, 3), dtype=np.uint8)
            robot_debug_points.append(debug_pts)
            if robot_debug_colors is not None:
                robot_debug_colors.append(debug_cols)

    return rgbs_out, depths_out, intrs_out, extrs_out, robot_debug_points, robot_debug_colors

def save_and_visualize(
    args,
    rgbs,
    depths,
    intrs,
    extrs,
    final_cam_ids,
    timestamps,
    per_camera_timestamps,
    robot_debug_points=None,
    robot_debug_colors=None,
):
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
        configs_path = args.config.resolve()
        rh20t_root = configs_path.parent.parent if configs_path.parent.name == "configs" else configs_path.parent

        # Log robot points separately (they survive better than going through reprojection)
        if robot_debug_points is not None and len(robot_debug_points) > 0:
            fps = 30.0
            print(f"[INFO] Logging {len(robot_debug_points)} robot point clouds to Rerun...")
            for idx, pts in enumerate(robot_debug_points):
                if pts is None or pts.size == 0:
                    continue
                cols = robot_debug_colors[idx] if robot_debug_colors and idx < len(robot_debug_colors) else None
                rr.set_time_seconds("frame", idx / fps)
                # Log to top-level robot entity for easy toggling (NOT under sequence-0)
                rr.log(
                    f"robot",
                    rr.Points3D(pts.astype(np.float32, copy=False), colors=cols),
                )
                # Also log to debug entity if in debug mode
                if getattr(args, "debug_mode", False):
                    rr.log(
                        f"robot_debug",
                        rr.Points3D(pts.astype(np.float32, copy=False), colors=cols),
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
    parser.add_argument("--config", default="rh20t_api_safety/configs/configs.json", type=Path, help="Path to RH20T robot configs JSON.")
    parser.add_argument("--max-frames", type=int, default=50, help="Limit frames to process (0 for all).")
    parser.add_argument("--frame-selection", choices=["first", "last", "middle"], default="middle", help="Method for selecting frames.")
    parser.add_argument(
        "--color-alignment-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable color-based filtering of reprojected points (disable with --no-color-alignment-check)."
    )
    parser.add_argument("--color-threshold", type=float, default=5.0, help="Max average color difference (0-255) for a point to be removed.")
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
        default=0.01,
        help="Radius (meters) for the Open3D radius outlier removal filter."
    )
    parser.add_argument(
        "--pc-clean-min-points",
        type=int,
        default=20,
        help="Minimum number of neighbors within radius to keep a point during cleaning."
    )
    parser.add_argument("--no-pointcloud", action="store_true", help="Only generate the .npz file, skip visualization.")
    parser.add_argument("--sync-fps", type=float, default=10.0, help="Target FPS for synchronization output timeline.")
    parser.add_argument("--sync-min-density", type=float, default=0.6, help="Minimum density ratio required per camera during synchronization.")
    parser.add_argument("--sync-max-drift", type=float, default=0.05, help="Maximum tolerated fractional FPS shortfall before warning.")
    parser.add_argument("--sync-tolerance-ms", type=float, default=50.0, help="Maximum timestamp deviation (ms) when matching frames; defaults to half frame period.")
    parser.add_argument("--add-robot", action="store_true", help="Include robot model in Rerun visualization.")
    parser.add_argument(
        "--debug-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable additional debug outputs such as red robot point overlays in Rerun.",
    )
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
    rgbs, depths, intrs, extrs, robot_debug_points, robot_debug_colors = process_frames(
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
    save_and_visualize(
        args,
        rgbs,
        depths,
        intrs,
        extrs,
        final_cam_ids,
        timestamps,
        per_cam_for_npz,
        robot_debug_points,
        robot_debug_colors,
    )

    if not args.no_pointcloud:
        rrd_path = args.out_dir / f"{args.task_folder.name}_reprojected.rrd"
        rr.save(str(rrd_path))
        print(f"✅ [OK] Saved Rerun visualization to: {rrd_path}")

if __name__ == "__main__":
    main()
