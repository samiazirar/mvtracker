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


# --- Standard Library Imports ---
# Importing modules for argument parsing, regex, warnings, and file path handling.
import argparse
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --- Third-Party Library Imports ---
# Importing libraries for numerical computation, image processing, 3D data handling, and progress tracking.
import numpy as np
import torch
import torch.nn as nn
import matplotlib
from PIL import Image
import cv2
import open3d as o3d
import rerun as rr
from tqdm import tqdm

# --- Project-Specific Imports ---
# Importing utilities and configurations specific to the RH20T dataset and robot model.
from mvtracker.models.core.monocular_baselines import (
    CoTrackerOfflineWrapper,
    CoTrackerOnlineWrapper,
    MonocularToMultiViewAdapter,
)
from mvtracker.utils.visualizer_rerun import log_pointclouds_to_rerun, log_tracks_to_rerun
from RH20T.rh20t_api.configurations import get_conf_from_dir_name, load_conf
from RH20T.rh20t_api.scene import RH20TScene
from RH20T.utils.robot import RobotModel

# --- Constants ---
# Precompiled regex for extracting numbers from filenames.
NUM_RE = re.compile(r"(\d+)")

# Mapping of robot types to their end-effector (EE) link names.
ROBOT_EE_LINK_MAP = {
    "ur5": "ee_link",
    "flexiv": "link7",
    "kuka": "lbr_iiwa_link_7",
    "franka": "panda_link8",
}

# Candidate gripper links for different robot types.
GRIPPER_LINK_CANDIDATES = {
    "ur5": ["robotiq_arg2f_base_link", "ee_link", "wrist_3_link"],
    "flexiv": ["flange", "link7"],
    "kuka": ["lbr_iiwa_link_7", "lbr_iiwa_link_6"],
    "franka": ["panda_hand", "panda_link8"],
}

# Default dimensions for various gripper models.
GRIPPER_DIMENSIONS = {
    "Robotiq 2F-85": {"length": 0.16, "height": 0.06, "default_width": 0.08},
    "WSG-50": {"length": 0.14, "height": 0.06, "default_width": 0.05},
    "Dahuan AG-95": {"length": 0.18, "height": 0.07, "default_width": 0.09},
    "franka": {"length": 0.14, "height": 0.05, "default_width": 0.08},
}

# Default dimensions for grippers if no specific model is provided.
DEFAULT_GRIPPER_DIMS = {"length": 0.15, "height": 0.06, "default_width": 0.08}

# Presets for contact surface dimensions for different gripper models.
CONTACT_SURFACE_PRESETS = {
    "Robotiq 2F-85": {"length": 0.045, "height": 0.02, "clearance": 0.004},
    "WSG-50": {"length": 0.040, "height": 0.018, "clearance": 0.0035},
    "Dahuan AG-95": {"length": 0.050, "height": 0.022, "clearance": 0.004},
    "franka": {"length": 0.038, "height": 0.018, "clearance": 0.003},
}

# Default contact surface dimensions.
DEFAULT_CONTACT_SURFACE = {"length": 0.042, "height": 0.02, "clearance": 0.004}

# Minimum dimensions for contact width and length.
MIN_CONTACT_WIDTH = 0.008
MIN_CONTACT_LENGTH = 0.015

# Fallback scaling factors for contact width and height.
CONTACT_WIDTH_SCALE_FALLBACK = 0.65
CONTACT_HEIGHT_SCALE_FALLBACK = 0.45

GLOBAL_RNG = np.random.default_rng(42)

# --- Utility Functions ---
# Functions for mathematical transformations, gripper bounding box computation, and 3D geometry handling.

def _rotation_matrix_to_xyzw(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a quaternion [x, y, z, w]."""
    # Validate the input early so downstream math can assume a proper rotation matrix.
    if R.shape != (3, 3):
        raise ValueError("Rotation matrix must be 3x3.")
    trace = np.trace(R)
    if trace > 0.0:
        # Positive trace gives the most numerically stable branch; compute quaternion directly.
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        # Otherwise pick the dominant diagonal entry to compute the quaternion reliably.
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    quat = np.array([qx, qy, qz, qw], dtype=np.float32)
    norm = np.linalg.norm(quat)
    if norm == 0.0:
        # Degenerate rotation: fall back to identity quaternion.
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return quat / norm


def _quaternion_xyzw_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert a quaternion [x, y, z, w] to a rotation matrix."""
    quat = np.asarray(quat, dtype=np.float32)
    if quat.shape[0] != 4:
        raise ValueError("Quaternion must have four components [x, y, z, w].")
    x, y, z, w = quat
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        return np.eye(3, dtype=np.float32)
    x /= norm
    y /= norm
    z /= norm
    w /= norm

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def _pose_7d_to_matrix(pose_7d: np.ndarray) -> np.ndarray:
    """Convert a 7D pose [x, y, z, qx, qy, qz, qw] to a 4x4 transformation matrix.
    
    Args:
        pose_7d: 7-element array with position (x,y,z) and quaternion (qx,qy,qz,qw)
        
    Returns:
        4x4 transformation matrix
    """
    pose_7d = np.asarray(pose_7d, dtype=np.float32)
    if pose_7d.shape[0] != 7:
        raise ValueError("Pose must have 7 components [x, y, z, qx, qy, qz, qw].")
    
    position = pose_7d[:3]
    quaternion = pose_7d[3:]  # [qx, qy, qz, qw]
    
    # Convert quaternion to rotation matrix
    rotation = _quaternion_xyzw_to_rotation_matrix(quaternion)
    
    # Construct 4x4 transformation matrix
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    
    return transform


def _compute_gripper_bbox(
    robot_model: RobotModel,
    robot_conf: Optional[Any],
    gripper_width_mm: Optional[float],
    contact_height_m: Optional[float] = None,
    contact_length_m: Optional[float] = None,
    tcp_transform: Optional[np.ndarray] = None,
) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[Dict[str, np.ndarray]], Optional[Dict[str, np.ndarray]]]:
    """Estimate contact bbox, full gripper bbox, and fingertip bbox aligned with the jaw frame."""
    if robot_conf is None:
        return None, None, None

    # Grab the most recent FK results so we can derive pad transforms.
    fk_map = getattr(robot_model, "latest_transforms", None) or {}
    robot_type = getattr(robot_conf, "robot", None)
    candidate_links = GRIPPER_LINK_CANDIDATES.get(robot_type, [])
    ee_link = ROBOT_EE_LINK_MAP.get(robot_type)

    tf_matrix = None
    # Prioritize the high-precision TCP transform from the API if it's provided.
    if tcp_transform is not None:
        tf_matrix = tcp_transform
    elif fk_map:
        for link_name in candidate_links:
            if link_name in fk_map:
                tf_matrix = fk_map[link_name].matrix()
                break
        if tf_matrix is None and ee_link and ee_link in fk_map:
            tf_matrix = fk_map[ee_link].matrix()

    if tf_matrix is None:
        return None, None, None

    tf_matrix = np.asarray(tf_matrix, dtype=np.float32)
    rotation = tf_matrix[:3, :3]
    translation = tf_matrix[:3, 3].astype(np.float32)

    # Resolve gripper geometry to determine nominal pad and body dimensions.
    gripper_name = getattr(robot_conf, "gripper", "")
    dims = DEFAULT_GRIPPER_DIMS
    contact_preset = CONTACT_SURFACE_PRESETS.get(gripper_name, DEFAULT_CONTACT_SURFACE)
    if gripper_name:
        for name, values in GRIPPER_DIMENSIONS.items():
            if name.lower() == gripper_name.lower():
                dims = values
                break

    height_m = contact_height_m if contact_height_m is not None else contact_preset.get(
        "height", dims["height"] * CONTACT_HEIGHT_SCALE_FALLBACK
    )
    base_length_m = contact_length_m if contact_length_m is not None else contact_preset.get(
        "length", dims["length"] * CONTACT_HEIGHT_SCALE_FALLBACK
    )
    clearance = contact_preset.get("clearance", DEFAULT_CONTACT_SURFACE["clearance"])

    height_m = float(max(height_m, 0.005))
    base_length_m = float(max(base_length_m, MIN_CONTACT_LENGTH))
    base_half_length = base_length_m / 2.0

    def _find_link(keyword_groups):
        for keywords in keyword_groups:
            for name in fk_map.keys():
                lname = name.lower()
                if all(k in lname for k in keywords):
                    return name
        return None

    left_link = _find_link([
        ("left", "inner", "finger", "pad"),
        ("left", "finger", "pad"),
        ("left", "inner", "finger"),
        ("left", "finger"),
        ("left", "knuckle"),
    ])
    right_link = _find_link([
        ("right", "inner", "finger", "pad"),
        ("right", "finger", "pad"),
        ("right", "inner", "finger"),
        ("right", "finger"),
        ("right", "knuckle"),
    ])

    left_tf = fk_map[left_link].matrix().astype(np.float32) if left_link and left_link in fk_map else None
    right_tf = fk_map[right_link].matrix().astype(np.float32) if right_link and right_link in fk_map else None
    left_pos = left_tf[:3, 3] if left_tf is not None else None
    right_pos = right_tf[:3, 3] if right_tf is not None else None

    # If both pads available, derive a stable coordinate frame directly from their transforms
    if left_pos is not None and right_pos is not None:
        # Width axis: from right to left (jaw separation direction)
        width_vec = left_pos - right_pos
        width_norm = np.linalg.norm(width_vec)
        if width_norm < 1e-6:
            return None, None, None
        width_axis = (width_vec / width_norm).astype(np.float32)
        measured_width = float(width_norm)
        
        # Pad midpoint is the center point between the two finger pads
        pad_midpoint = ((left_pos + right_pos) * 0.5).astype(np.float32)
        
        # Approach axis: average Z-axis (forward direction) from both finger pads
        # This is stable and points toward the object being grasped
        approach_axis = ((left_tf[:3, 2] + right_tf[:3, 2]) * 0.5).astype(np.float32)
        approach_norm = np.linalg.norm(approach_axis)
        if approach_norm < 1e-6:
            return None, None, None
        approach_axis = (approach_axis / approach_norm).astype(np.float32)
        
        # Height axis: perpendicular to both width and approach (completes right-handed frame)
        # This should point "up" relative to gripper (from tips toward palm)
        height_axis = np.cross(approach_axis, width_axis).astype(np.float32)
        height_norm = np.linalg.norm(height_axis)
        if height_norm < 1e-6:
            return None, None, None
        height_axis = (height_axis / height_norm).astype(np.float32)
        
        # Re-orthogonalize to ensure perfect right-handed frame
        approach_axis = np.cross(width_axis, height_axis).astype(np.float32)
        approach_axis = (approach_axis / np.linalg.norm(approach_axis)).astype(np.float32)
        
    elif left_tf is not None:
        # Fallback: use only left pad
        pad_midpoint = left_tf[:3, 3].astype(np.float32)
        width_axis = left_tf[:3, 0].astype(np.float32)  # X-axis of pad frame
        approach_axis = left_tf[:3, 2].astype(np.float32)  # Z-axis of pad frame
        height_axis = left_tf[:3, 1].astype(np.float32)  # Y-axis of pad frame
        measured_width = None
    elif right_tf is not None:
        # Fallback: use only right pad
        pad_midpoint = right_tf[:3, 3].astype(np.float32)
        width_axis = -right_tf[:3, 0].astype(np.float32)  # Flip X to match left orientation
        approach_axis = right_tf[:3, 2].astype(np.float32)  # Z-axis of pad frame
        height_axis = right_tf[:3, 1].astype(np.float32)  # Y-axis of pad frame
        measured_width = None
    else:
        # No pad transforms available
        return None, None, None

    # Calculate contact width from measured or commanded gripper width
    width_candidates = []
    if measured_width is not None:
        width_candidates.append(measured_width)
    if gripper_width_mm is not None:
        width_candidates.append(max(gripper_width_mm / 1000.0, MIN_CONTACT_WIDTH))
    if width_candidates:
        width_m = min(width_candidates)
    else:
        width_m = dims["default_width"] * CONTACT_WIDTH_SCALE_FALLBACK
    width_m = max(width_m - 2.0 * clearance, MIN_CONTACT_WIDTH)

    # Assemble the oriented box with half-extents
    basis = np.column_stack((width_axis, height_axis, approach_axis)).astype(np.float32)
    contact_half_sizes = np.array(
        [width_m / 2.0, height_m / 2.0, base_half_length],
        dtype=np.float32,
    )
    full_half_sizes = np.array(
        [
            max(contact_half_sizes[0] + clearance, dims["default_width"] / 2.0),
            max(contact_half_sizes[1] + clearance * 0.5, dims["height"] / 2.0),
            max(contact_half_sizes[2] + clearance, dims["length"] / 2.0),
        ],
        dtype=np.float32,
    )

    # Position the contact bbox:
    # - Back face at pad midpoint (fingertips)
    # - Box extends forward from the gripper tips along the approach direction
    contact_center = (pad_midpoint + approach_axis * contact_half_sizes[2]).astype(np.float32)
    
    # Full box shares the same front face position
    full_center = (pad_midpoint + approach_axis * full_half_sizes[2]).astype(np.float32)

    # Create a third bbox (blue) positioned at the TOP of the body bbox (opposite side from orange)
    # Same size as contact bbox (orange), but positioned inside the red body bbox
    # so its TOP face aligns with the red bbox's TOP face (toward the palm/wrist)
    fingertip_half_sizes = contact_half_sizes.copy()  # Same size as orange contact bbox
    
    # Position: top face aligned with body bbox top face (opposite direction from orange)
    # Body bbox top is at: full_center + approach_axis * full_half_sizes[2]
    # Blue bbox top should be at the same position
    # So: blue_center + approach_axis * fingertip_half_sizes[2] = full_center + approach_axis * full_half_sizes[2]
    # Therefore: blue_center = full_center + approach_axis * (full_half_sizes[2] - fingertip_half_sizes[2])
    fingertip_center = (full_center + approach_axis * (full_half_sizes[2] - fingertip_half_sizes[2])).astype(np.float32)

    # Provide both tight contact box and larger safety box consumers can choose between.
    quat = _rotation_matrix_to_xyzw(basis)
    contact_box = {
        "center": contact_center.astype(np.float32),
        "half_sizes": contact_half_sizes.astype(np.float32),
        "quat_xyzw": quat,
        "basis": basis,
    }
    full_box = {
        "center": full_center.astype(np.float32),
        "half_sizes": full_half_sizes.astype(np.float32),
        "quat_xyzw": quat,
        "basis": basis,
    }
    fingertip_box = {
        "center": fingertip_center.astype(np.float32),
        "half_sizes": fingertip_half_sizes.astype(np.float32),
        "quat_xyzw": quat,
        "basis": basis,
    }
    return contact_box, full_box, fingertip_box


def _compute_gripper_body_bbox(
    robot_model: RobotModel,
    robot_conf: Optional[Any],
    ref_bbox: Optional[Dict[str, np.ndarray]],
    body_width_m: Optional[float] = None,
    body_height_m: Optional[float] = None,
    body_length_m: Optional[float] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """Compute a larger gripper body bbox with fixed width, sharing the same frame as the contact bbox."""
    if robot_conf is None or ref_bbox is None:
        return None
    # Mirror the contact frame but enlarge the body dimensions to wrap the entire gripper.
    dims = DEFAULT_GRIPPER_DIMS
    gripper_name = getattr(robot_conf, "gripper", "")
    if gripper_name:
        for name, values in GRIPPER_DIMENSIONS.items():
            if name.lower() == gripper_name.lower():
                dims = values
                break
    # Use provided or default fixed width
    width_m = body_width_m if body_width_m is not None else dims["default_width"]
    # Use provided or default body length
    length_m = body_length_m if body_length_m is not None else dims["length"]
    # Keep the same vertical size as contact bbox
    contact_half_sizes = ref_bbox["half_sizes"]
    if body_height_m is not None:
        height_m = float(body_height_m)
    else:
        height_m = float(contact_half_sizes[1] * 2.0)
    half_sizes = np.array([width_m / 2.0, height_m / 2.0, length_m / 2.0], dtype=np.float32)
    
    # Share the same basis as contact bbox
    basis = ref_bbox.get("basis")
    if basis is None:
        return None
    basis = np.asarray(basis, dtype=np.float32)
    
    # Position body box to share the same front face as contact box
    # Get the contact box's front face position (pad_midpoint)
    contact_center = ref_bbox["center"]
    approach_axis = basis[:, 2]
    
    # The contact box front face is at: contact_center - approach_axis * contact_half_sizes[2]
    # The body box front face should be at the same position
    # So: body_center = front_face + approach_axis * body_half_sizes[2]
    front_face = contact_center - approach_axis * contact_half_sizes[2]
    center = (front_face + approach_axis * half_sizes[2]).astype(np.float32)
    
    quat = _rotation_matrix_to_xyzw(basis)
    return {
        "center": center.astype(np.float32),
        "half_sizes": half_sizes.astype(np.float32),
        "quat_xyzw": quat,
        "basis": basis,
    }


def _compute_gripper_pad_points(
    robot_model: RobotModel,
    robot_conf: Optional[Any],
) -> Optional[np.ndarray]:
    """Return FK-based centers for left/right gripper pads when available."""
    if robot_conf is None:
        return None
    fk_map = getattr(robot_model, "latest_transforms", None)
    if not fk_map:
        return None

    def _find_link(keyword_groups):
        # Search through FK entries looking for the best textual match for each pad.
        for keywords in keyword_groups:
            for name in fk_map.keys():
                lname = name.lower()
                if all(k in lname for k in keywords):
                    return name
        return None

    left_link = _find_link([
        ("left", "inner", "finger", "pad"),
        ("left", "finger", "pad"),
        ("left", "inner", "finger"),
        ("left", "finger"),
        ("left", "knuckle"),
    ])
    right_link = _find_link([
        ("right", "inner", "finger", "pad"),
        ("right", "finger", "pad"),
        ("right", "inner", "finger"),
        ("right", "finger"),
        ("right", "knuckle"),
    ])

    points = []
    for link in (left_link, right_link):
        if link and link in fk_map:
            T = fk_map[link].matrix()
            p = T[:3, 3].astype(np.float32)
            points.append(p)
    if not points:
        return None
    return np.stack(points, axis=0).astype(np.float32)


def _align_bbox_with_point_cloud_com(
    bbox: Dict[str, np.ndarray],
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    search_radius_scale: float = 2.0,
    min_points_required: int = 10,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Align bounding box with center of mass of nearby points.
    
    This function adjusts the bbox center and rotation to match the point cloud,
    but preserves the z-coordinate (height) and bbox dimensions.
    
    Args:
        bbox: Dictionary with 'center', 'half_sizes', 'quat_xyzw', 'basis'
        points: Nx3 array of 3D points in world coordinates
        colors: Optional Nx3 array of RGB colors
        search_radius_scale: Scale factor for search radius (relative to bbox diagonal)
        min_points_required: Minimum points needed for alignment
        
    Returns:
        Aligned bbox dictionary or None if alignment fails
    """
    if bbox is None or points is None or len(points) == 0:
        return bbox
    
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] < min_points_required:
        return bbox
    
    # Extract bbox properties
    center = np.asarray(bbox["center"], dtype=np.float32)
    half_sizes = np.asarray(bbox["half_sizes"], dtype=np.float32)
    basis = np.asarray(bbox.get("basis", np.eye(3)), dtype=np.float32)
    
    # Define search radius based on bbox size
    bbox_diagonal = np.linalg.norm(half_sizes * 2.0)
    search_radius = bbox_diagonal * search_radius_scale
    
    # Find points near the bbox
    distances = np.linalg.norm(points - center[None, :], axis=1)
    nearby_mask = distances <= search_radius
    nearby_points = points[nearby_mask]
    
    if len(nearby_points) < min_points_required:
        # Not enough points for reliable alignment
        return bbox
    
    # Compute center of mass (but only adjust x-y, preserve z)
    com = np.mean(nearby_points, axis=0).astype(np.float32)
    aligned_center = center.copy()
    aligned_center[0] = com[0]  # Update x
    aligned_center[1] = com[1]  # Update y
    # aligned_center[2] stays the same (preserve height)
    
    # Estimate rotation from point cloud covariance (only in x-y plane)
    # Project points to x-y plane
    points_xy = nearby_points[:, :2] - com[:2]
    
    if len(points_xy) >= 3:
        # Compute covariance matrix in 2D
        cov = np.cov(points_xy.T)
        
        # Eigen decomposition to find principal axes
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # Sort by eigenvalue (descending)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, idx]
            
            # Build 3D rotation from 2D principal axes
            # Primary axis becomes x-axis, secondary becomes y-axis
            aligned_basis = basis.copy()
            aligned_basis[:2, 0] = eigenvectors[:, 0]  # Primary direction -> x
            aligned_basis[:2, 1] = eigenvectors[:, 1]  # Secondary direction -> y
            # Normalize to ensure orthonormal
            aligned_basis[:, 0] = aligned_basis[:, 0] / np.linalg.norm(aligned_basis[:, 0])
            aligned_basis[:, 1] = aligned_basis[:, 1] / np.linalg.norm(aligned_basis[:, 1])
            # Z-axis remains unchanged (preserve approach direction)
            
            # Recompute quaternion from aligned basis
            aligned_quat = _rotation_matrix_to_xyzw(aligned_basis)
        except np.linalg.LinAlgError:
            # If eigendecomposition fails, keep original rotation
            aligned_basis = basis
            aligned_quat = bbox["quat_xyzw"]
    else:
        aligned_basis = basis
        aligned_quat = bbox["quat_xyzw"]
    
    # Return aligned bbox with same dimensions
    return {
        "center": aligned_center,
        "half_sizes": half_sizes,
        "quat_xyzw": aligned_quat,
        "basis": aligned_basis,
    }


def _compute_bbox_corners_world(bbox: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    """Compute the 3D corners of a bounding box in world coordinates."""
    basis = bbox.get("basis")
    if basis is None:
        quat = bbox.get("quat_xyzw")
        if quat is None:
            return None
        basis = _quaternion_xyzw_to_rotation_matrix(quat)
    basis = np.asarray(basis, dtype=np.float32)
    center = np.asarray(bbox.get("center"), dtype=np.float32)
    half_sizes = np.asarray(bbox.get("half_sizes"), dtype=np.float32)
    if basis.shape != (3, 3) or center.shape != (3,) or half_sizes.shape != (3,):
        return None
    offsets = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    local = offsets * half_sizes[None, :]
    corners = center[None, :] + local @ basis.T
    return corners


def _project_bbox_pixels(corners_world: np.ndarray, intr: np.ndarray, extr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Project 3D bounding box corners into 2D image pixels."""
    corners_world = np.asarray(corners_world, dtype=np.float32)
    intr = np.asarray(intr, dtype=np.float32)
    extr = np.asarray(extr, dtype=np.float32)
    num = corners_world.shape[0]
    pixels = np.zeros((num, 2), dtype=np.float32)
    valid = np.zeros((num,), dtype=bool)
    if num == 0:
        return pixels, valid

    corners_h = np.concatenate([corners_world, np.ones((num, 1), dtype=np.float32)], axis=1)
    cam = (extr @ corners_h.T).T
    z = cam[:, 2]
    valid = z > 1e-6
    if not np.any(valid):
        return pixels, valid
    cam_valid = cam[valid]
    proj = (intr @ cam_valid.T).T
    proj_xy = proj[:, :2] / proj[:, 2:3]
    pixels[valid] = proj_xy
    return pixels, valid


def _generate_query_points_from_bbox(
    bbox: Dict[str, np.ndarray],
    timestamp: int,
    num_points: int = 8,
) -> np.ndarray:
    """
    Generate query points from bbox corners for tracking.
    
    Args:
        bbox: Bounding box dictionary
        timestamp: Frame timestamp for query points
        num_points: Number of points to generate (8 for corners)
        
    Returns:
        Query points array of shape [N, 4] with [t, x, y, z]
    """
    if bbox is None:
        return np.zeros((0, 4), dtype=np.float32)
    
    # Get bbox corners in world space
    corners = _compute_bbox_corners_world(bbox)
    if corners is None or len(corners) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    
    # Limit to requested number of points
    if num_points < len(corners):
        # Sample evenly
        indices = np.linspace(0, len(corners) - 1, num_points, dtype=int)
        corners = corners[indices]
    
    # Add timestamp column
    timestamps = np.full((len(corners), 1), float(timestamp), dtype=np.float32)
    query_points = np.hstack([timestamps, corners])

    return query_points


def _sample_points_within_bbox(
    points_world: np.ndarray,
    bbox: Optional[Dict[str, np.ndarray]],
    margin: float,
    surface: Optional[str] = None,
    surface_margin: float = 0.01,
) -> np.ndarray:
    """Return points from `points_world` that lie inside (or near) the bbox."""
    if bbox is None or points_world.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    center = np.asarray(bbox["center"], dtype=np.float32)
    half_sizes = np.asarray(bbox["half_sizes"], dtype=np.float32)
    basis = bbox.get("basis")
    if basis is None:
        quat = bbox.get("quat_xyzw")
        if quat is None:
            return np.empty((0, 3), dtype=np.float32)
        basis = _quaternion_xyzw_to_rotation_matrix(np.asarray(quat, dtype=np.float32))
    basis = np.asarray(basis, dtype=np.float32)

    local = (points_world - center[None, :]) @ basis
    within = np.all(np.abs(local) <= (half_sizes + margin), axis=1)

    if surface == "positive_z":
        within &= local[:, 2] >= (half_sizes[2] - surface_margin)
    elif surface == "negative_z":
        within &= local[:, 2] <= (-half_sizes[2] + surface_margin)
    elif surface == "mid_z":
        within &= np.abs(local[:, 2]) <= surface_margin

    return points_world[within]


def _random_sample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    """Randomly select up to `max_points` rows from `points`."""
    if points.shape[0] <= max_points:
        return points.astype(np.float32, copy=False)
    idx = GLOBAL_RNG.choice(points.shape[0], size=max_points, replace=False)
    return points[idx].astype(np.float32, copy=False)


def _extract_points_for_bbox(
    points_world: np.ndarray,
    bbox: Optional[Dict[str, np.ndarray]],
    max_samples: int,
    margins: Sequence[float],
    surface: Optional[str] = None,
    surface_margin: float = 0.01,
) -> np.ndarray:
    """Extract up to `max_samples` points from the cloud near the bbox."""
    for margin in margins:
        subset = _sample_points_within_bbox(
            points_world,
            bbox,
            margin=margin,
            surface=surface,
            surface_margin=surface_margin,
        )
        if subset.shape[0] > 0:
            return _random_sample_points(subset, max_samples)
    return np.empty((0, 3), dtype=np.float32)


def _generate_query_points_from_samples(
    samples_per_frame: Sequence[Optional[np.ndarray]],
    total_samples: int,
    max_frames: Optional[int] = None,
) -> np.ndarray:
    """Generate query points [t, x, y, z] from sampled point clouds."""
    if not samples_per_frame:
        return np.empty((0, 4), dtype=np.float32)

    frame_limit = len(samples_per_frame) if max_frames is None else min(len(samples_per_frame), max_frames)
    collected: List[np.ndarray] = []
    for t in range(frame_limit):
        pts = samples_per_frame[t]
        if pts is None or pts.size == 0:
            continue
        pts = np.asarray(pts, dtype=np.float32)
        timestamps = np.full((pts.shape[0], 1), float(t), dtype=np.float32)
        collected.append(np.hstack([timestamps, pts]))

    if not collected:
        return np.empty((0, 4), dtype=np.float32)

    stacked = np.concatenate(collected, axis=0)
    if stacked.shape[0] <= total_samples:
        return stacked.astype(np.float32, copy=False)
    idx = GLOBAL_RNG.choice(stacked.shape[0], size=total_samples, replace=False)
    return stacked[idx].astype(np.float32, copy=False)


def _sample_points_near_fingertip(
    body_bbox: Dict[str, np.ndarray],
    fingertip_bbox: Dict[str, np.ndarray],
    num_samples: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample points inside the body bbox near the fingertip bbox side faces."""
    if num_samples <= 0 or body_bbox is None or fingertip_bbox is None:
        return np.zeros((0, 3), dtype=np.float32)

    def _ensure_basis(box: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        basis_arr = box.get("basis")
        if basis_arr is not None:
            return np.asarray(basis_arr, dtype=np.float32)
        quat = box.get("quat_xyzw")
        if quat is None:
            return None
        return _quaternion_xyzw_to_rotation_matrix(np.asarray(quat, dtype=np.float32))

    body_basis = _ensure_basis(body_bbox)
    if body_basis is None:
        return np.zeros((0, 3), dtype=np.float32)
    fingertip_basis = _ensure_basis(fingertip_bbox)
    if fingertip_basis is None:
        return np.zeros((0, 3), dtype=np.float32)

    # Assume fingertip bbox shares the body frame; fall back to aligning via body basis.
    if not np.allclose(body_basis, fingertip_basis, atol=1e-4):
        fingertip_basis = body_basis

    body_center = np.asarray(body_bbox["center"], dtype=np.float32)
    body_half = np.asarray(body_bbox["half_sizes"], dtype=np.float32)
    fingertip_center = np.asarray(fingertip_bbox["center"], dtype=np.float32)
    fingertip_half = np.asarray(fingertip_bbox["half_sizes"], dtype=np.float32)

    rng = rng or np.random.default_rng()

    # Compute fingertip center in body-local coordinates.
    fingertip_local_center = body_basis.T @ (fingertip_center - body_center)

    # Margin so that sampled points hug the fingertip surfaces but stay inside the body.
    max_margins = np.maximum(body_half - fingertip_half, 0.0)
    base_margin = np.maximum(fingertip_half * 0.25, 0.002)
    margins = np.minimum(np.maximum(max_margins, 0.0), base_margin + max_margins)
    margins = np.maximum(margins, 0.002)

    samples_local = np.zeros((num_samples, 3), dtype=np.float32)
    face_indices = rng.integers(0, 6, size=num_samples)
    for idx, face in enumerate(face_indices):
        axis = face // 2  # 0:x, 1:y, 2:z
        sign = -1.0 if face % 2 == 0 else 1.0

        coord = fingertip_local_center.copy()
        # Sample coordinates along the two orthogonal axes within fingertip extents.
        for other_axis in (a for a in range(3) if a != axis):
            extent = fingertip_half[other_axis]
            coord[other_axis] = fingertip_local_center[other_axis] + rng.uniform(-extent, extent)

        # Offset slightly outward from the fingertip surface along the chosen axis.
        available_margin = margins[axis]
        offset = rng.uniform(0.0, available_margin)
        coord[axis] = fingertip_local_center[axis] + sign * (fingertip_half[axis] + offset)

        # Clamp to remain inside the body bbox.
        coord = np.clip(coord, -body_half + 1e-4, body_half - 1e-4)
        samples_local[idx] = coord

    samples_world = body_center[None, :] + samples_local @ body_basis.T
    return samples_world.astype(np.float32)


def _generate_body_query_points(
    body_bboxes: Sequence[Optional[Dict[str, np.ndarray]]],
    fingertip_bboxes: Sequence[Optional[Dict[str, np.ndarray]]],
    total_samples: int = 500,
    max_frames: int = 3,
) -> np.ndarray:
    """Generate query points from early frames by sampling the body bbox near the fingertip."""
    if total_samples <= 0 or not body_bboxes or not fingertip_bboxes:
        return np.zeros((0, 4), dtype=np.float32)

    frames = min(max_frames, len(body_bboxes), len(fingertip_bboxes))
    if frames <= 0:
        return np.zeros((0, 4), dtype=np.float32)

    rng = np.random.default_rng()
    queries: List[np.ndarray] = []
    collected = 0
    per_frame = max(1, int(np.ceil(total_samples / frames)))

    for t in range(frames):
        if collected >= total_samples:
            break
        body_bbox = body_bboxes[t]
        fingertip_bbox = fingertip_bboxes[t]
        if body_bbox is None or fingertip_bbox is None:
            continue
        remaining = total_samples - collected
        samples_needed = min(remaining, per_frame)
        samples_world = _sample_points_near_fingertip(
            body_bbox,
            fingertip_bbox,
            samples_needed,
            rng=rng,
        )
        if samples_world.size == 0:
            continue
        timestamps = np.full((samples_world.shape[0], 1), float(t), dtype=np.float32)
        queries.append(np.hstack([timestamps, samples_world]))
        collected += samples_world.shape[0]

    if not queries:
        return np.zeros((0, 4), dtype=np.float32)

    query_points = np.concatenate(queries, axis=0)
    if query_points.shape[0] > total_samples:
        # Evenly subsample to the requested total.
        indices = np.linspace(0, query_points.shape[0] - 1, total_samples, dtype=int)
        query_points = query_points[indices]

    return query_points.astype(np.float32)


def _load_gripper_tracker(
    tracker_name: str,
    device: str,
) -> Optional[nn.Module]:
    """Load the requested tracker backend."""
    tracker_name = tracker_name.lower()
    try:
        if tracker_name == "mvtracker":
            model = torch.hub.load(
                "ethz-vlg/mvtracker",
                "mvtracker",
                pretrained=True,
                device=device,
            )
            model.eval()
            return model
        if tracker_name == "cotracker3_online":
            base = CoTrackerOnlineWrapper(model_name="cotracker3_online")
        elif tracker_name == "cotracker3_offline":
            base = CoTrackerOfflineWrapper(model_name="cotracker3_offline")
        else:
            print(f"[WARN] Unsupported tracker '{tracker_name}'.")
            return None
        base = base.to(device)
        base.eval()
        wrapper = MonocularToMultiViewAdapter(base).to(device)
        wrapper.eval()
        return wrapper
    except Exception as exc:
        print(f"[WARN] Failed to load tracker '{tracker_name}': {exc}")
        return None


def _track_gripper(
    tracker_name: str,
    rgbs: torch.Tensor,
    depths: torch.Tensor,
    intrs: torch.Tensor,
    extrs: torch.Tensor,
    gripper_bboxes: Optional[List[Dict[str, np.ndarray]]],
    body_bboxes: Optional[List[Dict[str, np.ndarray]]],
    fingertip_bboxes: Optional[List[Dict[str, np.ndarray]]],
    point_samples: Optional[Dict[str, Sequence[Optional[np.ndarray]]]] = None,
    device: str = "cuda",
) -> Dict[str, Optional[torch.Tensor]]:
    """
    Track gripper bboxes using the requested tracker backend.
    """
    tracker_name = tracker_name.lower()
    print(f"[INFO] Tracking gripper with '{tracker_name}' backend...")
    tracker_model = _load_gripper_tracker(tracker_name, device=device)
    if tracker_model is None:
        return {
            "gripper_tracks": None,
            "gripper_vis": None,
            "body_tracks": None,
            "body_vis": None,
            "fingertip_tracks": None,
            "fingertip_vis": None,
        }

    result = {
        "gripper_tracks": None,
        "gripper_vis": None,
        "gripper_query_points": None,
        "body_tracks": None,
        "body_vis": None,
        "body_query_points": None,
        "fingertip_tracks": None,
        "fingertip_vis": None,
        "fingertip_query_points": None,
    }
    bbox_types = [
        ("gripper", gripper_bboxes),
        ("body", body_bboxes),
        ("fingertip", fingertip_bboxes),
    ]
    
    sample_config = {
        "gripper": (128, 3),
        "body": (500, 3),
        "fingertip": (128, 3),
    }

    for bbox_name, bboxes in bbox_types:
        if bboxes is None or len(bboxes) == 0:
            continue

        total_samples, frames_limit = sample_config.get(bbox_name, (128, 3))
        query_points_np = np.empty((0, 4), dtype=np.float32)

        samples_per_frame = None
        if point_samples is not None:
            samples_per_frame = point_samples.get(bbox_name)
            if samples_per_frame is not None:
                has_samples = any((pts is not None and np.asarray(pts).size > 0) for pts in samples_per_frame)
                if not has_samples:
                    samples_per_frame = None

        if samples_per_frame is not None:
            query_points_np = _generate_query_points_from_samples(
                samples_per_frame,
                total_samples=total_samples,
                max_frames=frames_limit,
            )

        if query_points_np.size == 0:
            if (
                bbox_name == "body"
                and body_bboxes is not None
                and fingertip_bboxes is not None
            ):
                query_points_np = _generate_body_query_points(
                    body_bboxes=body_bboxes,
                    fingertip_bboxes=fingertip_bboxes,
                    total_samples=total_samples,
                    max_frames=frames_limit,
                )
            else:
                query_points_collected = []
                for t, bbox in enumerate(bboxes):
                    if bbox is None:
                        continue
                    qp = _generate_query_points_from_bbox(bbox, timestamp=t, num_points=8)
                    if qp.size:
                        query_points_collected.append(qp)
                query_points_np = (
                    query_points_collected[0]
                    if query_points_collected
                    else np.zeros((0, 4), dtype=np.float32)
                )

        if query_points_np.size == 0:
            continue

        query_points = torch.from_numpy(query_points_np).float()
        result[f"{bbox_name}_query_points"] = query_points.clone().detach().cpu()
        
        print(f"  Tracking {bbox_name} bbox with {len(query_points)} query points...")

        rgbs_input = rgbs[None].to(device).float() / 255.0
        depths_input = depths[None].to(device).float()
        intrs_input = intrs[None].to(device).float()
        extrs_input = extrs[None].to(device).float()

        tracker_kwargs: Dict[str, torch.Tensor]
        if tracker_name == "mvtracker":
            tracker_kwargs = {"query_points_3d": query_points[None].to(device)}
        else:
            tracker_kwargs = {"query_points": query_points[None].to(device)}

        try:
            with torch.no_grad():
                tracker_result = tracker_model(
                    rgbs=rgbs_input,
                    depths=depths_input,
                    intrs=intrs_input,
                    extrs=extrs_input,
                    **tracker_kwargs,
                )

            result[f"{bbox_name}_tracks"] = tracker_result["traj_e"].cpu()
            result[f"{bbox_name}_vis"] = tracker_result["vis_e"].cpu()
            result[f"{bbox_name}_query_points"] = query_points.clone().detach().cpu()
            print(f"  ✓ {bbox_name} tracking complete: {tracker_result['traj_e'].shape}")
            
        except Exception as e:
            print(f"  ✗ Failed to track {bbox_name}: {e}")
            result[f"{bbox_name}_tracks"] = None
            result[f"{bbox_name}_vis"] = None
    
    return result


def _segment_object_with_sam(
    rgb_image: np.ndarray,
    bbox_prompt: Dict[str, float],
    model_type: str = "vit_b",
    sam_checkpoint: Optional[Path] = None,
) -> Optional[np.ndarray]:
    """
    Segment object using SAM (Segment Anything Model).
    
    Args:
        rgb_image: RGB image [H, W, 3] uint8
        bbox_prompt: Dictionary with 'center' [x, y] and 'width', 'height'
        model_type: SAM model type ('vit_b', 'vit_l', 'vit_h')
        sam_checkpoint: Path to SAM checkpoint
        
    Returns:
        Binary mask [H, W] or None if segmentation fails
    """
    try:
        from segment_anything import sam_model_registry, SamPredictor
    except ImportError:
        print("[WARN] segment_anything not installed. Install with: pip install segment-anything")
        return None
    
    # Load SAM model
    if sam_checkpoint is None:
        # Try to find checkpoint in common locations
        sam_checkpoint = Path.home() / ".cache" / "sam" / f"sam_{model_type}.pth"
        if not sam_checkpoint.exists():
            print(f"[WARN] SAM checkpoint not found at {sam_checkpoint}")
            return None
    
    try:
        sam = sam_model_registry[model_type](checkpoint=str(sam_checkpoint))
        sam.eval()
        predictor = SamPredictor(sam)
    except Exception as e:
        print(f"[WARN] Failed to load SAM model: {e}")
        return None
    
    # Set image
    predictor.set_image(rgb_image)
    
    # Convert bbox prompt to xyxy format
    cx, cy = bbox_prompt["center"]
    w, h = bbox_prompt["width"], bbox_prompt["height"]
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2
    bbox_xyxy = np.array([x1, y1, x2, y2])
    
    # Predict mask
    try:
        masks, scores, logits = predictor.predict(
            box=bbox_xyxy,
            multimask_output=False,
        )
        # Return best mask
        return masks[0] if len(masks) > 0 else None
    except Exception as e:
        print(f"[WARN] SAM prediction failed: {e}")
        return None


def _track_gripper_contact_objects_with_sam(
    rgbs: np.ndarray,
    gripper_bboxes_2d: List[Dict[str, float]],
    contact_threshold_pixels: float = 50.0,
    sam_model_type: str = "vit_b",
) -> Dict[str, List[Optional[np.ndarray]]]:
    """
    Track objects in contact with gripper using SAM.
    
    Segments objects near the gripper in each frame and tracks them
    across the sequence.
    
    Args:
        rgbs: [T, H, W, 3] RGB images
        gripper_bboxes_2d: List of T 2D bbox projections
        contact_threshold_pixels: Distance threshold for "contact"
        sam_model_type: SAM model type
        
    Returns:
        Dictionary with 'object_masks' (list of T masks)
    """
    print("[INFO] Tracking gripper contact objects with SAM...")
    
    object_masks = []
    
    for t, (rgb, bbox_2d) in enumerate(zip(rgbs, gripper_bboxes_2d)):
        if bbox_2d is None:
            object_masks.append(None)
            continue
        
        # Expand bbox to include nearby objects
        expanded_bbox = {
            "center": bbox_2d["center"],
            "width": bbox_2d["width"] + 2 * contact_threshold_pixels,
            "height": bbox_2d["height"] + 2 * contact_threshold_pixels,
        }
        
        # Segment with SAM
        mask = _segment_object_with_sam(
            rgb_image=rgb,
            bbox_prompt=expanded_bbox,
            model_type=sam_model_type,
        )
        
        object_masks.append(mask)
        
        if t % 10 == 0:
            print(f"  Segmented frame {t}/{len(rgbs)}")
    
    print(f"  ✓ SAM tracking complete: {len(object_masks)} frames")
    
    return {"object_masks": object_masks}


def _export_gripper_bbox_videos(
    args,
    rgbs: np.ndarray,
    intrs: np.ndarray,
    extrs: np.ndarray,
    bboxes: Optional[List[Optional[Dict[str, np.ndarray]]]],
    camera_ids: Sequence[str],
    clip_fps: float,
) -> None:
    """Export videos with gripper bounding boxes overlaid on RGB frames."""
    if bboxes is None or len(bboxes) == 0 or all(b is None for b in bboxes):
        print("[INFO] Skipping bbox video export: no gripper boxes available.")
        return

    video_dir = Path(args.out_dir) / "bbox_videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    fps_override = getattr(args, "bbox_video_fps", None)
    if fps_override is not None and fps_override > 0:
        fps = float(fps_override)
    else:
        fps = float(clip_fps if clip_fps > 0 else 30.0)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    color = (0, 165, 255)

    rgbs = np.asarray(rgbs)
    intrs = np.asarray(intrs)
    extrs = np.asarray(extrs)

    num_cams, num_frames = rgbs.shape[0], rgbs.shape[1]

    for ci in range(num_cams):
        cam_id = str(camera_ids[ci]) if camera_ids is not None else str(ci)
        frame_shape = rgbs[ci, 0].shape
        if len(frame_shape) != 3:
            continue
        height, width = frame_shape[0], frame_shape[1]
        video_path = video_dir / f"{args.task_folder.name}_cam_{cam_id}_bbox.mp4"
        writer = cv2.VideoWriter(str(video_path), fourcc, float(fps), (width, height))
        if not writer.isOpened():
            print(f"[WARN] Could not open video writer for {video_path}.")
            continue

        for ti in range(num_frames):
            frame_rgb = rgbs[ci, ti]
            frame_bgr = np.ascontiguousarray(frame_rgb[:, :, ::-1])
            bbox = bboxes[ti]
            if bbox is not None:
                corners_world = _compute_bbox_corners_world(bbox)
                if corners_world is not None:
                    pixels, valid = _project_bbox_pixels(corners_world, intrs[ci, ti], extrs[ci, ti])
                    for a, b in edges:
                        if valid[a] and valid[b]:
                            pt1 = tuple(np.round(pixels[a]).astype(int))
                            pt2 = tuple(np.round(pixels[b]).astype(int))
                            cv2.line(frame_bgr, pt1, pt2, color, 2)
            writer.write(frame_bgr)

        writer.release()
        print(f"[INFO] Wrote bbox overlay video: {video_path}")


def _project_points_to_pixels(
    points_world: np.ndarray,
    intr: np.ndarray,
    extr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project 3D world points into image pixel coordinates."""
    points_world = np.asarray(points_world, dtype=np.float32)
    if points_world.ndim != 2 or points_world.shape[1] != 3 or points_world.size == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=bool)

    intr = np.asarray(intr, dtype=np.float32)
    extr = np.asarray(extr, dtype=np.float32)

    ones = np.ones((points_world.shape[0], 1), dtype=np.float32)
    homo = np.concatenate([points_world, ones], axis=1)
    cam = (extr @ homo.T).T  # [N, 3]
    z = cam[:, 2]
    valid = z > 1e-6
    pixels = np.zeros((points_world.shape[0], 2), dtype=np.float32)
    if np.any(valid):
        proj = (intr @ cam[valid].T).T
        denom = np.maximum(proj[:, 2:3], 1e-6)
        pixels_valid = proj[:, :2] / denom
        pixels[valid] = pixels_valid
    return pixels, valid


def _export_tracking_videos(
    args,
    rgbs: np.ndarray,
    intrs: np.ndarray,
    extrs: np.ndarray,
    camera_ids: Sequence[str],
    track_sets: Sequence[Dict[str, Any]],
    clip_fps: float,
) -> None:
    """Export per-camera RGB videos with tracking overlays."""
    if not track_sets:
        return

    rgbs = np.asarray(rgbs)
    intrs = np.asarray(intrs)
    extrs = np.asarray(extrs)
    num_cams = rgbs.shape[0]
    num_frames = rgbs.shape[1] if rgbs.ndim >= 2 else 0

    if num_cams == 0 or num_frames == 0:
        return

    video_root = Path(args.out_dir) / "track_videos"
    video_root.mkdir(parents=True, exist_ok=True)

    fps_override = getattr(args, "track_video_fps", None)
    if fps_override is not None and fps_override > 0:
        video_fps = float(fps_override)
    else:
        video_fps = float(clip_fps if clip_fps > 0 else 30.0)

    point_radius = int(getattr(args, "track_video_point_radius", 3))
    line_thickness = int(getattr(args, "track_video_line_thickness", 1))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    for track_cfg in track_sets:
        name = track_cfg.get("name", "tracks")
        kind = track_cfg.get("kind", "3d")
        colors_bgr = np.asarray(track_cfg.get("colors_bgr"), dtype=np.uint8)
        start_frames = np.asarray(track_cfg.get("start_frames"), dtype=np.int64)

        if kind == "3d":
            tracks_world = np.asarray(track_cfg.get("tracks_world"), dtype=np.float32)
            vis = np.asarray(track_cfg.get("vis"), dtype=np.float32)
            if tracks_world.ndim != 3 or vis.ndim != 2:
                print(f"[WARN] Skipping track video '{name}': invalid 3D track shapes {tracks_world.shape}, {vis.shape}")
                continue
        elif kind == "2d":
            tracks_pixels = np.asarray(track_cfg.get("tracks_pixels"), dtype=np.float32)
            vis = np.asarray(track_cfg.get("vis"), dtype=np.float32)
            if tracks_pixels.ndim != 4:
                print(f"[WARN] Skipping track video '{name}': invalid 2D track shape {tracks_pixels.shape}")
                continue
        else:
            print(f"[WARN] Skipping track video '{name}': unknown track type '{kind}'.")
            continue

        num_points = int(colors_bgr.shape[0]) if colors_bgr.size else (tracks_world.shape[2] if kind == "3d" else tracks_pixels.shape[2])
        if colors_bgr.size == 0:
            cmap = matplotlib.colormaps["gist_rainbow"]
            palette = cmap(np.linspace(0.0, 1.0, max(1, num_points)))[:, :3]
            colors_rgb = (palette * 255.0).astype(np.uint8)
            colors_bgr = colors_rgb[:, ::-1]

        if start_frames.size == 0:
            start_frames = np.zeros((num_points,), dtype=np.int64)

        track_dir = video_root / name
        track_dir.mkdir(parents=True, exist_ok=True)

        for ci in range(num_cams):
            cam_id = str(camera_ids[ci]) if camera_ids is not None and ci < len(camera_ids) else str(ci)
            frame_shape = rgbs[ci, 0].shape
            if len(frame_shape) != 3:
                continue
            height, width = frame_shape[0], frame_shape[1]

            video_path = track_dir / f"{args.task_folder.name}_cam_{cam_id}_{name}.mp4"
            writer = cv2.VideoWriter(str(video_path), fourcc, video_fps, (width, height))
            if not writer.isOpened():
                print(f"[WARN] Could not open video writer for {video_path}.")
                continue

            prev_pixels = np.full((num_points, 2), np.nan, dtype=np.float32)

            for ti in range(num_frames):
                frame_rgb = rgbs[ci, ti]
                frame_bgr = np.ascontiguousarray(frame_rgb[:, :, ::-1])

                if kind == "3d":
                    pixels, depth_valid = _project_points_to_pixels(tracks_world[ti], intrs[ci, ti], extrs[ci, ti])
                    visibility = (vis[ti] > 0.5) & depth_valid & (ti >= start_frames)
                else:  # 2d
                    pixels = tracks_pixels[ci, ti]
                    visibility = (vis[ci, ti] > 0.5) if vis.ndim == 4 else np.ones((num_points,), dtype=bool)
                    visibility = visibility & (ti >= start_frames)

                for idx in range(num_points):
                    if idx >= pixels.shape[0]:
                        continue
                    if not visibility[idx]:
                        prev_pixels[idx, :] = np.nan
                        continue

                    px, py = float(pixels[idx, 0]), float(pixels[idx, 1])
                    if not (0 <= px < width and 0 <= py < height):
                        prev_pixels[idx, :] = np.nan
                        continue

                    color = tuple(int(c) for c in colors_bgr[idx])
                    pt = (int(round(px)), int(round(py)))
                    cv2.circle(frame_bgr, pt, point_radius, color, thickness=-1, lineType=cv2.LINE_AA)

                    if not np.isnan(prev_pixels[idx, 0]):
                        prev_pt = (int(round(prev_pixels[idx, 0])), int(round(prev_pixels[idx, 1])))
                        cv2.line(frame_bgr, prev_pt, pt, color, thickness=line_thickness, lineType=cv2.LINE_AA)
                    prev_pixels[idx, :] = (px, py)

                writer.write(frame_bgr)

            writer.release()
            print(f"[INFO] Wrote tracking overlay video: {video_path}")


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
    """Convert a robot model at a specific joint state to a point cloud."""
    # Update the robot model with the current joint angles
    # This transforms the internal meshes to the correct pose
    try:
        # Ensure joint_angles is a numpy array and convert to list for robot_model.update()
        if isinstance(joint_angles, np.ndarray):
            joint_angles_list = joint_angles.tolist()
        else:
            joint_angles_list = list(joint_angles)
        robot_model.update(joint_angles_list, first_time=is_first_time)
    except Exception as e:
        print(f"[ERROR] robot_model.update() failed: {e}")
        import traceback
        traceback.print_exc()
        return o3d.geometry.PointCloud()

    fk = None
    joint_sequence = getattr(robot_model, "joint_sequence", [])
    if joint_sequence:
        # Ensure joint_angles is array-like for indexing
        joint_angles_arr = np.asarray(joint_angles) if not isinstance(joint_angles, np.ndarray) else joint_angles
        if len(joint_sequence) != len(joint_angles_arr):
            warnings.warn(
                f"Joint angle length ({len(joint_angles_arr)}) does not match robot joint sequence ({len(joint_sequence)})."
            )
        try:
            joint_map = {name: float(joint_angles_arr[i]) for i, name in enumerate(joint_sequence)}
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
                if mesh_basename and mesh_basename in mtl_colors:
                    num_points = len(mesh_pcd.points)
                    mtl_color = mtl_colors[mesh_basename]
                    mesh_pcd.colors = o3d.utility.Vector3dVector(np.tile(mtl_color, (num_points, 1)))
                    color_assigned = True
            
            # Second try: URDF material colors
            if not color_assigned and urdf_colors:
                link_name = robot_model.get_link_for_mesh(mesh)
                if link_name and link_name in urdf_colors:
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
    robot_gripper_boxes: Optional[List[Optional[Dict[str, np.ndarray]]]] = [] if (args.add_robot and getattr(args, "gripper_bbox", False)) else None
    robot_gripper_body_boxes: Optional[List[Optional[Dict[str, np.ndarray]]]] = [] if (args.add_robot and getattr(args, "gripper_body_bbox", False)) else None
    robot_gripper_fingertip_boxes: Optional[List[Optional[Dict[str, np.ndarray]]]] = [] if (args.add_robot and getattr(args, "gripper_fingertip_bbox", False)) else None
    robot_gripper_pad_points: Optional[List[Optional[np.ndarray]]] = [] if (args.add_robot and getattr(args, "gripper_pad_points", False)) else None
    robot_tcp_points: Optional[List[Optional[np.ndarray]]] = [] if (args.add_robot and getattr(args, "tcp_points", False)) else None
    robot_object_points: Optional[List[Optional[np.ndarray]]] = [] if getattr(args, "object_points", False) else None

    track_gripper_flag = getattr(args, "track_gripper_with_mvtracker", False) or getattr(args, "track_gripper", True)
    contact_samples_per_frame: Optional[List[Optional[np.ndarray]]] = [] if track_gripper_flag else None
    body_samples_per_frame: Optional[List[Optional[np.ndarray]]] = [] if track_gripper_flag else None
    fingertip_samples_per_frame: Optional[List[Optional[np.ndarray]]] = [] if track_gripper_flag else None

    color_lookup_low = [list_frames(cdir / 'color') for cdir in cam_dirs_low]
    depth_lookup_low = [list_frames(cdir / 'depth') for cdir in cam_dirs_low]

    if args.high_res_folder and cam_dirs_high:
        color_lookup_high = [list_frames(cdir / 'color') for cdir in cam_dirs_high]
    else:
        color_lookup_high = None

    # Create robot model if requested
    robot_model = None
    robot_conf: Optional[Any] = None
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
                robot_conf = conf
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
        bbox_entry_for_frame: Optional[Dict[str, np.ndarray]] = None
        full_bbox_for_frame: Optional[Dict[str, np.ndarray]] = None
        pad_pts_for_frame: Optional[np.ndarray] = None
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
                current_gripper_width_mm: Optional[float] = None
                # Get the timestamp for the current frame
                t_low = int(per_cam_low_ts[0][ti])
                # Get joint angles at this timestamp
                # Use scene_high if available (it has the robot data), otherwise fall back to scene_low
                robot_scene = scene_low if scene_low is not None else scene_high
                joint_angles = robot_scene.get_joint_angles_aligned(t_low)
                
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
                # Use scene_high if available (it has the robot data), otherwise fall back to scene_low
                robot_scene = scene_low if scene_low is not None else scene_high
                if hasattr(robot_scene, 'gripper') and len(robot_scene.gripper) > 0:
                    # Find first camera with gripper data
                    for cam_id in sorted(robot_scene.gripper.keys()):
                        if len(robot_scene.gripper[cam_id]) > 0:
                            gripper_data_source = robot_scene.gripper[cam_id]
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
                    # Convert gripper width to finger_joint angle in radians
                    # finger_joint: 0 rad (open) to ~0.8 rad (closed)
                    max_width_mm = 85.0
                    max_angle_rad = 0.8  # ~45 degrees when fully closed
                    # Invert: larger width = smaller angle (more open)
                    gripper_joint_angle = max_angle_rad * (1.0 - gripper_width_mm / max_width_mm)
                    current_gripper_width_mm = float(gripper_width_mm)
                    
                    # Update the robot_joint_angles array with the calculated angle
                    robot_joint_angles[-1] = gripper_joint_angle
                    
                    if ti == 0:
                        print(f"[INFO] Gripper width at frame 0: {gripper_width_mm:.2f} mm (finger_joint: {gripper_joint_angle:.3f} rad)")
                
                robot_pcd = _robot_model_to_pointcloud(
                    robot_model,
                    robot_joint_angles,
                    is_first_time=(ti == 0),
                    points_per_mesh=10000,
                    debug_mode=getattr(args, "debug_mode", False),
                    urdf_colors=robot_urdf_colors,
                    mtl_colors=robot_mtl_colors,
                )
                if robot_gripper_boxes is not None:
                    # Use API's get_tcp_aligned for precise TCP pose (more accurate than FK)
                    #is it base ?
                    tcp_transform = None
                    try:
                        print("Getting TCP from API...")
                        # Get the official TCP pose from the dataset (7D: position + quaternion)
                        # This is the pre-processed, high-fidelity pose from the robot's controller
                        # Use scene_high if available (it has the robot data), otherwise fall back to scene_low
                        robot_scene = scene_low if scene_low is not None else scene_high
                        tcp_pose_7d = robot_scene.get_tcp_aligned(t_low)
                        # Validate that we got valid data
                        if tcp_pose_7d is not None and len(tcp_pose_7d) == 7:
                            # Convert 7D pose [x, y, z, qx, qy, qz, qw] to 4x4 matrix
                            tcp_transform = _pose_7d_to_matrix(tcp_pose_7d)
                            if ti == 0:
                                print(f"[INFO] Using API TCP pose for gripper bbox: position={tcp_pose_7d[:3]}, quat={tcp_pose_7d[3:]}")
                        else:
                            if ti == 0:
                                print(f"[WARN] API returned invalid TCP data: {tcp_pose_7d}")
                                print(f"[INFO] Falling back to FK-based TCP computation")
                            raise ValueError("Invalid TCP data from API")
                            
                    except TypeError as e:
                        if ti == 0 and "Invalid TCP data" not in str(e):
                            print(f"[WARN] Could not get TCP from API: {e}")
                            print(f"[INFO] Falling back to FK-based TCP computation")
                        # Fallback to FK if API method fails
                        if hasattr(robot_model, 'chain') and hasattr(robot_model, 'joint_sequence'):
                            joint_map = {}
                            for idx, joint_name in enumerate(robot_model.joint_sequence):
                                if idx < len(robot_joint_angles):
                                    joint_map[joint_name] = float(robot_joint_angles[idx])
                            
                            robot_type_str = getattr(robot_conf, "robot", None)
                            ee_link_name = ROBOT_EE_LINK_MAP.get(robot_type_str, "ee_link")
                            fk_result = robot_model.chain.forward_kinematics(joint_map)
                            
                            if ee_link_name in fk_result:
                                tcp_transform = fk_result[ee_link_name].matrix().astype(np.float32)
                                if ti == 0:
                                    print(f"[INFO] Using FK-based TCP from link '{ee_link_name}'")
                            elif ti == 0:
                                print(f"[WARN] EE link '{ee_link_name}' not found in FK result")
                    
                    bbox_entry_for_frame, base_full_bbox, fingertip_bbox_for_frame = _compute_gripper_bbox(
                        robot_model,
                        robot_conf,
                        current_gripper_width_mm,
                        contact_height_m=getattr(args, "gripper_bbox_contact_height_m", None),
                        contact_length_m=getattr(args, "gripper_bbox_contact_length_m", None),
                        tcp_transform=tcp_transform,
                    )
                    full_bbox_for_frame = base_full_bbox
                    if robot_gripper_body_boxes is not None and bbox_entry_for_frame is not None:
                        body_width_override = getattr(args, "gripper_body_width_m", None)
                        body_height_override = getattr(args, "gripper_body_height_m", None)
                        body_length_override = getattr(args, "gripper_body_length_m", None)
                        if (
                            body_width_override is not None
                            or body_height_override is not None
                            or body_length_override is not None
                        ):
                            full_bbox_for_frame = _compute_gripper_body_bbox(
                                robot_model,
                                robot_conf,
                                bbox_entry_for_frame,
                                body_width_m=body_width_override,
                                body_height_m=body_height_override,
                                body_length_m=body_length_override,
                            )
                    elif robot_gripper_body_boxes is None and full_bbox_for_frame is not None:
                        print("[WARN] Computed full gripper body bbox but no output list to store it")
                    elif robot_gripper_body_boxes is not None and full_bbox_for_frame is None:
                        print("[WARN] robot_gripper_body_boxes is not None but full_bbox_for_frame is None")
                    else:
                        print("[WARN] Could not compute gripper bbox for this frame")
                    
                    points_world_np = np.asarray(combined_pcd.points, dtype=np.float32)
                    colors_world_np = np.asarray(combined_pcd.colors, dtype=np.float32) if combined_pcd.has_colors() else None

                    if getattr(args, "align_bbox_with_points", True) and points_world_np.size > 0:
                        search_radius_scale = getattr(args, "align_bbox_search_radius_scale", 2.0)

                        # Align contact bbox
                        if bbox_entry_for_frame is not None:
                            aligned_bbox = _align_bbox_with_point_cloud_com(
                                bbox=bbox_entry_for_frame,
                                points=points_world_np,
                                colors=colors_world_np,
                                search_radius_scale=search_radius_scale,
                            )
                            if aligned_bbox is not None:
                                bbox_entry_for_frame = aligned_bbox
                                if ti == 0:
                                    print(f"[INFO] Aligned contact bbox with point cloud COM")

                        # Align body bbox
                        if full_bbox_for_frame is not None:
                            aligned_body = _align_bbox_with_point_cloud_com(
                                bbox=full_bbox_for_frame,
                                points=points_world_np,
                                colors=colors_world_np,
                                search_radius_scale=search_radius_scale,
                            )
                            if aligned_body is not None:
                                full_bbox_for_frame = aligned_body

                        # Align fingertip bbox
                        if fingertip_bbox_for_frame is not None:
                            aligned_fingertip = _align_bbox_with_point_cloud_com(
                                bbox=fingertip_bbox_for_frame,
                                points=points_world_np,
                                colors=colors_world_np,
                                search_radius_scale=search_radius_scale,
                            )
                            if aligned_fingertip is not None:
                                fingertip_bbox_for_frame = aligned_fingertip

                    if contact_samples_per_frame is not None:
                        contact_samples = _extract_points_for_bbox(
                            points_world_np,
                            bbox_entry_for_frame,
                            max_samples=64,
                            margins=[0.002, 0.005, 0.01],
                        )
                        contact_samples_per_frame.append(contact_samples if contact_samples.size else None)

                        body_samples = _extract_points_for_bbox(
                            points_world_np,
                            full_bbox_for_frame,
                            max_samples=500,
                            margins=[0.005, 0.01, 0.02],
                            surface="positive_z",
                            surface_margin=0.025,
                        )
                        body_samples_per_frame.append(body_samples if body_samples.size else None)

                        fingertip_samples = _extract_points_for_bbox(
                            points_world_np,
                            fingertip_bbox_for_frame,
                            max_samples=64,
                            margins=[0.002, 0.005, 0.01],
                        )
                        fingertip_samples_per_frame.append(fingertip_samples if fingertip_samples.size else None)

                if robot_gripper_pad_points is not None:
                    pad_pts_for_frame = _compute_gripper_pad_points(robot_model, robot_conf)
                
                # Get TCP point from API if requested
                tcp_pt_for_frame: Optional[np.ndarray] = None
                if robot_tcp_points is not None:
                    try:
                        robot_scene = scene_low if scene_low is not None else scene_high
                        tcp_pose_7d = robot_scene.get_tcp_aligned(t_low)
                        if tcp_pose_7d is not None and len(tcp_pose_7d) >= 3:
                            # Extract position (first 3 elements: x, y, z)
                            tcp_pt_for_frame = np.array(tcp_pose_7d[:3], dtype=np.float32)
                    except Exception as e:
                        if ti == 0:
                            print(f"[WARN] Could not get TCP point from API: {e}")
                
                # Get object point from API if requested and available
                obj_pt_for_frame: Optional[np.ndarray] = None
                if robot_object_points is not None:
                    try:
                        robot_scene = scene_low if scene_low is not None else scene_high
                        # Try to get object pose if the API has it
                        # This is speculative - the exact method name may vary
                        if hasattr(robot_scene, 'get_object_pose'):
                            obj_pose = robot_scene.get_object_pose(t_low)
                            if obj_pose is not None and len(obj_pose) >= 3:
                                obj_pt_for_frame = np.array(obj_pose[:3], dtype=np.float32)
                    except Exception as e:
                        if ti == 0:
                            print(f"[INFO] Object pose not available from API: {e}")
                
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

                    
                    width_info = (
                        f"{current_gripper_width_mm:.2f} mm"
                        if current_gripper_width_mm is not None
                        else "N/A"
                    )
                    if ti == 0:
                        print(
                            f"[INFO] Added robot with {len(robot_pcd.points)} points for frame {ti} "
                            f"(gripper width: {width_info})"
                        )
                    elif ti % 5 == 0:
                        print(
                            f"[INFO] Frame {ti}: Robot has {len(robot_pcd.points)} points "
                            f"(gripper width: {width_info})"
                        )
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

        if robot_gripper_boxes is not None:
            robot_gripper_boxes.append(bbox_entry_for_frame)
        if robot_gripper_body_boxes is not None:
            robot_gripper_body_boxes.append(full_bbox_for_frame)
        if robot_gripper_fingertip_boxes is not None:
            robot_gripper_fingertip_boxes.append(fingertip_bbox_for_frame)
        if contact_samples_per_frame is not None:
            if len(contact_samples_per_frame) < ti + 1:
                contact_samples_per_frame.append(None)
            if len(body_samples_per_frame) < ti + 1:
                body_samples_per_frame.append(None)
            if len(fingertip_samples_per_frame) < ti + 1:
                fingertip_samples_per_frame.append(None)
        if robot_gripper_pad_points is not None:
            # Ensure we append even if None to keep timeline alignment
            robot_gripper_pad_points.append(pad_pts_for_frame)
        if robot_tcp_points is not None:
            robot_tcp_points.append(tcp_pt_for_frame)
        if robot_object_points is not None:
            robot_object_points.append(obj_pt_for_frame)

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

    gripper_point_samples = None
    if contact_samples_per_frame is not None:
        gripper_point_samples = {
            "gripper": contact_samples_per_frame,
            "body": body_samples_per_frame,
            "fingertip": fingertip_samples_per_frame,
        }

    return (
        rgbs_out,
        depths_out,
        intrs_out,
        extrs_out,
        robot_debug_points,
        robot_debug_colors,
        robot_gripper_boxes,
        robot_gripper_body_boxes,
        robot_gripper_fingertip_boxes,
        robot_gripper_pad_points,
        robot_tcp_points,
        robot_object_points,
        gripper_point_samples,
    )

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
    robot_gripper_boxes=None,
    robot_gripper_body_boxes=None,
    robot_gripper_fingertip_boxes=None,
    robot_gripper_pad_points=None,
    robot_tcp_points=None,
    robot_object_points=None,
    gripper_point_samples=None,
    mvtracker_results=None,
    sam_results=None,
):
    """Saves the processed data to an NPZ file and generates a Rerun visualization."""
    timestamps = np.asarray(timestamps)
    frame_count = int(timestamps.size)
    clip_fps = getattr(args, "sync_fps", 0.0) if getattr(args, "sync_fps", 0.0) > 0 else 30.0
    if frame_count > 1:
        duration_ms = float(timestamps[-1] - timestamps[0])
        if duration_ms > 0.0:
            clip_fps = max(1e-6, (frame_count - 1) / (duration_ms / 1000.0))
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
        if robot_gripper_boxes:
            valid_box_count = sum(1 for box in robot_gripper_boxes if box)
            if valid_box_count > 0:
                fps = 30.0
                print(f"[INFO] Logging {valid_box_count} gripper bounding boxes to Rerun...")
                for idx, box in enumerate(robot_gripper_boxes):
                    if not box:
                        continue
                    rr.set_time_seconds("frame", idx / fps)
                    centers = np.asarray(box["center"], dtype=np.float32)[None, :]
                    half_sizes = np.asarray(box["half_sizes"], dtype=np.float32)[None, :]
                    rr.log(
                        "robot/gripper_bbox",
                        rr.Boxes3D(
                            centers=centers,
                            half_sizes=half_sizes,
                            quaternions=np.asarray(box["quat_xyzw"], dtype=np.float32)[None, :],
                            colors=np.array([[255, 128, 0]], dtype=np.uint8),
                        ),
                    )
        if robot_gripper_body_boxes:
            valid_box_count = sum(1 for box in robot_gripper_body_boxes if box)
            if valid_box_count > 0:
                fps = 30.0
                print(f"[INFO] Logging {valid_box_count} gripper BODY bounding boxes to Rerun...")
                for idx, box in enumerate(robot_gripper_body_boxes):
                    if not box:
                        continue
                    rr.set_time_seconds("frame", idx / fps)
                    centers = np.asarray(box["center"], dtype=np.float32)[None, :]
                    half_sizes = np.asarray(box["half_sizes"], dtype=np.float32)[None, :]
                    rr.log(
                        "robot/gripper_bbox_body",
                        rr.Boxes3D(
                            centers=centers,
                            half_sizes=half_sizes,
                            quaternions=np.asarray(box["quat_xyzw"], dtype=np.float32)[None, :],
                            colors=np.array([[255, 0, 0]], dtype=np.uint8),
                        ),
                    )
        if robot_gripper_fingertip_boxes:
            valid_box_count = sum(1 for box in robot_gripper_fingertip_boxes if box)
            if valid_box_count > 0:
                fps = 30.0
                print(f"[INFO] Logging {valid_box_count} gripper FINGERTIP bounding boxes to Rerun...")
                for idx, box in enumerate(robot_gripper_fingertip_boxes):
                    if not box:
                        continue
                    rr.set_time_seconds("frame", idx / fps)
                    centers = np.asarray(box["center"], dtype=np.float32)[None, :]
                    half_sizes = np.asarray(box["half_sizes"], dtype=np.float32)[None, :]
                    rr.log(
                        "robot/gripper_bbox_fingertip",
                        rr.Boxes3D(
                            centers=centers,
                            half_sizes=half_sizes,
                            quaternions=np.asarray(box["quat_xyzw"], dtype=np.float32)[None, :],
                            colors=np.array([[0, 0, 255]], dtype=np.uint8),  # Blue color
                        ),
                    )
        if robot_gripper_pad_points:
            fps = 30.0
            valid_pts = any(pts is not None and len(pts) > 0 for pts in robot_gripper_pad_points)
            if valid_pts:
                count = sum(1 for pts in robot_gripper_pad_points if pts is not None and len(pts) > 0)
                print(f"[INFO] Logging {count} gripper pad point sets to Rerun (magenta)...")
                for idx, pts in enumerate(robot_gripper_pad_points):
                    if pts is None or len(pts) == 0:
                        continue
                    rr.set_time_seconds("frame", idx / fps)
                    cols = np.tile(np.array([[255, 0, 255]], dtype=np.uint8), (len(pts), 1))
                    rr.log(
                        "robot/gripper_pad_points",
                        rr.Points3D(pts.astype(np.float32, copy=False), colors=cols),
                    )
        
        # Log TCP points from API (cyan spheres)
        if robot_tcp_points:
            fps = args.sync_fps if args.sync_fps > 0 else 30.0
            valid_pts = any(pt is not None and len(pt) == 3 for pt in robot_tcp_points)
            if valid_pts:
                count = sum(1 for pt in robot_tcp_points if pt is not None and len(pt) == 3)
                tcp_radius = getattr(args, "tcp_point_radius", 0.01)
                print(f"[INFO] Logging {count} TCP points from API to Rerun (cyan, radius={tcp_radius}m)...")
                for idx, pt in enumerate(robot_tcp_points):
                    if pt is None or len(pt) != 3:
                        continue
                    rr.set_time_seconds("frame", idx / fps)
                    # Log as a single cyan point with larger radius
                    rr.log(
                        "points/tcp_point",
                        rr.Points3D(
                            pt.reshape(1, 3).astype(np.float32, copy=False),
                            colors=np.array([[0, 255, 255]], dtype=np.uint8),  # Cyan
                            radii=np.array([0.02], dtype=np.float32),  # Increased for visibility
                        ),
                    )
        
        # Log object points from API (yellow spheres)
        if robot_object_points:
            fps = args.sync_fps if args.sync_fps > 0 else 30.0
            valid_pts = any(pt is not None and len(pt) == 3 for pt in robot_object_points)
            if valid_pts:
                count = sum(1 for pt in robot_object_points if pt is not None and len(pt) == 3)
                print(f"[INFO] Logging {count} object points from API to Rerun (yellow)...")
                for idx, pt in enumerate(robot_object_points):
                    if pt is None or len(pt) != 3:
                        continue
                    rr.set_time_seconds("frame", idx / fps)
                    # Log as a single yellow point
                    rr.log(
                        "points/object_point",
                        rr.Points3D(
                            pt.reshape(1, 3).astype(np.float32, copy=False),
                            colors=np.array([[255, 255, 0]], dtype=np.uint8),  # Yellow
                            radii=np.array([0.025], dtype=np.float32),  # Increased for visibility
                        ),
                    )

    if getattr(args, "export_bbox_video", False):
        try:
            _export_gripper_bbox_videos(
                args,
                rgbs,
                intrs,
                extrs,
                robot_gripper_boxes,
                final_cam_ids,
                clip_fps,
            )
        except Exception as exc:
            print(f"[WARN] Failed to export bounding box videos: {exc}")
    
    # Log MVTracker results if available
    if mvtracker_results is not None:
        print("[INFO] Logging MVTracker results to Rerun...")
        track_specs = [
            ("gripper", [255, 165, 0], 0),
            ("body", [255, 0, 0], 1),
            ("fingertip", [0, 0, 255], 2),
        ]
        track_video_specs: List[Dict[str, Any]] = []

        for track_name, color_rgb, method_id in track_specs:
            tracks = mvtracker_results.get(f"{track_name}_tracks")
            vis = mvtracker_results.get(f"{track_name}_vis")
            query_points = mvtracker_results.get(f"{track_name}_query_points")

            if tracks is None or vis is None or query_points is None:
                continue

            tracks_tensor = tracks if isinstance(tracks, torch.Tensor) else torch.from_numpy(np.asarray(tracks))
            vis_tensor = vis if isinstance(vis, torch.Tensor) else torch.from_numpy(np.asarray(vis))
            query_tensor = query_points if isinstance(query_points, torch.Tensor) else torch.from_numpy(np.asarray(query_points))

            if tracks_tensor.dim() == 3:
                tracks_batch = tracks_tensor.unsqueeze(0)
            elif tracks_tensor.dim() == 4:
                tracks_batch = tracks_tensor
            else:
                print(f"[WARN] Unexpected track tensor shape for '{track_name}' ({tuple(tracks_tensor.shape)}); skipping logging.")
                continue

            if vis_tensor.dim() == 2:
                vis_batch = vis_tensor.unsqueeze(0)
            elif vis_tensor.dim() == 3:
                vis_batch = vis_tensor
            else:
                print(f"[WARN] Unexpected visibility tensor shape for '{track_name}' ({tuple(vis_tensor.shape)}); skipping logging.")
                continue

            tracks_batch = tracks_batch.to(torch.float32)
            vis_batch = vis_batch.to(torch.float32)

            if query_tensor.dim() == 3 and query_tensor.shape[0] == 1:
                query_tensor = query_tensor[0]
            query_tensor = query_tensor.to(torch.float32)
            if query_tensor.dim() != 2 or query_tensor.shape[1] != 4:
                print(f"[WARN] Unexpected query tensor shape for '{track_name}' ({tuple(query_tensor.shape)}); skipping logging.")
                continue

            if frame_count > 0:
                frame_indices = query_tensor[:, 0].round().to(torch.long)
                frame_indices = torch.clamp(frame_indices, 0, frame_count - 1)
                query_tensor[:, 0] = frame_indices.to(torch.float32)
            else:
                frame_indices = torch.zeros((query_tensor.shape[0],), dtype=torch.long)

            num_frames = tracks_batch.shape[1]
            num_points = tracks_batch.shape[2]

            print(f"  Logging {track_name} tracks: {num_frames} frames, {num_points} points")

            log_tracks_to_rerun(
                dataset_name="rh20t_reprojection",
                datapoint_idx=0,
                predictor_name=f"mvtracker_{track_name}",
                gt_trajectories_3d_worldspace=None,
                gt_visibilities_any_view=None,
                query_points_3d=query_tensor.unsqueeze(0),
                pred_trajectories=tracks_batch,
                pred_visibilities=vis_batch,
                per_track_results=None,
                radii_scale=1.0,
                fps=clip_fps,
                sphere_radius_crop=None,
                sphere_center_crop=None,
                log_per_interval_results=False,
                max_tracks_to_log=None,
                track_batch_size=max(1, num_points),
                method_id=method_id,
                color_per_method_id={method_id: tuple(color_rgb)},
                memory_lightweight_logging=False,
            )

            color_rgb_arr = np.array(color_rgb, dtype=np.uint8)
            colors_bgr = np.tile(color_rgb_arr[::-1], (num_points, 1))

            track_video_specs.append(
                {
                    "name": f"mvtracker_{track_name}",
                    "kind": "3d",
                    "tracks_world": tracks_batch.squeeze(0).cpu().numpy(),
                    "vis": vis_batch.squeeze(0).cpu().numpy(),
                    "start_frames": frame_indices.cpu().numpy().astype(np.int64),
                    "colors_bgr": colors_bgr,
                }
            )

        if track_video_specs and getattr(args, "export_track_video", True):
            _export_tracking_videos(
                args=args,
                rgbs=rgbs,
                intrs=intrs,
                extrs=extrs,
                camera_ids=final_cam_ids,
                track_sets=track_video_specs,
                clip_fps=clip_fps,
            )
    
    # Log SAM results if available
    if sam_results is not None and "object_masks" in sam_results:
        print("[INFO] Logging SAM segmentation masks to Rerun...")
        fps = args.sync_fps if args.sync_fps > 0 else 30.0
        object_masks = sam_results["object_masks"]
        
        # Log masks as segmentation images
        num_masks = sum(1 for mask in object_masks if mask is not None)
        print(f"  Logging {num_masks} segmentation masks")
        
        for t, mask in enumerate(object_masks):
            if mask is None:
                continue
            
            rr.set_time_seconds("frame", t / fps)
            
            # Convert boolean mask to uint8 (0 or 255)
            mask_uint8 = (mask.astype(np.uint8) * 255)
            
            # Log as segmentation image (can be overlaid on RGB)
            rr.log(
                "segmentation/contact_objects",
                rr.SegmentationImage(mask_uint8),
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
    parser.add_argument("--config", default="RH20T/configs/configs.json", type=Path, help="Path to RH20T robot configs JSON.")
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
    parser.add_argument("--add-robot", action="store_true", help="Include robot arm model in Rerun visualization.")
    parser.add_argument(
        "--gripper-bbox",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Log an approximate gripper contact bbox in Rerun (requires --add-robot).",
    )
    parser.add_argument(
        "--gripper-bbox-contact-height-m",
        type=float,
        default=None,
        help="Override contact bbox vertical size in meters (full size, not half).",
    )
    parser.add_argument(
        "--gripper-bbox-contact-length-m",
        type=float,
        default=None,
        help="Override contact bbox length (approach axis) in meters (full size).",
    )
    parser.add_argument(
        "--gripper-body-bbox",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also log a larger gripper body bbox (fixed width).",
    )
    parser.add_argument(
        "--gripper-fingertip-bbox",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Log a blue bbox at the bottom of the body bbox (fingertip position).",
    )
    parser.add_argument(
        "--gripper-body-width-m",
        type=float,
        default=None,
        help="Body bbox width along jaw-separation axis (full size, fixed across frames).",
    )
    parser.add_argument(
        "--gripper-body-height-m",
        type=float,
        default=None,
        help="Body bbox thickness along the pad-normal axis (full size).",
    )
    parser.add_argument(
        "--gripper-body-length-m",
        type=float,
        default=None,
        help="Body bbox length along approach axis (full size).",
    )
    parser.add_argument(
        "--gripper-pad-points",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, log magenta points at the left/right gripper pad centers (FK-based).",
    )
    parser.add_argument(
        "--tcp-points",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, log cyan spheres at TCP positions from the API (requires --add-robot).",
    )
    parser.add_argument(
        "--tcp-point-radius",
        type=float,
        default=0.05,
        help="Radius of TCP point spheres in meters.",
    )
    parser.add_argument(
        "--object-points",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, log yellow spheres at object positions from the API if available.",
    )
    parser.add_argument(
        "--export-bbox-video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Export per-camera RGB videos with gripper bounding boxes overlaid.",
    )
    parser.add_argument(
        "--bbox-video-fps",
        type=float,
        default=30.0,
        help="Frame rate used when exporting gripper bounding box videos (requires --export-bbox-video).",
    )
    parser.add_argument(
        "--export-track-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export per-camera RGB videos with tracking overlays when tracking results are available.",
    )
    parser.add_argument(
        "--track-video-fps",
        type=float,
        default=None,
        help="Override FPS used for tracking overlay videos (defaults to bbox video FPS or the clip FPS).",
    )
    parser.add_argument(
        "--track-video-point-radius",
        type=int,
        default=3,
        help="Radius in pixels for drawing tracked points in tracking overlay videos.",
    )
    parser.add_argument(
        "--track-video-line-thickness",
        type=int,
        default=1,
        help="Line thickness (pixels) for connecting track points frame-to-frame in tracking overlay videos.",
    )
    parser.add_argument(
        "--debug-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable additional debug outputs such as red robot point overlays in Rerun.",
    )
    parser.add_argument(
        "--align-bbox-with-points",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Align gripper bboxes with center of mass of nearby point cloud (x-y and rotation only). Disable with --no-align-bbox-with-points.",
    )
    parser.add_argument(
        "--align-bbox-search-radius-scale",
        type=float,
        default=2.0,
        help="Scale factor for bbox alignment search radius (relative to bbox diagonal).",
    )
    parser.add_argument(
        "--track-gripper-with-mvtracker",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Deprecated alias for --track-gripper --tracker mvtracker. Kept for backwards compatibility.",
    )
    parser.add_argument(
        "--track-gripper",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable gripper tracking pipeline.",
    )
    parser.add_argument(
        "--tracker",
        choices=["mvtracker", "cotracker3_online", "cotracker3_offline"],
        default="mvtracker",
        help="Tracker backend to use for gripper tracking.",
    )
    parser.add_argument(
        "--track-objects-with-sam",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use SAM to track objects that the gripper will touch if visible.",
    )
    parser.add_argument(
        "--sam-model-type",
        type=str,
        default="vit_b",
        choices=["vit_b", "vit_l", "vit_h"],
        help="SAM model type to use for object segmentation.",
    )
    parser.add_argument(
        "--sam-contact-threshold",
        type=float,
        default=50.0,
        help="Distance threshold (pixels) for considering objects in contact with gripper.",
    )
    args = parser.parse_args()

    if getattr(args, "export_bbox_video", False) and not getattr(args, "gripper_bbox", False):
        print("[INFO] --export-bbox-video requires bounding boxes; enabling --gripper-bbox.")
        args.gripper_bbox = True

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
        #error here? hould be per_cam_high_full TODO: check
        per_cam_high = [arr[idx_high] for arr in per_cam_high_full] if per_cam_high_full is not None else None
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
    (rgbs,
     depths,
     intrs,
     extrs,
     robot_debug_points,
     robot_debug_colors,
     robot_gripper_boxes,
     robot_gripper_body_boxes,
     robot_gripper_fingertip_boxes,
     robot_gripper_pad_points,
     robot_tcp_points,
     robot_object_points,
     gripper_point_samples) = process_frames(
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

    # --- Step 2.5: Track gripper with MVTracker if requested ---
    mvtracker_results = None
    track_gripper_flag = getattr(args, "track_gripper", True)
    if getattr(args, "track_gripper_with_mvtracker", False):
        track_gripper_flag = True
        args.tracker = "mvtracker"
    if track_gripper_flag:
        tracker_name = getattr(args, "tracker", "mvtracker")
        if robot_gripper_boxes or robot_gripper_body_boxes or robot_gripper_fingertip_boxes:
            print(f"[INFO] Invoking gripper tracker ({tracker_name})...")
            try:
                # Convert to torch tensors
                rgbs_torch = torch.from_numpy(rgbs).permute(0, 1, 4, 2, 3)  # [V, T, H, W, 3] -> [V, T, 3, H, W]
                depths_torch = torch.from_numpy(depths).unsqueeze(2)  # [V, T, H, W] -> [V, T, 1, H, W]
                intrs_torch = torch.from_numpy(intrs)  # [V, T, 3, 3]
                extrs_torch = torch.from_numpy(extrs)  # [V, T, 3, 4]

                device = "cuda" if torch.cuda.is_available() else "cpu"
                mvtracker_results = _track_gripper(
                    tracker_name=tracker_name,
                    rgbs=rgbs_torch,
                    depths=depths_torch,
                    intrs=intrs_torch,
                    extrs=extrs_torch,
                    gripper_bboxes=robot_gripper_boxes,
                    body_bboxes=robot_gripper_body_boxes,
                    fingertip_bboxes=robot_gripper_fingertip_boxes,
                    point_samples=gripper_point_samples,
                    device=device,
                )
                print("[INFO] ✓ Gripper tracking complete")
            except Exception as e:
                print(f"[WARN] Gripper tracking failed: {e}")
                mvtracker_results = None
        else:
            print("[WARN] Gripper tracking requested but no bboxes computed. Enable --gripper-bbox.")

    # --- Step 2.6: Track contact objects with SAM if requested ---
    sam_results = None
    if getattr(args, "track_objects_with_sam", False):
        if robot_gripper_boxes:
            print("[INFO] Tracking contact objects with SAM...")
            try:
                # Project gripper bboxes to 2D for each camera view
                # We'll use the first camera for simplicity (can be extended to multi-view)
                cam_idx = 0
                gripper_bboxes_2d = []
                
                for t in range(len(robot_gripper_boxes)):
                    bbox_3d = robot_gripper_boxes[t]
                    if bbox_3d is None:
                        gripper_bboxes_2d.append(None)
                        continue
                    
                    # Get bbox corners in 3D
                    corners_3d = _compute_bbox_corners_world(bbox_3d)
                    if corners_3d is None:
                        gripper_bboxes_2d.append(None)
                        continue
                    
                    # Project to 2D
                    intr = intrs[cam_idx, t]
                    extr = extrs[cam_idx, t]
                    pixels, valid = _project_bbox_pixels(corners_3d, intr, extr)
                    
                    if valid.any():
                        # Compute 2D bounding box from projected corners
                        valid_pixels = pixels[valid]
                        x_min, y_min = valid_pixels.min(axis=0)
                        x_max, y_max = valid_pixels.max(axis=0)
                        center_x = (x_min + x_max) / 2
                        center_y = (y_min + y_max) / 2
                        width = x_max - x_min
                        height = y_max - y_min
                        
                        bbox_2d = {
                            "center": np.array([center_x, center_y], dtype=np.float32),
                            "width": float(width),
                            "height": float(height),
                        }
                        gripper_bboxes_2d.append(bbox_2d)
                    else:
                        gripper_bboxes_2d.append(None)
                
                # Extract RGB sequence for first camera
                rgbs_cam0 = rgbs[cam_idx]  # [T, H, W, 3]
                
                sam_results = _track_gripper_contact_objects_with_sam(
                    rgbs=rgbs_cam0,
                    gripper_bboxes_2d=gripper_bboxes_2d,
                    contact_threshold_pixels=getattr(args, "sam_contact_threshold", 50.0),
                    sam_model_type=getattr(args, "sam_model_type", "vit_b"),
                )
                print("[INFO] ✓ SAM object tracking complete")
            except Exception as e:
                print(f"[WARN] SAM tracking failed: {e}")
                import traceback
                traceback.print_exc()
                sam_results = None
        else:
            print("[WARN] --track-objects-with-sam enabled but no gripper bboxes computed. Enable --gripper-bbox.")

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
        robot_gripper_boxes,
        robot_gripper_body_boxes,
        robot_gripper_fingertip_boxes,
        robot_gripper_pad_points,
        robot_tcp_points,
        robot_object_points,
        gripper_point_samples,
        mvtracker_results=mvtracker_results,
        sam_results=sam_results,
    )

    if not args.no_pointcloud:
        rrd_path = args.out_dir / f"{args.task_folder.name}_reprojected.rrd"
        rr.save(str(rrd_path))
        print(f"✅ [OK] Saved Rerun visualization to: {rrd_path}")

if __name__ == "__main__":
    main()
