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
from PIL import Image
import cv2
import open3d as o3d
import rerun as rr
from tqdm import tqdm

# --- Project-Specific Imports ---
# Importing utilities and configurations specific to the RH20T dataset and robot model.
from mvtracker.utils.visualizer_rerun import log_pointclouds_to_rerun
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
MIN_CONTACT_WIDTH = 0.0001
MIN_CONTACT_LENGTH = 0.015

# Fallback scaling factors for contact width and height.
CONTACT_WIDTH_SCALE_FALLBACK = 0.65
CONTACT_HEIGHT_SCALE_FALLBACK = 0.45

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
    tcp_transform: Optional[np.ndarray] = None, #not used smh
) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[Dict[str, np.ndarray]], Optional[Dict[str, np.ndarray]]]:
    """Estimate contact bbox, full gripper bbox, and fingertip bbox aligned with the jaw frame."""
    # Bail out immediately if we have no robot configuration to describe the gripper.
    if robot_conf is None:
        return None, None, None

    # Grab the most recent FK results so we can derive pad transforms.
    # `latest_transforms` caches forward-kinematics results keyed by link names.
    fk_map = getattr(robot_model, "latest_transforms", None) or {}
    # Identify which robot variant we are working with to look up gripper metadata.
    robot_type = getattr(robot_conf, "robot", None)
    # Candidate finger link names to check for transform matrices.
    candidate_links = GRIPPER_LINK_CANDIDATES.get(robot_type, [])
    # Fallback end-effector link name if finger links are not available.

    #Not used
    ee_link = ROBOT_EE_LINK_MAP.get(robot_type)

    # 1. Consolidate all possible links we might use from the FK map.
    all_possible_links = candidate_links + ([ee_link] if ee_link else [])

    # 2. Check if the fk_map exists and contains at least one of our needed links.
    # The `fk_map and ...` prevents errors if fk_map is None.
    has_usable_fk = fk_map and any(link in fk_map for link in all_possible_links)

    # 3. If we have NEITHER a usable FK link NOR a tcp_transform, we cannot continue.
    if not has_usable_fk and tcp_transform is None:
        print("[Warning] No usable FK link data or TCP transform available; cannot compute gripper bbox.")
        return None, None, None

    # FK matrices are expressed in the world frame; keep both rotation and translation so all boxes stay world-aligned.
    # Extract the rotation submatrix that defines the gripper orientation.

    # Resolve gripper geometry to determine nominal pad and body dimensions.
    # Get the configured gripper name so we can look up size presets.
    gripper_name = getattr(robot_conf, "gripper", "")
    # Default dimensions when no specific data is found.
    dims = DEFAULT_GRIPPER_DIMS
    # Optional preset overrides for contact surface sizing.
    contact_preset = CONTACT_SURFACE_PRESETS.get(gripper_name, DEFAULT_CONTACT_SURFACE)
    if gripper_name:
        # Look for a case-insensitive match in the known gripper dimension table.
        for name, values in GRIPPER_DIMENSIONS.items():
            if name.lower() == gripper_name.lower():
                dims = values
                break

    # Determine the contact pad height either from override, preset, or scaled default.
    height_m = contact_height_m if contact_height_m is not None else contact_preset.get(
        "height", dims["height"] * CONTACT_HEIGHT_SCALE_FALLBACK
    )
    # Same logic for the length of the contact region along the approach axis.
    base_length_m = contact_length_m if contact_length_m is not None else contact_preset.get(
        "length", dims["length"] * CONTACT_HEIGHT_SCALE_FALLBACK
    )
    # Clearance defines how much extra distance to leave between contact and body boxes.
    clearance = contact_preset.get("clearance", DEFAULT_CONTACT_SURFACE["clearance"])

    # Enforce minimum size thresholds to avoid degenerate boxes.
    height_m = float(max(height_m, 0.005))
    base_length_m = float(max(base_length_m, MIN_CONTACT_LENGTH))
    # Store half-length for convenience when building half-extents.
    base_half_length = base_length_m / 2.0

    def _find_link(keyword_groups):
        # Search the FK cache for a link whose name contains all keywords in a group.
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
    # Repeat the link search for the right finger side.
    right_link = _find_link([
        ("right", "inner", "finger", "pad"),
        ("right", "finger", "pad"),
        ("right", "inner", "finger"),
        ("right", "finger"),
        ("right", "knuckle"),
    ])

    # Fetch homogeneous transforms for both finger pads if available.
    left_tf = fk_map[left_link].matrix().astype(np.float32) if left_link and left_link in fk_map else None
    right_tf = fk_map[right_link].matrix().astype(np.float32) if right_link and right_link in fk_map else None
    # Extract the position components for convenience.
    left_pos = left_tf[:3, 3] if left_tf is not None else None
    right_pos = right_tf[:3, 3] if right_tf is not None else None

    # If both pads available, derive a stable coordinate frame directly from their transforms
    if left_pos is not None and right_pos is not None:
        print("[Info] Both gripper pad transforms available; computing gripper frame from finger pads.")
        # Width axis: from right to left (jaw separation direction)
        # Difference between pad centers gives the opening direction vector.
        width_vec = left_pos - right_pos
        # Compute the magnitude so we can normalize and measure the opening.
        width_norm = np.linalg.norm(width_vec)
        if width_norm < 1e-6:
            return None, None, None
        # Normalized vector from right to left defines the width axis.
        width_axis = (width_vec / width_norm).astype(np.float32)
        # `width_axis` points from the right jaw to the left jaw, matching the physical opening direction.
        # Actual pad separation gives us the measured gripper width in meters.
        measured_width = float(width_norm)
        
        # Pad midpoint is the center point between the two finger pads
        # Averaging pad positions yields the natural center of the grasp.
        pad_midpoint = ((left_pos + right_pos) * 0.5).astype(np.float32)
        # Origin the frame at the midpoint so the bbox center coincides with the actual grasp center.
        
        # Approach axis: average Z-axis (forward direction) from both finger pads
        # This is stable and points toward the object being grasped
        # Average the local Z axes for a robust approach direction.
        approach_axis = ((left_tf[:3, 2] + right_tf[:3, 2]) * 0.5).astype(np.float32)
        # Normalize to ensure unit length.
        approach_norm = np.linalg.norm(approach_axis)
        if approach_norm < 1e-6:
            return None, None, None
        approach_axis = (approach_axis / approach_norm).astype(np.float32)
        
        # Height axis: perpendicular to both width and approach (completes right-handed frame)
        # This should point "up" relative to gripper (from tips toward palm)
        # Use the cross product to create the third axis of the orthonormal frame.
        height_axis = np.cross(approach_axis, width_axis).astype(np.float32)
        # Cross product yields the palm-up direction; together the three axes form a right-handed basis.
        # Confirm the resulting vector is valid by checking its length.
        height_norm = np.linalg.norm(height_axis)
        if height_norm < 1e-6:
            return None, None, None
        height_axis = (height_axis / height_norm).astype(np.float32)
        
        # Re-orthogonalize to ensure perfect right-handed frame
        # Recompute approach axis using the other two to eliminate accumulated error.
        approach_axis = np.cross(width_axis, height_axis).astype(np.float32)
        approach_axis = (approach_axis / np.linalg.norm(approach_axis)).astype(np.float32)
        # Recompute approach to guarantee orthogonality after any numeric drift from averaging pad transforms.
        
    elif left_tf is not None:
        print("[Warning] Only left gripper pad transform available; using left pad frame directly.")
        # Fallback: use only left pad
        # Use the left pad pose directly when the right pad is unavailable.
        pad_midpoint = left_tf[:3, 3].astype(np.float32)
        # Pad frames follow X = width, Y = height, Z = approach in URDF convention.
        # Read axes straight from the transform.
        width_axis = left_tf[:3, 0].astype(np.float32)  # X-axis of pad frame
        approach_axis = left_tf[:3, 2].astype(np.float32)  # Z-axis of pad frame
        height_axis = left_tf[:3, 1].astype(np.float32)  # Y-axis of pad frame
        measured_width = None
    elif right_tf is not None:
        print("[Warning] Only right gripper pad transform available; negating X axis to maintain left-to-right convention.")
        # Fallback: use only right pad
        # Use right pad pose when only that side is known.
        pad_midpoint = right_tf[:3, 3].astype(np.float32)
        # Negate the local X axis so right finger coordinates produce the same left-to-right convention.
        width_axis = -right_tf[:3, 0].astype(np.float32)  # Flip X to match left orientation
        approach_axis = right_tf[:3, 2].astype(np.float32)  # Z-axis of pad frame
        height_axis = right_tf[:3, 1].astype(np.float32)  # Y-axis of pad frame
        measured_width = None
    else:
        print("[Warning] No gripper pad transforms available; cannot compute gripper bbox.")
        # No pad transforms available
        # Without any finger pose we cannot build a reasonable frame.
        return None, None, None

    # Calculate contact width from measured or commanded gripper width
    # Collect all available width candidates before choosing the tightest value.
    width_candidates = []
    if measured_width is not None and gripper_width_mm is None:
        #use measured width only if no commanded width is given
        width_candidates.append(measured_width)
    if gripper_width_mm is not None:
        width_candidates.append(max(gripper_width_mm / 1000.0, MIN_CONTACT_WIDTH))
    if width_candidates:
        width_m = min(width_candidates)
    else:
        width_m = dims["default_width"] * CONTACT_WIDTH_SCALE_FALLBACK
    width_m = max(width_m - 2.0 * clearance, MIN_CONTACT_WIDTH)
    # Assemble the oriented box with half-extents
    # Compose the rotation matrix whose columns define the gripper's width/height/approach directions.
    # Stack axis vectors into a rotation matrix consistent with our constructed frame.
    basis = np.column_stack((width_axis, height_axis, approach_axis)).astype(np.float32)
    # Half-sizes for the tightest contact box around the fingertips.
    contact_half_sizes = np.array(
        [width_m / 2.0, height_m / 2.0, base_half_length],
        dtype=np.float32,
    )
    # Larger half-sizes for a safety box that encompasses the full gripper body.
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
    # Shift the box forward along the approach axis so its rear face coincides with the fingertip plane.
    # Move the contact box so its rear plane aligns with the fingertip surface.
    contact_center = (pad_midpoint + approach_axis * contact_half_sizes[2]).astype(np.float32)
    
    # Full box shares the same front face position
    # Position the full body box using the same logic but with its own depth.
    full_center = (pad_midpoint - approach_axis * full_half_sizes[2]).astype(np.float32)

    # Create a third bbox (blue) positioned at the TOP of the body bbox (opposite side from orange)
    # Same size as contact bbox (orange), but positioned inside the red body bbox
    # so its TOP face aligns with the red bbox's TOP face (toward the palm/wrist)
    # The fingertip overlay box mirrors the contact dimensions.
    fingertip_half_sizes = contact_half_sizes.copy()  # Same size as orange contact bbox
    
    # Position: top face aligned with body bbox top face (opposite direction from orange)
    # Body bbox top is at: full_center + approach_axis * full_half_sizes[2]
    # Blue bbox top should be at the same position
    # So: blue_center + approach_axis * fingertip_half_sizes[2] = full_center + approach_axis * full_half_sizes[2]
    # Therefore: blue_center = full_center + approach_axis * (full_half_sizes[2] - fingertip_half_sizes[2])
    # Offset inward so the fingertip box shares the same top (palm-side) plane as the larger body box.
    # Slide the fingertip box inward so its top plane matches the body box top plane.
    fingertip_center = (full_center + approach_axis * (full_half_sizes[2] - fingertip_half_sizes[2])).astype(np.float32)

    # Provide both tight contact box and larger safety box consumers can choose between.
    # Convert the basis to a quaternion for downstream consumers that expect quaternion orientation.
    quat = _rotation_matrix_to_xyzw(basis)
    # Package the contact region box.
    contact_box = {
        "center": contact_center.astype(np.float32),
        "half_sizes": contact_half_sizes.astype(np.float32),
        "quat_xyzw": quat,
        "basis": basis,
    }
    # Package the full body box.
    full_box = {
        "center": full_center.astype(np.float32),
        "half_sizes": full_half_sizes.astype(np.float32),
        "quat_xyzw": quat,
        "basis": basis,
    }
    # Package the fingertip overlay box.
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
    # March forward along the approach axis so the larger body box shares the same fingertip plane.
    center = (front_face - approach_axis * half_sizes[2]).astype(np.float32)
    
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
    """Align a bbox with the center of mass of nearby point-cloud samples (x/y translation, planar rotation)."""
    if bbox is None or points is None or len(points) == 0:
        return bbox

    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] < min_points_required:
        return bbox

    center = np.asarray(bbox["center"], dtype=np.float32)
    half_sizes = np.asarray(bbox["half_sizes"], dtype=np.float32)
    basis = np.asarray(bbox.get("basis", np.eye(3, dtype=np.float32)), dtype=np.float32)

    bbox_diagonal = np.linalg.norm(half_sizes * 2.0)
    search_radius = bbox_diagonal * search_radius_scale

    distances = np.linalg.norm(points - center[None, :], axis=1)
    nearby_mask = distances <= search_radius
    nearby_points = points[nearby_mask]
    if nearby_points.shape[0] < min_points_required:
        return bbox

    com = np.mean(nearby_points, axis=0).astype(np.float32)
    aligned_center = center.copy()
    # Only slide the box in the plane; trust the original Z so we do not fight gravity offsets.
    aligned_center[0] = com[0]
    aligned_center[1] = com[1]

    points_xy = nearby_points[:, :2] - com[:2]
    if points_xy.shape[0] >= 3:
        cov = np.cov(points_xy.T)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, idx]
            aligned_basis = basis.copy()
            aligned_basis[:2, 0] = eigenvectors[:, 0]
            aligned_basis[:2, 1] = eigenvectors[:, 1]
            aligned_basis[:, 0] /= np.linalg.norm(aligned_basis[:, 0]) + 1e-12
            aligned_basis[:, 1] /= np.linalg.norm(aligned_basis[:, 1]) + 1e-12
            # Replace yaw so the X/Y axes follow the dominant spread of nearby points.
            aligned_quat = _rotation_matrix_to_xyzw(aligned_basis)
        except np.linalg.LinAlgError:
            aligned_basis = basis
            aligned_quat = bbox["quat_xyzw"]
    else:
        aligned_basis = basis
        aligned_quat = bbox["quat_xyzw"]

    return {
        "center": aligned_center,
        "half_sizes": half_sizes,
        "quat_xyzw": aligned_quat,
        "basis": aligned_basis,
    }


def _ensure_basis(box: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    """Return a 3x3 rotation basis for a bbox, preferring stored basis over quaternion."""
    basis = box.get("basis")
    if basis is not None:
        basis_arr = np.asarray(basis, dtype=np.float32)
        if basis_arr.shape == (3, 3):
            return basis_arr
    quat = box.get("quat_xyzw")
    if quat is None:
        return None
    return _quaternion_xyzw_to_rotation_matrix(np.asarray(quat, dtype=np.float32))

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
    # Columns of `basis` are the box's local X/Y/Z directions expressed in world space.
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
    # Rotate the canonical cube by the local basis, then translate to the world-space center.
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
    # Extrinsics are world-to-camera; transform corners into the camera frame.
    cam = (extr @ corners_h.T).T
    z = cam[:, 2]
    valid = z > 1e-6
    if not np.any(valid):
        return pixels, valid
    cam_valid = cam[valid]
    proj = (intr @ cam_valid.T).T
    # Divide by depth to land in pixel coordinates using the pinhole model.
    proj_xy = proj[:, :2] / proj[:, 2:3]
    pixels[valid] = proj_xy
    return pixels, valid


def _export_gripper_bbox_videos(
    args,
    rgbs: np.ndarray,
    intrs: np.ndarray,
    extrs: np.ndarray,
    bboxes: Optional[List[Optional[Dict[str, np.ndarray]]]],
    camera_ids: Sequence[str],
) -> None:
    """Export videos with gripper bounding boxes overlaid on RGB frames."""
    if bboxes is None or len(bboxes) == 0 or all(b is None for b in bboxes):
        print("[INFO] Skipping bbox video export: no gripper boxes available.")
        return

    video_dir = Path(args.out_dir) / "bbox_videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    fps = getattr(args, "bbox_video_fps", 30.0)
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
                    # Project the world-space OBB back into the current camera to draw a 2D outline.
                    pixels, valid = _project_bbox_pixels(corners_world, intrs[ci, ti], extrs[ci, ti])
                    for a, b in edges:
                        if valid[a] and valid[b]:
                            pt1 = tuple(np.round(pixels[a]).astype(int))
                            pt2 = tuple(np.round(pixels[b]).astype(int))
                            cv2.line(frame_bgr, pt1, pt2, color, 2)
            writer.write(frame_bgr)

        writer.release()
        print(f"[INFO] Wrote bbox overlay video: {video_path}")


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
            # Grab the closest timestamps on or around the desired grid slot so we can apply jitter tolerance.
            if idx < arr_sub.size:
                candidate_indices.append(start_idx + idx)
            if idx > 0:
                # Also consider the frame just before the insertion point; it may still fall within tolerance.
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

    # Decompose the world-to-camera matrix so OpenCV can consume its Rodrigues representation.
    R, t = E[:3, :3], E[:3, 3]
    rvec, _ = cv2.Rodrigues(R)
    tvec = t.reshape(3, 1)

    # Project all 3D points into the 2D image plane of the target camera
    projected_pts, _ = cv2.projectPoints(pts_world, rvec, tvec, K, distCoeffs=None)
    projected_pts = projected_pts.squeeze(1)
    
    # Calculate the depth of each point relative to the new camera
    # Re-use the same transform to express points in the camera frame for depth/Z buffering.
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
    # empty list if defined in args, else None
    robot_gripper_body_boxes: Optional[List[Optional[Dict[str, np.ndarray]]]] = [] if (args.add_robot and getattr(args, "gripper_body_bbox", False)) else None
    robot_gripper_fingertip_boxes: Optional[List[Optional[Dict[str, np.ndarray]]]] = [] if (args.add_robot and getattr(args, "gripper_fingertip_bbox", False)) else None
    robot_gripper_pad_points: Optional[List[Optional[np.ndarray]]] = [] if (args.add_robot and getattr(args, "gripper_pad_points", False)) else None
    robot_tcp_points: Optional[List[Optional[np.ndarray]]] = [] if (args.add_robot and getattr(args, "tcp_points", False)) else None
    robot_object_points: Optional[List[Optional[np.ndarray]]] = [] if getattr(args, "object_points", False) else None

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
                # `+=` performs an in-place concatenation of geometry in Open3D, so we avoid repeated numpy conversions here.
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
                    # if we want the griper bbox
                    if robot_gripper_body_boxes is not None and bbox_entry_for_frame is not None:
                        #if size is set the override
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

        points_world_np: Optional[np.ndarray] = None
        colors_world_np: Optional[np.ndarray] = None
        if combined_pcd.has_points():
            points_world_np = np.asarray(combined_pcd.points, dtype=np.float32)
            if combined_pcd.has_colors():
                colors_world_np = np.asarray(combined_pcd.colors, dtype=np.float32)

        if (
            getattr(args, "align_bbox_with_points", True)
            and points_world_np is not None
            and points_world_np.size > 0
        ):
            search_radius_scale = getattr(args, "align_bbox_search_radius_scale", 2.0)
            if bbox_entry_for_frame is not None:
                aligned_contact = _align_bbox_with_point_cloud_com(
                    bbox_entry_for_frame,
                    points_world_np,
                    colors=colors_world_np,
                    search_radius_scale=search_radius_scale,
                )
                if aligned_contact is not None:
                    bbox_entry_for_frame = aligned_contact
            if full_bbox_for_frame is not None:
                aligned_body = _align_bbox_with_point_cloud_com(
                    full_bbox_for_frame,
                    points_world_np,
                    colors=colors_world_np,
                    search_radius_scale=search_radius_scale,
                )
                if aligned_body is not None:
                    full_bbox_for_frame = aligned_body
            if fingertip_bbox_for_frame is not None:
                aligned_tip = _align_bbox_with_point_cloud_com(
                    fingertip_bbox_for_frame,
                    points_world_np,
                    colors=colors_world_np,
                    search_radius_scale=search_radius_scale,
                )
                if aligned_tip is not None:
                    fingertip_bbox_for_frame = aligned_tip

        if robot_gripper_boxes is not None:
            robot_gripper_boxes.append(bbox_entry_for_frame)
        if robot_gripper_body_boxes is not None:
            robot_gripper_body_boxes.append(full_bbox_for_frame)
        if robot_gripper_fingertip_boxes is not None:
            robot_gripper_fingertip_boxes.append(fingertip_bbox_for_frame)
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

            # Persist the world-to-camera [R|t] so downstream consumers can project the world-space point cloud.
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
):
    """Saves the processed data to an NPZ file and generates a Rerun visualization."""
    # Convert to channels-first format for NPZ
    rgbs_final = np.moveaxis(rgbs, -1, 2)
    depths_final = depths[:, :, None, :, :]

    per_cam_ts_arr = np.stack(per_camera_timestamps, axis=0).astype(np.int64)
    # Persist all modalities together so downstream tools can reload the full synchronized packet.
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
        # Cast to tensors because the visualizer expects batched torch inputs.
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
                # Keep robot color palette aligned with the optional debug stream.
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
                    basis = _ensure_basis(box)
                    if basis is not None:
                        axes = np.asarray(basis, dtype=np.float32)
                        half_sizes_vec = np.asarray(box["half_sizes"], dtype=np.float32)
                        center_vec = np.asarray(box["center"], dtype=np.float32)

                        approach_axis = axes[:, 2]
                        approach_axis_norm = np.linalg.norm(approach_axis) + 1e-12
                        approach_axis = approach_axis / approach_axis_norm
                        # Draw helper arrows to confirm the oriented bounding box aligns with the jaw approach direction.
                        origin = center_vec - approach_axis * half_sizes_vec[2]
                        vector = approach_axis * (half_sizes_vec[2] * 2.0)
                        rr.log(
                            "robot/gripper_bbox_centerline",
                            rr.Arrows3D(
                                origins=origin[np.newaxis, :],
                                vectors=vector[np.newaxis, :],
                                colors=np.array([[255, 128, 0]], dtype=np.uint8),
                                radii=0.004,
                            ),
                        )
                        up_axis = axes[:, 1]
                        up_axis_norm = np.linalg.norm(up_axis) + 1e-12
                        up_axis = up_axis / up_axis_norm
                        rr.log(
                            "robot/gripper_bbox_axis_height",
                            rr.Arrows3D(
                                origins=center_vec[np.newaxis, :],
                                vectors=(up_axis * half_sizes_vec[1])[np.newaxis, :],
                                colors=np.array([[0, 200, 0]], dtype=np.uint8),
                                radii=0.003,
                            ),
                        )
                        width_axis = axes[:, 0]
                        width_axis_norm = np.linalg.norm(width_axis) + 1e-12
                        width_axis = width_axis / width_axis_norm
                        rr.log(
                            "robot/gripper_bbox_axis_width",
                            rr.Arrows3D(
                                origins=center_vec[np.newaxis, :],
                                vectors=(width_axis * half_sizes_vec[0])[np.newaxis, :],
                                colors=np.array([[0, 150, 255]], dtype=np.uint8),
                                radii=0.003,
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
                    # Namespace logs under "robot/..." so the inspector groups all overlays together.
                    rr.log(
                        "robot/gripper_bbox_body",
                        rr.Boxes3D(
                            centers=centers,
                            half_sizes=half_sizes,
                            quaternions=np.asarray(box["quat_xyzw"], dtype=np.float32)[None, :],
                            colors=np.array([[255, 0, 0]], dtype=np.uint8),
                        ),
                    )
                    basis = _ensure_basis(box)
                    if basis is not None:
                        axes = np.asarray(basis, dtype=np.float32)
                        half_sizes_vec = np.asarray(box["half_sizes"], dtype=np.float32)
                        center_vec = np.asarray(box["center"], dtype=np.float32)

                        approach_axis = axes[:, 2]
                        approach_axis_norm = np.linalg.norm(approach_axis) + 1e-12
                        approach_axis = approach_axis / approach_axis_norm
                        origin = center_vec - approach_axis * half_sizes_vec[2]
                        vector = approach_axis * (half_sizes_vec[2] * 2.0)
                        rr.log(
                            "robot/gripper_bbox_body_centerline",
                            rr.Arrows3D(
                                origins=origin[np.newaxis, :],
                                vectors=vector[np.newaxis, :],
                                colors=np.array([[255, 0, 0]], dtype=np.uint8),
                                radii=0.004,
                            ),
                        )
                        up_axis = axes[:, 1]
                        up_axis_norm = np.linalg.norm(up_axis) + 1e-12
                        up_axis = up_axis / up_axis_norm
                        rr.log(
                            "robot/gripper_bbox_body_axis_height",
                            rr.Arrows3D(
                                origins=center_vec[np.newaxis, :],
                                vectors=(up_axis * half_sizes_vec[1])[np.newaxis, :],
                                colors=np.array([[0, 200, 0]], dtype=np.uint8),
                                radii=0.003,
                            ),
                        )
                        width_axis = axes[:, 0]
                        width_axis_norm = np.linalg.norm(width_axis) + 1e-12
                        width_axis = width_axis / width_axis_norm
                        rr.log(
                            "robot/gripper_bbox_body_axis_width",
                            rr.Arrows3D(
                                origins=center_vec[np.newaxis, :],
                                vectors=(width_axis * half_sizes_vec[0])[np.newaxis, :],
                                colors=np.array([[0, 150, 255]], dtype=np.uint8),
                                radii=0.003,
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
                    basis = _ensure_basis(box)
                    if basis is not None:
                        axes = np.asarray(basis, dtype=np.float32)
                        half_sizes_vec = np.asarray(box["half_sizes"], dtype=np.float32)
                        center_vec = np.asarray(box["center"], dtype=np.float32)

                        approach_axis = axes[:, 2]
                        approach_axis_norm = np.linalg.norm(approach_axis) + 1e-12
                        approach_axis = approach_axis / approach_axis_norm
                        origin = center_vec - approach_axis * half_sizes_vec[2]
                        vector = approach_axis * (half_sizes_vec[2] * 2.0)
                        rr.log(
                            "robot/gripper_bbox_fingertip_centerline",
                            rr.Arrows3D(
                                origins=origin[np.newaxis, :],
                                vectors=vector[np.newaxis, :],
                                colors=np.array([[0, 0, 255]], dtype=np.uint8),
                                radii=0.004,
                            ),
                        )
                        up_axis = axes[:, 1]
                        up_axis_norm = np.linalg.norm(up_axis) + 1e-12
                        up_axis = up_axis / up_axis_norm
                        rr.log(
                            "robot/gripper_bbox_fingertip_axis_height",
                            rr.Arrows3D(
                                origins=center_vec[np.newaxis, :],
                                vectors=(up_axis * half_sizes_vec[1])[np.newaxis, :],
                                colors=np.array([[0, 200, 0]], dtype=np.uint8),
                                radii=0.003,
                            ),
                        )
                        width_axis = axes[:, 0]
                        width_axis_norm = np.linalg.norm(width_axis) + 1e-12
                        width_axis = width_axis / width_axis_norm
                        rr.log(
                            "robot/gripper_bbox_fingertip_axis_width",
                            rr.Arrows3D(
                                origins=center_vec[np.newaxis, :],
                                vectors=(width_axis * half_sizes_vec[0])[np.newaxis, :],
                                colors=np.array([[0, 150, 255]], dtype=np.uint8),
                                radii=0.003,
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
            )
        except Exception as exc:
            print(f"[WARN] Failed to export bounding box videos: {exc}")


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
        "--debug-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable additional debug outputs such as red robot point overlays in Rerun.",
    )
    parser.add_argument(
        "--align-bbox-with-points",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Align gripper bboxes with nearby point cloud COM (disable with --no-align-bbox-with-points).",
    )
    parser.add_argument(
        "--align-bbox-search-radius-scale",
        type=float,
        default=2.0,
        help="Scale factor for alignment search radius relative to bbox diagonal.",
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
     robot_object_points) = process_frames(
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
    # Set the desired coordinate system for the 'world' space
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # --- ADD THIS DIAGNOSTIC CODE ---
    # Log visible arrows to represent the axes of the 'world' space
    rr.log(
        "world/axes",
        rr.Arrows3D(
            origins=[[0, 0, 0]],
            vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]] # X=Red, Y=Green, Z=Blue
        ),
        static=True
    )
    # --- END OF DIAGNOSTIC CODE ---


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
    )

    if not args.no_pointcloud:
        rrd_path = args.out_dir / f"{args.task_folder.name}_reprojected.rrd"
        rr.save(str(rrd_path))
        print(f"✅ [OK] Saved Rerun visualization to: {rrd_path}")

if __name__ == "__main__":
    main()
