"""
Gripper Utilities for Robotic Manipulation

This module provides utilities for computing and manipulating gripper-related geometry including:
- Gripper bounding box computation from forward kinematics
- Gripper bounding box computation from TCP (Tool Center Point) pose
- Gripper pad point extraction
- Gripper body bbox computation

Constants:
    ROBOT_EE_LINK_MAP: Mapping of robot types to end-effector link names
    GRIPPER_LINK_CANDIDATES: Candidate gripper link names for different robots
    GRIPPER_DIMENSIONS: Default dimensions for various gripper models
    CONTACT_SURFACE_PRESETS: Contact surface dimensions for different grippers

Functions:
    _compute_gripper_bbox_from_tcp: Compute gripper bbox using TCP transform from API
    _compute_gripper_bbox: Compute gripper bbox using forward kinematics
    _compute_gripper_body_bbox: Compute larger gripper body bbox
    _compute_gripper_pad_points: Extract gripper pad center points from FK
"""

import numpy as np
import warnings
from typing import Any, Dict, Optional, Tuple
from utils.geometry_utils import _rotation_matrix_to_xyzw, _quaternion_xyzw_to_rotation_matrix
from RH20T.utils.robot import RobotModel


# Mapping of robot types to their end-effector (EE) link names.
ROBOT_EE_LINK_MAP = {
    "ur5": "ee_link",
    "flexiv": "link7",
    "kuka": "lbr_iiwa_link_7",
    "franka": "panda_link8",
}

# Candidate gripper links for different robot types.
GRIPPER_LINK_CANDIDATES = {
    "ur5": ["robotiq_arg2f_base_link", "ee_link", "wrist_3_link","wsg_50_base_link"],
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


def _compute_gripper_bbox_from_tcp(
    tcp_transform: np.ndarray,
    robot_conf: Optional[Any],
    gripper_width_mm: Optional[float],
    contact_height_m: Optional[float] = None,
    contact_length_m: Optional[float] = None,
) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[Dict[str, np.ndarray]], Optional[Dict[str, np.ndarray]]]:
    """
    Compute gripper bbox using TCP transform from API.
    
    This function uses the precise TCP (Tool Center Point) pose from the robot API
    to compute gripper bounding boxes instead of relying on forward kinematics.
    
    Args:
        tcp_transform: 4x4 transformation matrix for TCP pose
        robot_conf: Robot configuration object
        gripper_width_mm: Gripper width in millimeters
        contact_height_m: Override for contact bbox height
        contact_length_m: Override for contact bbox length
        
    Returns:
        Tuple of (contact_box, full_box, fingertip_box) dictionaries or (None, None, None)
    """
    if tcp_transform is None or robot_conf is None:
        if tcp_transform is None:
            print("[Warning] No TCP transform provided; cannot compute gripper bbox from TCP. Returning None.")
        if robot_conf is None:
            print("[Warning] No robot configuration provided; cannot compute gripper bbox from TCP. Returning None.")
        return None, None, None
    
    # Extract position and rotation from TCP transform
    tcp_position = tcp_transform[:3, 3].astype(np.float32)
    tcp_rotation = tcp_transform[:3, :3].astype(np.float32)
    
    # Get gripper dimensions
    gripper_name = getattr(robot_conf, "gripper", "")
    dims = DEFAULT_GRIPPER_DIMS
    contact_preset = CONTACT_SURFACE_PRESETS.get(gripper_name, DEFAULT_CONTACT_SURFACE)
    if gripper_name:
        for name, values in GRIPPER_DIMENSIONS.items():
            if name.lower() == gripper_name.lower():
                dims = values
                break
    
    # Determine contact dimensions
    height_m = contact_height_m if contact_height_m is not None else contact_preset.get(
        "height", dims["height"] * CONTACT_HEIGHT_SCALE_FALLBACK
    )
    base_length_m = contact_length_m if contact_length_m is not None else contact_preset.get(
        "length", dims["length"] * CONTACT_HEIGHT_SCALE_FALLBACK
    )
    clearance = contact_preset.get("clearance", DEFAULT_CONTACT_SURFACE["clearance"])
    
    # Enforce minimum sizes
    height_m = float(max(height_m, 0.005))
    base_length_m = float(max(base_length_m, MIN_CONTACT_LENGTH))
    base_half_length = base_length_m / 2.0
    
    # Calculate contact width
    if gripper_width_mm is not None:
        width_m = max(gripper_width_mm / 1000.0, MIN_CONTACT_WIDTH)
    else:
        width_m = dims["default_width"] * CONTACT_WIDTH_SCALE_FALLBACK
    width_m = max(width_m - 2.0 * clearance, MIN_CONTACT_WIDTH)
    
    # Use TCP rotation as the gripper frame
    # Assuming TCP frame follows convention: X=width, Y=height, Z=approach
    width_axis = tcp_rotation[:, 0].astype(np.float32)
    height_axis = tcp_rotation[:, 1].astype(np.float32)
    approach_axis = tcp_rotation[:, 2].astype(np.float32)
    
    # Build basis matrix
    basis = np.column_stack((width_axis, height_axis, approach_axis)).astype(np.float32)
    
    # Define half-sizes
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
    
    # Position bboxes relative to TCP
    # Contact box extends forward from TCP along approach axis
    contact_center = (tcp_position + approach_axis * contact_half_sizes[2]).astype(np.float32)
    full_center = (tcp_position + approach_axis * full_half_sizes[2]).astype(np.float32)
    fingertip_center = (full_center + approach_axis * (full_half_sizes[2] - contact_half_sizes[2])).astype(np.float32)
    
    # Convert basis to quaternion
    quat = _rotation_matrix_to_xyzw(basis)
    
    # Package results
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
        "half_sizes": contact_half_sizes.astype(np.float32),
        "quat_xyzw": quat,
        "basis": basis,
    }
    
    return contact_box, full_box, fingertip_box


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
    
    # FALLBACK: If no finger links found, try using end-effector link (e.g., for WSG-50)
    ee_tf = None
    if left_tf is None and right_tf is None and ee_link and ee_link in fk_map:
        print(f"[Info] No finger pad links found; using end-effector link '{ee_link}' as gripper frame.")
        ee_tf = fk_map[ee_link].matrix().astype(np.float32)
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
    elif ee_tf is not None:
        print("[Info] Using end-effector link as gripper frame (no finger pads available).")
        # Fallback: use end-effector link when no finger links are found
        # This is common for grippers like WSG-50 where the URDF doesn't include finger details
        pad_midpoint = ee_tf[:3, 3].astype(np.float32)
        # Use standard EE frame convention: X = width, Y = height, Z = approach
        width_axis = ee_tf[:3, 0].astype(np.float32)  # X-axis of EE frame
        approach_axis = ee_tf[:3, 2].astype(np.float32)  # Z-axis of EE frame
        height_axis = ee_tf[:3, 1].astype(np.float32)  # Y-axis of EE frame
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
        width_m = DEFAULT_GRIPPER_DIMS["default_width"] * CONTACT_WIDTH_SCALE_FALLBACK
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
    full_center = (pad_midpoint + approach_axis * full_half_sizes[2]).astype(np.float32)

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
        if robot_conf is None:
            print("[Warning] No robot configuration provided; cannot compute gripper body bbox.")
        if ref_bbox is None:
            print("[Warning] No reference bbox provided; cannot compute gripper body bbox.")
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
        print("[Warning] Reference bbox has no basis; cannot compute body bbox.")
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
    
    # FALLBACK: If no finger pads found, use end-effector link
    if not points:
        print("[Info] No gripper pad links found; falling back to end-effector link for single gripper point.") 
        robot_type = getattr(robot_conf, "robot", None)
        ee_link = ROBOT_EE_LINK_MAP.get(robot_type)
        if ee_link and ee_link in fk_map:
            T = fk_map[ee_link].matrix()
            p = T[:3, 3].astype(np.float32)
            points.append(p)
            print(f"[Info] Using end-effector link '{ee_link}' for gripper pad point.")
    
    if not points:
        return None
    return np.stack(points, axis=0).astype(np.float32)
