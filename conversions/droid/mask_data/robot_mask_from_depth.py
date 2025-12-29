#!/usr/bin/env python3
"""
Generate robot masks by:
1. Computing FK to get robot mesh positions
2. Classifying 3D sensor points (from depth) that belong to the robot
3. Reprojecting those classified points to RGB

This approach uses the 3D model to classify sensor points rather than
directly projecting the mesh.

Stage 1: Generate full robot body mask using FK from joint_positions
Stage 2: Classify sensor points using 3D robot mesh, reproject to RGB

Usage:
    python robot_mask_from_depth.py \
        --h5_path /data/droid/data/droid_raw/1.0.1/IRIS/success/2023-05-12/Fri_May_12_10:51:00_2023/trajectory.h5 \
        --processed_dir /workspace/droid_processed/IRIS/success/2023-05-12/Fri_May_12_10:51:00_2023 \
        --output_dir ./robot_mask_output
"""

import argparse
import numpy as np
import os
import sys
import json
import h5py
import cv2
import trimesh
from typing import Dict, List, Tuple, Optional
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree

# Add parent for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# MESH PATHS (Robotiq gripper only - no Panda hand)
# =============================================================================
GRIPPER_MESH_BASE = "/workspace/external/robotiq_arg85_description/meshes"
if not os.path.exists(GRIPPER_MESH_BASE):
    GRIPPER_MESH_BASE = "/workspace/third_party/robotiq_arg85_description/meshes"

ROBOT_MESH_BASE = "/workspace/third_party/CtRNet-X/urdfs/Panda/meshes/collision"

# =============================================================================
# PANDA FORWARD KINEMATICS
# =============================================================================

# Panda DH-like parameters from URDF
PANDA_JOINT_ORIGINS = [
    ([0, 0, 0.333], [0, 0, 0]),           # joint1: link0 -> link1
    ([0, 0, 0], [-np.pi/2, 0, 0]),        # joint2: link1 -> link2
    ([0, -0.316, 0], [np.pi/2, 0, 0]),    # joint3: link2 -> link3
    ([0.0825, 0, 0], [np.pi/2, 0, 0]),    # joint4: link3 -> link4
    ([-0.0825, 0.384, 0], [-np.pi/2, 0, 0]),  # joint5: link4 -> link5
    ([0, 0, 0], [np.pi/2, 0, 0]),         # joint6: link5 -> link6
    ([0.088, 0, 0], [np.pi/2, 0, 0]),     # joint7: link6 -> link7
    ([0, 0, 0.107], [0, 0, -np.pi/4]),    # flange: link7 -> hand/EE
]

PANDA_JOINT_AXES = [[0, 0, 1]] * 7  # All joints rotate around Z


def make_transform(trans: List[float], rpy: List[float]) -> np.ndarray:
    """Create 4x4 transform from translation and RPY euler angles."""
    T = np.eye(4)
    T[:3, 3] = trans
    T[:3, :3] = R.from_euler('xyz', rpy).as_matrix()
    return T


def panda_forward_kinematics(
    joint_angles: np.ndarray,
    T_world_base: np.ndarray = None
) -> Dict[str, np.ndarray]:
    """
    Compute forward kinematics for Panda arm.
    
    Args:
        joint_angles: 7-element array of joint angles in radians
        T_world_base: Base transform in world frame (default: identity)
        
    Returns:
        Dictionary mapping link names to 4x4 transforms in world frame
    """
    if T_world_base is None:
        T_world_base = np.eye(4)
    
    transforms = {}
    
    # Base link (link0) at world origin
    T_current = T_world_base.copy()
    transforms["link0"] = T_current.copy()
    
    # Iterate through joints
    for i in range(7):
        trans, rpy = PANDA_JOINT_ORIGINS[i]
        
        # Apply joint origin transform
        T_origin = make_transform(trans, rpy)
        T_current = T_current @ T_origin
        
        # Apply joint rotation
        axis = PANDA_JOINT_AXES[i]
        angle = joint_angles[i]
        T_joint = np.eye(4)
        T_joint[:3, :3] = R.from_rotvec(np.array(axis) * angle).as_matrix()
        T_current = T_current @ T_joint
        
        transforms[f"link{i+1}"] = T_current.copy()
    
    # Flange transform (link7 -> hand/EE)
    trans, rpy = PANDA_JOINT_ORIGINS[7]
    T_flange = make_transform(trans, rpy)
    T_current = T_current @ T_flange
    transforms["hand"] = T_current.copy()
    
    return transforms


# =============================================================================
# ROBOTIQ GRIPPER KINEMATICS (matches GripperVisualizer exactly)
# =============================================================================

def robotiq_gripper_transforms(T_world_ee: np.ndarray, gripper_pos: float) -> Dict[str, np.ndarray]:
    """
    Compute transforms for all Robotiq 85 gripper components.
    This matches GripperVisualizer.get_o3d_mesh() exactly.
    
    Args:
        T_world_ee: End-effector transform in world frame
        gripper_pos: Gripper position (0.0=open, 1.0=closed)
        
    Returns:
        Dictionary mapping component names to 4x4 transforms in world frame
    """
    val = gripper_pos[0] if isinstance(gripper_pos, (list, np.ndarray)) else gripper_pos
    theta = val * 0.8  # Joint angle
    
    transforms = {}
    
    # Base (directly at EE)
    transforms["base"] = T_world_ee.copy()
    
    # Left Outer Knuckle
    T_lok = np.eye(4)
    T_lok[:3, 3] = [0.03060114, 0, 0.06279202]
    T_lok[:3, :3] = R.from_rotvec([0, -theta, 0]).as_matrix()
    transforms["left_outer_knuckle"] = T_world_ee @ T_lok
    
    # Left Outer Finger
    T_lof = np.eye(4)
    T_lof[:3, 3] = [0.03169104, 0, -0.00193396]
    transforms["left_outer_finger"] = T_world_ee @ T_lok @ T_lof
    
    # Left Inner Knuckle
    T_lik = np.eye(4)
    T_lik[:3, 3] = [0.0127, 0, 0.0693]
    T_lik[:3, :3] = R.from_rotvec([0, -theta, 0]).as_matrix()
    transforms["left_inner_knuckle"] = T_world_ee @ T_lik
    
    # Left Inner Finger
    T_lif = np.eye(4)
    T_lif[:3, 3] = [0.03458531, 0, 0.04549702]
    T_lif[:3, :3] = R.from_rotvec([0, theta, 0]).as_matrix()
    transforms["left_inner_finger"] = T_world_ee @ T_lik @ T_lif
    
    # Right Outer Knuckle
    T_rok_origin = np.eye(4)
    T_rok_origin[:3, 3] = [-0.03060114, 0, 0.06279202]
    T_rok_origin[:3, :3] = R.from_euler('z', np.pi).as_matrix()
    R_rok_joint = R.from_rotvec([0, -theta, 0]).as_matrix()
    T_rok = T_rok_origin.copy()
    T_rok[:3, :3] = T_rok[:3, :3] @ R_rok_joint
    transforms["right_outer_knuckle"] = T_world_ee @ T_rok
    
    # Right Outer Finger
    T_rof = np.eye(4)
    T_rof[:3, 3] = [0.03169104, 0, -0.00193396]
    transforms["right_outer_finger"] = T_world_ee @ T_rok @ T_rof
    
    # Right Inner Knuckle
    T_rik_origin = np.eye(4)
    T_rik_origin[:3, 3] = [-0.0127, 0, 0.0693]
    T_rik_origin[:3, :3] = R.from_euler('z', np.pi).as_matrix()
    R_rik_joint = R.from_rotvec([0, -theta, 0]).as_matrix()
    T_rik = T_rik_origin.copy()
    T_rik[:3, :3] = T_rik[:3, :3] @ R_rik_joint
    transforms["right_inner_knuckle"] = T_world_ee @ T_rik
    
    # Right Inner Finger
    T_rif = np.eye(4)
    T_rif[:3, 3] = [0.03410605, 0, 0.04585739]
    T_rif[:3, :3] = R.from_rotvec([0, theta, 0]).as_matrix()
    transforms["right_inner_finger"] = T_world_ee @ T_rik @ T_rif
    
    return transforms


# =============================================================================
# MESH LOADING
# =============================================================================

def load_mesh(path: str) -> Optional[trimesh.Trimesh]:
    """Load a mesh file, handling scenes."""
    if not os.path.exists(path):
        print(f"[WARN] Mesh not found: {path}")
        return None
    try:
        mesh = trimesh.load(path, force='mesh')
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(
                [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            )
        return mesh
    except Exception as e:
        print(f"[WARN] Failed to load mesh {path}: {e}")
        return None


def load_robot_meshes() -> Dict[str, trimesh.Trimesh]:
    """Load all Panda arm meshes (link0-link7, no hand)."""
    meshes = {}
    for i in range(8):
        path = f"{ROBOT_MESH_BASE}/link{i}.obj"
        mesh = load_mesh(path)
        if mesh:
            meshes[f"link{i}"] = mesh
    return meshes


def load_gripper_meshes() -> Dict[str, trimesh.Trimesh]:
    """Load Robotiq 85 gripper meshes."""
    mesh_files = {
        "base": f"{GRIPPER_MESH_BASE}/robotiq_85_base_link_fine.STL",
        "outer_knuckle": f"{GRIPPER_MESH_BASE}/outer_knuckle_fine.STL",
        "outer_finger": f"{GRIPPER_MESH_BASE}/outer_finger_fine.STL",
        "inner_knuckle": f"{GRIPPER_MESH_BASE}/inner_knuckle_fine.STL",
        "inner_finger": f"{GRIPPER_MESH_BASE}/inner_finger_fine.STL",
    }
    meshes = {}
    for name, path in mesh_files.items():
        mesh = load_mesh(path)
        if mesh:
            meshes[name] = mesh
    return meshes


# =============================================================================
# ROBOT MESH IN WORLD FRAME
# =============================================================================

def get_robot_mesh_in_world(
    joint_angles: np.ndarray,
    gripper_pos: float,
    T_world_base: np.ndarray = None,
    arm_meshes: Dict[str, trimesh.Trimesh] = None,
    gripper_meshes: Dict[str, trimesh.Trimesh] = None
) -> trimesh.Trimesh:
    """
    Get combined robot mesh (arm + gripper) in world frame.
    
    Args:
        joint_angles: 7-element array of joint angles
        gripper_pos: Gripper position (0-1)
        T_world_base: Robot base transform in world
        arm_meshes: Pre-loaded arm meshes
        gripper_meshes: Pre-loaded gripper meshes
        
    Returns:
        Combined mesh in world frame
    """
    if T_world_base is None:
        T_world_base = np.eye(4)
    
    # Get link transforms
    link_transforms = panda_forward_kinematics(joint_angles, T_world_base)
    
    combined = []
    
    # Transform arm links
    if arm_meshes:
        for link_name, mesh in arm_meshes.items():
            if link_name in link_transforms:
                transformed = mesh.copy()
                transformed.apply_transform(link_transforms[link_name])
                combined.append(transformed)
    
    # Transform gripper
    if gripper_meshes:
        T_world_ee = link_transforms.get("hand", np.eye(4))
        gripper_transforms = robotiq_gripper_transforms(T_world_ee, gripper_pos)
        
        # Map mesh names to gripper components
        mesh_to_component = {
            "base": ["base"],
            "outer_knuckle": ["left_outer_knuckle", "right_outer_knuckle"],
            "outer_finger": ["left_outer_finger", "right_outer_finger"],
            "inner_knuckle": ["left_inner_knuckle", "right_inner_knuckle"],
            "inner_finger": ["left_inner_finger", "right_inner_finger"],
        }
        
        for mesh_name, mesh in gripper_meshes.items():
            if mesh_name in mesh_to_component:
                for comp_name in mesh_to_component[mesh_name]:
                    if comp_name in gripper_transforms:
                        transformed = mesh.copy()
                        transformed.apply_transform(gripper_transforms[comp_name])
                        combined.append(transformed)
    
    if combined:
        return trimesh.util.concatenate(combined)
    return trimesh.Trimesh()


def sample_points_from_mesh(mesh: trimesh.Trimesh, num_points: int = 10000) -> np.ndarray:
    """Sample points uniformly from mesh surface."""
    if mesh is None or len(mesh.vertices) == 0:
        return np.array([]).reshape(0, 3)
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points


# =============================================================================
# STAGE 2: CLASSIFY SENSOR POINTS USING ROBOT MESH
# =============================================================================

def classify_sensor_points_by_robot(
    sensor_points_world: np.ndarray,
    robot_mesh: trimesh.Trimesh,
    distance_threshold: float = 0.02  # 2cm
) -> np.ndarray:
    """
    Classify which sensor points belong to the robot.
    
    Uses the robot mesh to find sensor points that are close to the robot surface.
    
    Args:
        sensor_points_world: (N, 3) sensor points in world frame
        robot_mesh: Robot mesh in world frame
        distance_threshold: Maximum distance to consider a point as robot (meters)
        
    Returns:
        Boolean mask (N,) where True = robot point
    """
    if len(sensor_points_world) == 0:
        return np.array([], dtype=bool)
    
    # Sample dense points from robot mesh for comparison
    robot_points = sample_points_from_mesh(robot_mesh, num_points=50000)
    
    if len(robot_points) == 0:
        return np.zeros(len(sensor_points_world), dtype=bool)
    
    # Build KD-tree for fast nearest neighbor lookup
    tree = KDTree(robot_points)
    
    # Find distance to nearest robot mesh point for each sensor point
    distances, _ = tree.query(sensor_points_world, k=1)
    
    # Classify as robot if within threshold
    is_robot = distances < distance_threshold
    
    return is_robot


def project_points_to_image(
    points_world: np.ndarray,
    K: np.ndarray,
    T_cam_world: np.ndarray,
    width: int,
    height: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project 3D points to 2D image coordinates.
    
    Args:
        points_world: (N, 3) points in world frame
        K: 3x3 camera intrinsics
        T_cam_world: 4x4 world-to-camera transform
        width, height: Image dimensions
        
    Returns:
        uv: (N, 2) pixel coordinates
        valid: (N,) boolean mask for valid projections
        depths: (N,) depths in camera frame
    """
    if len(points_world) == 0:
        return np.array([]).reshape(0, 2), np.array([], dtype=bool), np.array([])
    
    # Transform to camera frame
    points_hom = np.hstack([points_world, np.ones((len(points_world), 1))])
    points_cam = (T_cam_world @ points_hom.T).T[:, :3]
    
    # Get depths
    depths = points_cam[:, 2]
    
    # Project to 2D
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    u = (fx * points_cam[:, 0] / points_cam[:, 2]) + cx
    v = (fy * points_cam[:, 1] / points_cam[:, 2]) + cy
    
    uv = np.stack([u, v], axis=1)
    
    # Valid if in front of camera and within bounds
    valid = (depths > 0.01) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    
    return uv, valid, depths


def create_mask_from_points(
    uv: np.ndarray,
    valid: np.ndarray,
    width: int,
    height: int,
    radius: int = 3
) -> np.ndarray:
    """
    Create binary mask from projected points.
    
    Args:
        uv: (N, 2) pixel coordinates
        valid: (N,) boolean mask
        width, height: Image dimensions
        radius: Radius of each point in pixels
        
    Returns:
        Binary mask (height, width) as uint8
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    
    valid_uv = uv[valid].astype(int)
    for pt in valid_uv:
        cv2.circle(mask, (pt[0], pt[1]), radius, 255, -1)
    
    return mask


# =============================================================================
# DIRECT MESH PROJECTION (Stage 1 - for comparison)
# =============================================================================

def render_mesh_mask_direct(
    mesh: trimesh.Trimesh,
    K: np.ndarray,
    T_cam_world: np.ndarray,
    width: int,
    height: int,
    min_depth: float = 0.01
) -> np.ndarray:
    """
    Render mesh mask by projecting mesh triangles to 2D.
    
    Args:
        mesh: Mesh in world frame
        K: Camera intrinsics
        T_cam_world: World-to-camera transform
        width, height: Image dimensions
        min_depth: Minimum depth threshold
        
    Returns:
        Binary mask (height, width) as uint8
    """
    if mesh is None or len(mesh.vertices) == 0:
        return np.zeros((height, width), dtype=np.uint8)
    
    # Transform vertices to camera frame
    vertices_hom = np.hstack([mesh.vertices, np.ones((len(mesh.vertices), 1))])
    vertices_cam = (T_cam_world @ vertices_hom.T).T[:, :3]
    
    # Project to 2D
    depths = vertices_cam[:, 2]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        u = (fx * vertices_cam[:, 0] / depths) + cx
        v = (fy * vertices_cam[:, 1] / depths) + cy
    
    vertices_2d = np.stack([u, v], axis=1)
    
    # Render mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    for face in mesh.faces:
        pts = vertices_2d[face]
        face_depths = depths[face]
        
        # Skip faces behind camera
        if np.any(face_depths < min_depth):
            continue
        
        # Check if any vertex is in bounds
        if np.any((pts[:, 0] >= 0) & (pts[:, 0] < width) & 
                  (pts[:, 1] >= 0) & (pts[:, 1] < height)):
            pts_int = pts.astype(np.int32)
            cv2.fillPoly(mask, [pts_int], 255)
    
    return mask


# =============================================================================
# DEPTH TO POINT CLOUD
# =============================================================================

def depth_to_pointcloud(
    depth: np.ndarray,
    K: np.ndarray,
    T_world_cam: np.ndarray,
    rgb: np.ndarray = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Convert depth image to 3D point cloud in world frame.
    
    Args:
        depth: (H, W) depth image in meters
        K: 3x3 camera intrinsics
        T_world_cam: 4x4 camera-to-world transform (camera pose)
        rgb: Optional (H, W, 3) RGB image for colors
        
    Returns:
        points_world: (N, 3) points in world frame
        colors: (N, 3) RGB colors if rgb provided, else None
    """
    h, w = depth.shape
    
    # Create pixel coordinates
    u = np.arange(w)
    v = np.arange(h)
    u, v = np.meshgrid(u, v)
    
    # Unproject to camera frame
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # Stack and filter valid points
    points_cam = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    valid = points_cam[:, 2] > 0.01  # Minimum depth
    points_cam = points_cam[valid]
    
    # Transform to world frame
    points_hom = np.hstack([points_cam, np.ones((len(points_cam), 1))])
    points_world = (T_world_cam @ points_hom.T).T[:, :3]
    
    colors = None
    if rgb is not None:
        colors = rgb.reshape(-1, 3)[valid]
    
    return points_world, colors


# =============================================================================
# OVERLAY UTILITIES
# =============================================================================

def overlay_mask(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.5) -> np.ndarray:
    """Overlay colored mask on image."""
    overlay = image.copy()
    mask_bool = mask > 0
    overlay[mask_bool] = (
        (1 - alpha) * overlay[mask_bool] + alpha * np.array(color)
    ).astype(np.uint8)
    return overlay


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_episode(
    h5_path: str,
    processed_dir: str,
    output_dir: str,
    max_frames: int = 500,
    method: str = "direct"  # "direct" or "depth"
):
    """
    Process a DROID episode to generate robot masks.
    
    Args:
        h5_path: Path to trajectory.h5 with joint_positions
        processed_dir: Path to processed episode with extrinsics, RGB frames
        output_dir: Output directory
        max_frames: Maximum frames to process
        method: "direct" = project mesh directly, "depth" = classify sensor points
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("ROBOT MASK GENERATION")
    print("=" * 60)
    print(f"H5 path: {h5_path}")
    print(f"Processed dir: {processed_dir}")
    print(f"Output: {output_dir}")
    print(f"Method: {method}")
    print()
    
    # 1. Load robot data from H5
    print("[1/5] Loading robot state...")
    with h5py.File(h5_path, 'r') as f:
        joint_positions = f['observation/robot_state/joint_positions'][:]
        gripper_positions = f['observation/robot_state/gripper_position'][:]
        cartesian_positions = f['observation/robot_state/cartesian_position'][:]
    
    num_frames = min(len(joint_positions), max_frames)
    print(f"  Joint positions: {joint_positions.shape}")
    print(f"  Frames to process: {num_frames}")
    
    # 2. Load extrinsics
    print("\n[2/5] Loading camera extrinsics...")
    extr_path = os.path.join(processed_dir, 'extrinsics.npz')
    if not os.path.exists(extr_path):
        print(f"  [ERROR] Extrinsics not found: {extr_path}")
        return
    
    extr = np.load(extr_path, allow_pickle=True)
    
    # Get wrist camera info
    wrist_serial = None
    wrist_extrinsics = None
    if 'wrist_serial' in extr.files:
        wrist_serial = str(extr['wrist_serial'].item())
        if 'wrist_extrinsics' in extr.files:
            wrist_extrinsics = extr['wrist_extrinsics']
            print(f"  Wrist camera: {wrist_serial} ({len(wrist_extrinsics)} frames)")
    
    # Find cameras
    cameras = {}
    recordings_dir = os.path.join(processed_dir, 'recordings')
    for key in extr.files:
        if key.startswith('external_'):
            serial = key.replace('external_', '')
            cam_dir = os.path.join(recordings_dir, serial)
            if os.path.exists(cam_dir):
                # Load intrinsics
                intr_path = os.path.join(cam_dir, 'intrinsics.json')
                if os.path.exists(intr_path):
                    with open(intr_path, 'r') as f:
                        intr = json.load(f)
                    
                    cameras[serial] = {
                        'world_T_cam': extr[key],
                        'intrinsics': intr,
                        'rgb_dir': os.path.join(cam_dir, 'rgb'),
                        'is_wrist': False
                    }
                    print(f"  Found camera: {serial}")
    
    # Add wrist camera if available
    if wrist_serial and wrist_extrinsics is not None:
        cam_dir = os.path.join(recordings_dir, wrist_serial)
        if os.path.exists(cam_dir):
            intr_path = os.path.join(cam_dir, 'intrinsics.json')
            if os.path.exists(intr_path):
                with open(intr_path, 'r') as f:
                    intr = json.load(f)
                cameras[wrist_serial] = {
                    'world_T_cam': wrist_extrinsics,  # Per-frame extrinsics
                    'intrinsics': intr,
                    'rgb_dir': os.path.join(cam_dir, 'rgb'),
                    'is_wrist': True
                }
                print(f"  Found wrist camera: {wrist_serial}")
    
    if not cameras:
        print("  [ERROR] No cameras found")
        return
    
    # 3. Load meshes
    print("\n[3/5] Loading meshes...")
    arm_meshes = load_robot_meshes()
    gripper_meshes = load_gripper_meshes()
    print(f"  Arm links loaded: {list(arm_meshes.keys())}")
    print(f"  Gripper parts loaded: {list(gripper_meshes.keys())}")
    
    # 4. Process each camera
    print("\n[4/5] Processing cameras...")
    
    for serial, cam_info in cameras.items():
        print(f"\n  Camera: {serial}")
        
        # Get camera params
        is_wrist = cam_info.get('is_wrist', False)
        if is_wrist:
            print(f"    Type: Wrist (per-frame extrinsics)")
            # world_T_cam is an array of per-frame transforms
            wrist_cam_extrinsics = cam_info['world_T_cam']
        else:
            print(f"    Type: External (static extrinsics)")
            world_T_cam = cam_info['world_T_cam']
            cam_T_world = np.linalg.inv(world_T_cam)
        
        intr = cam_info['intrinsics']
        
        fx = intr.get('focal_length_x', intr.get('fx', 640))
        fy = intr.get('focal_length_y', intr.get('fy', 640))
        cx = intr.get('cx', intr.get('principal_point_x', 640))
        cy = intr.get('cy', intr.get('principal_point_y', 360))
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        
        # Get image size from first frame
        rgb_dir = cam_info['rgb_dir']
        rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')])
        if not rgb_files:
            print(f"    [WARN] No RGB frames found")
            continue
        
        first_img = cv2.imread(os.path.join(rgb_dir, rgb_files[0]))
        height, width = first_img.shape[:2]
        
        # Setup video writer
        cam_output_dir = os.path.join(output_dir, serial)
        os.makedirs(cam_output_dir, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        arm_writer = cv2.VideoWriter(
            os.path.join(cam_output_dir, 'arm_mask.mp4'),
            fourcc, 30, (width, height)
        )
        gripper_writer = cv2.VideoWriter(
            os.path.join(cam_output_dir, 'gripper_mask.mp4'),
            fourcc, 30, (width, height)
        )
        full_writer = cv2.VideoWriter(
            os.path.join(cam_output_dir, 'full_robot_mask.mp4'),
            fourcc, 30, (width, height)
        )
        
        # Process frames
        for frame_idx in range(min(num_frames, len(rgb_files))):
            if frame_idx % 10 == 0:
                print(f"    Frame {frame_idx}/{num_frames}")
            
            # Load RGB
            rgb_path = os.path.join(rgb_dir, rgb_files[frame_idx])
            rgb = cv2.imread(rgb_path)
            
            # Get camera extrinsics for this frame
            if is_wrist:
                # Wrist camera: per-frame extrinsics
                frame_world_T_cam = wrist_cam_extrinsics[min(frame_idx, len(wrist_cam_extrinsics)-1)]
                frame_cam_T_world = np.linalg.inv(frame_world_T_cam)
            else:
                # External camera: static extrinsics
                frame_cam_T_world = cam_T_world
            
            # Get robot state
            joints = joint_positions[frame_idx][:7]
            gripper_pos = gripper_positions[frame_idx]
            
            # Get link transforms
            link_transforms = panda_forward_kinematics(joints)
            T_world_ee = link_transforms.get("hand", np.eye(4))
            
            # Apply R_fix rotation to EE frame (90° Z rotation)
            # This matches the convention used in gripper_poses stored in tracks.npz
            R_fix = R.from_euler('z', 90, degrees=True).as_matrix()
            T_world_ee_rotated = T_world_ee.copy()
            T_world_ee_rotated[:3, :3] = T_world_ee[:3, :3] @ R_fix
            
            # Render arm mask
            arm_mask = np.zeros((height, width), dtype=np.uint8)
            for link_name, mesh in arm_meshes.items():
                if link_name in link_transforms:
                    transformed = mesh.copy()
                    transformed.apply_transform(link_transforms[link_name])
                    link_mask = render_mesh_mask_direct(
                        transformed, K, frame_cam_T_world, width, height
                    )
                    arm_mask = np.maximum(arm_mask, link_mask)
            
            # Render gripper mask (use rotated EE frame)
            gripper_transforms = robotiq_gripper_transforms(T_world_ee_rotated, gripper_pos)
            gripper_mask = np.zeros((height, width), dtype=np.uint8)
            
            mesh_to_component = {
                "base": ["base"],
                "outer_knuckle": ["left_outer_knuckle", "right_outer_knuckle"],
                "outer_finger": ["left_outer_finger", "right_outer_finger"],
                "inner_knuckle": ["left_inner_knuckle", "right_inner_knuckle"],
                "inner_finger": ["left_inner_finger", "right_inner_finger"],
            }
            
            for mesh_name, mesh in gripper_meshes.items():
                if mesh_name in mesh_to_component:
                    for comp_name in mesh_to_component[mesh_name]:
                        if comp_name in gripper_transforms:
                            transformed = mesh.copy()
                            transformed.apply_transform(gripper_transforms[comp_name])
                            comp_mask = render_mesh_mask_direct(
                                transformed, K, frame_cam_T_world, width, height
                            )
                            gripper_mask = np.maximum(gripper_mask, comp_mask)
            
            # Combined mask
            full_mask = np.maximum(arm_mask, gripper_mask)
            
            # Create overlays
            arm_overlay = overlay_mask(rgb, arm_mask, (255, 100, 0), 0.5)  # Orange
            gripper_overlay = overlay_mask(rgb, gripper_mask, (0, 255, 0), 0.5)  # Green
            full_overlay = overlay_mask(rgb, full_mask, (0, 200, 255), 0.5)  # Cyan
            
            # Write frames
            arm_writer.write(arm_overlay)
            gripper_writer.write(gripper_overlay)
            full_writer.write(full_overlay)
        
        arm_writer.release()
        gripper_writer.release()
        full_writer.release()
        print(f"    Saved to {cam_output_dir}/")
    
    print("\n[5/5] Done!")


def main():
    parser = argparse.ArgumentParser(description="Generate robot masks from DROID data")
    parser.add_argument('--h5_path', required=True, help='Path to trajectory.h5')
    parser.add_argument('--processed_dir', required=True, help='Path to processed episode directory')
    parser.add_argument('--output_dir', default='./robot_mask_output', help='Output directory')
    parser.add_argument('--max_frames', type=int, default=500, help='Maximum frames to process')
    parser.add_argument('--method', choices=['direct', 'depth'], default='direct',
                       help='Method: direct mesh projection or depth-based classification')
    
    args = parser.parse_args()
    
    process_episode(
        h5_path=args.h5_path,
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        method=args.method
    )


if __name__ == "__main__":
    main()
