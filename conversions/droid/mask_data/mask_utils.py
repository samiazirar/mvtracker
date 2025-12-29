"""
Utility functions for generating 2D masks from 3D robot meshes.

This module provides functions for:
- Loading robot and gripper meshes
- Forward kinematics for Panda arm + Robotiq gripper
- Projecting 3D meshes to 2D image coordinates
- Rendering binary masks from projected meshes
"""

import numpy as np
import cv2
import trimesh
import os
from typing import Dict, List, Tuple, Optional
from scipy.spatial.transform import Rotation as R


# =============================================================================
# MESH PATHS
# =============================================================================

# Default mesh paths (relative to workspace root)
# Primary: external/ (symlinked in main repo)
# Fallback: third_party/ (CtRNet-X submodule)
GRIPPER_MESH_BASE = "/workspace/external/robotiq_arg85_description/meshes"
ROBOT_MESH_BASE = "/workspace/third_party/CtRNet-X/urdfs/Panda/meshes/collision"

# Alternative paths for flexibility
GRIPPER_MESH_BASE_ALT = "/workspace/third_party/robotiq_arg85_description/meshes"

GRIPPER_MESHES = {
    "base": f"{GRIPPER_MESH_BASE}/robotiq_85_base_link_fine.STL",
    "outer_knuckle": f"{GRIPPER_MESH_BASE}/outer_knuckle_fine.STL",
    "outer_finger": f"{GRIPPER_MESH_BASE}/outer_finger_fine.STL",
    "inner_knuckle": f"{GRIPPER_MESH_BASE}/inner_knuckle_fine.STL",
    "inner_finger": f"{GRIPPER_MESH_BASE}/inner_finger_fine.STL",
}

ROBOT_ARM_MESHES = {
    "link0": f"{ROBOT_MESH_BASE}/link0.obj",
    "link1": f"{ROBOT_MESH_BASE}/link1.obj",
    "link2": f"{ROBOT_MESH_BASE}/link2.obj",
    "link3": f"{ROBOT_MESH_BASE}/link3.obj",
    "link4": f"{ROBOT_MESH_BASE}/link4.obj",
    "link5": f"{ROBOT_MESH_BASE}/link5.obj",
    "link6": f"{ROBOT_MESH_BASE}/link6.obj",
    "link7": f"{ROBOT_MESH_BASE}/link7.obj",
    "hand": f"{ROBOT_MESH_BASE}/hand.obj",
}


# =============================================================================
# MESH LOADING
# =============================================================================

def load_meshes(mesh_dict: Dict[str, str]) -> Dict[str, trimesh.Trimesh]:
    """
    Load meshes from a dictionary of paths.
    
    Args:
        mesh_dict: Dictionary mapping name -> path
        
    Returns:
        Dictionary mapping name -> trimesh.Trimesh
    """
    meshes = {}
    for name, path in mesh_dict.items():
        if os.path.exists(path):
            try:
                mesh = trimesh.load(path, force='mesh')
                if isinstance(mesh, trimesh.Scene):
                    # Combine all meshes in scene
                    mesh = trimesh.util.concatenate(
                        [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
                    )
                meshes[name] = mesh
            except Exception as e:
                print(f"[WARN] Failed to load mesh {path}: {e}")
        else:
            print(f"[WARN] Mesh not found: {path}")
    return meshes


def load_gripper_meshes() -> Dict[str, trimesh.Trimesh]:
    """Load all Robotiq 85 gripper meshes."""
    return load_meshes(GRIPPER_MESHES)


def load_robot_arm_meshes() -> Dict[str, trimesh.Trimesh]:
    """Load all Panda robot arm meshes (without gripper)."""
    return load_meshes(ROBOT_ARM_MESHES)


# =============================================================================
# FORWARD KINEMATICS - PANDA ARM
# =============================================================================

# Panda DH parameters (from URDF)
# Joint origins relative to parent link
PANDA_JOINT_ORIGINS = [
    # (translation, rpy)
    ([0, 0, 0.333], [0, 0, 0]),           # joint1: link0 -> link1
    ([0, 0, 0], [-np.pi/2, 0, 0]),        # joint2: link1 -> link2
    ([0, -0.316, 0], [np.pi/2, 0, 0]),    # joint3: link2 -> link3
    ([0.0825, 0, 0], [np.pi/2, 0, 0]),    # joint4: link3 -> link4
    ([-0.0825, 0.384, 0], [-np.pi/2, 0, 0]),  # joint5: link4 -> link5
    ([0, 0, 0], [np.pi/2, 0, 0]),         # joint6: link5 -> link6
    ([0.088, 0, 0], [np.pi/2, 0, 0]),     # joint7: link6 -> link7
    ([0, 0, 0.107], [0, 0, -np.pi/4]),    # flange: link7 -> hand
]

# Joint axes (all rotate around Z in their local frame after applying origin)
PANDA_JOINT_AXES = [
    [0, 0, 1],  # joint1
    [0, 0, 1],  # joint2
    [0, 0, 1],  # joint3
    [0, 0, 1],  # joint4
    [0, 0, 1],  # joint5
    [0, 0, 1],  # joint6
    [0, 0, 1],  # joint7
]


def pose6_to_T(pose: np.ndarray) -> np.ndarray:
    """Convert [x, y, z, roll, pitch, yaw] to 4x4 transformation matrix."""
    x, y, z, roll, pitch, yaw = pose
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    T[:3, :3] = R.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    return T


def make_transform(translation: List[float], rpy: List[float]) -> np.ndarray:
    """Create 4x4 transform from translation and roll-pitch-yaw."""
    T = np.eye(4)
    T[:3, 3] = translation
    T[:3, :3] = R.from_euler('xyz', rpy).as_matrix()
    return T


def panda_forward_kinematics(
    joint_angles: np.ndarray,
    T_world_base: np.ndarray = np.eye(4)
) -> Dict[str, np.ndarray]:
    """
    Compute forward kinematics for Panda arm.
    
    Args:
        joint_angles: 7-element array of joint angles in radians
        T_world_base: Base transform in world frame
        
    Returns:
        Dictionary mapping link names to 4x4 transforms in world frame
    """
    transforms = {}
    
    # Base link (link0) at world origin (or provided base transform)
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
    
    # Hand/flange transform
    trans, rpy = PANDA_JOINT_ORIGINS[7]
    T_flange = make_transform(trans, rpy)
    T_current = T_current @ T_flange
    transforms["hand"] = T_current.copy()
    
    return transforms


# =============================================================================
# FORWARD KINEMATICS - ROBOTIQ 85 GRIPPER
# =============================================================================

def robotiq_gripper_transforms(
    T_world_ee: np.ndarray,
    gripper_pos: float
) -> Dict[str, np.ndarray]:
    """
    Compute transforms for all Robotiq 85 gripper components.
    
    Args:
        T_world_ee: End-effector transform in world frame (4x4)
        gripper_pos: Gripper position (0.0=open, 1.0=closed)
        
    Returns:
        Dictionary mapping component names to 4x4 transforms in world frame
    """
    val = gripper_pos[0] if isinstance(gripper_pos, (list, np.ndarray)) else gripper_pos
    theta = val * 0.8  # Map 0-1 to 0-0.8 radians
    
    transforms = {}
    
    # Base is at end-effector
    transforms["base"] = T_world_ee.copy()
    
    # Left Outer Knuckle
    T_lok = np.eye(4)
    T_lok[:3, 3] = [0.03060114, 0, 0.06279202]
    T_lok[:3, :3] = R.from_rotvec([0, -theta, 0]).as_matrix()
    transforms["left_outer_knuckle"] = T_world_ee @ T_lok
    
    # Left Outer Finger (Fixed relative to knuckle)
    T_lof = np.eye(4)
    T_lof[:3, 3] = [0.03169104, 0, -0.00193396]
    transforms["left_outer_finger"] = transforms["left_outer_knuckle"] @ T_lof
    
    # Left Inner Knuckle
    T_lik = np.eye(4)
    T_lik[:3, 3] = [0.0127, 0, 0.0693]
    T_lik[:3, :3] = R.from_rotvec([0, -theta, 0]).as_matrix()
    transforms["left_inner_knuckle"] = T_world_ee @ T_lik
    
    # Left Inner Finger
    T_lif = np.eye(4)
    T_lif[:3, 3] = [0.03458531, 0, 0.04549702]
    T_lif[:3, :3] = R.from_rotvec([0, theta, 0]).as_matrix()
    transforms["left_inner_finger"] = transforms["left_inner_knuckle"] @ T_lif
    
    # Right Outer Knuckle (mirrored)
    T_rok = np.eye(4)
    T_rok[:3, 3] = [-0.03060114, 0, 0.06279202]
    T_rok[:3, :3] = R.from_euler('z', np.pi).as_matrix()
    T_rok[:3, :3] = T_rok[:3, :3] @ R.from_rotvec([0, -theta, 0]).as_matrix()
    transforms["right_outer_knuckle"] = T_world_ee @ T_rok
    
    # Right Outer Finger
    T_rof = np.eye(4)
    T_rof[:3, 3] = [0.03169104, 0, -0.00193396]
    transforms["right_outer_finger"] = transforms["right_outer_knuckle"] @ T_rof
    
    # Right Inner Knuckle (mirrored)
    T_rik = np.eye(4)
    T_rik[:3, 3] = [-0.0127, 0, 0.0693]
    T_rik[:3, :3] = R.from_euler('z', np.pi).as_matrix()
    T_rik[:3, :3] = T_rik[:3, :3] @ R.from_rotvec([0, -theta, 0]).as_matrix()
    transforms["right_inner_knuckle"] = T_world_ee @ T_rik
    
    # Right Inner Finger
    T_rif = np.eye(4)
    T_rif[:3, 3] = [0.03410605, 0, 0.04585739]
    T_rif[:3, :3] = R.from_rotvec([0, theta, 0]).as_matrix()
    transforms["right_inner_finger"] = transforms["right_inner_knuckle"] @ T_rif
    
    return transforms


# =============================================================================
# MESH TO 2D PROJECTION
# =============================================================================

def transform_mesh(mesh: trimesh.Trimesh, T: np.ndarray) -> np.ndarray:
    """
    Transform mesh vertices by a 4x4 matrix.
    
    Args:
        mesh: Trimesh object
        T: 4x4 transformation matrix
        
    Returns:
        Transformed vertices (Nx3)
    """
    vertices = np.asarray(mesh.vertices)
    # Convert to homogeneous coordinates
    vertices_h = np.hstack([vertices, np.ones((len(vertices), 1))])
    # Transform
    vertices_transformed = (T @ vertices_h.T).T[:, :3]
    return vertices_transformed


def project_vertices_to_2d(
    vertices: np.ndarray,
    K: np.ndarray,
    T_cam_world: np.ndarray,
    width: int,
    height: int,
    min_depth: float = 0.01
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D vertices to 2D image coordinates.
    
    Args:
        vertices: Nx3 array of 3D points in world frame
        K: 3x3 camera intrinsic matrix
        T_cam_world: 4x4 transform from world to camera frame
        width: Image width
        height: Image height
        min_depth: Minimum valid depth
        
    Returns:
        Tuple of (uv coordinates Mx2, valid mask)
    """
    if len(vertices) == 0:
        return np.empty((0, 2)), np.zeros(0, dtype=bool)
    
    # Transform to camera frame
    vertices_h = np.hstack([vertices, np.ones((len(vertices), 1))])
    vertices_cam = (T_cam_world @ vertices_h.T).T[:, :3]
    
    # Check depth
    z = vertices_cam[:, 2]
    valid = z > min_depth
    
    if not np.any(valid):
        return np.empty((0, 2)), valid
    
    # Project to 2D
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    u = (vertices_cam[:, 0] * fx / vertices_cam[:, 2]) + cx
    v = (vertices_cam[:, 1] * fy / vertices_cam[:, 2]) + cy
    
    # Check bounds
    in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height) & valid
    
    uv = np.stack([u, v], axis=-1)
    return uv, in_bounds


def render_mesh_mask(
    mesh: trimesh.Trimesh,
    T_world_mesh: np.ndarray,
    K: np.ndarray,
    T_cam_world: np.ndarray,
    width: int,
    height: int,
    min_depth: float = 0.01
) -> np.ndarray:
    """
    Render a binary mask for a single mesh.
    
    Args:
        mesh: Trimesh object
        T_world_mesh: Transform of mesh in world frame
        K: Camera intrinsics
        T_cam_world: Camera extrinsics (world to camera)
        width: Image width
        height: Image height
        min_depth: Minimum valid depth
        
    Returns:
        Binary mask (height x width) as uint8
    """
    # Transform mesh vertices to world frame
    vertices_world = transform_mesh(mesh, T_world_mesh)
    
    # Project to 2D
    uv, valid = project_vertices_to_2d(
        vertices_world, K, T_cam_world, width, height, min_depth
    )
    
    if not np.any(valid):
        return np.zeros((height, width), dtype=np.uint8)
    
    # Get valid triangles
    faces = np.asarray(mesh.faces)
    
    # Create mask by filling triangles
    mask = np.zeros((height, width), dtype=np.uint8)
    
    for face in faces:
        # Check if all vertices of the face are valid
        if not all(valid[face]):
            continue
        
        # Get triangle vertices in 2D
        pts = uv[face].astype(np.int32)
        
        # Fill triangle
        cv2.fillConvexPoly(mask, pts, 255)
    
    return mask


def render_multiple_meshes_mask(
    meshes: Dict[str, trimesh.Trimesh],
    transforms: Dict[str, np.ndarray],
    K: np.ndarray,
    T_cam_world: np.ndarray,
    width: int,
    height: int,
    min_depth: float = 0.01
) -> np.ndarray:
    """
    Render combined mask for multiple meshes.
    
    Args:
        meshes: Dictionary mapping names to Trimesh objects
        transforms: Dictionary mapping names to 4x4 transforms
        K: Camera intrinsics
        T_cam_world: Camera extrinsics
        width: Image width
        height: Image height
        min_depth: Minimum valid depth
        
    Returns:
        Combined binary mask (height x width) as uint8
    """
    combined_mask = np.zeros((height, width), dtype=np.uint8)
    
    for name, mesh in meshes.items():
        if name not in transforms:
            continue
        
        T = transforms[name]
        mask = render_mesh_mask(mesh, T, K, T_cam_world, width, height, min_depth)
        combined_mask = np.maximum(combined_mask, mask)
    
    return combined_mask


# =============================================================================
# HIGH-LEVEL API
# =============================================================================

def compute_hand_transform_from_ee(T_world_ee: np.ndarray) -> np.ndarray:
    """
    Compute the Panda hand transform given the end-effector (gripper base) pose.
    
    The Robotiq gripper is mounted on top of the Panda hand. The hand mesh needs
    to be rendered behind/below the gripper base. According to URDF:
    - link8 is at the flange (EE reference frame for Panda)
    - hand is attached to link8 with rotation -45° around Z
    
    In DROID, gripper_poses is T_base_ee where EE = gripper base frame.
    We need to go backwards to find where the Panda hand would be.
    
    The hand mesh center is approximately 0.058m below the gripper mounting point.
    """
    # The hand transform relative to EE: mostly same orientation, offset in -Z
    # The hand mesh is oriented with its grip area pointing in +Z
    T_ee_hand = np.eye(4)
    # Hand is below the gripper base by approximately 6cm
    T_ee_hand[:3, 3] = [0, 0, -0.058]
    
    return T_world_ee @ T_ee_hand


class GripperMaskRenderer:
    """Render 2D masks for Robotiq 85 gripper, optionally with Panda hand."""
    
    def __init__(
        self, 
        mesh_base_path: str = GRIPPER_MESH_BASE,
        arm_mesh_path: str = ROBOT_MESH_BASE,
        include_hand: bool = False
    ):
        """Initialize with gripper and optionally Panda hand mesh files."""
        self.mesh_paths = {
            "base": f"{mesh_base_path}/robotiq_85_base_link_fine.STL",
            "outer_knuckle": f"{mesh_base_path}/outer_knuckle_fine.STL",
            "outer_finger": f"{mesh_base_path}/outer_finger_fine.STL",
            "inner_knuckle": f"{mesh_base_path}/inner_knuckle_fine.STL",
            "inner_finger": f"{mesh_base_path}/inner_finger_fine.STL",
        }
        self.meshes = load_meshes(self.mesh_paths)
        
        # Optionally load Panda hand mesh
        self.include_hand = include_hand
        self.hand_mesh = None
        if include_hand:
            hand_path = f"{arm_mesh_path}/hand.obj"
            if os.path.exists(hand_path):
                try:
                    mesh = trimesh.load(hand_path, force='mesh')
                    if isinstance(mesh, trimesh.Scene):
                        mesh = trimesh.util.concatenate(
                            [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
                        )
                    self.hand_mesh = mesh
                except Exception as e:
                    print(f"[WARN] Could not load Panda hand mesh: {e}")
        
        # Map mesh names to gripper component names
        self.mesh_to_component = {
            "base": ["base"],
            "outer_knuckle": ["left_outer_knuckle", "right_outer_knuckle"],
            "outer_finger": ["left_outer_finger", "right_outer_finger"],
            "inner_knuckle": ["left_inner_knuckle", "right_inner_knuckle"],
            "inner_finger": ["left_inner_finger", "right_inner_finger"],
        }
    
    def render_mask(
        self,
        T_world_ee: np.ndarray,
        gripper_pos: float,
        K: np.ndarray,
        T_cam_world: np.ndarray,
        width: int,
        height: int,
        min_depth: float = 0.01,
        render_hand: bool = True
    ) -> np.ndarray:
        """
        Render gripper mask for a single frame.
        
        Args:
            T_world_ee: End-effector transform in world frame
            gripper_pos: Gripper position (0.0=open, 1.0=closed)
            K: Camera intrinsics (3x3)
            T_cam_world: Camera extrinsics (world to camera, 4x4)
            width: Image width
            height: Image height
            min_depth: Minimum depth threshold
            render_hand: If True and hand mesh is loaded, render Panda hand
            
        Returns:
            Binary mask (height x width) as uint8
        """
        # Get component transforms
        component_transforms = robotiq_gripper_transforms(T_world_ee, gripper_pos)
        
        # Build expanded mesh-transform mapping
        meshes_expanded = {}
        transforms_expanded = {}
        
        for mesh_name, mesh in self.meshes.items():
            components = self.mesh_to_component.get(mesh_name, [])
            for comp in components:
                if comp in component_transforms:
                    meshes_expanded[comp] = mesh
                    transforms_expanded[comp] = component_transforms[comp]
        
        # Add Panda hand if requested
        if render_hand and self.include_hand and self.hand_mesh is not None:
            T_world_hand = compute_hand_transform_from_ee(T_world_ee)
            meshes_expanded["hand"] = self.hand_mesh
            transforms_expanded["hand"] = T_world_hand
        
        return render_multiple_meshes_mask(
            meshes_expanded, transforms_expanded,
            K, T_cam_world, width, height, min_depth
        )


class RobotArmMaskRenderer:
    """Render 2D masks for Panda robot arm (without gripper)."""
    
    def __init__(self, mesh_base_path: str = ROBOT_MESH_BASE):
        """Initialize with robot arm mesh files."""
        self.mesh_paths = {
            f"link{i}": f"{mesh_base_path}/link{i}.obj" for i in range(8)
        }
        self.mesh_paths["hand"] = f"{mesh_base_path}/hand.obj"
        self.meshes = load_meshes(self.mesh_paths)
    
    def render_mask(
        self,
        joint_angles: np.ndarray,
        K: np.ndarray,
        T_cam_world: np.ndarray,
        width: int,
        height: int,
        T_world_base: np.ndarray = np.eye(4),
        min_depth: float = 0.01,
        exclude_hand: bool = True
    ) -> np.ndarray:
        """
        Render robot arm mask for a single frame.
        
        Args:
            joint_angles: 7-element array of joint angles
            K: Camera intrinsics (3x3)
            T_cam_world: Camera extrinsics (world to camera, 4x4)
            width: Image width
            height: Image height
            T_world_base: Base transform in world frame
            min_depth: Minimum depth threshold
            exclude_hand: If True, exclude hand mesh (gripper attachment)
            
        Returns:
            Binary mask (height x width) as uint8
        """
        # Get link transforms
        link_transforms = panda_forward_kinematics(joint_angles, T_world_base)
        
        # Filter meshes if needed
        meshes_to_render = self.meshes.copy()
        if exclude_hand and "hand" in meshes_to_render:
            del meshes_to_render["hand"]
        
        return render_multiple_meshes_mask(
            meshes_to_render, link_transforms,
            K, T_cam_world, width, height, min_depth
        )


class CombinedRobotMaskRenderer:
    """Render combined masks for full robot (arm + gripper)."""
    
    def __init__(
        self,
        gripper_mesh_path: str = GRIPPER_MESH_BASE,
        arm_mesh_path: str = ROBOT_MESH_BASE
    ):
        """Initialize with both gripper and arm meshes."""
        self.gripper_renderer = GripperMaskRenderer(gripper_mesh_path)
        self.arm_renderer = RobotArmMaskRenderer(arm_mesh_path)
    
    def render_masks(
        self,
        joint_angles: np.ndarray,
        gripper_pos: float,
        K: np.ndarray,
        T_cam_world: np.ndarray,
        width: int,
        height: int,
        T_world_base: np.ndarray = np.eye(4),
        min_depth: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Render separate and combined masks.
        
        Args:
            joint_angles: 7-element array of joint angles
            gripper_pos: Gripper position (0.0=open, 1.0=closed)
            K: Camera intrinsics
            T_cam_world: Camera extrinsics
            width: Image width
            height: Image height
            T_world_base: Base transform
            min_depth: Minimum depth threshold
            
        Returns:
            Tuple of (gripper_mask, arm_mask, combined_mask)
        """
        # Get end-effector transform for gripper
        link_transforms = panda_forward_kinematics(joint_angles, T_world_base)
        T_world_ee = link_transforms.get("hand", np.eye(4))
        
        # Render arm mask
        arm_mask = self.arm_renderer.render_mask(
            joint_angles, K, T_cam_world, width, height, T_world_base, min_depth
        )
        
        # Render gripper mask
        gripper_mask = self.gripper_renderer.render_mask(
            T_world_ee, gripper_pos, K, T_cam_world, width, height, min_depth
        )
        
        # Combined mask
        combined_mask = np.maximum(arm_mask, gripper_mask)
        
        return gripper_mask, arm_mask, combined_mask
