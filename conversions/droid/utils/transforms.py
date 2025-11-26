"""Transformation utilities for DROID point cloud generation.

This module contains all transformation logic for converting points between
different coordinate frames in the DROID setup. All transformations use
pytransform3d for the mathematical operations.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from pytransform3d import transformations as pt


# =============================================================================
# Basic Transform Utilities
# =============================================================================

def pose6_to_T(p):
    """
    Convert [x, y, z, roll, pitch, yaw] to 4x4 transformation matrix.
    
    Args:
        p: Array-like of 6 elements [x, y, z, roll, pitch, yaw]
        
    Returns:
        4x4 numpy array representing the homogeneous transformation matrix
    """
    x, y, z, roll, pitch, yaw = p
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    T[:3, :3] = R.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    return T


def rvec_tvec_to_matrix(val):
    """
    Convert rotation vector and translation vector to 4x4 transformation matrix.
    
    Args:
        val: Array-like of 6 elements [tx, ty, tz, rx, ry, rz] where
             first 3 are translation and last 3 are euler angles
             
    Returns:
        4x4 numpy array representing the homogeneous transformation matrix
    """
    pos = np.array(val[0:3])
    euler = np.array(val[3:6])
    R_mat = R.from_euler("xyz", euler).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = pos
    return T


def transform_points(points, transform_matrix):
    """
    Transform 3D points using a 4x4 transformation matrix.
    
    Args:
        points: Nx3 numpy array of 3D points
        transform_matrix: 4x4 homogeneous transformation matrix
        
    Returns:
        Nx3 numpy array of transformed points
    """
    if points.shape[0] == 0:
        return points
    
    # Convert to homogeneous coordinates (Nx4)
    points_homo = np.hstack([points, np.ones((points.shape[0], 1))])
    
    # Apply transformation: (4x4) @ (4xN) = (4xN)
    points_transformed = (transform_matrix @ points_homo.T).T
    
    # Return only XYZ (drop homogeneous coordinate)
    return points_transformed[:, :3]


def decompose_transform(transform_matrix):
    """
    Decompose a 4x4 transformation matrix into translation and rotation components.
    
    Args:
        transform_matrix: 4x4 homogeneous transformation matrix
        
    Returns:
        tuple: (translation, rotation_matrix) where
               - translation is a 3-element array [x, y, z]
               - rotation_matrix is a 3x3 rotation matrix
    """
    translation = transform_matrix[:3, 3]
    rotation_matrix = transform_matrix[:3, :3]
    return translation, rotation_matrix


def invert_transform(transform_matrix):
    """
    Invert a 4x4 homogeneous transformation matrix using pytransform3d.
    
    Args:
        transform_matrix: 4x4 homogeneous transformation matrix
        
    Returns:
        4x4 numpy array representing the inverted transformation
    """
    return pt.invert_transform(transform_matrix)


# =============================================================================
# DROID-Specific Transformation Pipelines
# =============================================================================

def compute_wrist_cam_offset(wrist_pose_t0, ee_pose_t0):
    """
    Compute the constant transform from end-effector to wrist camera.
    
    This is calculated once at t=0 and remains constant throughout the trajectory.
    
    Args:
        wrist_pose_t0: [x, y, z, roll, pitch, yaw] of wrist camera at t=0
        ee_pose_t0: [x, y, z, roll, pitch, yaw] of end-effector at t=0
        
    Returns:
        T_ee_cam: 4x4 transform from end-effector frame to camera frame
    """
    T_base_cam0 = pose6_to_T(wrist_pose_t0)
    T_base_ee0 = pose6_to_T(ee_pose_t0)
    
    # Constant offset: EE -> Camera
    # T_base_cam = T_base_ee @ T_ee_cam
    # => T_ee_cam = inv(T_base_ee) @ T_base_cam
    T_ee_cam = pt.invert_transform(T_base_ee0) @ T_base_cam0
    
    return T_ee_cam


def wrist_cam_to_world(ee_pose, T_ee_cam, gripper_alignment_fix=True):
    """
    Compute the world transform for the wrist camera at a given timestep.
    
    Args:
        ee_pose: [x, y, z, roll, pitch, yaw] of end-effector at current timestep
        T_ee_cam: 4x4 constant transform from end-effector to camera (from compute_wrist_cam_offset)
        gripper_alignment_fix: If True, apply 90-degree Z rotation to align gripper visualization
        
    Returns:
        T_world_cam: 4x4 transform from camera frame to world frame
    """
    T_base_ee = pose6_to_T(ee_pose)
    
    # Apply gripper alignment fix if needed
    if gripper_alignment_fix:
        R_fix = R.from_euler('z', 90, degrees=True).as_matrix()
        T_base_ee[:3, :3] = T_base_ee[:3, :3] @ R_fix
    
    # T_world_cam = T_base_ee @ T_ee_cam
    T_world_cam = T_base_ee @ T_ee_cam
    
    return T_world_cam


def wrist_points_to_world(points_cam, ee_pose, T_ee_cam, gripper_alignment_fix=True):
    """
    Transform wrist camera points to world coordinates.
    
    Complete pipeline: Camera Frame -> End-Effector Frame -> World Frame
    
    Args:
        points_cam: Nx3 array of points in camera frame
        ee_pose: [x, y, z, roll, pitch, yaw] of end-effector at current timestep
        T_ee_cam: 4x4 constant transform from end-effector to camera
        gripper_alignment_fix: If True, apply 90-degree Z rotation to align gripper
        
    Returns:
        points_world: Nx3 array of points in world frame
    """
    # Get world transform for wrist camera
    T_world_cam = wrist_cam_to_world(ee_pose, T_ee_cam, gripper_alignment_fix)
    
    # Transform points
    points_world = transform_points(points_cam, T_world_cam)
    
    return points_world


def external_cam_to_world(extrinsic_params):
    """
    Get the static world transform for an external camera.
    
    Args:
        extrinsic_params: [tx, ty, tz, rx, ry, rz] extrinsic calibration parameters
        
    Returns:
        T_world_cam: 4x4 transform from camera frame to world frame
    """
    T_world_cam = rvec_tvec_to_matrix(extrinsic_params)
    return T_world_cam


def external_points_to_world(points_cam, extrinsic_params):
    """
    Transform external camera points to world coordinates.
    
    Complete pipeline: Camera Frame -> World Frame (static transform)
    
    Args:
        points_cam: Nx3 array of points in camera frame
        extrinsic_params: [tx, ty, tz, rx, ry, rz] extrinsic calibration parameters
        
    Returns:
        points_world: Nx3 array of points in world frame
    """
    T_world_cam = external_cam_to_world(extrinsic_params)
    points_world = transform_points(points_cam, T_world_cam)
    
    return points_world


def precompute_wrist_trajectory(cartesian_positions, wrist_pose_t0, gripper_alignment_fix=True):
    """
    Precompute all wrist camera transforms for the entire trajectory.
    
    Args:
        cartesian_positions: Nx6 array of [x, y, z, roll, pitch, yaw] for all timesteps
        wrist_pose_t0: [x, y, z, roll, pitch, yaw] of wrist camera at t=0
        gripper_alignment_fix: If True, apply 90-degree Z rotation to align gripper
        
    Returns:
        wrist_transforms: List of 4x4 transforms for each timestep
    """
    # Compute constant offset
    T_ee_cam = compute_wrist_cam_offset(wrist_pose_t0, cartesian_positions[0])
    
    # Compute transform for each timestep
    wrist_transforms = []
    for ee_pose in cartesian_positions:
        T_world_cam = wrist_cam_to_world(ee_pose, T_ee_cam, gripper_alignment_fix)
        wrist_transforms.append(T_world_cam)
    
    return wrist_transforms
