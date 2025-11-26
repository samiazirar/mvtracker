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
# DROID Specific Transform Logic
# =============================================================================

def compute_wrist_cam_offset(wrist_pose_t0, cartesian_position_t0):
    """
    Calculate the constant offset from End-Effector to Wrist Camera.
    
    Args:
        wrist_pose_t0: 6-element array [x,y,z,r,p,y] of wrist camera at t=0
        cartesian_position_t0: 6-element array [x,y,z,r,p,y] of EE at t=0
        
    Returns:
        4x4 numpy array representing T_ee_cam
    """
    T_base_cam0 = pose6_to_T(wrist_pose_t0)
    T_base_ee0 = pose6_to_T(cartesian_position_t0)
    # T_ee_cam = inv(T_base_ee0) @ T_base_cam0
    return np.linalg.inv(T_base_ee0) @ T_base_cam0


def wrist_cam_to_world(cartesian_position_t, T_ee_cam):
    """
    Calculate Wrist Camera pose in World frame at time t.
    
    Args:
        cartesian_position_t: 6-element array [x,y,z,r,p,y] of EE at time t
        T_ee_cam: 4x4 offset matrix (EE -> Camera)
        
    Returns:
        4x4 numpy array representing T_world_cam
    """
    T_base_ee_t = pose6_to_T(cartesian_position_t)
    return T_base_ee_t @ T_ee_cam


def wrist_points_to_world(points, cartesian_position_t, T_ee_cam):
    """
    Transform points from Wrist Camera frame to World frame.
    
    Args:
        points: Nx3 numpy array of points in camera frame
        cartesian_position_t: 6-element array of EE pose
        T_ee_cam: 4x4 offset matrix
        
    Returns:
        Nx3 numpy array of points in world frame
    """
    T_world_cam = wrist_cam_to_world(cartesian_position_t, T_ee_cam)
    return transform_points(points, T_world_cam)


def external_cam_to_world(transform_list):
    """
    Calculate External Camera pose in World frame.
    
    Args:
        transform_list: 6-element array [tx,ty,tz,rx,ry,rz]
        
    Returns:
        4x4 numpy array representing T_world_cam
    """
    return rvec_tvec_to_matrix(transform_list)


def external_points_to_world(points, transform_list):
    """
    Transform points from External Camera frame to World frame.
    
    Args:
        points: Nx3 numpy array of points in camera frame
        transform_list: 6-element array defining camera pose
        
    Returns:
        Nx3 numpy array of points in world frame
    """
    T_world_cam = external_cam_to_world(transform_list)
    return transform_points(points, T_world_cam)


def precompute_wrist_trajectory(cartesian_positions, T_ee_cam):
    """
    Precompute wrist camera poses for the entire trajectory.
    
    Args:
        cartesian_positions: Nx6 array of EE poses
        T_ee_cam: 4x4 offset matrix
        
    Returns:
        List of 4x4 numpy arrays representing T_world_cam for each frame
    """
    transforms = []
    for pos in cartesian_positions:
        transforms.append(wrist_cam_to_world(pos, T_ee_cam))
    return transforms
