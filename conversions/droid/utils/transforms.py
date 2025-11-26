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
