"""Transformation utilities for DROID point cloud generation."""

import numpy as np
from scipy.spatial.transform import Rotation as R


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
