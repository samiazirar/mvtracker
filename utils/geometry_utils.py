"""
Geometry Utilities for 3D Transformations

This module provides fundamental 3D geometry operations including:
- Quaternion <-> Rotation matrix conversions
- 7D pose <-> 4x4 transformation matrix conversions
- Rotation matrix manipulations
- Oriented bounding box operations

Functions:
    _rotation_matrix_to_xyzw: Convert 3x3 rotation matrix to quaternion [x,y,z,w]
    _quaternion_xyzw_to_rotation_matrix: Convert quaternion [x,y,z,w] to 3x3 rotation matrix
    _pose_7d_to_matrix: Convert 7D pose [x,y,z,qx,qy,qz,qw] to 4x4 matrix
    _rotation_matrix_to_quaternion: Convert rotation matrix to quaternion [w,x,y,z] (COLMAP format)
    _ensure_basis: Extract rotation basis from bbox dict
    _compute_bbox_corners_world: Compute 8 corners of oriented bounding box in world coordinates
"""

import numpy as np
from typing import Dict, Optional


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


def _rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    return np.array([w, x, y, z])


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
