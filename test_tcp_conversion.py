#!/usr/bin/env python3
"""
Quick test to verify the TCP pose conversion function works correctly.
"""

import numpy as np
import sys
sys.path.insert(0, '/workspace')

from create_sparse_depth_map import _pose_7d_to_matrix, _quaternion_xyzw_to_rotation_matrix


def test_pose_7d_to_matrix():
    """Test conversion of 7D pose to 4x4 transformation matrix."""
    print("Testing _pose_7d_to_matrix()...")
    
    # Test case 1: Identity pose (origin, no rotation)
    pose_identity = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
    matrix_identity = _pose_7d_to_matrix(pose_identity)
    
    expected_identity = np.eye(4, dtype=np.float32)
    assert np.allclose(matrix_identity, expected_identity), "Identity pose failed"
    print("✓ Identity pose test passed")
    
    # Test case 2: Translation only
    pose_translation = np.array([1.0, 2.0, 3.0, 0, 0, 0, 1], dtype=np.float32)
    matrix_translation = _pose_7d_to_matrix(pose_translation)
    
    expected_translation = np.eye(4, dtype=np.float32)
    expected_translation[:3, 3] = [1.0, 2.0, 3.0]
    assert np.allclose(matrix_translation, expected_translation), "Translation pose failed"
    print("✓ Translation pose test passed")
    
    # Test case 3: 90-degree rotation around Z-axis
    # Quaternion for 90° rotation around Z: [0, 0, sin(45°), cos(45°)] ≈ [0, 0, 0.707, 0.707]
    pose_rotation = np.array([0, 0, 0, 0, 0, 0.7071068, 0.7071068], dtype=np.float32)
    matrix_rotation = _pose_7d_to_matrix(pose_rotation)
    
    # Check that it's a valid rotation matrix (orthonormal)
    rotation_part = matrix_rotation[:3, :3]
    identity_check = rotation_part @ rotation_part.T
    assert np.allclose(identity_check, np.eye(3)), "Rotation matrix not orthonormal"
    print("✓ Rotation pose test passed")
    
    # Test case 4: Combined translation and rotation
    pose_combined = np.array([1.0, 2.0, 3.0, 0, 0, 0.7071068, 0.7071068], dtype=np.float32)
    matrix_combined = _pose_7d_to_matrix(pose_combined)
    
    # Check translation part
    assert np.allclose(matrix_combined[:3, 3], [1.0, 2.0, 3.0]), "Combined pose translation failed"
    # Check that bottom row is [0, 0, 0, 1]
    assert np.allclose(matrix_combined[3, :], [0, 0, 0, 1]), "Bottom row incorrect"
    print("✓ Combined pose test passed")
    
    # Test case 5: Real-world example (typical TCP pose)
    pose_real = np.array([0.5, 0.3, 0.8, 0.1, 0.2, 0.3, 0.9], dtype=np.float32)
    matrix_real = _pose_7d_to_matrix(pose_real)
    
    # Verify it's a valid transformation matrix
    assert matrix_real.shape == (4, 4), "Output shape incorrect"
    assert np.allclose(matrix_real[3, :], [0, 0, 0, 1]), "Bottom row incorrect"
    
    # Check rotation part is orthonormal
    R = matrix_real[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-5), "Rotation not orthonormal"
    assert np.allclose(np.linalg.det(R), 1.0, atol=1e-5), "Rotation determinant not 1"
    print("✓ Real-world pose test passed")
    
    print("\n✅ All tests passed!")
    return True


if __name__ == "__main__":
    try:
        test_pose_7d_to_matrix()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
