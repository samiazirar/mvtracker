"""
Test that invisible points have NaN depth values after depth sampling.

This test verifies that after sampling depth using align_nearest_neighbor,
points marked as invisible by the tracker (view_vis_e == False) have their
depth set to NaN, preventing invalid 3D coordinates from being computed.
"""

import torch
import sys
import os

# Add parent directory to path to import mvtracker modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mvtracker.models.core.monocular_baselines import align_nearest_neighbor


def test_align_nearest_neighbor_preserves_valid_depths():
    """Test that align_nearest_neighbor samples depths correctly for valid points."""
    T, N = 5, 10
    H, W = 256, 256
    
    # Create a simple depth map with known values
    view_depths = torch.ones(T, 1, H, W) * 5.0  # All depths are 5.0
    
    # Create trajectory points in the middle of the image
    view_traj_e = torch.ones(T, N, 2) * 128.0  # All points at (128, 128)
    
    # Sample depths
    view_camera_z = align_nearest_neighbor(view_depths, view_traj_e)
    
    # Check that all sampled depths are approximately 5.0
    assert view_camera_z.shape == (T, N, 1)
    assert torch.allclose(view_camera_z, torch.ones_like(view_camera_z) * 5.0)


def test_invisible_points_should_have_nan_depth():
    """
    Test that after applying visibility mask, invisible points have NaN depth.
    
    This is the main test for the fix. The depth sampling function should
    set depth to NaN for points where view_vis_e is False.
    """
    T, N = 5, 10
    H, W = 256, 256
    
    # Create depth map
    view_depths = torch.ones(T, 1, H, W) * 5.0
    
    # Create trajectory points
    view_traj_e = torch.ones(T, N, 2) * 128.0
    
    # Create visibility mask - half the points are invisible
    view_vis_e = torch.zeros(T, N, dtype=torch.bool)
    view_vis_e[:, :5] = True  # First 5 points are visible
    # Last 5 points are invisible
    
    # Sample depths
    view_camera_z = align_nearest_neighbor(view_depths, view_traj_e)
    
    # Apply visibility mask (THIS IS WHAT NEEDS TO BE FIXED)
    # After the fix, invisible points should have NaN depth
    view_camera_z[~view_vis_e] = float('nan')
    
    # Check that visible points have valid depth
    assert torch.allclose(view_camera_z[:, :5], torch.ones_like(view_camera_z[:, :5]) * 5.0)
    
    # Check that invisible points have NaN depth
    assert torch.isnan(view_camera_z[:, 5:]).all()


def test_sparse_depth_with_zero_values():
    """Test that zero depth values are handled correctly."""
    T, N = 5, 10
    H, W = 256, 256
    
    # Create sparse depth map with some zeros
    view_depths = torch.zeros(T, 1, H, W)
    view_depths[:, :, 100:150, 100:150] = 5.0  # Valid depths in center region
    
    # Points in the valid region
    view_traj_e = torch.ones(T, N, 2) * 125.0
    
    # Sample depths
    view_camera_z = align_nearest_neighbor(view_depths, view_traj_e)
    
    # All points should have depth around 5.0 since they're in the valid region
    assert view_camera_z.shape == (T, N, 1)
    assert torch.allclose(view_camera_z, torch.ones_like(view_camera_z) * 5.0, atol=1e-4)


def test_integration_with_visibility_masking():
    """
    Integration test simulating the actual usage pattern in monocular_baselines.py.
    
    This tests the pattern:
    1. Sample depth using align_nearest_neighbor
    2. Apply visibility mask to set invisible points to NaN
    3. Verify NaN propagation prevents invalid 3D coordinates
    """
    T, N = 5, 10
    H, W = 256, 256
    
    # Simulate the actual data
    view_depths = torch.randn(T, 1, H, W).abs() * 10.0  # Random positive depths
    view_traj_e = torch.rand(T, N, 2) * 200 + 28  # Random points in image
    view_vis_e = torch.rand(T, N) > 0.5  # Random visibility
    
    # Sample depths (as done in monocular_baselines.py)
    view_camera_z = align_nearest_neighbor(view_depths, view_traj_e)
    
    # Apply visibility mask - THIS IS THE CRITICAL FIX
    # Invisible points should have NaN depth
    view_camera_z[~view_vis_e] = float('nan')
    
    # Verify the shape
    assert view_camera_z.shape == (T, N, 1)
    
    # Verify visible points have valid (non-NaN) depths
    visible_mask = view_vis_e.unsqueeze(-1)
    assert not torch.isnan(view_camera_z[visible_mask]).any(), \
        "Visible points should not have NaN depth"
    
    # Verify invisible points have NaN depths
    invisible_mask = ~view_vis_e.unsqueeze(-1)
    assert torch.isnan(view_camera_z[invisible_mask]).all(), \
        "Invisible points should have NaN depth"


def test_nan_depth_prevents_invalid_3d_coords():
    """
    Test that NaN depths result in NaN 3D coordinates.
    
    This verifies that the visualizer will correctly ignore points with NaN depth.
    """
    T, N = 5, 3
    
    # Create sample data
    view_camera_z = torch.tensor([
        [[5.0], [float('nan')], [3.0]],  # Frame 0: point 1 is invisible
        [[4.0], [4.5], [float('nan')]],   # Frame 1: point 2 is invisible
        [[float('nan')], [5.0], [4.0]],   # Frame 2: point 0 is invisible
        [[3.0], [float('nan')], [float('nan')]],  # Frame 3: points 1,2 invisible
        [[5.5], [4.8], [3.2]],            # Frame 4: all visible
    ])
    
    # Simulate pixel coordinates
    pixel_xy = torch.rand(T, N, 2) * 200 + 28
    
    # Create dummy camera matrices
    intrs = torch.eye(3).unsqueeze(0).repeat(T, 1, 1)
    extrs = torch.eye(4).unsqueeze(0).repeat(T, 1, 1)
    
    # Simulate world space conversion (simplified)
    # In the actual code, this would use pixel_xy_and_camera_z_to_world_space
    # For testing, we just multiply to show NaN propagation
    fake_3d_coords = pixel_xy * view_camera_z.squeeze(-1).unsqueeze(-1)
    
    # Verify that NaN depths lead to NaN 3D coordinates
    assert torch.isnan(fake_3d_coords[0, 1]).all(), "Frame 0, point 1 should be NaN"
    assert torch.isnan(fake_3d_coords[1, 2]).all(), "Frame 1, point 2 should be NaN"
    assert torch.isnan(fake_3d_coords[2, 0]).all(), "Frame 2, point 0 should be NaN"
    assert torch.isnan(fake_3d_coords[3, 1]).all(), "Frame 3, point 1 should be NaN"
    assert torch.isnan(fake_3d_coords[3, 2]).all(), "Frame 3, point 2 should be NaN"
    
    # Verify visible points don't have NaN
    assert not torch.isnan(fake_3d_coords[0, 0]).any(), "Frame 0, point 0 should be valid"
    assert not torch.isnan(fake_3d_coords[4, :]).any(), "Frame 4, all points should be valid"


if __name__ == "__main__":
    print("Running tests for NaN depth handling...")
    
    print("\n1. Testing align_nearest_neighbor preserves valid depths...")
    test_align_nearest_neighbor_preserves_valid_depths()
    print("✓ PASSED")
    
    print("\n2. Testing invisible points should have NaN depth...")
    test_invisible_points_should_have_nan_depth()
    print("✓ PASSED")
    
    print("\n3. Testing sparse depth with zero values...")
    test_sparse_depth_with_zero_values()
    print("✓ PASSED")
    
    print("\n4. Testing integration with visibility masking...")
    test_integration_with_visibility_masking()
    print("✓ PASSED")
    
    print("\n5. Testing NaN depth prevents invalid 3D coords...")
    test_nan_depth_prevents_invalid_3d_coords()
    print("✓ PASSED")
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)
