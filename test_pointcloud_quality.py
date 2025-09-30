#!/usr/bin/env python3
"""
Test script to compare the new demo.py-style point cloud creation
with the old multi-camera approach.
"""

import sys
sys.path.append('/workspace')

import numpy as np
from pathlib import Path
from create_sparse_depth_map import (
    create_confidence_filtered_point_cloud,
    unproject_to_world_o3d,
    read_depth,
    read_rgb
)

def test_point_cloud_quality():
    """Compare point cloud creation methods."""
    
    # Test data paths (adjust these to your actual data)
    test_task_folder = Path("/data/rh20t_api/data/test_data_full_rgb_upscaled_depth/uncompressed_low_res_data/task_0065_user_0010_scene_0009_cfg_0004")
    
    if not test_task_folder.exists():
        print("Test data not found, skipping comparison")
        return
    
    # Find a camera directory
    cam_dirs = [p for p in test_task_folder.glob("cam_*") if p.is_dir()]
    if not cam_dirs:
        print("No camera directories found")
        return
    
    cam_dir = cam_dirs[0]
    print(f"Testing with camera: {cam_dir.name}")
    
    # Find first color and depth files
    color_files = list((cam_dir / "color").glob("*.jpg"))
    depth_files = list((cam_dir / "depth").glob("*.png"))
    
    if not color_files or not depth_files:
        print("No color or depth files found")
        return
    
    # Load test data
    rgb = read_rgb(color_files[0])
    depth = read_depth(depth_files[0], is_l515=False)  # Assuming not L515
    
    # Create dummy calibration matrices
    H, W = depth.shape
    K = np.array([
        [500, 0, W/2],
        [0, 500, H/2],
        [0, 0, 1]
    ], dtype=np.float32)
    
    E_inv = np.eye(4, dtype=np.float32)  # Identity transform for testing
    
    print(f"Input depth shape: {depth.shape}")
    print(f"Valid depth pixels: {np.sum(depth > 0)} / {depth.size}")
    
    # Test old method
    print("\n=== OLD METHOD (Basic Open3D) ===")
    pcd_old = unproject_to_world_o3d.__wrapped__(depth, rgb, K, E_inv)  # Call the legacy version
    print(f"Old method points: {len(pcd_old.points)}")
    
    # Test new method
    print("\n=== NEW METHOD (Demo.py style with confidence filtering) ===")
    pcd_new = create_confidence_filtered_point_cloud(
        depth, rgb, K, E_inv,
        confidence_threshold=0.1,
        edge_margin=10,
        gradient_threshold=0.1,
        smooth_kernel=3
    )
    print(f"New method points: {len(pcd_new.points)}")
    
    # Compare point counts
    reduction_ratio = len(pcd_new.points) / len(pcd_old.points) if len(pcd_old.points) > 0 else 0
    print(f"\nPoint reduction: {reduction_ratio:.3f} ({len(pcd_new.points)}/{len(pcd_old.points)})")
    
    if reduction_ratio < 0.5:
        print("✅ Good: New method significantly filters unreliable points")
    elif reduction_ratio < 0.8:
        print("⚠️  Moderate filtering")
    else:
        print("❌ Warning: New method didn't filter much")
    
    print(f"New method should produce higher quality point clouds for reprojection!")

if __name__ == "__main__":
    test_point_cloud_quality()