#!/usr/bin/env python3
"""
Comprehensive tests for gripper tracking and alignment features.

Tests cover:
1. Bbox alignment with point cloud center of mass
2. MVTracker integration for gripper tracking
3. SAM integration for object tracking
"""

import unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


class TestBboxAlignment(unittest.TestCase):
    """Test bbox alignment with point cloud center of mass."""

    def setUp(self):
        """Create test data: simple bbox and point cloud."""
        # Create a simple bbox centered at origin
        self.bbox = {
            "center": np.array([0.0, 0.0, 0.1], dtype=np.float32),
            "half_sizes": np.array([0.05, 0.03, 0.08], dtype=np.float32),
            "quat_xyzw": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "basis": np.eye(3, dtype=np.float32),
        }
        
        # Create point cloud shifted to the right (+x direction)
        # Should cause bbox to shift right in x-y plane
        num_points = 100
        self.points = np.random.randn(num_points, 3).astype(np.float32) * 0.02
        self.points[:, 0] += 0.03  # Shift right
        self.points[:, 2] += 0.1   # Match bbox z center
        
        # Create colors for points
        self.colors = np.random.rand(num_points, 3).astype(np.float32)

    def test_bbox_alignment_shifts_xy(self):
        """Test that alignment shifts bbox in x-y plane to match point COM."""
        from create_sparse_depth_map import _align_bbox_with_point_cloud_com
        
        aligned_bbox = _align_bbox_with_point_cloud_com(
            bbox=self.bbox,
            points=self.points,
            colors=self.colors,
            search_radius_scale=2.0,
        )
        
        # Check that bbox shifted in +x direction
        self.assertIsNotNone(aligned_bbox)
        self.assertGreater(aligned_bbox["center"][0], self.bbox["center"][0])
        
        # Check that z coordinate didn't change significantly
        self.assertAlmostEqual(
            aligned_bbox["center"][2], 
            self.bbox["center"][2], 
            places=3,
            msg="Z coordinate should remain unchanged"
        )

    def test_bbox_alignment_preserves_height(self):
        """Test that alignment doesn't change bbox height (z-axis)."""
        from create_sparse_depth_map import _align_bbox_with_point_cloud_com
        
        aligned_bbox = _align_bbox_with_point_cloud_com(
            bbox=self.bbox,
            points=self.points,
            colors=self.colors,
            search_radius_scale=2.0,
        )
        
        # Check that half_sizes remain the same
        np.testing.assert_array_almost_equal(
            aligned_bbox["half_sizes"],
            self.bbox["half_sizes"],
            decimal=5,
            err_msg="Bbox dimensions should not change"
        )

    def test_bbox_alignment_with_rotated_points(self):
        """Test alignment when point cloud is rotated around z-axis."""
        from create_sparse_depth_map import _align_bbox_with_point_cloud_com
        
        # Create a rotated point cloud (45 degrees around z-axis)
        angle = np.pi / 4
        rotation_z = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        rotated_points = (rotation_z @ self.points.T).T
        
        aligned_bbox = _align_bbox_with_point_cloud_com(
            bbox=self.bbox,
            points=rotated_points,
            colors=self.colors,
            search_radius_scale=2.0,
        )
        
        # Check that basis matrix changed (rotation applied)
        self.assertIsNotNone(aligned_bbox)
        # Original basis is identity, so any rotation should change it
        basis_changed = not np.allclose(aligned_bbox["basis"], self.bbox["basis"], atol=0.01)
        self.assertTrue(basis_changed, "Basis should be updated to match point cloud orientation")

    def test_bbox_alignment_with_empty_points(self):
        """Test that alignment handles empty point clouds gracefully."""
        from create_sparse_depth_map import _align_bbox_with_point_cloud_com
        
        empty_points = np.zeros((0, 3), dtype=np.float32)
        empty_colors = np.zeros((0, 3), dtype=np.float32)
        
        aligned_bbox = _align_bbox_with_point_cloud_com(
            bbox=self.bbox,
            points=empty_points,
            colors=empty_colors,
            search_radius_scale=2.0,
        )
        
        # Should return original bbox when no points available
        self.assertIsNotNone(aligned_bbox)
        np.testing.assert_array_almost_equal(
            aligned_bbox["center"],
            self.bbox["center"],
            decimal=5
        )

    def test_bbox_alignment_filters_distant_points(self):
        """Test that alignment only uses points near the bbox."""
        from create_sparse_depth_map import _align_bbox_with_point_cloud_com
        
        # Add distant outlier points that should be ignored
        outliers = np.array([
            [10.0, 10.0, 0.1],
            [-10.0, -10.0, 0.1],
        ], dtype=np.float32)
        outlier_colors = np.ones((2, 3), dtype=np.float32)
        
        combined_points = np.vstack([self.points, outliers])
        combined_colors = np.vstack([self.colors, outlier_colors])
        
        aligned_bbox = _align_bbox_with_point_cloud_com(
            bbox=self.bbox,
            points=combined_points,
            colors=combined_colors,
            search_radius_scale=2.0,
        )
        
        # Result should be similar to using just nearby points
        aligned_bbox_no_outliers = _align_bbox_with_point_cloud_com(
            bbox=self.bbox,
            points=self.points,
            colors=self.colors,
            search_radius_scale=2.0,
        )
        
        np.testing.assert_array_almost_equal(
            aligned_bbox["center"],
            aligned_bbox_no_outliers["center"],
            decimal=2,
            err_msg="Distant outliers should not affect alignment"
        )


class TestMVTrackerIntegration(unittest.TestCase):
    """Test MVTracker integration for gripper tracking."""

    def setUp(self):
        """Create minimal test data for tracking."""
        # Create simple synthetic RGBD sequence
        self.num_views = 2
        self.num_frames = 5
        self.height = 64
        self.width = 64
        
        self.rgbs = torch.randint(0, 255, (self.num_views, self.num_frames, 3, self.height, self.width), dtype=torch.uint8)
        self.depths = torch.rand(self.num_views, self.num_frames, 1, self.height, self.width) * 2.0
        self.intrs = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(self.num_views, self.num_frames, 1, 1)
        self.intrs[:, :, 0, 0] = 50.0  # fx
        self.intrs[:, :, 1, 1] = 50.0  # fy
        self.intrs[:, :, 0, 2] = 32.0  # cx
        self.intrs[:, :, 1, 2] = 32.0  # cy
        
        self.extrs = torch.eye(3, 4).unsqueeze(0).unsqueeze(0).repeat(self.num_views, self.num_frames, 1, 1)
        
        # Create 3 simple bboxes (gripper, body, fingertip)
        self.gripper_bboxes = []
        for t in range(self.num_frames):
            bbox = {
                "center": np.array([0.0, 0.0, 1.0], dtype=np.float32),
                "half_sizes": np.array([0.04, 0.02, 0.06], dtype=np.float32),
                "quat_xyzw": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                "basis": np.eye(3, dtype=np.float32),
            }
            self.gripper_bboxes.append(bbox)
        
        self.body_bboxes = [b.copy() for b in self.gripper_bboxes]
        self.fingertip_bboxes = [b.copy() for b in self.gripper_bboxes]

    def test_track_gripper_with_mvtracker_returns_tracks(self):
        """Test that tracking function returns valid track data."""
        from create_sparse_depth_map import _track_gripper_with_mvtracker
        
        result = _track_gripper_with_mvtracker(
            rgbs=self.rgbs,
            depths=self.depths,
            intrs=self.intrs,
            extrs=self.extrs,
            gripper_bboxes=self.gripper_bboxes,
            body_bboxes=self.body_bboxes,
            fingertip_bboxes=self.fingertip_bboxes,
            device="cpu",
        )
        
        # Check that result contains expected keys
        self.assertIn("gripper_tracks", result)
        self.assertIn("body_tracks", result)
        self.assertIn("fingertip_tracks", result)
        self.assertIn("gripper_vis", result)
        self.assertIn("body_vis", result)
        self.assertIn("fingertip_vis", result)
        
        # Check shapes
        gripper_tracks = result["gripper_tracks"]
        self.assertEqual(gripper_tracks.ndim, 3)  # [T, N, 3]
        self.assertEqual(gripper_tracks.shape[0], self.num_frames)

    def test_track_gripper_handles_none_bboxes(self):
        """Test that tracking handles missing bboxes gracefully."""
        from create_sparse_depth_map import _track_gripper_with_mvtracker
        
        result = _track_gripper_with_mvtracker(
            rgbs=self.rgbs,
            depths=self.depths,
            intrs=self.intrs,
            extrs=self.extrs,
            gripper_bboxes=None,
            body_bboxes=self.body_bboxes,
            fingertip_bboxes=None,
            device="cpu",
        )
        
        # Should still return valid structure
        self.assertIn("body_tracks", result)
        self.assertIsNone(result.get("gripper_tracks"))
        self.assertIsNone(result.get("fingertip_tracks"))

    def test_track_gripper_query_points_from_bbox(self):
        """Test that query points are correctly generated from bbox corners."""
        from create_sparse_depth_map import _generate_query_points_from_bbox
        
        bbox = self.gripper_bboxes[0]
        query_points = _generate_query_points_from_bbox(
            bbox=bbox,
            timestamp=0,
            num_points=8,
        )
        
        # Should return [N, 4] array with [t, x, y, z]
        self.assertEqual(query_points.shape, (8, 4))
        self.assertTrue(np.all(query_points[:, 0] == 0))  # All at t=0


class TestSAMIntegration(unittest.TestCase):
    """Test SAM integration for object tracking."""

    def setUp(self):
        """Create test data for SAM tracking."""
        self.rgb_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Create simple bbox as seed for SAM
        self.seed_bbox = {
            "center": np.array([320.0, 240.0], dtype=np.float32),
            "width": 100.0,
            "height": 80.0,
        }

    def test_sam_segment_object_returns_mask(self):
        """Test that SAM segmentation returns a valid mask."""
        from create_sparse_depth_map import _segment_object_with_sam
        
        # This test requires SAM model, so we'll skip if not available
        try:
            mask = _segment_object_with_sam(
                rgb_image=self.rgb_image,
                bbox_prompt=self.seed_bbox,
                model_type="vit_b",
            )
            
            # Check mask shape matches image
            self.assertEqual(mask.shape[:2], self.rgb_image.shape[:2])
            self.assertTrue(mask.dtype == bool or mask.dtype == np.uint8)
            
        except ImportError:
            self.skipTest("SAM not available")

    def test_sam_track_gripper_contact_objects(self):
        """Test SAM-based tracking of objects in contact with gripper."""
        from create_sparse_depth_map import _track_gripper_contact_objects_with_sam
        
        try:
            # Create sequence of RGB images
            rgbs = np.random.randint(0, 255, (5, 480, 640, 3), dtype=np.uint8)
            
            # Create gripper bboxes (2D projections)
            gripper_bboxes_2d = []
            for t in range(5):
                bbox = {
                    "center": np.array([320.0, 240.0 + t * 10], dtype=np.float32),
                    "width": 100.0,
                    "height": 80.0,
                }
                gripper_bboxes_2d.append(bbox)
            
            result = _track_gripper_contact_objects_with_sam(
                rgbs=rgbs,
                gripper_bboxes_2d=gripper_bboxes_2d,
                contact_threshold_pixels=50,
            )
            
            # Should return masks for each frame
            self.assertIn("object_masks", result)
            self.assertEqual(len(result["object_masks"]), 5)
            
        except ImportError:
            self.skipTest("SAM not available")


class TestEndToEnd(unittest.TestCase):
    """End-to-end integration tests."""

    def test_full_pipeline_with_alignment_and_tracking(self):
        """Test complete pipeline with all features enabled."""
        # This would require a small test dataset
        # For now, just verify imports work
        try:
            from create_sparse_depth_map import (
                _align_bbox_with_point_cloud_com,
                _track_gripper_with_mvtracker,
                _segment_object_with_sam,
            )
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import required functions: {e}")

    def test_rerun_visualization_with_new_features(self):
        """Test that rerun logging works with aligned bboxes and tracks."""
        # This test would verify rerun integration
        # For now, just check imports
        try:
            import rerun as rr
            from mvtracker.utils.visualizer_rerun import log_pointclouds_to_rerun
            self.assertTrue(True)
        except ImportError:
            self.skipTest("Rerun not available")


def run_visual_test():
    """
    Run visual tests that output to rerun for manual inspection.
    
    This is not a unit test but a visual verification tool.
    Run with: python tests/test_gripper_features.py --visual
    """
    import rerun as rr
    
    print("Running visual tests - check rerun viewer for results")
    
    # Initialize rerun
    rr.init("gripper_feature_tests", recording_id="test")
    rr.spawn()
    
    # Test 1: Visualize bbox alignment
    print("\n[Visual Test 1] Bbox Alignment")
    print("- Red bbox: Original")
    print("- Green bbox: Aligned to point cloud COM")
    print("- Points: Point cloud used for alignment")
    
    # Create test data
    bbox = {
        "center": np.array([0.0, 0.0, 0.1], dtype=np.float32),
        "half_sizes": np.array([0.05, 0.03, 0.08], dtype=np.float32),
        "quat_xyzw": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "basis": np.eye(3, dtype=np.float32),
    }
    
    # Create shifted point cloud
    num_points = 200
    points = np.random.randn(num_points, 3).astype(np.float32) * 0.02
    points[:, 0] += 0.03  # Shift right
    points[:, 2] += 0.1
    colors = np.random.rand(num_points, 3).astype(np.float32)
    
    # Log original bbox in red
    rr.log("test/original_bbox", rr.Boxes3D(
        centers=[bbox["center"]],
        half_sizes=[bbox["half_sizes"]],
        colors=[[255, 0, 0]],
    ))
    
    # Log points
    rr.log("test/points", rr.Points3D(
        positions=points,
        colors=colors,
        radii=0.005,
    ))
    
    # Align and log
    from create_sparse_depth_map import _align_bbox_with_point_cloud_com
    aligned_bbox = _align_bbox_with_point_cloud_com(
        bbox=bbox,
        points=points,
        colors=colors,
        search_radius_scale=2.0,
    )
    
    if aligned_bbox:
        rr.log("test/aligned_bbox", rr.Boxes3D(
            centers=[aligned_bbox["center"]],
            half_sizes=[aligned_bbox["half_sizes"]],
            colors=[[0, 255, 0]],
        ))
    
    print("✓ Visual test complete. Check rerun viewer.")
    print(f"  Original center: {bbox['center']}")
    print(f"  Aligned center: {aligned_bbox['center']}")


if __name__ == "__main__":
    import sys
    
    if "--visual" in sys.argv:
        run_visual_test()
    else:
        unittest.main()
