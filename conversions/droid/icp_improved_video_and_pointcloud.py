"""
ICP-based Wrist Camera Z-Offset Optimization and Video Generation.

This script optimizes the Z-offset of the wrist camera relative to the gripper
using point-to-plane ICP alignment against external camera point clouds.

Key features:
1. Assumes gripper pose is correct - only optimizes wrist camera Z offset
2. Excludes gripper points (< 15cm from camera based on the config) during ICP alignment
3. Generates two folders of reprojected videos:
   - videos_no_icp: Before ICP optimization
   - videos_icp: After ICP optimization

Usage:
    python conversions/droid/icp_improved_video_and_pointcloud.py
"""

