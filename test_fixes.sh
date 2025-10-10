#!/usr/bin/env bash
# Quick test script for the fixes

set -e  # Exit on error

echo "========================================="
echo "Testing Timing Synchronization Fix"
echo "========================================="

# Test with robot and bboxes
python create_sparse_depth_map.py \
  --task-folder /data/rh20t_api/data/test_data_full_rgb_upscaled_depth/uncompressed_low_res_data/task_0065_user_0010_scene_0009_cfg_0004 \
  --high-res-folder /data/rh20t_api/data/test_data_full_rgb_upscaled_depth/rgb_data/RH20T_cfg4/task_0065_user_0010_scene_0009_cfg_0004 \
  --out-dir ./data/test_timing_fix \
  --max-frames 25 \
  --add-robot \
  --gripper-bbox \
  --gripper-body-bbox \
  --gripper-fingertip-bbox \
  --track-gripper \
  --tracker mvtracker \
  --export-track-video \
  --no-sharpen-edges-with-mesh

echo ""
echo "✅ Timing sync test complete!"
echo "Check: ./data/test_timing_fix/task_0065_user_0010_scene_0009_cfg_0004_reprojected.rrd"
echo ""
echo "========================================="
echo "Testing CoTracker 2D-Only Mode"
echo "========================================="

# Test CoTracker 2D-only
python create_sparse_depth_map.py \
  --task-folder /data/rh20t_api/data/test_data_full_rgb_upscaled_depth/uncompressed_low_res_data/task_0065_user_0010_scene_0009_cfg_0004 \
  --high-res-folder /data/rh20t_api/data/test_data_full_rgb_upscaled_depth/rgb_data/RH20T_cfg4/task_0065_user_0010_scene_0009_cfg_0004 \
  --out-dir ./data/test_cotracker2d \
  --max-frames 25 \
  --tracker cotracker2d_only \
  --cotracker-grid-size 20 \
  --export-track-video \
  --no-add-robot \
  --no-gripper-bbox \
  --no-sharpen-edges-with-mesh

echo ""
echo "✅ CoTracker 2D test complete!"
echo "Check videos: ./data/test_cotracker2d/track_videos/cotracker2d_only/*.mp4"
echo ""
echo "========================================="
echo "All Tests Complete! 🎉"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Open .rrd files in Rerun viewer to verify timing sync"
echo "2. Play .mp4 videos to see 2D tracking"
echo "3. Compare with previous outputs to confirm fixes"
