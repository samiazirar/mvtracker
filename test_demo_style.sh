#!/bin/bash
# Test script for the updated demo.py-style point cloud creation

echo "=== Testing Demo.py-Style Point Cloud Creation ==="
echo ""

# Test with conservative confidence filtering (should reduce color bleeding significantly)
python create_sparse_depth_map.py \
  --task-folder /data/rh20t_api/data/test_data_full_rgb_upscaled_depth/uncompressed_low_res_data/task_0065_user_0010_scene_0009_cfg_0004 \
  --high-res-folder /data/rh20t_api/data/test_data_full_rgb_upscaled_depth/rgb_data/RH20T_cfg4/task_0065_user_0010_scene_0009_cfg_0004 \
  --out-dir ./data/high_res_filtered_demo_style \
  --max-frames 25 \
  --use-splatting \
  --splat-mode recolor \
  --splat-radius 0.002 \
  --confidence-threshold 0.3 \
  --edge-margin 15 \
  --gradient-threshold 0.08 \
  --smooth-kernel 3 \
  --clean-pointcloud

echo ""
echo "=== Key Changes Applied ==="
echo "1. Single-camera selection per frame (like demo.py)"
echo "2. Confidence-based depth filtering"  
echo "3. Edge margin removal"
echo "4. Depth gradient filtering"
echo "5. Confidence-weighted smoothing"
echo "6. Statistical outlier removal"
echo ""
echo "This should significantly reduce color bleeding compared to the multi-camera approach!"