#!/usr/bin/env bash

# Example command to process an RH20T sequence with ghost gripper overlays.
# Update the paths below to match your local dataset layout before running.

set -euo pipefail

python create_sparse_depth_map.py \
  --task-folder /data/rh20t_api/data/test_data_full_rgb_upscaled_depth/uncompressed_low_res_data/task_0065_user_0010_scene_0009_cfg_0004 \
  --high-res-folder /data/rh20t_api/data/test_data_full_rgb_upscaled_depth/rgb_data/RH20T_cfg4/task_0065_user_0010_scene_0009_cfg_0004 \
  --out-dir ./data/high_res_filtered \
  --max-frames 20 \
  --no-color-alignment-check \
  --no-sharpen-edges-with-mesh \
  --add-robot \
  --include-gripper-visuals \
  --gripper-bbox \
  --export-bbox-video \
  "${@}"

cp ./data/high_res_filtered/task_0065_user_0010_scene_0009_cfg_0004_reprojected.rrd /data/rh20t_api
