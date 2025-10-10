#!/usr/bin/env bash

# Example command to process an RH20T sequence with ghost gripper overlays.
# Update the paths below to match your local dataset layout before running.

set -euo pipefail

python create_sparse_depth_map.py \
  --task-folder /data/rh20t_api/data/test_data_full_rgb_upscaled_depth/uncompressed_low_res_data/task_0065_user_0010_scene_0009_cfg_0004 \
  --high-res-folder /data/rh20t_api/data/test_data_full_rgb_upscaled_depth/rgb_data/RH20T_cfg4/task_0065_user_0010_scene_0009_cfg_0004 \
  --out-dir ./data/high_res_filtered \
  --max-frames 20 \
  --no-sharpen-edges-with-mesh \
  --add-robot \
  --gripper-bbox \
  --gripper-body-bbox \
  --gripper-fingertip-bbox \
  --gripper-pad-points \
  --export-bbox-video \
  --tcp-points \
  --object-points \
  --gripper-body-width-m 0.05 \
  --align-bbox-with-points \
  --align-bbox-search-radius-scale 2.0 \
  --track-gripper-with-mvtracker \
  --track-objects-with-sam \
  --sam-model-type vit_b \
  --sam-contact-threshold 50.0 \
  "${@}"

cp ./data/high_res_filtered/task_0065_user_0010_scene_0009_cfg_0004_reprojected.rrd /data/rh20t_api

cp -r  ./data /data/rh20t_api

#--no-color-alignment-check \