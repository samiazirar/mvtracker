#!/usr/bin/env bash

# Example command to process an RH20T sequence with ghost gripper overlays.
# Update the paths below to match your local dataset layout before running.

set -euo pipefail

# python create_sparse_depth_map.py \
#   --task-folder /data/rh20t_api/data/test_data_full_rgb_upscaled_depth/uncompressed_low_res_data/task_0065_user_0010_scene_0009_cfg_0004 \
#   --high-res-folder /data/rh20t_api/data/test_data_full_rgb_upscaled_depth/rgb_data/RH20T_cfg4/task_0065_user_0010_scene_0009_cfg_0004 \
#   --out-dir ./data/high_res_filtered \
#   --max-frames 100 \
#   --frames-for-tracking 1 \
#   --no-sharpen-edges-with-mesh \
#   --add-robot \
#   --gripper-bbox \
#   --gripper-body-bbox \
#   --gripper-fingertip-bbox \
#   --gripper-pad-points \
#   --export-bbox-video \
#   --tcp-points \
#   --object-points \
#   --visualize-query-points \
#   --max-query-points 512 \
#   "${@}"


echo "Copying data to /data/rh20t_api"
cp ./data/high_res_filtered/task_0065_user_0010_scene_0009_cfg_0004_reprojected.rrd /data/rh20t_api
cp -r  ./data /data/rh20t_api/test_data_generated

echo "Running MVTracker demo"
python demo.py  --temporal_stride 1 --spatial_downsample 1 --depth_estimator gt --depth_cache_dir ./depth_cache --rerun save   

echo "Copying MVTracker demo results to /data/rh20t_api"
cp -r ./mvtracker_demo.rrd /data/rh20t_api/test_data_generated

#TODO: add a function to liimit the number of query poiints
# remove the max query points
# TODO: Make thre query points around the gripper ..
#--no-color-alignment-check \
  # --align-bbox-with-points \
#  --align-bbox-search-radius-scale 2.0 \
  # --gripper-body-length-m 0.15 \
  # --gripper-body-height-m 0.15 \
  # --gripper-body-width-m 0.15 \
  # check number and if it uses query points each time again
#--exclude-by-cluster -> dbscan anr remove what inside
#--exclude-inside-gripper -> remove what is inside gripper
# TODO: remove all the unnessary functions such as temporal stride etc.
#--tracker cotracker3_offline 