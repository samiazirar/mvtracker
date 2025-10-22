#!/usr/bin/env bash

# Example command to process an RH20T sequence with ghost gripper overlays.
# Update the paths below to match your local dataset layout before running.

set -euo pipefail


# TASK_FOLDER="task_0065_user_0010_scene_0009_cfg_0004" -> original which i test on 
  # --task-folder /data/rh20t_api/data/test_data_full_rgb_upscaled_depth/uncompressed_low_res_data/$TASK_FOLDER \
  # --high-res-folder /data/rh20t_api/data/test_data_full_rgb_upscaled_depth/rgb_data/RH20T_cfg4/$TASK_FOLDER \


#config 4
TASK_FOLDER="task_0065_user_0010_scene_0009_cfg_0004" 
DEPTH_FOLDER="/data/rh20t_api/data/test_data_full_rgb_upscaled_depth/uncompressed_low_res_data/$TASK_FOLDER"
RGB_FOLDER="/data/rh20t_api/data/test_data_full_rgb_upscaled_depth/rgb_data/RH20T_cfg4/$TASK_FOLDER"

#cfg 3 for both
# TASK_FOLDER="task_0024_user_0010_scene_0005_cfg_0003"
# DEPTH_FOLDER="/data/rh20t_api/data/low_res_data/RH20T_cfg3/$TASK_FOLDER"
# RGB_FOLDER="/data/rh20t_api/data/RH20T/RH20T_cfg3/$TASK_FOLDER"
#check if using scene high with tcp now means it is swapped for all?

#run low res both so dense low res
python create_sparse_depth_map.py \
  --task-folder $DEPTH_FOLDER \
  --high-res-folder $RGB_FOLDER \
  --out-dir ./data/high_res_filtered \
  --max-frames 60 \
  --frames-for-tracking 1 \
  --no-sharpen-edges-with-mesh \
  --add-robot \
  --gripper-bbox \
  --gripper-body-bbox \
  --gripper-fingertip-bbox \
  --gripper-pad-points \
  --export-bbox-video \
  --object-points \
  --gripper-body-length-m 0.15 \
  --gripper-body-height-m 0.15 \
  --gripper-body-width-m 0.15 \
  --visualize-query-points \
  --max-query-points 512 \
  --no-color-alignment-check \
  --refine-colmap \
  --limit-num-cameras 4 \
  "${@}"

# if not happy camera sselection may longer overlap
#check colmap refinement
#add colmap densificaiton TODO
# python create_sparse_depth_map.py \
#   --task-folder $DEPTH_FOLDER \
#   --high-res-folder $RGB_FOLDER \
#   --out-dir ./data/high_res_filtered \
#   --max-frames 240 \
#   --frames-for-tracking 1 \
#   --no-sharpen-edges-with-mesh \
#   --add-robot \
#   --gripper-bbox \
#   --gripper-body-bbox \
#   --gripper-fingertip-bbox \
#   --gripper-pad-points \
#   --export-bbox-video \
#   --object-points \
#   --gripper-body-length-m 0.15 \
#   --gripper-body-height-m 0.15 \
#   --gripper-body-width-m 0.15 \
#   --visualize-query-points \
#   --max-query-points 128 \
#   --no-color-alignment-check \
#   --sam2-tracking \
#   --sam2-checkpoint third_party/sam2/sam2/checkpoints/sam2.1_hiera_large.pt \
#   --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
#   --visualize-sam2-masks \
#   "${@}"

# --sam2-mask-as-mesh \

#  --use-tcp \
# SpatialTrackerV2 works good -> KNN to nearest depth pixel avaiable
echo "Copying data to /data/rh20t_api"
cp ./data/high_res_filtered/${TASK_FOLDER}_reprojected.rrd /data/rh20t_api
cp -r  ./data /data/rh20t_api/test_data_generated

SAMPLE_PATH="data/high_res_filtered/${TASK_FOLDER}_processed.npz"
echo "Running MVTracker demo"
# python demo.py  --temporal_stride 1 --spatial_downsample 1 --depth_estimator gt --depth_cache_dir ./depth_cache --rerun save  --sample-path $SAMPLE_PATH --tracker spatialtrackerv2 

# python demo.py  --temporal_stride 1 --spatial_downsample 1 --depth_estimator gt --depth_cache_dir ./depth_cache --rerun save  --sample-path $SAMPLE_PATH --tracker cotracker3_offline 

python demo.py  --temporal_stride 1 --spatial_downsample 1 --depth_estimator gt --depth_cache_dir ./depth_cache --rerun save  --sample-path $SAMPLE_PATH  


echo "Copying MVTracker demo results to /data/rh20t_api/test_data_generated"
cp -r ./mvtracker_demo.rrd /data/rh20t_api/test_data_generated

  
#  --use-tcp \


#TODO: add a function to liimit the number of query poiints
# remove the max query points
# TODO: Make thre query points around the gripper ..
#--no-color-alignment-check \
  # --align-bbox-with-points \
#  --align-bbox-search-radius-scale 2.0 \
  # check number and if it uses query points each time again
#--exclude-by-cluster -> dbscan anr remove what inside
#--exclude-inside-gripper -> remove what is inside gripper
# TODO: remove all the unnessary functions such as temporal stride etc.
#--tracker cotracker3_offline 
#check if the fps is relevant

# Load the RH20T dataset with memory management
# sample_path = "/data/rh20t_api/data/RH20T/packed_npz/task_0015_user_0011_scene_0006_cfg_0003.npz"

# upscaled:
# sample_path = "/data/rh20t_api/data/test_data_full_rgb_upscaled_depth/packed_npz/task_0065_user_0010_scene_0009_cfg_0004.npz"
# not upscaled:
# sample_path = "/data/rh20t_api/data/test_data_full_rgb_upscaled_depth/uncompressed_low_res_data/packed_npz/task_0065_user_0010_scene_0009_cfg_0004.npz"
# sample_path = "/data/rh20t_api/data/low_res_data/packed_npz/task_0001_user_0010_scene_0005_cfg_0004.npz"
# with mapanythign
# sample_path = "/data/npz_file/task_0065_user_0010_scene_0009_cfg_0004_pred.npz"
# Final mapanything
# sample_path = "/data/rh20t_api/mapanything_test/task_0065_user_0010_scene_0009_cfg_0004_processed.npz"

# reprojected without depth
# sample_path = "data/high_res_filtered/task_0065_user_0010_scene_0009_cfg_0004_processed.npz"
# input to mapanythibg
# sample_path = "/data/rh20t_api/test_npz/task_0065_user_0010_scene_0009_cfg_0004_processed.npz"
# sample_path = "data/high_res_filtered/task_0065_user_0010_scene_0009_cfg_0004_processed.npz"


# Install SAM2: cd sam2 && pip install -e .
# Download checkpoint: wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -P sam2/checkpoints/
# Run with --sam2-tracking flag

# mkdir -p /tmp/sam2_runner
# cd /tmp/sam2_runner
# PYTHONPATH="/workspace:/workspace/sam2/sam2" python -m runpy /workspace/create_sparse_depth_map.py …
