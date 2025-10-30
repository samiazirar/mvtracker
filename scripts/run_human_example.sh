#!/usr/bin/env bash

# Example command to process an RH20T human demonstration sequence.
# Update the paths below to match your local dataset layout before running.

set -euo pipefail

# Human example from configuration 3 (default)
#maybe calib files are wrong TODO:
# Lets use cfg 5...
TASK_FOLDER="task_0034_user_0014_scene_0004_cfg_0006_human"
# TASK_FOLDER="task_0092_user_0010_scene_0004_cfg_0003_human"
DEPTH_FOLDER="/data/rh20t_api/data/low_res_data/RH20T_cfg6/$TASK_FOLDER"
# RGB_FOLDER="/data/rh20t_api/data/RH20T/RH20T_cfg6/$TASK_FOLDER" # Does not exist rn
RGB_FOLDER="/data/rh20t_api/hf_download/RH20T/RH20T_cfg6/$TASK_FOLDER"

# Why does it not crash with the high res RGB data ? Calib not there
# Uncomment to try configuration 4 style data
# TASK_FOLDER="task_0065_user_0010_scene_0009_cfg_0004_human"
# DEPTH_FOLDER="/data/rh20t_api/data/test_data_full_rgb_upscaled_depth/uncompressed_low_res_data/$TASK_FOLDER"
# RGB_FOLDER="/data/rh20t_api/data/test_data_full_rgb_upscaled_depth/rgb_data/RH20T_cfg4/$TASK_FOLDER"

#seems for humans both depth data is bad

OUT_DIR="./data/human_high_res_filtered"
mkdir -p "${OUT_DIR}"
#spatracker on all then mask with sam
# Human recordings reuse the calibration bundle shipped with the RGB archive.
DEPTH_ROOT="$(dirname "$DEPTH_FOLDER")"
RGB_ROOT_PARENT="$(dirname "$RGB_FOLDER")"
#TODO: intrinsics?

echo "running both in low res for now"
# TODO: USE OLD Point CLoud! THEN FIX SAM MASK UPLIFING
# TODO: raise error if calib files not found
python create_sparse_depth_map.py \
  --task-folder "$DEPTH_FOLDER" \
  --high-res-folder "$DEPTH_FOLDER" \
  --out-dir "$OUT_DIR" \
  --dataset-type human \
  --max-frames 50 \
  --frame-selection first \
  --frames-for-tracking 1 \
  --no-sharpen-edges-with-mesh \
  --pc-clean-radius 0.05 \
  --pc-clean-min-points 5 \
  "$@"


#  --no-color-alignment-check \
  # --refine-colmap \
  # --limit-num-cameras 4 \


# echo "Adding the Sam Masks"
cd third_party/hamer
source .hamer/bin/activate
python add_hand_mask_from_sam_to_rh20t.py \
  --npz "../.$OUT_DIR/${TASK_FOLDER}_processed.npz" \
  --out-dir "../.$OUT_DIR"

pip install rerun-sdk==0.21.0
deactivate
cd ../..
# #use video sam?
# echo "Sam Masks added."


# echo "Copying data to /data/rh20t_api"
# cp "$OUT_DIR/${TASK_FOLDER}_reprojected.rrd" /data/rh20t_api
# cp -r ./data /data/rh20t_api/test_data_generated_human

SAMPLE_PATH="$OUT_DIR/${TASK_FOLDER}_processed.npz"
SAMPLE_PATH_HAND_TRACKED="$OUT_DIR/${TASK_FOLDER}_processed_hand_tracked.npz"
echo "Running MVTracker demo"
# python demo.py --temporal_stride 1 --spatial_downsample 1 --depth_estimator gt --depth_cache_dir ./depth_cache --rerun save --sample-path "$SAMPLE_PATH" --tracker spatialtrackerv2

# python demo.py --temporal_stride 1 --spatial_downsample 1 --depth_estimator gt --depth_cache_dir ./depth_cache --rerun save --sample-path "$SAMPLE_PATH"
# python demo.py --temporal_stride 1 --spatial_downsample 1 --depth_estimator gt --depth_cache_dir ./depth_cache --rerun save --sample-path "$SAMPLE_PATH" --tracker spatialtrackerv2
# python demo.py --temporal_stride 1 --spatial_downsample 1 --depth_estimator gt --depth_cache_dir ./depth_cache --rerun save --sample-path "$SAMPLE_PATH" --tracker cotracker3_offline


# erst mal ohne SAM
# TODO -> 
python demo.py --temporal_stride 1 --spatial_downsample 1 --depth_estimator gt --depth_cache_dir ./depth_cache --rerun save --sample-path "$SAMPLE_PATH_HAND_TRACKED" --tracker spatialtrackerv2 --rrd "./mvtracker_demo_hands_spatracker_${TASK_FOLDER}.rrd"

python demo.py --temporal_stride 1 --spatial_downsample 1 --depth_estimator gt --depth_cache_dir ./depth_cache --rerun save --sample-path "$SAMPLE_PATH_HAND_TRACKED" --tracker mvtracker --rrd "./mvtracker_demo_hands_mvtracker_${TASK_FOLDER}.rrd"

# python demo.py --temporal_stride 1 --spatial_downsample 1 --depth_estimator gt --depth_cache_dir ./depth_cache --rerun save --sample-path "$SAMPLE_PATH_HAND_TRACKED" --tracker cotracker3_offline
# python demo.py --temporal_stride 1 --spatial_downsample 1 --depth_estimator gt --depth_cache_dir ./depth_cache --rerun save --sample-path "$SAMPLE_PATH_HAND_TRACKED" --tracker mvtracker


echo "generating masks for objects manipulated by human hands - using HoistFormer"

cd third_party/HOISTFormer
# ./run_demo_hand_object.sh
# ./run_sam2_tracking.sh # for now we do not need to track the object for the whole time lets use the mask we already have
#but we need to know whan it starts..

#!/bin/bash
# Script to run combined hand + object visualization on hand_tracked NPZ files

source .hoist/bin/activate


# Example NPZ file path - replace with your actual NPZ file
NPZ_FILE="../../data/human_high_res_filtered/${TASK_FOLDER}_processed_hand_tracked.npz"

# # Run combined hand + object visualization
# python demo_hand_object_combined.py \
#     --npz "$NPZ_FILE" \
#     --out-dir ./hand_object_output \
#     --cfg ./configs/hoist/hoistformer.yaml \
#     --weights ./pretrained_models/trained_model.pth

# echo "Done! Check ./hand_object_output/output_results_video/ for videos"


# Run HOISTFormer on the NPZ file


echo "Running HOISTFormer on the NPZ file to produce object masks"
python demo_npz.py \
    --npz "$NPZ_FILE" \
    --out-dir ./hoist_output \
    --cfg ./configs/hoist/hoistformer.yaml \
    --weights ./pretrained_models/trained_model.pth
#output npz = ... ${TASK_FOLDER}_processed_hand_tracked_hoist.npz
echo "Done! Check ./hoist_output/output_results_video/ for videos"

# Example NPZ file path - should be output from demo_hand_object_combined.py
# NPZ_FILE="./hand_object_output/task_0045_user_0020_scene_0004_cfg_0006_human_processed_hand_object.npz"
NPZ_FILE="./hoist_output/${TASK_FOLDER}_processed_hand_tracked_hoist.npz"
# Run SAM2 object tracking - need backward pass
python demo_sam2_object_tracking_debug.py \
    --npz "$NPZ_FILE" \
    --out-dir ./sam2_tracking_output \
    --sam-config /workspace/third_party/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml \
    --sam-checkpoint ../sam2/sam2/checkpoints/sam2.1_hiera_large.pt \
    --device cuda

echo "Done! Check ./sam2_tracking_output/output_results_video/ for videos with all masks"

# output : NPZ_FILE="./sam2_tracking_output/${TASK_FOLDER}_processed_hand_tracked_hoist_sam2_tracked.npz"



deactivate

cd ../..
echo "Done with HoistFormer processing."


echo "Creating query points from masks and running MVTracker demo with GT depth estimator"
# python create_query_points_from_masks.py --npz third_party/HOISTFormer/ho
SAMPLE_PATH_SAM_HAND_TRACKED="third_party/HOISTFormer/sam2_tracking_output/${TASK_FOLDER}_processed_hand_tracked_hoist_sam2_query.npz"
SAMPLE_PATH_HAND_TRACKED="third_party/HOISTFormer/sam2_tracking_output/${TASK_FOLDER}_processed_hand_tracked_hoist_sam2.npz"

python create_query_points_from_masks.py --npz $SAMPLE_PATH_HAND_TRACKED --frames-before 3 --frames-after 5 --output $SAMPLE_PATH_HAND_TRACKED  --key sam2_masks --use-first-frame


echo "Running MVTracker demo with GT depth estimator"


python demo.py --temporal_stride 1 --spatial_downsample 1 --depth_estimator gt --depth_cache_dir ./depth_cache --rerun save --sample-path "$SAMPLE_PATH_HAND_TRACKED" --tracker spatialtrackerv2 --rrd mvtracker_demo_hands_mvtracker_${TASK_FOLDER}_hoist_sam2.rrd

python demo.py --temporal_stride 1 --spatial_downsample 1 --depth_estimator gt --depth_cache_dir ./depth_cache --rerun save --sample-path "$SAMPLE_PATH_HAND_TRACKED" --tracker mvtracker --rrd mvtracker_demo_hands_mvtracker_${TASK_FOLDER}_hoist_sam2.rrd


# python demo.py --temporal_stride 1 --spatial_downsample 1 --depth_estimator vggt_raw --depth_cache_dir ./depth_cache --rerun save --sample-path "$SAMPLE_PATH" --tracker mvtracker --rrd vggt_raw_mvtracker_demo.rrd

#TODO: Use only vggt to get the extrinsics 
# python demo.py --temporal_stride 1 --spatial_downsample 1 --depth_estimator vggt_raw --depth_cache_dir ./depth_cache --rerun save --sample-path "$SAMPLE_PATH" --tracker mvtracker

# echo "Copying MVTracker demo results to /data/rh20t_api/test_data_generated_human"
# cp -r ./mvtracker_demo.rrd /data/rh20t_api/test_data_generated_human
# cp -r ./vggt_raw_mvtracker_demo.rrd /data/rh20t_api/test_data_generated_human

# Human dataset references
# sample_path = "/data/rh20t_api/data/RH20T/packed_npz/task_0092_user_0010_scene_0004_cfg_0003_human.npz"



# echo "Running MVTracker demo with VGGt depth estimator"
# python demo.py --temporal_stride 1 --spatial_downsample 1 --depth_estimator vggt_raw --depth_cache_dir ./depth_cache --rerun save --sample-path "$SAMPLE_PATH_HAND_TRACKED" --tracker mvtracker --rrd "./vggt_raw_mvtracker_raw.rrd"
# cp -r ./vggt_raw_mvtracker_raw.rrd /data/rh20t_api/test_data_generated_human


# TODO: track per mask ->
# Max points