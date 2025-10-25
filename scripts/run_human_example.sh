#!/usr/bin/env bash

# Example command to process an RH20T human demonstration sequence.
# Update the paths below to match your local dataset layout before running.

set -euo pipefail

# Human example from configuration 3 (default)
#maybe calib files are wrong TODO:
# Lets use cfg 5...
TASK_FOLDER="task_0045_user_0020_scene_0004_cfg_0006_human"
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

# TODO: USE OLD Point CLoud! THEN FIX SAM MASK UPLIFING
# TODO: raise error if calib files not found
python create_sparse_depth_map.py \
  --task-folder "$DEPTH_FOLDER" \
  --high-res-folder "$RGB_FOLDER" \
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


echo "Copying data to /data/rh20t_api"
cp "$OUT_DIR/${TASK_FOLDER}_reprojected.rrd" /data/rh20t_api
cp -r ./data /data/rh20t_api/test_data_generated_human


SAMPLE_PATH="$OUT_DIR/${TASK_FOLDER}_processed.npz"
SAMPLE_PATH_HAND_TRACKED="$OUT_DIR/${TASK_FOLDER}_processed_hand_tracked.npz"
echo "Running MVTracker demo"
# python demo.py --temporal_stride 1 --spatial_downsample 1 --depth_estimator gt --depth_cache_dir ./depth_cache --rerun save --sample-path "$SAMPLE_PATH" --tracker spatialtrackerv2

# python demo.py --temporal_stride 1 --spatial_downsample 1 --depth_estimator gt --depth_cache_dir ./depth_cache --rerun save --sample-path "$SAMPLE_PATH"
# python demo.py --temporal_stride 1 --spatial_downsample 1 --depth_estimator gt --depth_cache_dir ./depth_cache --rerun save --sample-path "$SAMPLE_PATH" --tracker spatialtrackerv2
# python demo.py --temporal_stride 1 --spatial_downsample 1 --depth_estimator gt --depth_cache_dir ./depth_cache --rerun save --sample-path "$SAMPLE_PATH" --tracker cotracker3_offline

# TODO -> 
# python demo.py --temporal_stride 1 --spatial_downsample 1 --depth_estimator gt --depth_cache_dir ./depth_cache --rerun save --sample-path "$SAMPLE_PATH_HAND_TRACKED" --tracker spatialtrackerv2
python demo.py --temporal_stride 1 --spatial_downsample 1 --depth_estimator gt --depth_cache_dir ./depth_cache --rerun save --sample-path "$SAMPLE_PATH_HAND_TRACKED" --tracker cotracker3_offline
# python demo.py --temporal_stride 1 --spatial_downsample 1 --depth_estimator gt --depth_cache_dir ./depth_cache --rerun save --sample-path "$SAMPLE_PATH_HAND_TRACKED" --tracker mvtracker

#add also for with depth added 

# python demo.py --temporal_stride 1 --spatial_downsample 1 --depth_estimator vggt_raw --depth_cache_dir ./depth_cache --rerun save --sample-path "$SAMPLE_PATH" --tracker mvtracker --rrd vggt_raw_mvtracker_demo.rrd

#TODO: Use only vggt to get the extrinsics 
# python demo.py --temporal_stride 1 --spatial_downsample 1 --depth_estimator vggt_raw --depth_cache_dir ./depth_cache --rerun save --sample-path "$SAMPLE_PATH" --tracker mvtracker

echo "Copying MVTracker demo results to /data/rh20t_api/test_data_generated_human"
cp -r ./mvtracker_demo.rrd /data/rh20t_api/test_data_generated_human
# cp -r ./vggt_raw_mvtracker_demo.rrd /data/rh20t_api/test_data_generated_human

# Human dataset references
# sample_path = "/data/rh20t_api/data/RH20T/packed_npz/task_0092_user_0010_scene_0004_cfg_0003_human.npz"




