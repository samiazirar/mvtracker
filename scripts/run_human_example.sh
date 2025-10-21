#!/usr/bin/env bash

# Example command to process an RH20T human demonstration sequence.
# Update the paths below to match your local dataset layout before running.

set -euo pipefail

# Human example from configuration 3 (default)
TASK_FOLDER="task_0092_user_0010_scene_0004_cfg_0003_human"
DEPTH_FOLDER="/data/rh20t_api/data/low_res_data/RH20T_cfg3/$TASK_FOLDER"
RGB_FOLDER="/data/rh20t_api/data/RH20T/RH20T_cfg3/$TASK_FOLDER"

# Uncomment to try configuration 4 style data
# TASK_FOLDER="task_0065_user_0010_scene_0009_cfg_0004_human"
# DEPTH_FOLDER="/data/rh20t_api/data/test_data_full_rgb_upscaled_depth/uncompressed_low_res_data/$TASK_FOLDER"
# RGB_FOLDER="/data/rh20t_api/data/test_data_full_rgb_upscaled_depth/rgb_data/RH20T_cfg4/$TASK_FOLDER"

OUT_DIR="./data/human_high_res_filtered"
mkdir -p "${OUT_DIR}"

# Human recordings reuse the calibration bundle shipped with the RGB archive.
DEPTH_ROOT="$(dirname "$DEPTH_FOLDER")"
RGB_ROOT_PARENT="$(dirname "$RGB_FOLDER")"
CALIB_DST="${DEPTH_ROOT}/calib"
CALIB_SRC="${RGB_ROOT_PARENT}/calib"
if [[ ! -e "${CALIB_DST}" ]]; then
  if [[ -d "${CALIB_SRC}" ]]; then
    echo "Linking calibration data: ${CALIB_SRC} -> ${CALIB_DST}"
    ln -s "${CALIB_SRC}" "${CALIB_DST}"
  else
    echo "Warning: calibration source missing at ${CALIB_SRC}"
  fi
fi

python create_sparse_depth_map.py \
  --task-folder "$DEPTH_FOLDER" \
  --high-res-folder "$RGB_FOLDER" \
  --out-dir "$OUT_DIR" \
  --dataset-type human \
  --max-frames 60 \
  --frames-for-tracking 1 \
  --no-sharpen-edges-with-mesh \
  --no-color-alignment-check \
  "$@"

echo "Copying data to /data/rh20t_api"
cp "$OUT_DIR/${TASK_FOLDER}_reprojected.rrd" /data/rh20t_api
cp -r ./data /data/rh20t_api/test_data_generated_human

SAMPLE_PATH="$OUT_DIR/${TASK_FOLDER}_processed.npz"
echo "Running MVTracker demo"
python demo.py --temporal_stride 1 --spatial_downsample 1 --depth_estimator gt --depth_cache_dir ./depth_cache --rerun save --sample-path "$SAMPLE_PATH"
# python demo.py --temporal_stride 1 --spatial_downsample 1 --depth_estimator gt --depth_cache_dir ./depth_cache --rerun save --sample-path "$SAMPLE_PATH" --tracker spatialtrackerv2
# python demo.py --temporal_stride 1 --spatial_downsample 1 --depth_estimator gt --depth_cache_dir ./depth_cache --rerun save --sample-path "$SAMPLE_PATH" --tracker cotracker3_offline

echo "Copying MVTracker demo results to /data/rh20t_api"
cp -r ./mvtracker_demo.rrd /data/rh20t_api/test_data_generated_human

# Human dataset references
# sample_path = "/data/rh20t_api/data/RH20T/packed_npz/task_0092_user_0010_scene_0004_cfg_0003_human.npz"
