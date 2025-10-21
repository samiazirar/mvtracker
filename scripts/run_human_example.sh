#!/usr/bin/env bash

# Example command to process an RH20T human demonstration sequence.
# Update the paths below to match your local dataset layout before running.

set -euo pipefail

# Default sample (human) task identifiers can be overridden by exporting TASK_FOLDER.
: "${TASK_FOLDER:=task_0092_user_0010_scene_0004_cfg_0003_human}"

# Low-resolution depth source (typically RH20T_cfg3 for human data)
: "${DEPTH_ROOT:=/data/rh20t_api/data/low_res_data/RH20T_cfg3}"

# High-resolution RGB source
: "${RGB_ROOT:=/data/rh20t_api/data/RH20T/RH20T_cfg3}"

DEPTH_FOLDER="${DEPTH_ROOT}/${TASK_FOLDER}"
RGB_FOLDER="${RGB_ROOT}/${TASK_FOLDER}"

if [[ ! -d "${DEPTH_FOLDER}" ]]; then
  echo "Depth folder not found: ${DEPTH_FOLDER}" >&2
  exit 1
fi

if [[ ! -d "${RGB_FOLDER}" ]]; then
  echo "RGB folder not found: ${RGB_FOLDER}" >&2
  exit 1
fi

OUT_DIR="./data/human_processed"
mkdir -p "${OUT_DIR}"

echo "Processing human task: ${TASK_FOLDER}"
python create_sparse_depth_map.py \
  --task-folder "${DEPTH_FOLDER}" \
  --high-res-folder "${RGB_FOLDER}" \
  --out-dir "${OUT_DIR}" \
  --max-frames 60 \
  --frames-for-tracking 1 \
  --no-sharpen-edges-with-mesh \
  "$@"

RRD_PATH="${OUT_DIR}/${TASK_FOLDER}_reprojected.rrd"
NPZ_PATH="${OUT_DIR}/${TASK_FOLDER}_processed.npz"

STAGING_ROOT="/data/rh20t_api/test_data_generated_human"
mkdir -p "${STAGING_ROOT}"

if [[ -f "${RRD_PATH}" ]]; then
  echo "Copying RRD to ${STAGING_ROOT}"
  cp "${RRD_PATH}" "${STAGING_ROOT}/"
else
  echo "RRD file not found (skipping copy): ${RRD_PATH}"
fi

if [[ -f "${NPZ_PATH}" ]]; then
  echo "Sample NPZ available at ${NPZ_PATH}"
else
  echo "NPZ file not found: ${NPZ_PATH}" >&2
fi

echo "Human processing pipeline finished."
