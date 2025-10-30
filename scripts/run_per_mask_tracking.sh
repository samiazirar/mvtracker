#!/usr/bin/env bash

# Example script showing how to run per-mask tracking
# This is a template - adapt the paths and parameters to your needs

set -euo pipefail

# Configuration
TASK_FOLDER="task_0034_user_0014_scene_0004_cfg_0006_human"
BASE_DIR="third_party/HOISTFormer/sam2_tracking_output"

# Input NPZ with SAM2 masks (output from your existing pipeline)
INPUT_NPZ="${BASE_DIR}/${TASK_FOLDER}_processed_hand_tracked_hoist_sam2.npz"

# Output directory for per-mask tracking results
OUTPUT_DIR="./tracking_per_mask/${TASK_FOLDER}"

# Tracking parameters
# TRACKER="mvtracker"  # Options: mvtracker, spatialtrackerv2, cotracker3_offline
TRACKER="spatialtrackerv2"  # Options: mvtracker, spatialtrackerv2, cotracker3_offline

MASK_KEY="sam2_masks"  # Key for masks in NPZ

# Optional: Batch processing for large masks (set to empty string to disable)
# MAX_QUERY_POINTS_PER_BATCH=""  # Disabled
MAX_QUERY_POINTS_PER_BATCH="5000"  # Split masks with >5000 query points into batches

echo "=========================================="
echo "Per-Mask Independent Tracking Pipeline"
echo "=========================================="
echo "Input NPZ: ${INPUT_NPZ}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Tracker: ${TRACKER}"
if [ -n "${MAX_QUERY_POINTS_PER_BATCH}" ]; then
  echo "Batch processing: enabled (${MAX_QUERY_POINTS_PER_BATCH} points per batch)"
else
  echo "Batch processing: disabled"
fi
echo ""

# Build command
CMD="python process_masks_independently.py \
  --npz \"${INPUT_NPZ}\" \
  --mask-key \"${MASK_KEY}\" \
  --tracker \"${TRACKER}\" \
  --output-dir \"${OUTPUT_DIR}\" \
  --base-name \"${TASK_FOLDER}\" \
  --frames-before 3 \
  --frames-after 10 \
  --use-first-frame \
  --temporal-stride 1 \
  --spatial-downsample 1 \
  --depth-estimator gt \
  --depth-cache-dir ./depth_cache"

# Add batch limit if set
if [ -n "${MAX_QUERY_POINTS_PER_BATCH}" ]; then
  CMD="${CMD} --max-query-points-per-batch ${MAX_QUERY_POINTS_PER_BATCH}"
fi

# Run the per-mask tracking pipeline
eval $CMD

echo ""
echo "=========================================="
echo "Done! Results saved to: ${OUTPUT_DIR}"
echo "=========================================="
echo ""
echo "To view all tracking results together:"
echo "  bash ${OUTPUT_DIR}/view_${TASK_FOLDER}_${TRACKER}_combined.sh"
echo ""
echo "To view individual masks:"
echo "  rerun ${OUTPUT_DIR}/${TASK_FOLDER}_${TRACKER}_instance_0.rrd"
echo "  rerun ${OUTPUT_DIR}/${TASK_FOLDER}_${TRACKER}_instance_1.rrd"
echo ""

# Optional: Run with a different tracker
# echo "Running with spatialtrackerv2..."
# python process_masks_independently.py \
#   --npz "${INPUT_NPZ}" \
#   --mask-key "${MASK_KEY}" \
#   --tracker spatialtrackerv2 \
#   --output-dir "${OUTPUT_DIR}_spatracker" \
#   --base-name "${TASK_FOLDER}" \
#   --frames-before 3 \
#   --frames-after 10 \
#   --use-first-frame

# Optional: Compare both trackers
# echo ""
# echo "To compare both trackers:"
# echo "  rerun ${OUTPUT_DIR}/*.rrd ${OUTPUT_DIR}_spatracker/*.rrd"
