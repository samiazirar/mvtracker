#!/usr/bin/env bash

# Complete track refinement and video export pipeline
# 
# This script runs AFTER scripts/run_per_mask_tracking.sh
# 
# WORKFLOW:
# 1. Run: bash scripts/run_per_mask_tracking.sh
#    → Creates RRD visualization files AND NPZ tracking results
# 2. Run: THIS SCRIPT (bash scripts/run_track_refinement_pipeline.sh)
#    → Refines tracks (filters static, aligns to points)
#    → Exports to per-camera videos
# 
# INPUT: NPZ files with tracking results from per-mask tracking
# OUTPUT: Refined NPZ files + per-camera visualization videos

set -euo pipefail

# Configuration
TASK_FOLDER="task_0035_user_0020_scene_0006_cfg_0006_human"
TRACKER="spatialtrackerv2"  # Must match the tracker used in run_per_mask_tracking.sh

# Input: Tracking NPZ files from process_masks_independently.py
# These are created in the output directory with "_tracked.npz" suffix
INPUT_DIR="./tracking_per_mask/${TASK_FOLDER}"

# Output directories
REFINED_DIR="./refined_tracks/${TASK_FOLDER}"
VIDEOS_DIR="./track_videos/${TASK_FOLDER}"

# Refinement parameters
MOTION_THRESHOLD="0.01"  # Minimum motion in world units (meters)
MOTION_METHOD="total_displacement"  # Method to compute motion

# Video parameters
FPS="30.0"
TRAIL_LENGTH="10"  # Number of frames to show in trail

echo "=========================================="
echo "Track Refinement and Video Export Pipeline"
echo "=========================================="
echo "Task: ${TASK_FOLDER}"
echo "Tracker: ${TRACKER}"
echo ""
echo "Input directory: ${INPUT_DIR}"
echo "Refined output: ${REFINED_DIR}"
echo "Videos output: ${VIDEOS_DIR}"
echo ""

# Check if input directory exists
if [ ! -d "${INPUT_DIR}" ]; then
  echo "ERROR: Input directory not found: ${INPUT_DIR}"
  echo ""
  echo "Please run the per-mask tracking pipeline first:"
  echo "  bash scripts/run_per_mask_tracking.sh"
  echo ""
  exit 1
fi

# Find all tracking result NPZ files (with "_tracked.npz" suffix)
INSTANCE_FILES=($(find "${INPUT_DIR}" -name "*_${TRACKER}_*_tracked.npz" | sort))

if [ ${#INSTANCE_FILES[@]} -eq 0 ]; then
  echo "ERROR: No tracking NPZ files found in ${INPUT_DIR}"
  echo ""
  echo "Expected files matching pattern: *_${TRACKER}_*_tracked.npz"
  echo ""
  echo "This could mean:"
  echo "  1. Per-mask tracking hasn't been run yet"
  echo "  2. The tracking used a different tracker name"
  echo "  3. You're using an older version without NPZ export"
  echo ""
  echo "Please run: bash scripts/run_per_mask_tracking.sh"
  exit 1
fi

echo "Found ${#INSTANCE_FILES[@]} tracking result(s) to process:"
for f in "${INSTANCE_FILES[@]}"; do
  echo "  - $(basename "$f")"
done
echo ""
# Step 1: Refine tracks (filter static, align to points)
echo "=========================================="
echo "Step 1: Refining Tracks"
echo "=========================================="
echo "Motion threshold: ${MOTION_THRESHOLD}"
echo "Motion method: ${MOTION_METHOD}"
echo ""

# Process each instance independently
for input_npz in "${INSTANCE_FILES[@]}"; do
  instance_name=$(basename "$input_npz" .npz)
  echo "Processing: ${instance_name}"
  
  python refine_tracks.py \
    --input "${input_npz}" \
    --output "${REFINED_DIR}/${instance_name}_refined.npz" \
    --motion-threshold ${MOTION_THRESHOLD} \
    --motion-method ${MOTION_METHOD}
  
  echo "  ✓ Refined: ${REFINED_DIR}/${instance_name}_refined.npz"
  echo "  ✓ Stats: ${REFINED_DIR}/${instance_name}_refined_stats.json"
  echo ""
done

echo ""
echo "Refinement complete for all instances!"
echo ""

# Step 2: Export to per-camera videos
echo "=========================================="
echo "Step 2: Exporting to Per-Camera Videos"
echo "=========================================="
echo "FPS: ${FPS}"
echo "Trail length: ${TRAIL_LENGTH}"
echo ""

# Export videos for each refined instance
for input_npz in "${INSTANCE_FILES[@]}"; do
  instance_name=$(basename "$input_npz" .npz)
  refined_npz="${REFINED_DIR}/${instance_name}_refined.npz"
  
  if [ -f "${refined_npz}" ]; then
    echo "Exporting videos for: ${instance_name}"
    
    python export_tracks_to_videos.py \
      --input "${refined_npz}" \
      --output-dir "${VIDEOS_DIR}/${instance_name}" \
      --fps ${FPS} \
      --trail-length ${TRAIL_LENGTH}
    
    echo "  ✓ Videos saved to: ${VIDEOS_DIR}/${instance_name}/"
    echo ""
  else
    echo "  ⚠ Skipping ${instance_name} - refined NPZ not found"
    echo ""
  fi
done

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Processed ${#INSTANCE_FILES[@]} instance(s)"
echo ""
echo "Outputs:"
echo "  Refined NPZs: ${REFINED_DIR}/*_refined.npz"
echo "  Statistics: ${REFINED_DIR}/*_refined_stats.json"
echo "  Videos: ${VIDEOS_DIR}/*/cam_*.mp4"
echo ""
echo "To view statistics for an instance:"
echo "  cat ${REFINED_DIR}/*_instance_0_refined_stats.json"
echo ""
echo "To play videos for an instance:"
echo "  vlc ${VIDEOS_DIR}/*_instance_0/*.mp4"
echo ""

# Optional: Batch processing example (process all masks from per-mask tracking output)
# 
# echo "=========================================="
# echo "Batch Processing Example"
# echo "=========================================="
# 
# # Refine all per-mask NPZ files individually
# for mask_npz in ./tracking_per_mask/${TASK_FOLDER}/${TASK_FOLDER}_${TRACKER}_instance_*.npz; do
#   mask_id=$(basename "$mask_npz" .npz | sed "s/${TASK_FOLDER}_${TRACKER}_//")
#   echo "Processing $mask_id..."
#   python refine_tracks.py \
#     --input "$mask_npz" \
#     --output "./refined_tracks/${TASK_FOLDER}/${mask_id}_refined.npz" \
#     --motion-threshold 0.01
# done
# 
# # Export videos for all refined NPZ files
# for refined_npz in ./refined_tracks/${TASK_FOLDER}/*_refined.npz; do
#   mask_id=$(basename "$refined_npz" _refined.npz)
#   python export_tracks_to_videos.py \
#     --input "$refined_npz" \
#     --output-dir "./track_videos/${TASK_FOLDER}/${mask_id}" \
#     --fps 30 \
#     --trail-length 10
# done
