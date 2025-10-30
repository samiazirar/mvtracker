#!/usr/bin/env bash

# Complete track refinement and video export pipeline
# This script demonstrates the full workflow:
# 1. Run tracking (assumes already done)
# 2. Refine tracks (filter static, align to points)
# 3. Export to per-camera videos

set -euo pipefail

# Configuration
TASK_FOLDER="task_0034_user_0014_scene_0004_cfg_0006_human"
BASE_DIR="third_party/HOISTFormer/sam2_tracking_output"

# Input: Tracking results from demo.py or process_masks_independently.py
INPUT_NPZ="${BASE_DIR}/${TASK_FOLDER}_processed_hand_tracked_hoist_sam2.npz"

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
echo ""
echo "Input NPZ: ${INPUT_NPZ}"
echo "Refined output: ${REFINED_DIR}"
echo "Videos output: ${VIDEOS_DIR}"
echo ""

# Step 1: Refine tracks (filter static, align to points)
echo "=========================================="
echo "Step 1: Refining Tracks"
echo "=========================================="
echo "Motion threshold: ${MOTION_THRESHOLD}"
echo "Motion method: ${MOTION_METHOD}"
echo ""

python refine_tracks.py \
  --input "${INPUT_NPZ}" \
  --output "${REFINED_DIR}/${TASK_FOLDER}_refined.npz" \
  --motion-threshold ${MOTION_THRESHOLD} \
  --motion-method ${MOTION_METHOD}

echo ""
echo "Refinement complete!"
echo "Refined NPZ: ${REFINED_DIR}/${TASK_FOLDER}_refined.npz"
echo "Statistics: ${REFINED_DIR}/${TASK_FOLDER}_refined_stats.json"
echo ""

# Step 2: Export to per-camera videos
echo "=========================================="
echo "Step 2: Exporting to Per-Camera Videos"
echo "=========================================="
echo "FPS: ${FPS}"
echo "Trail length: ${TRAIL_LENGTH}"
echo ""

python export_tracks_to_videos.py \
  --input "${REFINED_DIR}/${TASK_FOLDER}_refined.npz" \
  --output-dir "${VIDEOS_DIR}" \
  --fps ${FPS} \
  --trail-length ${TRAIL_LENGTH}

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Outputs:"
echo "  Refined NPZ: ${REFINED_DIR}/${TASK_FOLDER}_refined.npz"
echo "  Statistics: ${REFINED_DIR}/${TASK_FOLDER}_refined_stats.json"
echo "  Videos: ${VIDEOS_DIR}/*.mp4"
echo ""
echo "To view the statistics:"
echo "  cat ${REFINED_DIR}/${TASK_FOLDER}_refined_stats.json"
echo ""
echo "To play a video:"
echo "  vlc ${VIDEOS_DIR}/${TASK_FOLDER}_refined_cam_00.mp4"
echo ""

# Optional: Batch processing example
# 
# echo "=========================================="
# echo "Batch Processing Example"
# echo "=========================================="
# 
# # Refine all NPZ files in a directory
# python refine_tracks.py \
#   --input-dir ./tracking_per_mask/${TASK_FOLDER} \
#   --output-dir ./refined_tracks/${TASK_FOLDER} \
#   --motion-threshold 0.01
# 
# # Export videos for all refined NPZ files
# python export_tracks_to_videos.py \
#   --input-dir ./refined_tracks/${TASK_FOLDER} \
#   --output-dir ./track_videos/${TASK_FOLDER} \
#   --fps 30 \
#   --trail-length 10
