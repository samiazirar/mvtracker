#!/usr/bin/env bash

# Example script to track grasped objects using SAM2.1
# This script takes a hand-tracked NPZ file and segments the object being grasped
# based on timestamps where gripping occurs.

set -euo pipefail

# ============================================================================
# CONFIGURATION - Update these paths to match your data
# ============================================================================

TASK_FOLDER="task_0045_user_0020_scene_0004_cfg_0006_human"
OUT_DIR="./data/human_high_res_filtered"

# Path to the hand-tracked NPZ file (output from add_hand_mask_from_sam_to_rh20t.py)
NPZ_PATH="${OUT_DIR}/${TASK_FOLDER}_processed_hand_tracked.npz"

# Timestamps (frame indices) where gripping occurs
# You can specify multiple frames where the hand is grasping the object
# The script will use the middle frame as the reference for SAM prompting
GRASP_TIMESTAMPS="10 15 20 25"

# Output directory for results
OBJECT_OUTPUT_DIR="${OUT_DIR}/object_tracking"

# ============================================================================
# RUN OBJECT TRACKING
# ============================================================================

echo "========================================"
echo "Starting Object Tracking with SAM2.1"
echo "========================================"
echo ""
echo "Input NPZ: ${NPZ_PATH}"
echo "Grasp frames: ${GRASP_TIMESTAMPS}"
echo "Output directory: ${OBJECT_OUTPUT_DIR}"
echo ""

# Check if input file exists
if [ ! -f "${NPZ_PATH}" ]; then
    echo "[ERROR] Input NPZ file not found: ${NPZ_PATH}"
    echo "Please run the hand tracking pipeline first (scripts/run_human_example.sh)"
    exit 1
fi

# Create output directory
mkdir -p "${OBJECT_OUTPUT_DIR}"

# Run the object tracking script
python scripts/filter_grasped_object_with_sam.py \
  --npz "${NPZ_PATH}" \
  --grasp-timestamps ${GRASP_TIMESTAMPS} \
  --output-dir "${OBJECT_OUTPUT_DIR}" \
  --sam-config "third_party/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml" \
  --sam-checkpoint "third_party/sam2/sam2/checkpoints/sam2.1_hiera_large.pt" \
  --device cuda

echo ""
echo "========================================"
echo "Object Tracking Complete!"
echo "========================================"
echo ""
echo "Outputs saved to: ${OBJECT_OUTPUT_DIR}"
echo ""
echo "Results:"
echo "  - NPZ with object masks: ${OBJECT_OUTPUT_DIR}/${TASK_FOLDER}_processed_hand_tracked_with_object_masks.npz"
echo "  - Video overlays: ${OBJECT_OUTPUT_DIR}/*_object_tracked.mp4"
echo ""
echo "To visualize the results, use:"
echo "  rerun ${OBJECT_OUTPUT_DIR}/*.rrd"
echo ""

# Optional: Copy results to shared data directory
if [ -d "/data/rh20t_api/test_data_generated_human" ]; then
    echo "Copying results to /data/rh20t_api/test_data_generated_human..."
    cp -r "${OBJECT_OUTPUT_DIR}" /data/rh20t_api/test_data_generated_human/
    echo "Done!"
fi
