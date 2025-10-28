#!/usr/bin/env bash

# ===========================================================================
# COMPLETE WORKFLOW: From Raw Data to Object Tracking
# ===========================================================================
# 
# This script demonstrates the complete pipeline for tracking grasped objects
# in human demonstrations using SAM2.1.
#
# Pipeline stages:
# 1. Process raw data to create sparse depth maps
# 2. Detect hands and generate SAM masks
# 3. Identify grasp timestamps automatically
# 4. Track grasped object with SAM2.1
# 5. Optional: Run 3D tracking with MVTracker
#
# ===========================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

TASK_FOLDER="task_0045_user_0020_scene_0004_cfg_0006_human"
DEPTH_FOLDER="/data/rh20t_api/data/low_res_data/RH20T_cfg6/$TASK_FOLDER"
RGB_FOLDER="/data/rh20t_api/hf_download/RH20T/RH20T_cfg6/$TASK_FOLDER"
OUT_DIR="./data/human_high_res_filtered"

echo "========================================"
echo "Complete Object Tracking Pipeline"
echo "========================================"
echo ""
echo "Task: $TASK_FOLDER"
echo "Output: $OUT_DIR"
echo ""

# ---------------------------------------------------------------------------
# STAGE 1: Create Sparse Depth Maps
# ---------------------------------------------------------------------------

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STAGE 1: Creating Sparse Depth Maps"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

mkdir -p "${OUT_DIR}"

if [ -f "${OUT_DIR}/${TASK_FOLDER}_processed.npz" ]; then
    echo "[SKIP] Processed NPZ already exists"
else
    echo "[RUN] Processing depth maps..."
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
      --pc-clean-min-points 5
    
    echo "[DONE] Created ${OUT_DIR}/${TASK_FOLDER}_processed.npz"
fi

# ---------------------------------------------------------------------------
# STAGE 2: Detect Hands and Generate SAM Masks
# ---------------------------------------------------------------------------

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STAGE 2: Detecting Hands with HaMeR + SAM"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -f "${OUT_DIR}/${TASK_FOLDER}_processed_hand_tracked.npz" ]; then
    echo "[SKIP] Hand tracking already complete"
else
    echo "[RUN] Detecting hands and generating masks..."
    
    cd third_party/hamer
    source .hamer/bin/activate
    
    python add_hand_mask_from_sam_to_rh20t.py \
      --npz "../../${OUT_DIR}/${TASK_FOLDER}_processed.npz" \
      --out-dir "../../${OUT_DIR}"
    
    deactivate
    cd ../..
    
    echo "[DONE] Created ${OUT_DIR}/${TASK_FOLDER}_processed_hand_tracked.npz"
fi

# ---------------------------------------------------------------------------
# STAGE 3: Identify Grasp Timestamps
# ---------------------------------------------------------------------------

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STAGE 3: Identifying Grasp Timestamps"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

NPZ_PATH="${OUT_DIR}/${TASK_FOLDER}_processed_hand_tracked.npz"

echo "[RUN] Analyzing finger distances..."

# Run timestamp identification and capture output
TIMESTAMP_OUTPUT=$(python scripts/identify_grasp_timestamps.py \
  --npz "$NPZ_PATH" \
  --threshold 0.05 \
  --min-duration 3 \
  --plot "${OUT_DIR}/grasp_analysis.png" | tee /dev/tty)

# Extract suggested timestamps from output
GRASP_TIMESTAMPS=$(echo "$TIMESTAMP_OUTPUT" | grep "Suggested grasp timestamps" -A 1 | tail -n 1 | xargs)

if [ -z "$GRASP_TIMESTAMPS" ]; then
    echo ""
    echo "[ERROR] Could not identify grasp timestamps automatically."
    echo "Please specify manually or adjust parameters."
    echo ""
    echo "Try one of these approaches:"
    echo "  1. Lower threshold: --threshold 0.03"
    echo "  2. Shorter duration: --min-duration 2"
    echo "  3. Manual specification: GRASP_TIMESTAMPS=\"10 15 20\""
    echo ""
    exit 1
fi

echo ""
echo "[DONE] Identified grasp timestamps: $GRASP_TIMESTAMPS"
echo "[INFO] Visualization saved to: ${OUT_DIR}/grasp_analysis.png"

# ---------------------------------------------------------------------------
# STAGE 4: Track Grasped Object with SAM2.1
# ---------------------------------------------------------------------------

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STAGE 4: Tracking Grasped Object"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

OBJECT_OUTPUT_DIR="${OUT_DIR}/object_tracking"

echo "[RUN] Running SAM2.1 video segmentation..."

python scripts/filter_grasped_object_with_sam.py \
  --npz "$NPZ_PATH" \
  --grasp-timestamps $GRASP_TIMESTAMPS \
  --output-dir "$OBJECT_OUTPUT_DIR" \
  --sam-config "third_party/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml" \
  --sam-checkpoint "third_party/sam2/checkpoints/sam2.1_hiera_large.pt" \
  --device cuda

echo "[DONE] Object tracking complete!"
echo ""

# ---------------------------------------------------------------------------
# STAGE 5 (Optional): Run MVTracker
# ---------------------------------------------------------------------------

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STAGE 5: 3D Tracking (Optional)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

SAMPLE_PATH_WITH_OBJECTS="${OBJECT_OUTPUT_DIR}/${TASK_FOLDER}_processed_hand_tracked_with_object_masks.npz"

read -p "Run MVTracker for 3D object tracking? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "[RUN] Running MVTracker..."
    
    python demo.py \
      --temporal_stride 1 \
      --spatial_downsample 1 \
      --depth_estimator gt \
      --depth_cache_dir ./depth_cache \
      --rerun save \
      --sample-path "$SAMPLE_PATH_WITH_OBJECTS" \
      --tracker cotracker3_offline
    
    echo "[DONE] MVTracker complete!"
else
    echo "[SKIP] Skipping MVTracker"
fi

# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------

echo ""
echo "========================================"
echo "Pipeline Complete!"
echo "========================================"
echo ""
echo "Generated Files:"
echo "  1. Processed NPZ: ${OUT_DIR}/${TASK_FOLDER}_processed.npz"
echo "  2. Hand tracked NPZ: ${OUT_DIR}/${TASK_FOLDER}_processed_hand_tracked.npz"
echo "  3. Grasp analysis: ${OUT_DIR}/grasp_analysis.png"
echo "  4. Object tracked NPZ: ${OBJECT_OUTPUT_DIR}/${TASK_FOLDER}_processed_hand_tracked_with_object_masks.npz"
echo "  5. Object tracking videos: ${OBJECT_OUTPUT_DIR}/*_object_tracked.mp4"
echo ""
echo "Next Steps:"
echo "  - View grasp analysis: open ${OUT_DIR}/grasp_analysis.png"
echo "  - Watch object videos: ${OBJECT_OUTPUT_DIR}/*_object_tracked.mp4"
echo "  - Run MVTracker for 3D tracking (if not done above)"
echo ""

# Optional: Copy to shared directory
if [ -d "/data/rh20t_api/test_data_generated_human" ]; then
    read -p "Copy results to /data/rh20t_api/test_data_generated_human? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "[COPY] Copying results..."
        cp -r "${OUT_DIR}/object_tracking" /data/rh20t_api/test_data_generated_human/
        cp "${OUT_DIR}/grasp_analysis.png" /data/rh20t_api/test_data_generated_human/
        echo "[DONE] Results copied!"
    fi
fi

echo ""
echo "All done! 🎉"
