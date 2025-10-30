#!/bin/bash
# Complete Pipeline: Track SAM Masks Using SpatialTrackerV2 and Visualize in 3D
#
# This script runs the complete pipeline to track SAM masks:
# 1. Track SAM mask points per camera using SpatialTrackerV2 (2D tracking)
# 2. Lift 2D tracks to 3D using depth maps
# 3. Generate .rrd visualization with all tracks
#
# Usage:
#   bash run_sam_mask_tracking_pipeline.sh
#
# Or edit the variables below and run directly

set -e  # Exit on error

# ============================================================================
# CONFIGURATION - EDIT THESE VARIABLES
# ============================================================================

TASK_FOLDER="task_0006_user_0014_scene_0007_cfg_0006_human"
# Input NPZ file with SAM2 masks (output from demo_sam2_object_tracking_debug.py)
NPZ_FILE="third_party/HOISTFormer/sam2_tracking_output/${TASK_FOLDER}_processed_hand_tracked_hoist_sam2.npz"

# Key for masks in NPZ file (usually "sam2_masks" or "sam2_predictions")
MASK_KEY="sam2_masks"

# Output directory for results
OUTPUT_DIR="third_party/HOISTFormer/sam2_tracking_output"

# Tracking parameters
TRACK_MODE="offline"  # "offline" or "online"
MAX_POINTS_PER_MASK=500  # Limit points per mask to avoid memory issues (set to high value or remove for all points)
FPS=1
TRACK_NUM=756

# Visualization parameters
VIZ_FPS=10.0
SPAWN_VIEWER=false  # Set to true to automatically open Rerun viewer
MAX_FRAMES=""  # Leave empty for all frames, or set to a number like "50"

# ============================================================================
# PIPELINE EXECUTION
# ============================================================================

echo "========================================"
echo "SAM Mask Tracking Pipeline"
echo "========================================"
echo ""
echo "Input NPZ: $NPZ_FILE"
echo "Mask key: $MASK_KEY"
echo "Track mode: $TRACK_MODE"
echo "Max points per mask: $MAX_POINTS_PER_MASK"
echo ""

# Check if input file exists
if [ ! -f "$NPZ_FILE" ]; then
    echo "ERROR: Input file not found: $NPZ_FILE"
    exit 1
fi

# Extract base name for output files
BASE_NAME=$(basename "$NPZ_FILE" .npz)
TRACKS_NPZ="${OUTPUT_DIR}/${BASE_NAME}_tracks_per_camera.npz"
TRACKS_RRD="${OUTPUT_DIR}/${BASE_NAME}_tracks_3d.rrd"

echo "========================================"
echo "Step 1/3: Track SAM masks per camera"
echo "========================================"
echo ""

# Build command for tracking
TRACK_CMD="python track_sam_masks_per_camera.py \
    --npz \"$NPZ_FILE\" \
    --mask-key \"$MASK_KEY\" \
    --track-mode \"$TRACK_MODE\" \
    --fps $FPS \
    --track-num $TRACK_NUM \
    --output \"$TRACKS_NPZ\""

# Add max points per mask if specified
if [ -n "$MAX_POINTS_PER_MASK" ]; then
    TRACK_CMD="$TRACK_CMD --max-points-per-mask $MAX_POINTS_PER_MASK"
fi

echo "Running: $TRACK_CMD"
echo ""
eval $TRACK_CMD

if [ $? -ne 0 ]; then
    echo "ERROR: Tracking failed!"
    exit 1
fi

echo ""
echo "Tracking complete! Output: $TRACKS_NPZ"
echo ""

echo "========================================"
echo "Step 2/3: Lift tracks to 3D and visualize"
echo "========================================"
echo ""

# Build command for lifting and visualization
LIFT_CMD="python lift_and_visualize_tracks.py \
    --npz \"$TRACKS_NPZ\" \
    --output \"$TRACKS_RRD\" \
    --fps $VIZ_FPS"

# Add spawn flag if enabled
if [ "$SPAWN_VIEWER" = true ]; then
    LIFT_CMD="$LIFT_CMD --spawn"
fi

# Add max frames if specified
if [ -n "$MAX_FRAMES" ]; then
    LIFT_CMD="$LIFT_CMD --max-frames $MAX_FRAMES"
fi

echo "Running: $LIFT_CMD"
echo ""
eval $LIFT_CMD

if [ $? -ne 0 ]; then
    echo "ERROR: Lifting/visualization failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "Step 3/3: Pipeline Complete!"
echo "========================================"
echo ""
echo "Results:"
echo "  - 2D Tracks NPZ: $TRACKS_NPZ"
echo "  - 3D Visualization: $TRACKS_RRD"
echo ""
echo "To view the visualization:"
echo "  rerun \"$TRACKS_RRD\" --web-viewer"
echo ""
echo "Or open in desktop viewer:"
echo "  rerun \"$TRACKS_RRD\""
echo ""
