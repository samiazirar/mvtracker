#!/bin/bash
# DROID Training Data Processing Pipeline
# Downloads, extracts RGB/depth, and generates tracks for episodes
#
# Usage:
#   ./run_pipeline.sh              # Process 10 episodes (default)
#   ./run_pipeline.sh 100          # Process 100 episodes
#   ./run_pipeline.sh -1           # Process all episodes

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================

LIMIT=${1:-10}
CAM2BASE_PATH="/data/droid/calib_and_annot/droid/cam2base_extrinsic_superset.json"
CONFIG_PATH="conversions/droid/training_data/config.yaml"
SCRIPT_DIR="conversions/droid/training_data"
LOG_DIR="logs/pipeline_$(date +%Y%m%d_%H%M%S)"
EPISODES_FILE="${LOG_DIR}/episodes.txt"
TIMING_FILE="${LOG_DIR}/timing.csv"

# GCS bucket for downloads
GCS_BUCKET="gs://gresearch/robotics/droid_raw/1.0.1"
LOCAL_DOWNLOAD_DIR="./droid_downloads"

# ============================================================================
# SETUP
# ============================================================================

mkdir -p "${LOG_DIR}"
mkdir -p "${LOCAL_DOWNLOAD_DIR}"

echo "=== DROID Training Data Pipeline ==="
echo "Limit: ${LIMIT}"
echo "Log dir: ${LOG_DIR}"
echo ""

# Initialize timing CSV
echo "episode_id,download_time_sec,rgb_depth_time_sec,tracks_time_sec,total_time_sec" > "${TIMING_FILE}"

# Running averages
TOTAL_DOWNLOAD_TIME=0
TOTAL_RGB_DEPTH_TIME=0
TOTAL_TRACKS_TIME=0
PROCESSED_COUNT=0

# ============================================================================
# GET EPISODES
# ============================================================================

echo "[1/4] Getting episodes sorted by quality..."
python "${SCRIPT_DIR}/get_episodes_by_quality.py" \
    --cam2base "${CAM2BASE_PATH}" \
    --limit "${LIMIT}" \
    --output "${EPISODES_FILE}"

EPISODE_COUNT=$(wc -l < "${EPISODES_FILE}")
echo "Found ${EPISODE_COUNT} episodes to process"
echo ""

# ============================================================================
# PROCESS EACH EPISODE
# ============================================================================

while IFS= read -r EPISODE_ID; do
    echo "============================================================"
    echo "Processing: ${EPISODE_ID}"
    echo "Progress: $((PROCESSED_COUNT + 1))/${EPISODE_COUNT}"
    echo "============================================================"
    
    EPISODE_LOG_DIR="${LOG_DIR}/${EPISODE_ID}"
    mkdir -p "${EPISODE_LOG_DIR}"
    
    EPISODE_START=$(date +%s)
    
    # ------------------------------------------------------------------------
    # STEP A: Download Episode
    # ------------------------------------------------------------------------
    echo "[A] Downloading episode..."
    DOWNLOAD_START=$(date +%s)
    
    python "${SCRIPT_DIR}/download_single_episode.py" \
        --episode_id "${EPISODE_ID}" \
        --cam2base "${CAM2BASE_PATH}" \
        --output_dir "${LOCAL_DOWNLOAD_DIR}" \
        --gcs_bucket "${GCS_BUCKET}" \
        > "${EPISODE_LOG_DIR}/download.log" 2>&1 || {
            echo "[ERROR] Download failed for ${EPISODE_ID}"
            continue
        }
    
    DOWNLOAD_END=$(date +%s)
    DOWNLOAD_TIME=$((DOWNLOAD_END - DOWNLOAD_START))
    echo "    Download completed in ${DOWNLOAD_TIME}s"
    
    # ------------------------------------------------------------------------
    # STEP B: Extract RGB and Depth (GPU)
    # ------------------------------------------------------------------------
    echo "[B] Extracting RGB and depth frames..."
    RGB_DEPTH_START=$(date +%s)
    
    python "${SCRIPT_DIR}/extract_rgb_depth.py" \
        --episode_id "${EPISODE_ID}" \
        --config "${CONFIG_PATH}" \
        > "${EPISODE_LOG_DIR}/rgb_depth.log" 2>&1 || {
            echo "[ERROR] RGB/depth extraction failed for ${EPISODE_ID}"
            RGB_DEPTH_TIME=0
            TRACKS_TIME=0
            continue
        }
    
    RGB_DEPTH_END=$(date +%s)
    RGB_DEPTH_TIME=$((RGB_DEPTH_END - RGB_DEPTH_START))
    echo "    RGB/depth extraction completed in ${RGB_DEPTH_TIME}s"
    
    # ------------------------------------------------------------------------
    # STEP C: Generate Tracks and Metadata (CPU)
    # ------------------------------------------------------------------------
    echo "[C] Generating tracks and metadata..."
    TRACKS_START=$(date +%s)
    
    python "${SCRIPT_DIR}/generate_tracks_and_metadata.py" \
        --episode_id "${EPISODE_ID}" \
        --config "${CONFIG_PATH}" \
        > "${EPISODE_LOG_DIR}/tracks.log" 2>&1 || {
            echo "[ERROR] Track generation failed for ${EPISODE_ID}"
            TRACKS_TIME=0
            continue
        }
    
    TRACKS_END=$(date +%s)
    TRACKS_TIME=$((TRACKS_END - TRACKS_START))
    echo "    Track generation completed in ${TRACKS_TIME}s"
    
    # ------------------------------------------------------------------------
    # STEP D: Cleanup downloaded files (optional)
    # ------------------------------------------------------------------------
    # Uncomment to delete downloaded files after processing
    # echo "[D] Cleaning up downloaded files..."
    # rm -rf "${LOCAL_DOWNLOAD_DIR}/${EPISODE_ID}"
    
    # ------------------------------------------------------------------------
    # Record Timing
    # ------------------------------------------------------------------------
    EPISODE_END=$(date +%s)
    TOTAL_TIME=$((EPISODE_END - EPISODE_START))
    
    echo "${EPISODE_ID},${DOWNLOAD_TIME},${RGB_DEPTH_TIME},${TRACKS_TIME},${TOTAL_TIME}" >> "${TIMING_FILE}"
    
    # Update running averages
    TOTAL_DOWNLOAD_TIME=$((TOTAL_DOWNLOAD_TIME + DOWNLOAD_TIME))
    TOTAL_RGB_DEPTH_TIME=$((TOTAL_RGB_DEPTH_TIME + RGB_DEPTH_TIME))
    TOTAL_TRACKS_TIME=$((TOTAL_TRACKS_TIME + TRACKS_TIME))
    PROCESSED_COUNT=$((PROCESSED_COUNT + 1))
    
    AVG_DOWNLOAD=$((TOTAL_DOWNLOAD_TIME / PROCESSED_COUNT))
    AVG_RGB_DEPTH=$((TOTAL_RGB_DEPTH_TIME / PROCESSED_COUNT))
    AVG_TRACKS=$((TOTAL_TRACKS_TIME / PROCESSED_COUNT))
    AVG_TOTAL=$((AVG_DOWNLOAD + AVG_RGB_DEPTH + AVG_TRACKS))
    
    echo ""
    echo "Episode total: ${TOTAL_TIME}s"
    echo "Running averages (${PROCESSED_COUNT} episodes):"
    echo "  Download:   ${AVG_DOWNLOAD}s"
    echo "  RGB/Depth:  ${AVG_RGB_DEPTH}s"
    echo "  Tracks:     ${AVG_TRACKS}s"
    echo "  Total:      ${AVG_TOTAL}s"
    echo ""
    
done < "${EPISODES_FILE}"

# ============================================================================
# SUMMARY
# ============================================================================

echo "============================================================"
echo "PIPELINE COMPLETE"
echo "============================================================"
echo "Processed: ${PROCESSED_COUNT} episodes"
echo "Timing log: ${TIMING_FILE}"
echo ""
echo "Final averages:"
if [ "${PROCESSED_COUNT}" -gt 0 ]; then
    echo "  Download:   $((TOTAL_DOWNLOAD_TIME / PROCESSED_COUNT))s"
    echo "  RGB/Depth:  $((TOTAL_RGB_DEPTH_TIME / PROCESSED_COUNT))s"
    echo "  Tracks:     $((TOTAL_TRACKS_TIME / PROCESSED_COUNT))s"
fi
