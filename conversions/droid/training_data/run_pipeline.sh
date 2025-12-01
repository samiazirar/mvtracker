#!/bin/bash
# DROID Training Data Processing Pipeline (Parallel Optimized)
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

# [MODIFICATION] Parallel & Storage Configuration
# ----------------------------------------------------------------------------
NUM_WORKERS=24                           # 24 Workers * ~4GB VRAM = ~96GB Usage
# FAST_LOCAL_DIR="/data/droid_scratch"     # Node-local fast storage (NVMe)
FAST_LOCAL_DIR="droid_processed"
PERMANENT_STORAGE_DIR="droid_processed" # Final destination for processed data
# ----------------------------------------------------------------------------

# Export variables for parallel workers
export CAM2BASE_PATH CONFIG_PATH SCRIPT_DIR GCS_BUCKET FAST_LOCAL_DIR PERMANENT_STORAGE_DIR TIMING_FILE

# ============================================================================
# SETUP
# ============================================================================

mkdir -p "${LOG_DIR}"
mkdir -p "${FAST_LOCAL_DIR}"
mkdir -p "${PERMANENT_STORAGE_DIR}"

echo "=== DROID Training Data Pipeline (Parallel) ==="
echo "Limit: ${LIMIT}"
echo "Workers: ${NUM_WORKERS}"
echo "Local Scratch: ${FAST_LOCAL_DIR}"
echo "Final Output: ${PERMANENT_STORAGE_DIR}"
echo "Log dir: ${LOG_DIR}"
echo ""

# Initialize timing CSV
echo "episode_id,download_time_sec,rgb_depth_time_sec,tracks_time_sec,total_time_sec" > "${TIMING_FILE}"

# ============================================================================
# WORKER FUNCTION (Parallel Execution)
# ============================================================================

process_episode_worker() {
    local EPISODE_ID=$1
    local WORKER_ID=$BASHPID  # Unique Process ID for isolation
    
    # Define unique local workspace for this worker
    local JOB_DIR="${FAST_LOCAL_DIR}/${EPISODE_ID}_${WORKER_ID}"
    local JOB_DATA="${JOB_DIR}/data"
    local JOB_OUTPUT="${JOB_DIR}/output"
    local JOB_LOGS="${JOB_DIR}/logs"
    local TEMP_CONFIG="${JOB_DIR}/config.yaml"
    
    mkdir -p "${JOB_DATA}" "${JOB_OUTPUT}" "${JOB_LOGS}"

    # Create Temp Config for this worker
    # We override 'droid_root', 'download_dir', and 'output_root' to point to fast local storage
    cp "${CONFIG_PATH}" "${TEMP_CONFIG}"
    echo "" >> "${TEMP_CONFIG}"
    echo "# Worker Overrides" >> "${TEMP_CONFIG}"
    echo "droid_root: \"${JOB_DATA}\"" >> "${TEMP_CONFIG}"
    echo "download_dir: \"${JOB_DATA}\"" >> "${TEMP_CONFIG}"
    echo "output_root: \"${JOB_OUTPUT}\"" >> "${TEMP_CONFIG}"

    # Start Timing
    local EPISODE_START=$(date +%s)
    
    echo "[Worker ${WORKER_ID}] Starting: ${EPISODE_ID}"

    # ------------------------------------------------------------------------
    # STEP A: Download Episode (To Fast Local /data)
    # ------------------------------------------------------------------------
    local DOWNLOAD_START=$(date +%s)
    
    python "${SCRIPT_DIR}/download_single_episode.py" \
        --episode_id "${EPISODE_ID}" \
        --cam2base "${CAM2BASE_PATH}" \
        --output_dir "${JOB_DATA}" \
        --gcs_bucket "${GCS_BUCKET}" \
        > "${JOB_LOGS}/download.log" 2>&1 || {
            echo "[ERROR] Download failed for ${EPISODE_ID} (See ${JOB_LOGS}/download.log)"
            rm -rf "${JOB_DIR}"
            return 1
        }
    
    local DOWNLOAD_END=$(date +%s)
    local DOWNLOAD_TIME=$((DOWNLOAD_END - DOWNLOAD_START))
    
    # ------------------------------------------------------------------------
    # STEP B: Extract RGB and Depth (GPU Step)
    # ------------------------------------------------------------------------
    local RGB_DEPTH_START=$(date +%s)
    
    # Uses TEMP_CONFIG which points input/output to local scratch
    python "${SCRIPT_DIR}/extract_rgb_depth.py" \
        --episode_id "${EPISODE_ID}" \
        --config "${TEMP_CONFIG}" \
        > "${JOB_LOGS}/rgb_depth.log" 2>&1 || {
            echo "[ERROR] Extraction failed for ${EPISODE_ID} (See ${JOB_LOGS}/rgb_depth.log)"
            rm -rf "${JOB_DIR}"
            return 1
        }
    
    local RGB_DEPTH_END=$(date +%s)
    local RGB_DEPTH_TIME=$((RGB_DEPTH_END - RGB_DEPTH_START))
    
    # ------------------------------------------------------------------------
    # STEP C: Generate Tracks and Metadata (CPU Step)
    # ------------------------------------------------------------------------
    local TRACKS_START=$(date +%s)
    
    python "${SCRIPT_DIR}/generate_tracks_and_metadata.py" \
        --episode_id "${EPISODE_ID}" \
        --config "${TEMP_CONFIG}" \
        > "${JOB_LOGS}/tracks.log" 2>&1 || {
            echo "[ERROR] Tracking failed for ${EPISODE_ID} (See ${JOB_LOGS}/tracks.log)"
            rm -rf "${JOB_DIR}"
            return 1
        }
    
    local TRACKS_END=$(date +%s)
    local TRACKS_TIME=$((TRACKS_END - TRACKS_START))
    
    # ------------------------------------------------------------------------
    # STEP D: Move Results & Cleanup
    # ------------------------------------------------------------------------
    # Sync the generated output folder structure to permanent storage
    # rsync is safer than cp for merging directory trees
    rsync -a "${JOB_OUTPUT}/" "${PERMANENT_STORAGE_DIR}/"
    
    # Cleanup local scratch
    rm -rf "${JOB_DIR}"
    
    # ------------------------------------------------------------------------
    # Record Timing
    # ------------------------------------------------------------------------
    local EPISODE_END=$(date +%s)
    local TOTAL_TIME=$((EPISODE_END - EPISODE_START))
    
    # Atomic append to CSV
    echo "${EPISODE_ID},${DOWNLOAD_TIME},${RGB_DEPTH_TIME},${TRACKS_TIME},${TOTAL_TIME}" >> "${TIMING_FILE}"
    
    echo "[Worker ${WORKER_ID}] Finished: ${EPISODE_ID} (Total: ${TOTAL_TIME}s)"
}

export -f process_episode_worker

# ============================================================================
# GET EPISODES
# ============================================================================

echo "[1/2] Getting episodes sorted by quality..."
python "${SCRIPT_DIR}/get_episodes_by_quality.py" \
    --cam2base "${CAM2BASE_PATH}" \
    --limit "${LIMIT}" \
    --output "${EPISODES_FILE}"

EPISODE_COUNT=$(wc -l < "${EPISODES_FILE}")
echo "Found ${EPISODE_COUNT} episodes to process"
echo ""

# ============================================================================
# EXECUTE PARALLEL PIPELINE
# ============================================================================

echo "[2/2] Launching ${NUM_WORKERS} parallel workers..."
echo "      Monitoring output in ${TIMING_FILE}"
echo "============================================================"

# xargs -P manages the pool of parallel processes
cat "${EPISODES_FILE}" | xargs -P "${NUM_WORKERS}" -I {} bash -c 'process_episode_worker "$@"' _ {}

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE"
echo "============================================================"

PROCESSED_COUNT=$(wc -l < "${TIMING_FILE}")
# Adjust for header line
PROCESSED_COUNT=$((PROCESSED_COUNT - 1))

echo "Processed: ${PROCESSED_COUNT} episodes"
echo "Timing log: ${TIMING_FILE}"
echo "Output dir: ${PERMANENT_STORAGE_DIR}"

if [ "${PROCESSED_COUNT}" -gt 0 ]; then
    # Calculate simple averages using awk from the CSV
    awk -F',' 'NR>1 {
        sum_dl+=$2; sum_rgb+=$3; sum_tracks+=$4; sum_total+=$5; count++
    } END {
        if (count > 0) {
            print "\nFinal Averages:";
            printf "  Download:   %.1fs\n", sum_dl/count;
            printf "  RGB/Depth:  %.1fs\n", sum_rgb/count;
            printf "  Tracks:     %.1fs\n", sum_tracks/count;
            printf "  Total:      %.1fs\n", sum_total/count;
        }
    }' "${TIMING_FILE}"
fi
