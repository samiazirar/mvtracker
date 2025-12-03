#!/bin/bash
# DROID Training Data Processing Pipeline (Multi-GPU Parallel Optimized + Validation Rendering)
# Downloads, extracts RGB/depth, generates tracks, and renders validation RRDs for episodes
#
# Pipeline Steps:
#   A. Download episode data from GCS
#   B. Extract RGB and Depth frames from SVO files (GPU)
#   C. Generate gripper tracks and camera extrinsics (CPU)
#   D. Render validation RRD and videos to point_clouds/ folder
#   E. Move results to permanent storage
#
# Output Structure:
#   {output_root}/{lab}/success/{date}/{timestamp}/
#       ├── tracks.npz              # Gripper contact tracks
#       ├── extrinsics.npz          # Camera extrinsics
#       ├── quality.json            # Episode metadata
#       ├── recordings/
#       │   └── {camera_serial}/
#       │       ├── rgb/            # PNG frames
#       │       ├── depth/          # NPY depth maps
#       │       └── intrinsics.json
#       └── point_clouds/           # Validation outputs
#           ├── validation.rrd      # Rerun visualization
#           └── videos/             # MP4 videos with tracks
#
# Usage:
#   ./run_pipeline_with_rendering.sh                    # Process 10 episodes, auto-detect GPUs, 3 workers/GPU
#   ./run_pipeline_with_rendering.sh 100                # Process 100 episodes (workers/GPU=3, auto GPUs)
#   ./run_pipeline_with_rendering.sh 100 6              # Process 100 episodes, 6 workers/GPU, auto GPUs
#   ./run_pipeline_with_rendering.sh 100 6 4            # Process 100 episodes, 6 workers/GPU, 4 GPUs
#   ./run_pipeline_with_rendering.sh -1                 # Process all episodes
#
# Environment Variables:
#   CUDA_VISIBLE_DEVICES    Override which GPUs to use (e.g., "0,1,2,3")
#   DROID_WORKERS_PER_GPU   Workers per GPU (default: 3)
#   DROID_NUM_GPUS          Number of GPUs to use (default: auto-detect)

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================

LIMIT=${1:-10}
WORKERS_PER_GPU=${2:-${DROID_WORKERS_PER_GPU:-3}}
NUM_GPUS=${3:-${DROID_NUM_GPUS:-0}}      # 0 = auto-detect

CAM2BASE_PATH="/data/droid/calib_and_annot/droid/cam2base_extrinsic_superset.json"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/config.yaml"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
LOG_DIR="/data/logs/pipeline_rendering_$(date +%Y%m%d_%H%M%S)"
EPISODES_FILE="${LOG_DIR}/episodes.txt"
TIMING_FILE="${LOG_DIR}/timing.csv"
STATUS_FILE="${LOG_DIR}/status.log"
ERROR_LOG="${LOG_DIR}/errors.log"

# GCS bucket for downloads
GCS_BUCKET="gs://gresearch/robotics/droid_raw/1.0.1"

# Storage Configuration
# ----------------------------------------------------------------------------
# FAST_LOCAL_DIR="/data/droid_scratch"     # Node-local fast storage (NVMe)
FAST_LOCAL_DIR="droid_processed"
PERMANENT_STORAGE_DIR="droid_processed" # Final destination for processed data
# ----------------------------------------------------------------------------

## Finger mesh, no need to set on mashine
# DEFAULT_INNER_FINGER_MESH="/workspace/third_party/robotiq_arg85_description/meshes/inner_finger_fine.STL"


# ----------------------------------------------------------------------------
# AUTO-DOWNLOAD EXTRINSICS (Hugging Face / Git LFS)
# ----------------------------------------------------------------------------
if [ ! -f "${CAM2BASE_PATH}" ]; then
    echo "[INFO] Extrinsic file not found at: ${CAM2BASE_PATH}"
    echo "[INFO] Downloading from Hugging Face (KarlP/droid)..."

    # Ensure parent directory exists
    mkdir -p "$(dirname "${CAM2BASE_PATH}")"

    # Check for git-lfs
    if ! command -v git-lfs &> /dev/null; then
        echo "[ERROR] git-lfs is required but not installed."
        exit 1
    fi

    # Create temp dir for cloning
    TEMP_CLONE_DIR=$(mktemp -d)
    
    # Clone the repo (depth 1 for speed)
    echo "[INFO] Cloning repo to ${TEMP_CLONE_DIR}..."
    git clone --depth 1 https://huggingface.co/KarlP/droid "${TEMP_CLONE_DIR}"
    
    # Ensure LFS objects are actually pulled
    pushd "${TEMP_CLONE_DIR}" > /dev/null
    git lfs install
    git lfs pull
    popd > /dev/null

    # Determine filename from the path variable and move it
    TARGET_FILENAME=$(basename "${CAM2BASE_PATH}")
    
    if [ -f "${TEMP_CLONE_DIR}/${TARGET_FILENAME}" ]; then
        mv "${TEMP_CLONE_DIR}/${TARGET_FILENAME}" "${CAM2BASE_PATH}"
        echo "[SUCCESS] Downloaded ${TARGET_FILENAME} to ${CAM2BASE_PATH}"
    else
        echo "[ERROR] File '${TARGET_FILENAME}' not found in Hugging Face repo."
        rm -rf "${TEMP_CLONE_DIR}"
        exit 1
    fi

    # Cleanup
    rm -rf "${TEMP_CLONE_DIR}"
fi

# ============================================================================
# GPU DETECTION AND SETUP
# ============================================================================

detect_gpus() {
    # Try nvidia-smi first
    if command -v nvidia-smi &> /dev/null; then
        local gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
        if [ "${gpu_count}" -gt 0 ]; then
            echo "${gpu_count}"
            return
        fi
    fi
    
    # Fallback: check /dev for nvidia devices
    local dev_count=$(ls /dev/nvidia[0-9]* 2>/dev/null | wc -l)
    if [ "${dev_count}" -gt 0 ]; then
        echo "${dev_count}"
        return
    fi
    
    # No GPUs found
    echo "0"
}

# Auto-detect GPUs if not specified
if [ "${NUM_GPUS}" -eq 0 ]; then
    NUM_GPUS=$(detect_gpus)
    if [ "${NUM_GPUS}" -eq 0 ]; then
        echo "[ERROR] No GPUs detected! Set CUDA_VISIBLE_DEVICES or DROID_NUM_GPUS"
        exit 1
    fi
fi

# Cap at 8 GPUs max
if [ "${NUM_GPUS}" -gt 8 ]; then
    echo "[WARN] Capping NUM_GPUS from ${NUM_GPUS} to 8"
    NUM_GPUS=8
fi

# Calculate total workers
TOTAL_WORKERS=$((NUM_GPUS * WORKERS_PER_GPU))

# Build GPU list (respect CUDA_VISIBLE_DEVICES if set)
if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
    IFS=',' read -ra GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
    # Limit to requested number
    GPU_LIST=()
    for ((i=0; i<NUM_GPUS && i<${#GPU_ARRAY[@]}; i++)); do
        GPU_LIST+=("${GPU_ARRAY[$i]}")
    done
else
    GPU_LIST=()
    for ((i=0; i<NUM_GPUS; i++)); do
        GPU_LIST+=("$i")
    done
fi

GPU_LIST_STR=$(IFS=,; echo "${GPU_LIST[*]}")

# Export variables for parallel workers
export CAM2BASE_PATH CONFIG_PATH SCRIPT_DIR GCS_BUCKET FAST_LOCAL_DIR PERMANENT_STORAGE_DIR TIMING_FILE
export NUM_GPUS WORKERS_PER_GPU GPU_LIST_STR
export STATUS_FILE ERROR_LOG

record_status() {
    local status=$1
    local episode=$2
    local stage=$3
    (
        flock -x 200
        printf "%s,%s,%s,%s\n" "$(date +%s)" "${status}" "${episode}" "${stage}" >> "${STATUS_FILE}"
    ) 200>"${STATUS_FILE}.lock"
}

record_error() {
    local episode=$1
    local stage=$2
    local message=$3
    (
        flock -x 201
        printf "[%s] %s | %s | %s\n" "$(date +'%Y-%m-%d %H:%M:%S')" "${episode}" "${stage}" "${message}" >> "${ERROR_LOG}"
    ) 201>"${ERROR_LOG}.lock"
}

export -f record_status record_error

# ============================================================================
# SETUP
# ============================================================================

mkdir -p "${LOG_DIR}"
mkdir -p "${FAST_LOCAL_DIR}"
mkdir -p "${PERMANENT_STORAGE_DIR}"

echo "=== DROID Training Data Pipeline (Multi-GPU Parallel) ==="
echo "Limit: ${LIMIT}"
echo "GPUs: ${NUM_GPUS} (${GPU_LIST_STR})"
echo "Workers/GPU: ${WORKERS_PER_GPU}"
echo "Total Workers: ${TOTAL_WORKERS}"
echo "Local Scratch: ${FAST_LOCAL_DIR}"
echo "Final Output: ${PERMANENT_STORAGE_DIR}"
echo "Log dir: ${LOG_DIR}"
echo ""

# Initialize timing CSV (now includes rendering time)
echo "episode_id,gpu_id,download_time_sec,rgb_depth_time_sec,tracks_time_sec,render_time_sec,total_time_sec" > "${TIMING_FILE}"

# ============================================================================
# WORKER FUNCTION (Multi-GPU Parallel Execution)
# ============================================================================

process_episode_worker() {
    local EPISODE_ID=$1
    local WORKER_NUM=$2          # Worker number (0 to TOTAL_WORKERS-1)
    local WORKER_ID=$BASHPID     # Unique Process ID for isolation
    local EPISODE_START=$(date +%s)
    local DEST_LOG_DIR="${LOG_DIR}/${EPISODE_ID}"
    
    # Determine which GPU this worker should use
    # Workers are distributed round-robin across GPUs
    IFS=',' read -ra GPU_ARRAY <<< "${GPU_LIST_STR}"
    local GPU_IDX=$((WORKER_NUM % NUM_GPUS))
    local ASSIGNED_GPU="${GPU_ARRAY[$GPU_IDX]}"
    
    # Set CUDA device for this worker
    export CUDA_VISIBLE_DEVICES="${ASSIGNED_GPU}"
    
    # Define unique local workspace for this worker
    local JOB_DIR="${FAST_LOCAL_DIR}/${EPISODE_ID}_${WORKER_ID}"
    local JOB_DATA="${JOB_DIR}/data"
    local JOB_OUTPUT="${JOB_DIR}/output"
    local JOB_LOGS="${JOB_DIR}/logs"
    local TEMP_CONFIG="${JOB_DIR}/config.yaml"
    
    mkdir -p "${JOB_DATA}" "${JOB_OUTPUT}" "${JOB_LOGS}" "${DEST_LOG_DIR}"

    # Create Temp Config for this worker
    # We override 'droid_root', 'download_dir', and 'output_root' to point to fast local storage
    cp "${CONFIG_PATH}" "${TEMP_CONFIG}"
    echo "" >> "${TEMP_CONFIG}"
    echo "# Worker Overrides" >> "${TEMP_CONFIG}"
    echo "droid_root: \"${JOB_DATA}\"" >> "${TEMP_CONFIG}"
    echo "download_dir: \"${JOB_DATA}\"" >> "${TEMP_CONFIG}"
    echo "output_root: \"${JOB_OUTPUT}\"" >> "${TEMP_CONFIG}"
    echo "log_dir: \"${LOG_DIR}/${EPISODE_ID}\"" >> "${TEMP_CONFIG}"
    echo "cam2base_extrinsics_path: \"${CAM2BASE_PATH}\"" >> "${TEMP_CONFIG}"

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
            cp -a "${JOB_LOGS}/." "${DEST_LOG_DIR}/" 2>/dev/null || true
            record_error "${EPISODE_ID}" "download" "Download failed (see ${DEST_LOG_DIR}/download.log)"
            record_status "failure" "${EPISODE_ID}" "download"
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
            cp -a "${JOB_LOGS}/." "${DEST_LOG_DIR}/" 2>/dev/null || true
            record_error "${EPISODE_ID}" "extract" "Extraction failed (see ${DEST_LOG_DIR}/rgb_depth.log)"
            record_status "failure" "${EPISODE_ID}" "extract"
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
            cp -a "${JOB_LOGS}/." "${DEST_LOG_DIR}/" 2>/dev/null || true
            record_error "${EPISODE_ID}" "tracks" "Tracking failed (see ${DEST_LOG_DIR}/tracks.log)"
            record_status "failure" "${EPISODE_ID}" "tracks"
            rm -rf "${JOB_DIR}"
            return 1
        }
    
    local TRACKS_END=$(date +%s)
    local TRACKS_TIME=$((TRACKS_END - TRACKS_START))
    
    # ------------------------------------------------------------------------
    # STEP D: Render Validation RRD and Videos (point_clouds folder)
    # ------------------------------------------------------------------------
    local RENDER_START=$(date +%s)
    
    python "${SCRIPT_DIR}/render_episode_validation.py" \
        --episode_id "${EPISODE_ID}" \
        --config "${TEMP_CONFIG}" \
        --headless \
        > "${JOB_LOGS}/render.log" 2>&1 || {
            # Don't fail the whole job if rendering fails - data is still valid
            record_error "${EPISODE_ID}" "render" "Rendering failed (non-fatal, see ${DEST_LOG_DIR}/render.log)"
        }
    
    local RENDER_END=$(date +%s)
    local RENDER_TIME=$((RENDER_END - RENDER_START))
    
    # ------------------------------------------------------------------------
    # STEP E: Move Results & Cleanup
    # ------------------------------------------------------------------------
    # Sync the generated output folder structure to permanent storage
    # rsync is safer than cp for merging directory trees
    rsync -a "${JOB_OUTPUT}/" "${PERMANENT_STORAGE_DIR}/"
    
    # Copy logs before cleanup
    cp -a "${JOB_LOGS}/." "${DEST_LOG_DIR}/" 2>/dev/null || true
    
    # Cleanup local scratch
    rm -rf "${JOB_DIR}"
    
    # ------------------------------------------------------------------------
    # Record Timing
    # ------------------------------------------------------------------------
    local EPISODE_END=$(date +%s)
    local TOTAL_TIME=$((EPISODE_END - EPISODE_START))
    
    # Atomic append to CSV (with GPU info and render time)
    echo "${EPISODE_ID},${ASSIGNED_GPU},${DOWNLOAD_TIME},${RGB_DEPTH_TIME},${TRACKS_TIME},${RENDER_TIME},${TOTAL_TIME}" >> "${TIMING_FILE}"
    record_status "success" "${EPISODE_ID}" "complete"
}

export -f process_episode_worker

# ============================================================================
# GET EPISODES
# ============================================================================

echo "[1/2] Getting episodes (balanced success/failure per lab)..."
python "${SCRIPT_DIR}/get_episodes_by_quality.py" \
    --cam2base "${CAM2BASE_PATH}" \
    --limit "${LIMIT}" \
    --gcs_bucket "${GCS_BUCKET}" \
    --output "${EPISODES_FILE}"

EPISODE_COUNT=$(wc -l < "${EPISODES_FILE}")
echo "Found ${EPISODE_COUNT} episodes to process"
echo ""

progress_monitor() {
    local total=$1
    local interval=${2:-5}
    while true; do
        local success=0
        local failure=0
        if [ -f "${STATUS_FILE}" ]; then
            success=$(awk -F, '$2=="success"{c++} END{print c+0}' "${STATUS_FILE}")
            failure=$(awk -F, '$2=="failure"{c++} END{print c+0}' "${STATUS_FILE}")
        fi
        local done=$((success + failure))
        local percent=0
        if [ "${total}" -gt 0 ]; then
            percent=$(( 100 * done / total ))
        fi
        printf "\r[progress] %d/%d (%d%%) | success: %d/%d | failures: %d/%d | workers: %d (per GPU: %d; GPUs: %s)" \
            "${done}" "${total}" "${percent}" \
            "${success}" "${total}" \
            "${failure}" "${total}" \
            "${TOTAL_WORKERS}" "${WORKERS_PER_GPU}" "${GPU_LIST_STR}"
        sleep "${interval}"
    done
}

PROGRESS_PID=""
start_progress_monitor() {
    progress_monitor "${EPISODE_COUNT}" 5 &
    PROGRESS_PID=$!
}

stop_progress_monitor() {
    if [ -n "${PROGRESS_PID}" ] && kill -0 "${PROGRESS_PID}" 2>/dev/null; then
        kill "${PROGRESS_PID}" 2>/dev/null || true
        wait "${PROGRESS_PID}" 2>/dev/null || true
        echo ""
    fi
}

trap stop_progress_monitor EXIT

# ============================================================================
# EXECUTE MULTI-GPU PARALLEL PIPELINE
# ============================================================================

echo "[2/2] Launching ${TOTAL_WORKERS} workers across ${NUM_GPUS} GPUs..."
echo "      Workers per GPU: ${WORKERS_PER_GPU}"
echo "      GPU IDs: ${GPU_LIST_STR}"
echo "      Monitoring output in ${TIMING_FILE}"
echo "============================================================"

start_progress_monitor

set +e
# Use GNU parallel if available (better load balancing), otherwise fallback to xargs
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel for optimal load balancing..."
    # GNU parallel with round-robin GPU assignment
    cat "${EPISODES_FILE}" | parallel -j "${TOTAL_WORKERS}" --line-buffer \
        'process_episode_worker {} {%}'
else
    # Fallback: xargs with numbered workers
    # We need to track worker numbers manually
    WORKER_COUNTER_FILE="${LOG_DIR}/worker_counter"
    echo "0" > "${WORKER_COUNTER_FILE}"
    
    get_next_worker_num() {
        # Atomic increment (using flock for thread safety)
        (
            flock -x 200
            local num=$(cat "${WORKER_COUNTER_FILE}")
            echo $((num + 1)) > "${WORKER_COUNTER_FILE}"
            echo "${num}"
        ) 200>"${WORKER_COUNTER_FILE}.lock"
    }
    export -f get_next_worker_num
    export WORKER_COUNTER_FILE
    
    process_episode_with_counter() {
        local EPISODE_ID=$1
        local WORKER_NUM=$(get_next_worker_num)
        process_episode_worker "${EPISODE_ID}" "${WORKER_NUM}"
    }
    export -f process_episode_with_counter
    
    cat "${EPISODES_FILE}" | xargs -P "${TOTAL_WORKERS}" -I {} bash -c 'process_episode_with_counter "$@"' _ {}
fi
PIPELINE_EXIT_CODE=$?
set -e

stop_progress_monitor

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE"
echo "============================================================"
SUCCESS_COUNT=0
FAILURE_COUNT=0
if [ -f "${STATUS_FILE}" ]; then
    SUCCESS_COUNT=$(awk -F, '$2=="success"{c++} END{print c+0}' "${STATUS_FILE}")
    FAILURE_COUNT=$(awk -F, '$2=="failure"{c++} END{print c+0}' "${STATUS_FILE}")
fi
echo "Successes: ${SUCCESS_COUNT}/${EPISODE_COUNT}"
echo "Failures : ${FAILURE_COUNT}/${EPISODE_COUNT}"
if [ -s "${ERROR_LOG}" ]; then
    echo "Failure details: ${ERROR_LOG}"
fi

PROCESSED_COUNT=$(wc -l < "${TIMING_FILE}")
# Adjust for header line
PROCESSED_COUNT=$((PROCESSED_COUNT - 1))

echo "Processed: ${PROCESSED_COUNT} episodes"
echo "GPUs used: ${NUM_GPUS} (${GPU_LIST_STR})"
echo "Workers per GPU: ${WORKERS_PER_GPU}"
echo "Total workers: ${TOTAL_WORKERS}"
echo "Timing log: ${TIMING_FILE}"
echo "Output dir: ${PERMANENT_STORAGE_DIR}"

if [ "${PROCESSED_COUNT}" -gt 0 ]; then
    # Calculate averages and per-GPU statistics using awk from the CSV
    awk -F',' 'NR>1 {
        sum_dl+=$3; sum_rgb+=$4; sum_tracks+=$5; sum_render+=$6; sum_total+=$7; count++;
        gpu_count[$2]++; gpu_time[$2]+=$7
    } END {
        if (count > 0) {
            print "\nFinal Averages:";
            printf "  Download:   %.1fs\n", sum_dl/count;
            printf "  RGB/Depth:  %.1fs\n", sum_rgb/count;
            printf "  Tracks:     %.1fs\n", sum_tracks/count;
            printf "  Render:     %.1fs\n", sum_render/count;
            printf "  Total:      %.1fs\n", sum_total/count;
            
            print "\nPer-GPU Statistics:";
            for (gpu in gpu_count) {
                printf "  GPU %s: %d episodes, avg %.1fs/episode\n", gpu, gpu_count[gpu], gpu_time[gpu]/gpu_count[gpu];
            }
        }
    }' "${TIMING_FILE}"
fi

exit "${PIPELINE_EXIT_CODE}"