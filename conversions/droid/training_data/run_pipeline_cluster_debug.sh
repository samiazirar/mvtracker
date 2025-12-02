#!/bin/bash
# DROID Training Data Processing Pipeline (Multi-GPU) with high-resolution timing.
# Mirrors run_pipeline_cluster.sh behavior but records millisecond timings for every stage.
#
# Usage: same as run_pipeline_cluster.sh
#   ./run_pipeline_cluster_debug.sh [LIMIT] [WORKERS_PER_GPU] [NUM_GPUS]

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================

LIMIT=${1:-10}
WORKERS_PER_GPU=${2:-${DROID_WORKERS_PER_GPU:-3}}
NUM_GPUS=${3:-${DROID_NUM_GPUS:-0}}      # 0 = auto-detect

# Paths
CAM2BASE_PATH="/data/cam2base_extrinsic_superset.json"
CONFIG_PATH="conversions/droid/training_data/config.yaml"
SCRIPT_DIR="conversions/droid/training_data"
LOG_DIR="logs/pipeline_debug_$(date +%Y%m%d_%H%M%S)"
EPISODES_FILE="${LOG_DIR}/episodes.txt"
TIMING_FILE="${LOG_DIR}/timing_ms.csv"
STATUS_FILE="${LOG_DIR}/status.log"
ERROR_LOG="${LOG_DIR}/errors.log"
DEFAULT_INNER_FINGER_MESH="/data/robotiq_arg85_description/meshes/inner_finger_fine.STL"

# GCS bucket for downloads
GCS_BUCKET="gs://gresearch/robotics/droid_raw/1.0.1"

# Storage Configuration
FAST_LOCAL_DIR="/data/"
PERMANENT_STORAGE_DIR="./droid_processed"

# ============================================================================
# HELPERS
# ============================================================================

ts_ms() {
    date +%s%3N
}

duration_ms() {
    local start=$1
    local end=$2
    echo $((end - start))
}

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

export -f record_status record_error ts_ms duration_ms

# ============================================================================
# AUTO-DOWNLOAD EXTRINSICS (Hugging Face / Git LFS)
# ============================================================================
if [ ! -f "${CAM2BASE_PATH}" ]; then
    echo "[INFO] Extrinsic file not found at: ${CAM2BASE_PATH}"
    echo "[INFO] Downloading from Hugging Face (KarlP/droid)..."

    mkdir -p "$(dirname "${CAM2BASE_PATH}")"

    if ! command -v git-lfs &> /dev/null; then
        echo "[ERROR] git-lfs is required but not installed."
        exit 1
    fi

    TEMP_CLONE_DIR=$(mktemp -d)
    echo "[INFO] Cloning repo to ${TEMP_CLONE_DIR}..."
    git clone --depth 1 https://huggingface.co/KarlP/droid "${TEMP_CLONE_DIR}"
    pushd "${TEMP_CLONE_DIR}" > /dev/null
    git lfs install
    git lfs pull
    popd > /dev/null

    TARGET_FILENAME=$(basename "${CAM2BASE_PATH}")
    if [ -f "${TEMP_CLONE_DIR}/${TARGET_FILENAME}" ]; then
        mv "${TEMP_CLONE_DIR}/${TARGET_FILENAME}" "${CAM2BASE_PATH}"
        echo "[SUCCESS] Downloaded ${TARGET_FILENAME} to ${CAM2BASE_PATH}"
    else
        echo "[ERROR] File '${TARGET_FILENAME}' not found in Hugging Face repo."
        rm -rf "${TEMP_CLONE_DIR}"
        exit 1
    fi
    rm -rf "${TEMP_CLONE_DIR}"
fi

# ============================================================================
# GPU DETECTION AND SETUP
# ============================================================================

detect_gpus() {
    if command -v nvidia-smi &> /dev/null; then
        local gpu_count
        gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
        if [ "${gpu_count}" -gt 0 ]; then
            echo "${gpu_count}"
            return
        fi
    fi
    local dev_count
    dev_count=$(ls /dev/nvidia[0-9]* 2>/dev/null | wc -l)
    if [ "${dev_count}" -gt 0 ]; then
        echo "${dev_count}"
        return
    fi
    echo "0"
}

if [ "${NUM_GPUS}" -eq 0 ]; then
    NUM_GPUS=$(detect_gpus)
    if [ "${NUM_GPUS}" -eq 0 ]; then
        echo "[ERROR] No GPUs detected! Set CUDA_VISIBLE_DEVICES or DROID_NUM_GPUS"
        exit 1
    fi
fi

if [ "${NUM_GPUS}" -gt 8 ]; then
    echo "[WARN] Capping NUM_GPUS from ${NUM_GPUS} to 8"
    NUM_GPUS=8
fi

TOTAL_WORKERS=$((NUM_GPUS * WORKERS_PER_GPU))

if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
    IFS=',' read -ra GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
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

export CAM2BASE_PATH CONFIG_PATH SCRIPT_DIR GCS_BUCKET FAST_LOCAL_DIR PERMANENT_STORAGE_DIR TIMING_FILE DEFAULT_INNER_FINGER_MESH
export NUM_GPUS WORKERS_PER_GPU GPU_LIST_STR STATUS_FILE ERROR_LOG

# ============================================================================
# SETUP
# ============================================================================

mkdir -p "${LOG_DIR}" "${FAST_LOCAL_DIR}" "${PERMANENT_STORAGE_DIR}"

echo "=== DROID Training Data Pipeline (Debug Timing) ==="
echo "Limit: ${LIMIT}"
echo "GPUs: ${NUM_GPUS} (${GPU_LIST_STR})"
echo "Workers/GPU: ${WORKERS_PER_GPU}"
echo "Total Workers: ${TOTAL_WORKERS}"
echo "Local Scratch: ${FAST_LOCAL_DIR}"
echo "Final Output: ${PERMANENT_STORAGE_DIR}"
echo "Log dir: ${LOG_DIR}"
echo ""

echo "episode_id,gpu_id,prep_ms,download_ms,rgb_depth_ms,tracks_ms,sync_ms,cleanup_ms,total_ms" > "${TIMING_FILE}"

# ============================================================================
# WORKER FUNCTION
# ============================================================================

process_episode_worker() {
    local EPISODE_ID=$1
    local WORKER_NUM=$2
    local WORKER_ID=$BASHPID
    local DEST_LOG_DIR="${LOG_DIR}/${EPISODE_ID}"

    IFS=',' read -ra GPU_ARRAY <<< "${GPU_LIST_STR}"
    local GPU_IDX=$((WORKER_NUM % NUM_GPUS))
    local ASSIGNED_GPU="${GPU_ARRAY[$GPU_IDX]}"
    export CUDA_VISIBLE_DEVICES="${ASSIGNED_GPU}"

    local PIPE_START_MS
    PIPE_START_MS=$(ts_ms)

    local JOB_DIR="${FAST_LOCAL_DIR}/${EPISODE_ID}_${WORKER_ID}"
    local JOB_DATA="${JOB_DIR}/data"
    local JOB_OUTPUT="${JOB_DIR}/output"
    local JOB_LOGS="${JOB_DIR}/logs"
    local TEMP_CONFIG="${JOB_DIR}/config.yaml"

    mkdir -p "${JOB_DATA}" "${JOB_OUTPUT}" "${JOB_LOGS}"
    mkdir -p "${DEST_LOG_DIR}"

    cp "${CONFIG_PATH}" "${TEMP_CONFIG}"
    {
        echo ""
        echo "# Worker Overrides"
        echo "droid_root: \"${JOB_DATA}\""
        echo "download_dir: \"${JOB_DATA}\""
        echo "output_root: \"${JOB_OUTPUT}\""
        echo "log_dir: \"${LOG_DIR}/${EPISODE_ID}\""
        echo "cam2base_extrinsics_path: \"${CAM2BASE_PATH}\""
        echo "finger_mesh_path: \"${DEFAULT_INNER_FINGER_MESH}\""
    } >> "${TEMP_CONFIG}"

    local PREP_END_MS
    PREP_END_MS=$(ts_ms)
    local PREP_MS
    PREP_MS=$(duration_ms "${PIPE_START_MS}" "${PREP_END_MS}")

    # Download
    local DL_START_MS
    DL_START_MS=$(ts_ms)
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
    local DL_END_MS
    DL_END_MS=$(ts_ms)
    local DL_MS
    DL_MS=$(duration_ms "${DL_START_MS}" "${DL_END_MS}")

    # Extract RGB/Depth
    local RGB_START_MS
    RGB_START_MS=$(ts_ms)
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
    local RGB_END_MS
    RGB_END_MS=$(ts_ms)
    local RGB_MS
    RGB_MS=$(duration_ms "${RGB_START_MS}" "${RGB_END_MS}")

    # Tracks
    local TRACK_START_MS
    TRACK_START_MS=$(ts_ms)
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
    local TRACK_END_MS
    TRACK_END_MS=$(ts_ms)
    local TRACK_MS
    TRACK_MS=$(duration_ms "${TRACK_START_MS}" "${TRACK_END_MS}")

    # Sync
    local SYNC_START_MS
    SYNC_START_MS=$(ts_ms)
    rsync -a "${JOB_OUTPUT}/" "${PERMANENT_STORAGE_DIR}/"
    local SYNC_END_MS
    SYNC_END_MS=$(ts_ms)
    local SYNC_MS
    SYNC_MS=$(duration_ms "${SYNC_START_MS}" "${SYNC_END_MS}")

    # Cleanup
    local CLEANUP_START_MS
    CLEANUP_START_MS=$(ts_ms)
    rm -rf "${JOB_DIR}"
    local CLEANUP_END_MS
    CLEANUP_END_MS=$(ts_ms)
    local CLEANUP_MS
    CLEANUP_MS=$(duration_ms "${CLEANUP_START_MS}" "${CLEANUP_END_MS}")

    # Persist logs before cleanup removal
    cp -a "${JOB_LOGS}/." "${DEST_LOG_DIR}/" 2>/dev/null || true

    local PIPE_END_MS
    PIPE_END_MS=$(ts_ms)
    local TOTAL_MS
    TOTAL_MS=$(duration_ms "${PIPE_START_MS}" "${PIPE_END_MS}")

    echo "${EPISODE_ID},${ASSIGNED_GPU},${PREP_MS},${DL_MS},${RGB_MS},${TRACK_MS},${SYNC_MS},${CLEANUP_MS},${TOTAL_MS}" >> "${TIMING_FILE}"
    record_status "success" "${EPISODE_ID}" "complete"
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
# EXECUTE
# ============================================================================

echo "[2/2] Launching ${TOTAL_WORKERS} workers across ${NUM_GPUS} GPUs..."
echo "      Workers per GPU: ${WORKERS_PER_GPU}"
echo "      GPU IDs: ${GPU_LIST_STR}"
echo "      Monitoring output in ${TIMING_FILE}"
echo "============================================================"

start_progress_monitor

set +e
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel for optimal load balancing..."
    cat "${EPISODES_FILE}" | parallel -j "${TOTAL_WORKERS}" --line-buffer \
        'process_episode_worker {} {%}'
else
    WORKER_COUNTER_FILE="${LOG_DIR}/worker_counter"
    echo "0" > "${WORKER_COUNTER_FILE}"
    
    get_next_worker_num() {
        (
            flock -x 200
            local num
            num=$(cat "${WORKER_COUNTER_FILE}")
            echo $((num + 1)) > "${WORKER_COUNTER_FILE}"
            echo "${num}"
        ) 200>"${WORKER_COUNTER_FILE}.lock"
    }
    export -f get_next_worker_num
    export WORKER_COUNTER_FILE
    
    process_episode_with_counter() {
        local EPISODE_ID=$1
        local WORKER_NUM
        WORKER_NUM=$(get_next_worker_num)
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

exit "${PIPELINE_EXIT_CODE}"