#!/bin/bash
# DROID Debug Pipeline (Sequential, Fault Tolerant, Quiet)
# - Runs episodes one by one
# - Streams step output to per-episode logs to keep stdout clean
# - If an episode fails, records it and continues to the next one

# Force Python to flush print statements immediately
export PYTHONUNBUFFERED=1

# Stop script if setup fails (variables, basic paths)
set -e

# ============================================================================
# CONFIGURATION
# ============================================================================

LIMIT=${1:-10} # Default to 10 episodes
echo "Running Debug Pipeline for ${LIMIT} episode(s)..."

# Hardcoded GPU 0
export CUDA_VISIBLE_DEVICES=0
WORKER_ID=${WORKER_ID:-1}
TOTAL_WORKERS=${TOTAL_WORKERS:-1}

# Paths
CAM2BASE_PATH="/data/cam2base_extrinsic_superset.json"
CONFIG_PATH="conversions/droid/training_data/config.yaml"
SCRIPT_DIR="conversions/droid/training_data"
LOG_DIR="logs/debug_$(date +%Y%m%d_%H%M%S)"
EPISODES_FILE="${LOG_DIR}/episodes.txt"
FAILURE_LOG="${LOG_DIR}/failures.log"

# Storage
FAST_LOCAL_DIR="/data/droid_debug_scratch"
PERMANENT_STORAGE_DIR="./droid_processed"
GCS_BUCKET="gs://gresearch/robotics/droid_raw/1.0.1"
DEFAULT_INNER_FINGER_MESH="/data/robotiq_arg85_description/meshes/inner_finger_fine.STL"
mkdir -p "${LOG_DIR}"
mkdir -p "${FAST_LOCAL_DIR}"
mkdir -p "${PERMANENT_STORAGE_DIR}"

export CAM2BASE_PATH CONFIG_PATH SCRIPT_DIR GCS_BUCKET FAST_LOCAL_DIR PERMANENT_STORAGE_DIR DEFAULT_INNER_FINGER_MESH

# ============================================================================
# PRE-FLIGHT: DOWNLOAD EXTRINSICS
# ============================================================================
if [ ! -f "${CAM2BASE_PATH}" ]; then
    echo "[INFO] Downloading Extrinsics..."
    mkdir -p "$(dirname "${CAM2BASE_PATH}")"
    TEMP_CLONE_DIR=$(mktemp -d)
    git clone --depth 1 https://huggingface.co/KarlP/droid "${TEMP_CLONE_DIR}"
    pushd "${TEMP_CLONE_DIR}" > /dev/null
    git lfs install
    git lfs pull
    popd > /dev/null
    mv "${TEMP_CLONE_DIR}/$(basename "${CAM2BASE_PATH}")" "${CAM2BASE_PATH}"
    rm -rf "${TEMP_CLONE_DIR}"
fi

# ============================================================================
# STEP 1: GET EPISODE LIST
# ============================================================================
echo "[INFO] Fetching episode list..."
python "${SCRIPT_DIR}/get_episodes_by_quality.py" \
    --cam2base "${CAM2BASE_PATH}" \
    --limit "${LIMIT}" \
    --output "${EPISODES_FILE}"

echo "List generated. Starting Processing Loop..."
echo "----------------------------------------------------------------"

# ============================================================================
# STEP 2: PROCESSING LOOP (SEQUENTIAL & FAULT TOLERANT)
# ============================================================================

# Progress tracking helpers
BAR_WIDTH=30
PROCESSED=0
SUCCEEDED=0
FAILED=0

progress_bar() {
    if (( TOTAL_EPISODES == 0 )); then
        return
    fi

    local percent=$(( PROCESSED * 100 / TOTAL_EPISODES ))
    local filled=$(( BAR_WIDTH * PROCESSED / TOTAL_EPISODES ))
    local empty=$(( BAR_WIDTH - filled ))
    local bar=""

    for ((i = 0; i < filled; i++)); do
        bar+="#"
    done
    for ((i = 0; i < empty; i++)); do
        bar+="-"
    done

    printf "\r[Worker %s/%s] [%s] %3d%% | processed %d/%d | success %d | failed %d" \
        "${WORKER_ID}" "${TOTAL_WORKERS}" "${bar}" "${percent}" "${PROCESSED}" "${TOTAL_EPISODES}" "${SUCCEEDED}" "${FAILED}"
}

# Disable exit-on-error so the loop continues even if python fails
set +e 

# Prepare episode list and counters
TOTAL_EPISODES=$(wc -l < "${EPISODES_FILE}")
: > "${FAILURE_LOG}"
echo "[INFO] Worker ${WORKER_ID}/${TOTAL_WORKERS} handling ${TOTAL_EPISODES} episode(s)."

if (( TOTAL_EPISODES == 0 )); then
    echo "[ERROR] No episodes found in ${EPISODES_FILE}. Exiting."
    exit 1
fi

while read -r EPISODE_ID; do
    echo ""
    EPISODE_LOG="${LOG_DIR}/${EPISODE_ID}.log"
    echo "[INFO] Worker ${WORKER_ID}/${TOTAL_WORKERS} processing ${EPISODE_ID} (log: ${EPISODE_LOG})"

    # Define Workspace
    JOB_DIR="${FAST_LOCAL_DIR}/${EPISODE_ID}_debug"
    JOB_DATA="${JOB_DIR}/data"
    JOB_OUTPUT="${JOB_DIR}/output"
    TEMP_CONFIG="${JOB_DIR}/config.yaml"

    mkdir -p "${JOB_DATA}" "${JOB_OUTPUT}"

    # Generate Config Override
    cp "${CONFIG_PATH}" "${TEMP_CONFIG}"
    {
        echo ""
        echo "# Debug Overrides"
        echo "droid_root: \"${JOB_DATA}\""
        echo "download_dir: \"${JOB_DATA}\""
        echo "output_root: \"${JOB_OUTPUT}\""
        echo "log_dir: \"${LOG_DIR}/${EPISODE_ID}\""
        echo "cam2base_extrinsics_path: \"${CAM2BASE_PATH}\""
        echo "finger_mesh_path: \"${DEFAULT_INNER_FINGER_MESH}\""
    } >> "${TEMP_CONFIG}"

    # --- BLOCK 1: DOWNLOAD ---
    if ! python "${SCRIPT_DIR}/download_single_episode.py" \
        --episode_id "${EPISODE_ID}" \
        --cam2base "${CAM2BASE_PATH}" \
        --output_dir "${JOB_DATA}" \
        --gcs_bucket "${GCS_BUCKET}" \
        >> "${EPISODE_LOG}" 2>&1; then
        
        echo "[${EPISODE_ID}] download failed (see ${EPISODE_LOG})" >> "${FAILURE_LOG}"
        FAILED=$((FAILED + 1))
        PROCESSED=$((PROCESSED + 1))
        progress_bar
        rm -rf "${JOB_DIR}"
        continue
    fi

    # --- BLOCK 2: EXTRACT ---
    if ! python "${SCRIPT_DIR}/extract_rgb_depth.py" \
        --episode_id "${EPISODE_ID}" \
        --config "${TEMP_CONFIG}" \
        >> "${EPISODE_LOG}" 2>&1; then
        
        echo "[${EPISODE_ID}] extraction failed (see ${EPISODE_LOG})" >> "${FAILURE_LOG}"
        FAILED=$((FAILED + 1))
        PROCESSED=$((PROCESSED + 1))
        progress_bar
        rm -rf "${JOB_DIR}"
        continue
    fi

    # --- BLOCK 3: TRACKS ---
    if ! python "${SCRIPT_DIR}/generate_tracks_and_metadata.py" \
        --episode_id "${EPISODE_ID}" \
        --config "${TEMP_CONFIG}" \
        >> "${EPISODE_LOG}" 2>&1; then
        
        echo "[${EPISODE_ID}] tracking failed (see ${EPISODE_LOG})" >> "${FAILURE_LOG}"
        FAILED=$((FAILED + 1))
        PROCESSED=$((PROCESSED + 1))
        progress_bar
        rm -rf "${JOB_DIR}"
        continue
    fi

    # --- BLOCK 4: FINALIZE ---
    rsync -a "${JOB_OUTPUT}/" "${PERMANENT_STORAGE_DIR}/" >> "${EPISODE_LOG}" 2>&1
    
    # CLEANUP
    rm -rf "${JOB_DIR}" >> "${EPISODE_LOG}" 2>&1
    
    SUCCEEDED=$((SUCCEEDED + 1))
    PROCESSED=$((PROCESSED + 1))
    progress_bar
done < "${EPISODES_FILE}"

echo ""
echo ""
echo "Debug run complete."
echo "Success: ${SUCCEEDED}/${TOTAL_EPISODES}"
echo "Failures: ${FAILED}/${TOTAL_EPISODES}"
if (( FAILED > 0 )); then
    echo "Failure log: ${FAILURE_LOG}"
fi
