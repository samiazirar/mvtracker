#!/bin/bash
# DROID Debug Pipeline (Sequential, Fault Tolerant, Unbuffered)
# - Runs episodes one by one
# - Prints directly to screen (no log files)
# - If an episode fails, it prints the error and moves to the next one

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

# Paths
CAM2BASE_PATH="/data/cam2base_extrinsic_superset.json"
CONFIG_PATH="conversions/droid/training_data/config.yaml"
SCRIPT_DIR="conversions/droid/training_data"
LOG_DIR="logs/debug_$(date +%Y%m%d_%H%M%S)"
EPISODES_FILE="${LOG_DIR}/episodes.txt"

# Storage
FAST_LOCAL_DIR="/data/droid_debug_scratch"
PERMANENT_STORAGE_DIR="./droid_processed"
GCS_BUCKET="gs://gresearch/robotics/droid_raw/1.0.1"

mkdir -p "${LOG_DIR}"
mkdir -p "${FAST_LOCAL_DIR}"
mkdir -p "${PERMANENT_STORAGE_DIR}"

export CAM2BASE_PATH CONFIG_PATH SCRIPT_DIR GCS_BUCKET FAST_LOCAL_DIR PERMANENT_STORAGE_DIR

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

# Disable exit-on-error so the loop continues even if python fails
set +e 

while read -r EPISODE_ID; do
    echo ""
    echo "################################################################"
    echo "PROCESSING: ${EPISODE_ID}"
    echo "################################################################"

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
    } >> "${TEMP_CONFIG}"

    # --- BLOCK 1: DOWNLOAD ---
    echo ">>> Step 1: Downloading..."
    if ! python "${SCRIPT_DIR}/download_single_episode.py" \
        --episode_id "${EPISODE_ID}" \
        --cam2base "${CAM2BASE_PATH}" \
        --output_dir "${JOB_DATA}" \
        --gcs_bucket "${GCS_BUCKET}"; then
        
        echo "!!!! [ERROR] Download Failed for ${EPISODE_ID}. Skipping..."
        rm -rf "${JOB_DIR}"
        continue
    fi

    # --- BLOCK 2: EXTRACT ---
    echo ">>> Step 2: Extracting RGB/Depth..."
    if ! python "${SCRIPT_DIR}/extract_rgb_depth.py" \
        --episode_id "${EPISODE_ID}" \
        --config "${TEMP_CONFIG}"; then
        
        echo "!!!! [ERROR] Extraction Failed for ${EPISODE_ID}. Skipping..."
        rm -rf "${JOB_DIR}"
        continue
    fi

    # --- BLOCK 3: TRACKS ---
    echo ">>> Step 3: Generating Tracks..."
    if ! python "${SCRIPT_DIR}/generate_tracks_and_metadata.py" \
        --episode_id "${EPISODE_ID}" \
        --config "${TEMP_CONFIG}"; then
        
        echo "!!!! [ERROR] Tracking Failed for ${EPISODE_ID}. Skipping..."
        rm -rf "${JOB_DIR}"
        continue
    fi

    # --- BLOCK 4: FINALIZE ---
    echo ">>> Step 4: Moving data to permanent storage..."
    rsync -a "${JOB_OUTPUT}/" "${PERMANENT_STORAGE_DIR}/"
    
    # CLEANUP
    echo ">>> Cleanup..."
    rm -rf "${JOB_DIR}"
    
    echo ">>> [SUCCESS] Finished ${EPISODE_ID}"

done < "${EPISODES_FILE}"

echo ""
echo "Debug run complete."