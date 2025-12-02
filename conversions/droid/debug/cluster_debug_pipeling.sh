#!/bin/bash
# DROID Debug Pipeline (Serial, Single GPU, Verbose)
set -e  # Exit immediately if a command exits with a non-zero status

# ============================================================================
# CONFIGURATION
# ============================================================================

LIMIT=${1:-1} # Default to just 1 episode for testing
echo "Running Debug Pipeline for ${LIMIT} episode(s)..."

# Hardcoded GPU 0 for debugging
export CUDA_VISIBLE_DEVICES=0

# Paths (Kept same as original)
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

# Export vars needed by python scripts
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
# STEP 2: PROCESSING LOOP (SERIAL)
# ============================================================================

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
    } >> "${TEMP_CONFIG}"

    # 1. DOWNLOAD
    echo ">>> Step 1: Downloading..."
    python "${SCRIPT_DIR}/download_single_episode.py" \
        --episode_id "${EPISODE_ID}" \
        --cam2base "${CAM2BASE_PATH}" \
        --output_dir "${JOB_DATA}" \
        --gcs_bucket "${GCS_BUCKET}"

    # 2. EXTRACT (GPU)
    echo ">>> Step 2: Extracting RGB/Depth (Using GPU)..."
    python "${SCRIPT_DIR}/extract_rgb_depth.py" \
        --episode_id "${EPISODE_ID}" \
        --config "${TEMP_CONFIG}"

    # 3. TRACKS (CPU)
    echo ">>> Step 3: Generating Tracks..."
    python "${SCRIPT_DIR}/generate_tracks_and_metadata.py" \
        --episode_id "${EPISODE_ID}" \
        --config "${TEMP_CONFIG}"

    # 4. FINALIZE
    echo ">>> Step 4: Moving data to permanent storage..."
    rsync -a "${JOB_OUTPUT}/" "${PERMANENT_STORAGE_DIR}/"
    
    # CLEANUP
    echo ">>> Cleanup..."
    rm -rf "${JOB_DIR}"
    
    echo ">>> [SUCCESS] Finished ${EPISODE_ID}"

done < "${EPISODES_FILE}"

echo ""
echo "Debug run complete."