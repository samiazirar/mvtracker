#!/bin/bash
# DROID Training Data Processing Pipeline (Multi-GPU Parallel Optimized)
# NPZ COMPRESSED MODE: RGB + depth stored as lossless .npz archives
#
# Downloads, extracts RGB/depth (as compressed .npz), and generates tracks for episodes
# Output format: rgb.npz (uint8 RGB frames), depth.npz (uint16 depth in mm)
#
# BATCH UPLOAD MODE: Episodes are staged locally and uploaded to HuggingFace
# every 10 minutes (configurable via BATCH_UPLOAD_INTERVAL) to avoid rate limits.
#
# Usage:
#   ./run_pipeline.sh                    # Process 10 episodes, auto-detect GPUs, 3 workers/GPU
#   ./run_pipeline.sh 100                # Process 100 episodes (workers/GPU=3, auto GPUs)
#   ./run_pipeline.sh 100 6              # Process 100 episodes, 6 workers/GPU, auto GPUs
#   ./run_pipeline.sh 100 6 4            # Process 100 episodes, 6 workers/GPU, 4 GPUs
#   ./run_pipeline.sh -1                 # Process all episodes
#
# Environment Variables:
#   CUDA_VISIBLE_DEVICES    Override which GPUs to use (e.g., "0,1,2,3")
#   DROID_WORKERS_PER_GPU   Workers per GPU (default: 3)
#   DROID_NUM_GPUS          Number of GPUs to use (default: auto-detect)
#   SKIP_HF_CHECK=1         Skip checking HuggingFace for existing episodes (fast mode)
#   BATCH_UPLOAD_INTERVAL   Seconds between batch uploads (default: 600 = 10 min)

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================

LIMIT=${1:-10}
WORKERS_PER_GPU=${2:-${DROID_WORKERS_PER_GPU:-3}}
NUM_GPUS=${3:-${DROID_NUM_GPUS:-0}}      # 0 = auto-detect
SKIP_HF_CHECK=${SKIP_HF_CHECK:-0}        # Set to 1 to skip HuggingFace existence checks (fast mode)

# Paths (all on fast /data storage)
CAM2BASE_PATH="/data/cam2base_extrinsic_superset.json"
CONFIG_PATH="conversions/droid/training_data/config.yaml"
SCRIPT_DIR="conversions/droid/training_data"
LOG_DIR="/data/logs/pipeline_huggingface_$(date +%Y%m%d_%H%M%S)"
EPISODES_FILE="${LOG_DIR}/episodes.txt"
TIMING_FILE="${LOG_DIR}/timing.csv"
STATUS_FILE="${LOG_DIR}/status.log"
ERROR_LOG="${LOG_DIR}/errors.log"
FAILED_EPISODES_FILE="${LOG_DIR}/failed_episodes.txt"
ERROR_COUNTS_FILE="${LOG_DIR}/error_counts.txt"
DEFAULT_INNER_FINGER_MESH="/data/robotiq_arg85_description/meshes/inner_finger_fine.STL"

# GCS bucket for downloads
GCS_BUCKET="gs://gresearch/robotics/droid_raw/1.0.1"

# Storage Configuration - ALL ON FAST /data SSD
# ----------------------------------------------------------------------------
FAST_LOCAL_DIR="/data/droid_scratch"      # Node-local fast storage (NVMe)
STAGING_DIR="/data/droid_staging"          # Staging area before HF upload
BATCH_UPLOAD_DIR="/data/droid_batch_upload" # Completed episodes waiting for batch upload
# ----------------------------------------------------------------------------

# Batch Upload Configuration
# ----------------------------------------------------------------------------
BATCH_UPLOAD_INTERVAL=${BATCH_UPLOAD_INTERVAL:-600}  # Upload every 10 minutes (600 seconds)
# ----------------------------------------------------------------------------

# Hugging Face Configuration
# ----------------------------------------------------------------------------
HF_REPO_ID="sazirarrwth99/trajectory_data"     # Your HF dataset repo
HF_REPO_TYPE="dataset"                      # Type: dataset, model, or space
# ----------------------------------------------------------------------------

# Load HF_TOKEN from .env file explicitly
ENV_FILE=".env"
if [ -f "${ENV_FILE}" ]; then
    # Extract HF_TOKEN from .env file (handles quotes)
    HF_TOKEN=$(grep -E '^HF_TOKEN=' "${ENV_FILE}" | sed 's/^HF_TOKEN=//; s/^"//; s/"$//')
    if [ -z "${HF_TOKEN}" ]; then
        echo "[ERROR] HF_TOKEN not found in ${ENV_FILE}"
        echo "Please add: HF_TOKEN=\"hf_your_token_here\" to ${ENV_FILE}"
        exit 1
    fi
    export HF_TOKEN
    echo "[INFO] Loaded HF_TOKEN from ${ENV_FILE}"
else
    echo "[ERROR] .env file not found at ${ENV_FILE}"
    exit 1
fi

# ----------------------------------------------------------------------------
# CREATE HUGGINGFACE REPO IF IT DOESN'T EXIST
# ----------------------------------------------------------------------------
echo "[INFO] Ensuring HuggingFace repo exists: ${HF_REPO_ID}..."
python3 << 'PYTHON_SCRIPT'
import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

token = os.environ['HF_TOKEN']
repo_id = os.environ.get('HF_REPO_ID', 'sazirarrwth99/trajectory_data')
repo_type = os.environ.get('HF_REPO_TYPE', 'dataset')

api = HfApi(token=token)

try:
    # Check if repo exists
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"[INFO] Repo '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f"[INFO] Creating new {repo_type} repo: {repo_id}")
    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=False,  # Set to True if you want a private repo
        token=token,
    )
    print(f"[SUCCESS] Created repo: https://huggingface.co/datasets/{repo_id}")
except Exception as e:
    print(f"[ERROR] Failed to check/create repo: {e}")
    exit(1)
PYTHON_SCRIPT

if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to create/verify HuggingFace repo"
    exit 1
fi
export HF_REPO_ID HF_REPO_TYPE
# ----------------------------------------------------------------------------



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
export CAM2BASE_PATH CONFIG_PATH SCRIPT_DIR GCS_BUCKET FAST_LOCAL_DIR STAGING_DIR BATCH_UPLOAD_DIR TIMING_FILE DEFAULT_INNER_FINGER_MESH
export NUM_GPUS WORKERS_PER_GPU GPU_LIST_STR LOG_DIR
export STATUS_FILE ERROR_LOG FAILED_EPISODES_FILE ERROR_COUNTS_FILE
export HF_TOKEN HF_REPO_ID HF_REPO_TYPE BATCH_UPLOAD_INTERVAL

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
        # Also record to failed episodes file for easy listing
        printf "%s,%s\n" "${episode}" "${stage}" >> "${FAILED_EPISODES_FILE}"
    ) 201>"${ERROR_LOG}.lock"
}

export -f record_status record_error

# ============================================================================
# SETUP
# ============================================================================

mkdir -p "${LOG_DIR}"
mkdir -p "${FAST_LOCAL_DIR}"
mkdir -p "${STAGING_DIR}"
mkdir -p "${BATCH_UPLOAD_DIR}"

echo "=== DROID Training Data Pipeline (NPZ Compressed + HuggingFace Upload) ==="
echo "Limit: ${LIMIT}"
echo "GPUs: ${NUM_GPUS} (${GPU_LIST_STR})"
echo "Workers/GPU: ${WORKERS_PER_GPU}"
echo "Total Workers: ${TOTAL_WORKERS}"
echo "Local Scratch: ${FAST_LOCAL_DIR}"
echo "Staging Dir: ${STAGING_DIR}"
echo "Batch Upload Dir: ${BATCH_UPLOAD_DIR}"
echo "Batch Upload Interval: ${BATCH_UPLOAD_INTERVAL}s (every $((BATCH_UPLOAD_INTERVAL / 60)) min)"
echo "HuggingFace Repo: ${HF_REPO_ID}"
echo "Log dir: ${LOG_DIR}"
echo ""

# Initialize timing CSV
echo "episode_id,gpu_id,prep_time_sec,download_time_sec,rgb_depth_time_sec,tracks_time_sec,sync_time_sec,cleanup_time_sec,total_time_sec" > "${TIMING_FILE}"

# ============================================================================
# WORKER FUNCTION (Multi-GPU Parallel Execution)
# ============================================================================

process_episode_worker() {
    local EPISODE_ID=$1
    local WORKER_NUM=$2          # Worker number (0 to TOTAL_WORKERS-1)
    local WORKER_ID=$BASHPID     # Unique Process ID for isolation
    local PIPELINE_START=$(date +%s)
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
    echo "finger_mesh_path: \"${DEFAULT_INNER_FINGER_MESH}\"" >> "${TEMP_CONFIG}"

    # Prep timing (directory + config setup)
    local PREP_END=$(date +%s)
    local PREP_TIME=$((PREP_END - PIPELINE_START))

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
    # STEP B: Extract RGB and Depth (GPU Step) - NPZ COMPRESSED MODE
    # Stores RGB and depth as lossless compressed .npz archives
    # ------------------------------------------------------------------------
    local RGB_DEPTH_START=$(date +%s)
    
    # Uses TEMP_CONFIG which points input/output to local scratch
    # --npz: save RGB + depth as compressed numpy archives (lossless)
    python "${SCRIPT_DIR}/extract_rgb_depth.py" \
        --episode_id "${EPISODE_ID}" \
        --config "${TEMP_CONFIG}" \
        --npz \
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
    # STEP D: Move to Batch Upload Directory & Cleanup
    # ------------------------------------------------------------------------
    # Move to batch upload directory instead of uploading directly
    # A background process will batch upload every BATCH_UPLOAD_INTERVAL seconds
    local SYNC_START=$(date +%s)
    
    # Move output to batch upload directory (atomic via rename within same filesystem)
    # Use rsync to merge into the batch upload directory (preserves existing content)
    rsync -a --remove-source-files "${JOB_OUTPUT}/" "${BATCH_UPLOAD_DIR}/" 2>"${JOB_LOGS}/sync.log" || {
        cp -a "${JOB_LOGS}/." "${DEST_LOG_DIR}/" 2>/dev/null || true
        record_error "${EPISODE_ID}" "sync" "Failed to move to batch upload dir (see ${DEST_LOG_DIR}/sync.log)"
        record_status "failure" "${EPISODE_ID}" "sync"
        rm -rf "${JOB_DIR}"
        return 1
    }
    echo "[BATCH] Staged ${EPISODE_ID} for batch upload" >> "${JOB_LOGS}/sync.log"
    
    local SYNC_END=$(date +%s)
    local SYNC_TIME=$((SYNC_END - SYNC_START))
    
    # Cleanup local scratch
    local CLEANUP_START=$(date +%s)
    cp -a "${JOB_LOGS}/." "${DEST_LOG_DIR}/" 2>/dev/null || true
    rm -rf "${JOB_DIR}"
    local CLEANUP_END=$(date +%s)
    local CLEANUP_TIME=$((CLEANUP_END - CLEANUP_START))
    
    # ------------------------------------------------------------------------
    # Record Timing
    # ------------------------------------------------------------------------
    local EPISODE_END=$(date +%s)
    local TOTAL_TIME=$((EPISODE_END - PIPELINE_START))
    
    # Atomic append to CSV (with GPU info)
    echo "${EPISODE_ID},${ASSIGNED_GPU},${PREP_TIME},${DOWNLOAD_TIME},${RGB_DEPTH_TIME},${TRACKS_TIME},${SYNC_TIME},${CLEANUP_TIME},${TOTAL_TIME}" >> "${TIMING_FILE}"
    record_status "success" "${EPISODE_ID}" "complete"
}

export -f process_episode_worker

# ============================================================================
# GET EPISODES
# ============================================================================
export EPISODES_FILE

echo "[1/2] Getting episodes sorted by quality..."
python "${SCRIPT_DIR}/get_episodes_by_quality.py" \
    --cam2base "${CAM2BASE_PATH}" \
    --limit "${LIMIT}" \
    --output "${EPISODES_FILE}"

EXISTING_LIST_FILE="${LOG_DIR}/hf_file_list.txt"
ALREADY_PROCESSED_FILE="${LOG_DIR}/already_processed.txt"
REMAINING_FILE="${LOG_DIR}/remaining_to_process.txt"

if [ "${SKIP_HF_CHECK}" -eq 1 ]; then
    echo "[1.5/2] SKIP_HF_CHECK=1: Skipping HuggingFace existence checks (fast mode)"
    echo "[HF] Processing all episodes without checking HuggingFace"
else
    echo "[1.5/2] Loading processed episodes from HuggingFace via git (fast skip check)..."
    echo "        (Set SKIP_HF_CHECK=1 to skip this entirely)"
    SKIP_START=$(date +%s)
    TEMP_GIT_DIR=$(mktemp -d)

    if git clone --depth 1 --filter=blob:none --no-checkout \
        "https://oauth2:${HF_TOKEN}@huggingface.co/datasets/${HF_REPO_ID}" \
        "${TEMP_GIT_DIR}" > /dev/null 2>&1; then
        pushd "${TEMP_GIT_DIR}" > /dev/null
        git ls-tree -r --name-only HEAD > "${EXISTING_LIST_FILE}"
        popd > /dev/null
        rm -rf "${TEMP_GIT_DIR}"

        FILE_COUNT=$(wc -l < "${EXISTING_LIST_FILE}")
        echo "      [SUCCESS] Retrieved list of ${FILE_COUNT} files."

        export EPISODES_FILE EXISTING_LIST_FILE ALREADY_PROCESSED_FILE REMAINING_FILE
        python3 << 'PYTHON_SCRIPT'
import os
import re
from datetime import datetime

episodes_file = os.environ["EPISODES_FILE"]
existing_list_file = os.environ["EXISTING_LIST_FILE"]
processed_file = os.environ["ALREADY_PROCESSED_FILE"]
remaining_file = os.environ["REMAINING_FILE"]

print("[HF] Building lookup table from git file list...")
processed_signatures = set()
with open(existing_list_file, "r") as f:
    for path in f:
        path = path.strip()
        parts = path.split("/")
        if len(parts) >= 4:
            lab, outcome, date, timestamp = parts[0], parts[1], parts[2], parts[3]
            if outcome in ("success", "failure"):
                processed_signatures.add(f"{lab}+{date}+{timestamp}")

def parse_episode_id(episode_id):
    parts = episode_id.split("+")
    if len(parts) != 3:
        return None
    m = re.match(r"(\d{4}-\d{2}-\d{2})-(\d+)h-(\d+)m-(\d+)s", parts[2])
    if not m:
        return None
    dt = datetime.strptime(
        f"{m.group(1)} {m.group(2)}:{m.group(3)}:{m.group(4)}",
        "%Y-%m-%d %H:%M:%S",
    )
    return {
        "lab": parts[0],
        "date": m.group(1),
        "timestamp_folder": dt.strftime("%a_%b_%e_%H:%M:%S_%Y"),
    }

already_processed = []
remaining = []
with open(episodes_file, "r") as f:
    for line in f:
        ep = line.strip()
        if not ep:
            continue
        parsed = parse_episode_id(ep)
        if parsed:
            sig = f"{parsed['lab']}+{parsed['date']}+{parsed['timestamp_folder']}"
            (already_processed if sig in processed_signatures else remaining).append(ep)
        else:
            remaining.append(ep)

with open(processed_file, "w") as f:
    f.write("\n".join(already_processed) + "\n")
with open(remaining_file, "w") as f:
    f.write("\n".join(remaining) + "\n")

print(f"[HF] Matched:   {len(already_processed)} (Already on HF)")
print(f"[HF] Remaining: {len(remaining)} (Need processing)")
PYTHON_SCRIPT

        cp "${REMAINING_FILE}" "${EPISODES_FILE}"
        PROCESSED_COUNT=$(wc -l < "${ALREADY_PROCESSED_FILE}")
        REMAINING_COUNT=$(wc -l < "${EPISODES_FILE}")
        echo "[HF] Skipped ${PROCESSED_COUNT} episodes already processed; remaining: ${REMAINING_COUNT}"
    else
        echo "[WARN] Could not fetch HuggingFace file list via git; processing all episodes"
        rm -rf "${TEMP_GIT_DIR}"
    fi
    SKIP_END=$(date +%s)
    SKIP_TIME=$((SKIP_END - SKIP_START))
    echo "[HF] Git fetch + filter time: ${SKIP_TIME}s"
fi

EPISODE_COUNT=$(wc -l < "${EPISODES_FILE}")
if [ "${EPISODE_COUNT}" -le 0 ]; then
    echo "[INFO] No episodes to process after skipping existing ones. Exiting."
    exit 0
fi
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

cleanup_on_exit() {
    stop_progress_monitor
    stop_batch_uploader
}
trap cleanup_on_exit EXIT

# ============================================================================
# BATCH UPLOADER (Background Process)
# ============================================================================

BATCH_UPLOAD_LOG="${LOG_DIR}/batch_upload.log"
BATCH_UPLOAD_LOCK="${BATCH_UPLOAD_DIR}/.upload.lock"
BATCH_UPLOAD_COUNT_FILE="${LOG_DIR}/batch_upload_count.txt"
echo "0" > "${BATCH_UPLOAD_COUNT_FILE}"
export BATCH_UPLOAD_LOG BATCH_UPLOAD_LOCK BATCH_UPLOAD_COUNT_FILE

do_batch_upload() {
    # Acquire lock to prevent concurrent uploads
    (
        flock -n 200 || {
            echo "[$(date +'%Y-%m-%d %H:%M:%S')] Skipping: another upload in progress" >> "${BATCH_UPLOAD_LOG}"
            return 0
        }
        
        # Check if there's anything to upload
        local file_count=$(find "${BATCH_UPLOAD_DIR}" -type f 2>/dev/null | wc -l)
        if [ "${file_count}" -eq 0 ]; then
            echo "[$(date +'%Y-%m-%d %H:%M:%S')] No files to upload" >> "${BATCH_UPLOAD_LOG}"
            return 0
        fi
        
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting batch upload of ${file_count} files..." >> "${BATCH_UPLOAD_LOG}"
        
        # Upload the batch
        python3 -c "
import os
import sys
from huggingface_hub import HfApi
from datetime import datetime

batch_dir = '${BATCH_UPLOAD_DIR}'
api = HfApi(token=os.environ['HF_TOKEN'])

try:
    api.upload_folder(
        folder_path=batch_dir,
        repo_id='${HF_REPO_ID}',
        repo_type='${HF_REPO_TYPE}',
        commit_message=f'Batch upload {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}',
        run_as_future=False,
    )
    print(f'[HF] Batch upload successful')
except Exception as e:
    print(f'[HF] Batch upload failed: {e}', file=sys.stderr)
    sys.exit(1)
" >> "${BATCH_UPLOAD_LOG}" 2>&1
        
        local upload_status=$?
        if [ ${upload_status} -eq 0 ]; then
            # Success: clear the batch upload directory
            echo "[$(date +'%Y-%m-%d %H:%M:%S')] Upload successful, clearing batch directory" >> "${BATCH_UPLOAD_LOG}"
            find "${BATCH_UPLOAD_DIR}" -mindepth 1 -delete 2>/dev/null || true
            
            # Increment upload counter
            local count=$(cat "${BATCH_UPLOAD_COUNT_FILE}")
            echo $((count + 1)) > "${BATCH_UPLOAD_COUNT_FILE}"
        else
            echo "[$(date +'%Y-%m-%d %H:%M:%S')] Upload failed, will retry next interval" >> "${BATCH_UPLOAD_LOG}"
        fi
    ) 200>"${BATCH_UPLOAD_LOCK}"
}
export -f do_batch_upload

batch_upload_loop() {
    local interval=$1
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Batch uploader started (interval: ${interval}s)" >> "${BATCH_UPLOAD_LOG}"
    while true; do
        sleep "${interval}"
        do_batch_upload
    done
}

BATCH_UPLOAD_PID=""
start_batch_uploader() {
    batch_upload_loop "${BATCH_UPLOAD_INTERVAL}" &
    BATCH_UPLOAD_PID=$!
    echo "[INFO] Started batch uploader (PID: ${BATCH_UPLOAD_PID}, interval: ${BATCH_UPLOAD_INTERVAL}s)"
}

stop_batch_uploader() {
    if [ -n "${BATCH_UPLOAD_PID}" ] && kill -0 "${BATCH_UPLOAD_PID}" 2>/dev/null; then
        kill "${BATCH_UPLOAD_PID}" 2>/dev/null || true
        wait "${BATCH_UPLOAD_PID}" 2>/dev/null || true
        echo "[INFO] Stopped batch uploader"
    fi
}

final_batch_upload() {
    echo ""
    echo "[INFO] Performing final batch upload..."
    do_batch_upload
    
    # Check if anything remains
    local remaining=$(find "${BATCH_UPLOAD_DIR}" -type f 2>/dev/null | wc -l)
    if [ "${remaining}" -gt 0 ]; then
        echo "[WARN] ${remaining} files still pending upload in ${BATCH_UPLOAD_DIR}"
        echo "[WARN] You may need to run the upload manually or retry"
    else
        echo "[INFO] All files uploaded successfully"
    fi
}

# ============================================================================
# EXECUTE MULTI-GPU PARALLEL PIPELINE
# ============================================================================

echo "[2/2] Launching ${TOTAL_WORKERS} workers across ${NUM_GPUS} GPUs..."
echo "      Workers per GPU: ${WORKERS_PER_GPU}"
echo "      GPU IDs: ${GPU_LIST_STR}"
echo "      Batch upload interval: ${BATCH_UPLOAD_INTERVAL}s"
echo "      Monitoring output in ${TIMING_FILE}"
echo "============================================================"

start_batch_uploader
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

stop_progress_monitor
stop_batch_uploader

# Perform final batch upload for any remaining files
final_batch_upload

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
echo ""
echo "=== SUCCESS / FAILURE ==="
echo "Successes: ${SUCCESS_COUNT}/${EPISODE_COUNT}"
echo "Failures : ${FAILURE_COUNT}/${EPISODE_COUNT}"
echo ""

# Show error breakdown by step
if [ -f "${FAILED_EPISODES_FILE}" ] && [ -s "${FAILED_EPISODES_FILE}" ]; then
    echo "=== ERRORS BY STEP ==="
    # Count errors by step
    DOWNLOAD_ERRORS=$(awk -F, '$2=="download"{c++} END{print c+0}' "${FAILED_EPISODES_FILE}")
    EXTRACT_ERRORS=$(awk -F, '$2=="extract"{c++} END{print c+0}' "${FAILED_EPISODES_FILE}")
    TRACKS_ERRORS=$(awk -F, '$2=="tracks"{c++} END{print c+0}' "${FAILED_EPISODES_FILE}")
    SYNC_ERRORS=$(awk -F, '$2=="sync"{c++} END{print c+0}' "${FAILED_EPISODES_FILE}")
    
    echo "  download:  ${DOWNLOAD_ERRORS}"
    echo "  extract:   ${EXTRACT_ERRORS}"
    echo "  tracks:    ${TRACKS_ERRORS}"
    echo "  sync:      ${SYNC_ERRORS}"
    echo ""
    
    echo "=== FAILED RUNS ==="
    echo "(episode_id, failed_step)"
    echo "------------------------------------------------------------"
    cat "${FAILED_EPISODES_FILE}"
    echo "------------------------------------------------------------"
    echo ""
    echo "Failed episodes file: ${FAILED_EPISODES_FILE}"
fi

if [ -s "${ERROR_LOG}" ]; then
    echo ""
    echo "=== DETAILED ERROR LOG ==="
    echo "Full error log: ${ERROR_LOG}"
    echo ""
    echo "Last 10 errors:"
    tail -n 10 "${ERROR_LOG}"
fi

# Show timing statistics
PROCESSED_COUNT=$(wc -l < "${TIMING_FILE}")
# Adjust for header line
PROCESSED_COUNT=$((PROCESSED_COUNT - 1))

echo ""
echo "=== TIMING STATISTICS ==="
echo "Processed: ${PROCESSED_COUNT} episodes"
echo "GPUs used: ${NUM_GPUS} (${GPU_LIST_STR})"
echo "Workers per GPU: ${WORKERS_PER_GPU}"
echo "Total workers: ${TOTAL_WORKERS}"
echo "Timing log: ${TIMING_FILE}"
echo "HuggingFace Repo: ${HF_REPO_ID}"

# Show batch upload statistics
BATCH_UPLOAD_COUNT=$(cat "${BATCH_UPLOAD_COUNT_FILE}" 2>/dev/null || echo "0")
echo "Batch uploads completed: ${BATCH_UPLOAD_COUNT}"
echo "Batch upload log: ${BATCH_UPLOAD_LOG}"

if [ "${PROCESSED_COUNT}" -gt 0 ]; then
    # Calculate averages and per-GPU statistics using awk from the CSV
    awk -F',' 'NR>1 {
        sum_prep+=$3; sum_dl+=$4; sum_rgb+=$5; sum_tracks+=$6; sum_sync+=$7; sum_cleanup+=$8; sum_total+=$9; count++;
        gpu_count[$2]++; gpu_time[$2]+=$9
    } END {
        if (count > 0) {
            print "\nFinal Averages:";
            printf "  Prep:       %.1fs\n", sum_prep/count;
            printf "  Download:   %.1fs\n", sum_dl/count;
            printf "  RGB/Depth:  %.1fs\n", sum_rgb/count;
            printf "  Tracks:     %.1fs\n", sum_tracks/count;
            printf "  Sync:       %.1fs\n", sum_sync/count;
            printf "  Cleanup:    %.1fs\n", sum_cleanup/count;
            printf "  Total:      %.1fs\n", sum_total/count;
            
            print "\nPer-GPU Statistics:";
            for (gpu in gpu_count) {
                printf "  GPU %s: %d episodes, avg %.1fs/episode\n", gpu, gpu_count[gpu], gpu_time[gpu]/gpu_count[gpu];
            }
        }
    }' "${TIMING_FILE}"
fi

exit "${PIPELINE_EXIT_CODE}"
