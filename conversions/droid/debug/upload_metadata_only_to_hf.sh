#!/bin/bash
# Upload only metadata files (tracks.npz, extrinsics.npz, quality.json) to a separate HF repo.
#
# This script filters a processed DROID output directory and uploads only the small
# metadata files, excluding large video/image data. Useful when full data is too large
# for HuggingFace Hub.
#
# Usage:
#   conversions/droid/debug/upload_metadata_only_to_hf.sh [SOURCE_DIR] [LOG_DIR]
#
# Defaults:
#   SOURCE_DIR: /data/droid_batch_upload
#   LOG_DIR:    /data/logs/metadata_hf_upload_<timestamp>
#
# Environment variables:
#   HF_REPO_ID           Target dataset repo (default: sazirarrwth99/trajectories_droid)
#   HF_REPO_TYPE         Repo type (default: dataset)
#   COMMIT_MESSAGE       Optional commit message for the upload
#   CLEAR_STAGING        If set to 1, delete staging dir contents after successful upload
#   DRY_RUN              If set to 1, only show what would be uploaded without actually uploading

set -euo pipefail

SOURCE_DIR=${1:-/data/droid_batch_upload}
LOG_DIR=${2:-"/data/logs/metadata_hf_upload_$(date +%Y%m%d_%H%M%S)"}
LOG_FILE="${LOG_DIR}/upload.log"

# Staging directory for filtered files
STAGING_DIR="${LOG_DIR}/staging"

mkdir -p "${LOG_DIR}" "${STAGING_DIR}"

# Tee stdout/stderr to the log file for easier debugging
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[INFO] Metadata-only Hugging Face upload"
echo "[INFO] Source: ${SOURCE_DIR}"
echo "[INFO] Staging: ${STAGING_DIR}"
echo "[INFO] Logs:   ${LOG_FILE}"

if [ ! -d "${SOURCE_DIR}" ]; then
    echo "[ERROR] Source directory does not exist: ${SOURCE_DIR}" >&2
    exit 1
fi

# Resolve repo root to find .env reliably
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
ENV_FILE="${REPO_ROOT}/.env"

if [ -f "${ENV_FILE}" ]; then
    HF_TOKEN=$(grep -E '^HF_TOKEN=' "${ENV_FILE}" | sed 's/^HF_TOKEN=//; s/^"//; s/"$//')
    if [ -z "${HF_TOKEN}" ]; then
        echo "[ERROR] HF_TOKEN is empty in ${ENV_FILE}" >&2
        exit 1
    fi
    export HF_TOKEN
    echo "[INFO] Loaded HF_TOKEN from ${ENV_FILE}"
else
    echo "[ERROR] .env file not found at ${ENV_FILE}" >&2
    exit 1
fi

# Default to a separate metadata repository
HF_REPO_ID=${HF_REPO_ID:-sazirarrwth99/trajectories_droid}
HF_REPO_TYPE=${HF_REPO_TYPE:-dataset}
COMMIT_MESSAGE=${COMMIT_MESSAGE:-"Metadata upload $(date +%Y-%m-%d_%H:%M:%S)"}
CLEAR_STAGING=${CLEAR_STAGING:-0}
DRY_RUN=${DRY_RUN:-0}
export SOURCE_DIR STAGING_DIR HF_REPO_ID HF_REPO_TYPE COMMIT_MESSAGE DRY_RUN

echo "[INFO] Target repo: ${HF_REPO_ID}"
echo "[INFO] Files to include: tracks.npz, extrinsics.npz, quality.json"
echo ""

# ----------------------------------------------------------------------------
# STEP 1: Filter and copy only metadata files to staging
# ----------------------------------------------------------------------------
echo "[INFO] Filtering metadata files from source..."

# Find and copy only the metadata files while preserving directory structure
METADATA_PATTERNS=("tracks.npz" "extrinsics.npz" "quality.json")

TOTAL_FILES=0
TOTAL_BYTES=0

for pattern in "${METADATA_PATTERNS[@]}"; do
    while IFS= read -r -d '' file; do
        # Get relative path from source
        rel_path="${file#${SOURCE_DIR}/}"
        dest_file="${STAGING_DIR}/${rel_path}"
        dest_dir=$(dirname "${dest_file}")
        
        mkdir -p "${dest_dir}"
        cp "${file}" "${dest_file}"
        
        file_size=$(stat -c%s "${file}" 2>/dev/null || stat -f%z "${file}" 2>/dev/null || echo 0)
        TOTAL_BYTES=$((TOTAL_BYTES + file_size))
        TOTAL_FILES=$((TOTAL_FILES + 1))
        
        echo "  [COPY] ${rel_path}"
    done < <(find "${SOURCE_DIR}" -type f -name "${pattern}" -print0 2>/dev/null)
done

if [ "${TOTAL_FILES}" -eq 0 ]; then
    echo "[ERROR] No metadata files found in ${SOURCE_DIR}. Nothing to upload." >&2
    echo "[INFO] Expected files: tracks.npz, extrinsics.npz, quality.json"
    exit 1
fi

# Human-readable size
human_size() {
    local bytes=$1
    if [ "${bytes}" -lt 1024 ]; then
        echo "${bytes} B"
    elif [ "${bytes}" -lt 1048576 ]; then
        echo "$((bytes / 1024)) KB"
    elif [ "${bytes}" -lt 1073741824 ]; then
        echo "$((bytes / 1048576)) MB"
    else
        echo "$((bytes / 1073741824)) GB"
    fi
}

echo ""
echo "[INFO] Staged ${TOTAL_FILES} metadata files ($(human_size ${TOTAL_BYTES}))"
echo ""

# ----------------------------------------------------------------------------
# STEP 2: Show summary by file type
# ----------------------------------------------------------------------------
echo "[INFO] File breakdown:"
for pattern in "${METADATA_PATTERNS[@]}"; do
    count=$(find "${STAGING_DIR}" -type f -name "${pattern}" 2>/dev/null | wc -l)
    echo "  ${pattern}: ${count} files"
done
echo ""

# ----------------------------------------------------------------------------
# STEP 3: Upload to HuggingFace (or dry run)
# ----------------------------------------------------------------------------
if [ "${DRY_RUN}" -eq 1 ]; then
    echo "[DRY RUN] Would upload ${TOTAL_FILES} files to ${HF_REPO_ID}"
    echo "[DRY RUN] Staging directory: ${STAGING_DIR}"
    echo "[DRY RUN] Files:"
    find "${STAGING_DIR}" -type f | head -20
    STAGING_COUNT=$(find "${STAGING_DIR}" -type f | wc -l)
    if [ "${STAGING_COUNT}" -gt 20 ]; then
        echo "  ... and $((STAGING_COUNT - 20)) more files"
    fi
    exit 0
fi

echo "[INFO] Preparing to upload ${TOTAL_FILES} files to ${HF_REPO_ID} (${HF_REPO_TYPE})"

python3 <<'PYTHON'
import os
import sys
from datetime import datetime
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

staging_dir = Path(os.environ['STAGING_DIR'])
repo_id = os.environ['HF_REPO_ID']
repo_type = os.environ['HF_REPO_TYPE']
commit_message = os.environ['COMMIT_MESSAGE']

api = HfApi(token=os.environ['HF_TOKEN'])

def human_readable_size(bytes_size: int) -> str:
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    size = float(bytes_size)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024

# Ensure repo exists
print(f"[INFO] Ensuring repo exists: {repo_id}")
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"[INFO] Repo '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f"[INFO] Creating new {repo_type} repo: {repo_id}")
    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=False,
        token=os.environ['HF_TOKEN'],
    )
    print(f"[SUCCESS] Created repo: https://huggingface.co/datasets/{repo_id}")

total_bytes = sum(f.stat().st_size for f in staging_dir.rglob('*') if f.is_file())
file_count = sum(1 for f in staging_dir.rglob('*') if f.is_file())
print(f"[INFO] Total size: {human_readable_size(total_bytes)} ({file_count} files)")
print(f"[INFO] Starting upload at {datetime.now().isoformat(timespec='seconds')} ...")

try:
    api.upload_folder(
        folder_path=str(staging_dir),
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=commit_message,
        run_as_future=False,
    )
    print(f"[SUCCESS] Upload complete at {datetime.now().isoformat(timespec='seconds')}")
except Exception as exc:  # noqa: BLE001
    print(f"[ERROR] Upload failed: {exc}", file=sys.stderr)
    sys.exit(1)
PYTHON

UPLOAD_STATUS=$?

if [ "${UPLOAD_STATUS}" -ne 0 ]; then
    echo "[ERROR] Upload failed. Staging directory preserved at: ${STAGING_DIR}"
    exit 1
fi

# ----------------------------------------------------------------------------
# STEP 4: Cleanup staging (optional)
# ----------------------------------------------------------------------------
if [ "${CLEAR_STAGING}" -eq 1 ]; then
    echo "[INFO] Clearing staging directory after successful upload..."
    rm -rf "${STAGING_DIR}"
fi

echo ""
echo "[INFO] Done. Detailed log saved to ${LOG_FILE}"
echo "[INFO] Uploaded to: https://huggingface.co/datasets/${HF_REPO_ID}"

