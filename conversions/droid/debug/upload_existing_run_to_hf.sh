#!/bin/bash
# Manually upload an already-processed run directory to Hugging Face.
#
# Usage:
#   conversions/droid/debug/upload_existing_run_to_hf.sh [SOURCE_DIR] [LOG_DIR]
#
# Defaults:
#   SOURCE_DIR: /data/droid_batch_upload
#   LOG_DIR:    /data/logs/manual_hf_upload_<timestamp>
#
# Environment variables:
#   HF_REPO_ID           Target dataset repo (default: sazirarrwth99/trajectory_data)
#   HF_REPO_TYPE         Repo type (default: dataset)
#   COMMIT_MESSAGE       Optional commit message for the upload
#   CLEAR_AFTER_UPLOAD   If set to 1, delete SOURCE_DIR contents after a successful upload

set -euo pipefail

SOURCE_DIR=${1:-/data/droid_batch_upload}
LOG_DIR=${2:-"/data/logs/manual_hf_upload_$(date +%Y%m%d_%H%M%S)"}
LOG_FILE="${LOG_DIR}/upload.log"

mkdir -p "${LOG_DIR}"

# Tee stdout/stderr to the log file for easier debugging
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[INFO] Manual Hugging Face upload"
echo "[INFO] Source: ${SOURCE_DIR}"
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

HF_REPO_ID=${HF_REPO_ID:-sazirarrwth99/trajectory_data}
HF_REPO_TYPE=${HF_REPO_TYPE:-dataset}
COMMIT_MESSAGE=${COMMIT_MESSAGE:-"Manual upload $(date +%Y-%m-%d_%H:%M:%S)"}
CLEAR_AFTER_UPLOAD=${CLEAR_AFTER_UPLOAD:-0}
export SOURCE_DIR HF_REPO_ID HF_REPO_TYPE COMMIT_MESSAGE

# Count files before attempting upload
FILE_COUNT=$(find "${SOURCE_DIR}" -type f 2>/dev/null | wc -l)
if [ "${FILE_COUNT}" -eq 0 ]; then
    echo "[ERROR] No files found in ${SOURCE_DIR}. Nothing to upload." >&2
    exit 1
fi

echo "[INFO] Preparing to upload ${FILE_COUNT} files to ${HF_REPO_ID} (${HF_REPO_TYPE})"

python3 <<'PYTHON'
import os
import sys
from datetime import datetime
from pathlib import Path
from huggingface_hub import HfApi

source_dir = Path(os.environ['SOURCE_DIR'])
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

total_bytes = sum(f.stat().st_size for f in source_dir.rglob('*') if f.is_file())
print(f"[INFO] Total size: {human_readable_size(total_bytes)}")
print(f"[INFO] Starting upload at {datetime.now().isoformat(timespec='seconds')} ...")

try:
    api.upload_folder(
        folder_path=str(source_dir),
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

if [ "${CLEAR_AFTER_UPLOAD}" -eq 1 ]; then
    echo "[INFO] Clearing source directory after successful upload..."
    find "${SOURCE_DIR}" -mindepth 1 -delete 2>/dev/null || true
fi

echo "[INFO] Done. Detailed log saved to ${LOG_FILE}"
