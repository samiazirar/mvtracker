#!/bin/bash
# Manual HuggingFace Upload Script
# ================================
# Use this script to manually upload pending files from the batch upload directory
# to HuggingFace after the main pipeline completes.
#
# Usage:
#   ./upload_to_huggingface.sh                           # Upload from default directory
#   ./upload_to_huggingface.sh /path/to/custom/dir       # Upload from custom directory
#   DRY_RUN=1 ./upload_to_huggingface.sh                 # Preview what would be uploaded
#
# Environment Variables:
#   HF_REPO_ID          Override the target repository (default: sazirarrwth99/droid_metadata_only)
#   DRY_RUN=1           Show what would be uploaded without actually uploading
#   DELETE_AFTER=1      Delete files after successful upload (default: 1)

set -e

# ============================================================================
# LOAD ENVIRONMENT VARIABLES FROM .ENV FILE
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Try multiple possible .env locations
ENV_LOCATIONS=(
    "/root/mvtracker/.env"           # Docker container location
    "${REPO_ROOT}/.env"              # Repository root
    "${HOME}/.env"                    # User home directory
)

HF_TOKEN_LOADED=0
for ENV_FILE in "${ENV_LOCATIONS[@]}"; do
    if [ -f "${ENV_FILE}" ]; then
        echo "[INFO] Loading environment variables from: ${ENV_FILE}"
        set -a  # automatically export all variables
        source "${ENV_FILE}"
        set +a  # turn off auto-export
        HF_TOKEN_LOADED=1
        break
    fi
done

# Verify HF_TOKEN is set
if [ -z "${HF_TOKEN}" ]; then
    echo "[ERROR] HF_TOKEN not set"
    echo "Please set HF_TOKEN in one of these locations:"
    for loc in "${ENV_LOCATIONS[@]}"; do
        echo "  - ${loc}"
    done
    echo "Or set it as an environment variable: export HF_TOKEN=\"hf_your_token\""
    exit 1
fi

BATCH_UPLOAD_DIR="${1:-/data/droid_batch_upload_metadata}"
HF_REPO_ID="${HF_REPO_ID:-sazirarrwth99/droid_metadata_only}"
HF_REPO_TYPE="dataset"
DRY_RUN="${DRY_RUN:-0}"
DELETE_AFTER="${DELETE_AFTER:-1}"

echo ""
echo "============================================================"
echo "HuggingFace Manual Upload"
echo "============================================================"
echo "Upload Directory: ${BATCH_UPLOAD_DIR}"
echo "Target Repo:      ${HF_REPO_ID}"
echo ""

[ ! -d "${BATCH_UPLOAD_DIR}" ] && { echo "[ERROR] Directory not found: ${BATCH_UPLOAD_DIR}"; exit 1; }

FILE_COUNT=$(find "${BATCH_UPLOAD_DIR}" -type f 2>/dev/null | wc -l)
TOTAL_SIZE=$(du -sh "${BATCH_UPLOAD_DIR}" 2>/dev/null | awk '{print $1}')
echo "Files to upload:  ${FILE_COUNT}"
echo "Total size:       ${TOTAL_SIZE}"
echo ""

[ "${FILE_COUNT}" -eq 0 ] && { echo "[INFO] No files to upload. Directory is empty."; exit 0; }

find "${BATCH_UPLOAD_DIR}" -type f | head -10 | while read f; do echo "  ${f#${BATCH_UPLOAD_DIR}/}"; done
[ "${FILE_COUNT}" -gt 10 ] && echo "  ... and $((FILE_COUNT - 10)) more"
echo ""

[ "${DRY_RUN}" -eq 1 ] && { echo "[DRY RUN] Would upload ${FILE_COUNT} files to ${HF_REPO_ID}"; exit 0; }

echo "[INFO] Starting upload..."
export BATCH_UPLOAD_DIR HF_REPO_ID HF_REPO_TYPE

python3 << 'PYTHON_SCRIPT'
import os, sys
from huggingface_hub import HfApi
from datetime import datetime

api = HfApi(token=os.environ['HF_TOKEN'])
try:
    api.upload_folder(
        folder_path=os.environ.get('BATCH_UPLOAD_DIR', '/data/droid_batch_upload_metadata'),
        repo_id=os.environ.get('HF_REPO_ID', 'sazirarrwth99/droid_metadata_only'),
        repo_type=os.environ.get('HF_REPO_TYPE', 'dataset'),
        commit_message=f'Manual upload {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
    )
    print('[SUCCESS] Upload completed!')
except Exception as e:
    print(f'[ERROR] Upload failed: {e}', file=sys.stderr)
    sys.exit(1)
PYTHON_SCRIPT

if [ $? -eq 0 ] && [ "${DELETE_AFTER}" -eq 1 ]; then
    echo "[INFO] Cleaning up..."
    find "${BATCH_UPLOAD_DIR}" -mindepth 1 -delete
fi
echo ""
echo "============================================================"
echo "Upload Complete!"
echo "============================================================"
echo "Repository: https://huggingface.co/datasets/${HF_REPO_ID}"
