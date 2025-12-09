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

# ============================================================================
# BATCH UPLOAD LOGIC
# ============================================================================
# HuggingFace has a 25k file limit per commit, so we upload in batches by episode
BATCH_SIZE="${BATCH_SIZE:-500}"  # Number of episodes per batch (each episode ~3-5 files)
echo "[INFO] Starting batch upload (batch size: ${BATCH_SIZE} episodes)..."
export BATCH_UPLOAD_DIR HF_REPO_ID HF_REPO_TYPE BATCH_SIZE DELETE_AFTER

python3 << 'PYTHON_SCRIPT'
import os
import sys
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
from huggingface_hub import HfApi

batch_dir = Path(os.environ.get('BATCH_UPLOAD_DIR', '/data/droid_batch_upload_metadata'))
repo_id = os.environ.get('HF_REPO_ID', 'sazirarrwth99/droid_metadata_only')
repo_type = os.environ.get('HF_REPO_TYPE', 'dataset')
batch_size = int(os.environ.get('BATCH_SIZE', '500'))
delete_after = os.environ.get('DELETE_AFTER', '1') == '1'
token = os.environ['HF_TOKEN']

api = HfApi(token=token)

# Find all unique episodes (top-level directories like "RAIL/success/2023-07-15/...")
# We group by the 4th level directory (timestamp folder)
def get_episode_dirs(base_path):
    """Get unique episode directories (lab/outcome/date/timestamp)"""
    episodes = set()
    for path in base_path.rglob('*'):
        if path.is_file():
            # Get relative path parts
            rel_parts = path.relative_to(base_path).parts
            if len(rel_parts) >= 4:
                # Episode path is first 4 components: lab/outcome/date/timestamp
                episode_path = Path(*rel_parts[:4])
                episodes.add(episode_path)
    return sorted(episodes)

print(f"[INFO] Scanning for episodes in {batch_dir}...")
episodes = get_episode_dirs(batch_dir)
total_episodes = len(episodes)
print(f"[INFO] Found {total_episodes} episodes to upload")

if total_episodes == 0:
    print("[INFO] No episodes found. Nothing to upload.")
    sys.exit(0)

# Split into batches
batches = [episodes[i:i + batch_size] for i in range(0, len(episodes), batch_size)]
total_batches = len(batches)
print(f"[INFO] Will upload in {total_batches} batches ({batch_size} episodes per batch)")
print()

successful_episodes = []
failed_batches = []

for batch_num, batch_episodes in enumerate(batches, 1):
    print(f"[BATCH {batch_num}/{total_batches}] Preparing {len(batch_episodes)} episodes...")
    
    # Create temporary directory for this batch
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Copy episode files to temp directory
        file_count = 0
        for episode in batch_episodes:
            src_episode = batch_dir / episode
            dst_episode = temp_path / episode
            
            if src_episode.exists():
                # Copy all files for this episode
                for src_file in src_episode.rglob('*'):
                    if src_file.is_file():
                        rel_path = src_file.relative_to(batch_dir)
                        dst_file = temp_path / rel_path
                        dst_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src_file, dst_file)
                        file_count += 1
        
        print(f"[BATCH {batch_num}/{total_batches}] Uploading {file_count} files...")
        
        try:
            api.upload_folder(
                folder_path=str(temp_path),
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=f'Batch {batch_num}/{total_batches} - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            )
            print(f"[BATCH {batch_num}/{total_batches}] SUCCESS - Uploaded {len(batch_episodes)} episodes")
            successful_episodes.extend(batch_episodes)
            
            # Delete uploaded episodes from source if DELETE_AFTER is enabled
            if delete_after:
                for episode in batch_episodes:
                    src_episode = batch_dir / episode
                    if src_episode.exists():
                        shutil.rmtree(src_episode, ignore_errors=True)
                # Clean up empty parent directories
                for episode in batch_episodes:
                    for parent in (batch_dir / episode).parents:
                        if parent == batch_dir:
                            break
                        try:
                            parent.rmdir()  # Only removes if empty
                        except OSError:
                            pass  # Not empty, skip
                            
        except Exception as e:
            print(f"[BATCH {batch_num}/{total_batches}] FAILED: {e}", file=sys.stderr)
            failed_batches.append((batch_num, batch_episodes, str(e)))
    
    print()

# Summary
print("=" * 60)
print("UPLOAD SUMMARY")
print("=" * 60)
print(f"Total episodes:     {total_episodes}")
print(f"Successful:         {len(successful_episodes)}")
print(f"Failed batches:     {len(failed_batches)}")
print()

if failed_batches:
    print("Failed batches:")
    for batch_num, episodes, error in failed_batches:
        print(f"  Batch {batch_num}: {len(episodes)} episodes - {error[:100]}")
    print()
    print("[WARN] Some batches failed. Re-run the script to retry.")
    sys.exit(1)
else:
    print("[SUCCESS] All episodes uploaded successfully!")
    print(f"Repository: https://huggingface.co/datasets/{repo_id}")
PYTHON_SCRIPT
