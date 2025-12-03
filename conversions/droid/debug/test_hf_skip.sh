#!/bin/bash
# Test script to check which episodes are already processed on HuggingFace
# Usage: ./test_hf_skip.sh [limit]
#   limit: number of episodes to check (default: 100, use -1 for all)

set -e

LIMIT=${1:-100}
OUTPUT_DIR="/data/logs/hf_skip_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUTPUT_DIR}"

# Output files
EPISODES_FILE="${OUTPUT_DIR}/all_episodes.txt"
PROCESSED_FILE="${OUTPUT_DIR}/already_processed.txt"
REMAINING_FILE="${OUTPUT_DIR}/remaining_to_process.txt"

# Paths
CAM2BASE_PATH="/data/cam2base_extrinsic_superset.json"
SCRIPT_DIR="conversions/droid/training_data"

# HuggingFace config
HF_REPO_ID="sazirarrwth99/trajectory_data"
HF_REPO_TYPE="dataset"

# Load HF_TOKEN from .env
ENV_FILE=".env"
if [ -f "${ENV_FILE}" ]; then
    HF_TOKEN=$(grep -E '^HF_TOKEN=' "${ENV_FILE}" | sed 's/^HF_TOKEN=//; s/^"//; s/"$//')
    if [ -z "${HF_TOKEN}" ]; then
        echo "[ERROR] HF_TOKEN not found in ${ENV_FILE}"
        exit 1
    fi
    export HF_TOKEN
else
    echo "[ERROR] .env file not found"
    exit 1
fi

export HF_REPO_ID HF_REPO_TYPE EPISODES_FILE PROCESSED_FILE REMAINING_FILE

echo "=============================================="
echo "HuggingFace Skip Test"
echo "=============================================="
echo "Limit: ${LIMIT}"
echo "Output dir: ${OUTPUT_DIR}"
echo "HF Repo: ${HF_REPO_ID}"
echo ""

# Step 1: Get episodes
echo "[1/2] Getting episodes..."
python "${SCRIPT_DIR}/get_episodes_by_quality.py" \
    --cam2base "${CAM2BASE_PATH}" \
    --limit "${LIMIT}" \
    --output "${EPISODES_FILE}"

TOTAL_EPISODES=$(wc -l < "${EPISODES_FILE}")
echo "Found ${TOTAL_EPISODES} episodes to check"
echo ""

# Step 2: Check HuggingFace
echo "[2/2] Checking HuggingFace for already processed episodes..."
SKIP_START=$(date +%s)

python3 << 'PYTHON_SCRIPT'
import os
import re
from datetime import datetime
from huggingface_hub import HfApi

episodes_file = os.environ["EPISODES_FILE"]
processed_file = os.environ["PROCESSED_FILE"]
remaining_file = os.environ["REMAINING_FILE"]
repo_id = os.environ.get("HF_REPO_ID")
repo_type = os.environ.get("HF_REPO_TYPE", "dataset")
token = os.environ.get("HF_TOKEN")

api = HfApi(token=token)

def parse_episode_id(episode_id: str):
    """Convert episode_id to the path format used in HuggingFace."""
    parts = episode_id.split("+")
    if len(parts) != 3:
        return None
    m = re.match(r"(\d{4}-\d{2}-\d{2})-(\d+)h-(\d+)m-(\d+)s", parts[2])
    if not m:
        return None
    date = m.group(1)
    hour, minute, second = m.group(2), m.group(3), m.group(4)
    dt = datetime.strptime(f"{date} {hour}:{minute}:{second}", "%Y-%m-%d %H:%M:%S")
    timestamp_folder = dt.strftime("%a_%b_%e_%H:%M:%S_%Y")
    return {
        "lab": parts[0],
        "date": date,
        "timestamp_folder": timestamp_folder,
    }

# ============================================================================
# STEP 1: Bulk fetch ALL paths from HuggingFace
# ============================================================================
print("[HF] Fetching all existing paths from HuggingFace repo...")

existing_paths = set()
try:
    for item in api.list_repo_tree(
        repo_id=repo_id,
        repo_type=repo_type,
        recursive=True,
    ):
        existing_paths.add(item.path)
except Exception as e:
    print(f"[ERROR] Could not fetch HuggingFace repo tree: {e}")
    exit(1)

print(f"[HF] Found {len(existing_paths)} total paths in repo")

# Build processed signatures
processed_signatures = set()
for path in existing_paths:
    parts = path.split("/")
    if len(parts) >= 4:
        lab, outcome, date, timestamp = parts[0], parts[1], parts[2], parts[3]
        if outcome in ("success", "failure"):
            sig = f"{lab}+{date}+{timestamp}"
            processed_signatures.add(sig)

print(f"[HF] Found {len(processed_signatures)} unique processed episodes")

# ============================================================================
# STEP 2: Filter episodes
# ============================================================================
already_processed = []
remaining = []

with open(episodes_file, "r") as f:
    for line in f:
        ep = line.strip()
        if not ep:
            continue
        parsed = parse_episode_id(ep)
        if parsed is None:
            remaining.append(ep)
            continue
        
        sig = f"{parsed['lab']}+{parsed['date']}+{parsed['timestamp_folder']}"
        
        if sig in processed_signatures:
            already_processed.append(ep)
        else:
            remaining.append(ep)

# Save results
with open(processed_file, "w") as f:
    if already_processed:
        f.write("\n".join(already_processed) + "\n")

with open(remaining_file, "w") as f:
    if remaining:
        f.write("\n".join(remaining) + "\n")

print("")
print("=" * 50)
print("RESULTS")
print("=" * 50)
print(f"Total episodes checked:    {len(already_processed) + len(remaining)}")
print(f"Already processed on HF:   {len(already_processed)}")
print(f"Remaining to process:      {len(remaining)}")
print("")
print(f"Processed episodes saved to: {processed_file}")
print(f"Remaining episodes saved to: {remaining_file}")
PYTHON_SCRIPT

SKIP_END=$(date +%s)
SKIP_TIME=$((SKIP_END - SKIP_START))

echo ""
echo "=============================================="
echo "Time taken: ${SKIP_TIME}s"
echo "=============================================="
echo ""
echo "Output files:"
echo "  All episodes:      ${EPISODES_FILE}"
echo "  Already processed: ${PROCESSED_FILE}"
echo "  Remaining:         ${REMAINING_FILE}"
echo ""
echo "Quick preview of already processed (first 10):"
head -n 10 "${PROCESSED_FILE}" 2>/dev/null || echo "(none)"
echo ""
echo "Quick preview of remaining (first 10):"
head -n 10 "${REMAINING_FILE}" 2>/dev/null || echo "(none)"

