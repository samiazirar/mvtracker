#!/bin/bash
# Test script using GIT to check existing files (Fast & Timeout-proof)

set -e

LIMIT=${1:-100}
OUTPUT_DIR="/data/logs/hf_skip_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUTPUT_DIR}"

# Files
EPISODES_FILE="${OUTPUT_DIR}/all_episodes.txt"
EXISTING_LIST_FILE="${OUTPUT_DIR}/hf_file_list.txt"
PROCESSED_FILE="${OUTPUT_DIR}/already_processed.txt"
REMAINING_FILE="${OUTPUT_DIR}/remaining_to_process.txt"

# Config
CAM2BASE_PATH="/data/cam2base_extrinsic_superset.json"
SCRIPT_DIR="conversions/droid/training_data"
HF_REPO_ID="sazirarrwth99/trajectory_data"

# Load Token
if [ -f ".env" ]; then
    HF_TOKEN=$(grep -E '^HF_TOKEN=' .env | sed 's/^HF_TOKEN=//; s/^"//; s/"$//')
    export HF_TOKEN
else
    echo "[ERROR] .env file not found"
    exit 1
fi

echo "=============================================="
echo "HuggingFace Skip Test (GIT Method)"
echo "=============================================="

# 1. Get Episodes to check
echo "[1/3] Generating episode list..."
python "${SCRIPT_DIR}/get_episodes_by_quality.py" \
    --cam2base "${CAM2BASE_PATH}" \
    --limit "${LIMIT}" \
    --output "${EPISODES_FILE}"

# 2. Get File List via Git (The Robust Fix)
echo "[2/3] Fetching file list from HuggingFace via Git..."
TEMP_GIT_DIR=$(mktemp -d)

# Clone ONLY the history/tree, NO files (extremely fast)
# We embed the token in the URL for authentication
git clone --depth 1 --filter=blob:none --no-checkout \
    "https://oauth2:${HF_TOKEN}@huggingface.co/datasets/${HF_REPO_ID}" \
    "${TEMP_GIT_DIR}" > /dev/null 2>&1

# List all files in the tree
pushd "${TEMP_GIT_DIR}" > /dev/null
git ls-tree -r --name-only HEAD > "${EXISTING_LIST_FILE}"
popd > /dev/null

rm -rf "${TEMP_GIT_DIR}"

FILE_COUNT=$(wc -l < "${EXISTING_LIST_FILE}")
echo "      [SUCCESS] Retrieved list of ${FILE_COUNT} files."

# 3. Python Filter Logic (Local & Fast)
echo "[3/3] Cross-referencing local episodes with HF list..."
export EPISODES_FILE EXISTING_LIST_FILE PROCESSED_FILE REMAINING_FILE

python3 << 'PYTHON_SCRIPT'
import os
import re
from datetime import datetime

episodes_file = os.environ["EPISODES_FILE"]
existing_list_file = os.environ["EXISTING_LIST_FILE"]
processed_file = os.environ["PROCESSED_FILE"]
remaining_file = os.environ["REMAINING_FILE"]

# 1. Load the HF file list into a set of signatures
print("[Internal] Building lookup table...")
processed_signatures = set()
with open(existing_list_file, "r") as f:
    for path in f:
        path = path.strip()
        parts = path.split("/")
        # Path structure: lab/outcome/date/timestamp
        if len(parts) >= 4:
            lab, outcome, date, timestamp = parts[0], parts[1], parts[2], parts[3]
            if outcome in ("success", "failure"):
                sig = f"{lab}+{date}+{timestamp}"
                processed_signatures.add(sig)

# 2. Check our episodes
def parse_episode_id(episode_id):
    parts = episode_id.split("+")
    if len(parts) != 3: return None
    m = re.match(r"(\d{4}-\d{2}-\d{2})-(\d+)h-(\d+)m-(\d+)s", parts[2])
    if not m: return None
    dt = datetime.strptime(f"{m.group(1)} {m.group(2)}:{m.group(3)}:{m.group(4)}", "%Y-%m-%d %H:%M:%S")
    # Linux style space-padding for days < 10 (e.g., "Jan  1")
    return {
        "lab": parts[0],
        "date": m.group(1),
        "timestamp_folder": dt.strftime("%a_%b_%e_%H:%M:%S_%Y")
    }

already_processed = []
remaining = []

with open(episodes_file, "r") as f:
    for line in f:
        ep = line.strip()
        if not ep: continue
        
        parsed = parse_episode_id(ep)
        if parsed:
            sig = f"{parsed['lab']}+{parsed['date']}+{parsed['timestamp_folder']}"
            if sig in processed_signatures:
                already_processed.append(ep)
            else:
                remaining.append(ep)
        else:
            remaining.append(ep)

# Save
with open(processed_file, "w") as f:
    f.write("\n".join(already_processed) + "\n")
with open(remaining_file, "w") as f:
    f.write("\n".join(remaining) + "\n")

print(f"      Matched:   {len(already_processed)} (Already on HF)")
print(f"      Remaining: {len(remaining)} (Need processing)")
PYTHON_SCRIPT

echo ""
echo "Done. Results saved to: ${OUTPUT_DIR}"