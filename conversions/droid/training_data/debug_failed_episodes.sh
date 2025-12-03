#!/bin/bash
# Quick debug script for failed episode downloads
#
# Usage:
#   ./debug_failed_episodes.sh                          # Check failed episodes from latest log
#   ./debug_failed_episodes.sh /path/to/failed.txt      # Check specific file
#   ./debug_failed_episodes.sh --single "ILIAD+xxx"     # Check single episode

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DATA="/data/droid/data/droid_raw/1.0.1"
GCS_BUCKET="gs://gresearch/robotics/droid_raw/1.0.1"

# Find the check script
CHECK_SCRIPT="${SCRIPT_DIR}/check_episode_availability.py"

if [ ! -f "${CHECK_SCRIPT}" ]; then
    echo "[ERROR] check_episode_availability.py not found at ${CHECK_SCRIPT}"
    exit 1
fi

# Parse arguments
if [ "$1" == "--single" ]; then
    # Single episode mode
    EPISODE_ID="$2"
    if [ -z "${EPISODE_ID}" ]; then
        echo "Usage: $0 --single <episode_id>"
        exit 1
    fi
    
    echo "Checking single episode: ${EPISODE_ID}"
    echo ""
    python3 "${CHECK_SCRIPT}" \
        --episode_id "${EPISODE_ID}" \
        --local_roots "${LOCAL_DATA}" \
        --gcs_bucket "${GCS_BUCKET}" \
        --verbose
        
elif [ -n "$1" ]; then
    # File provided
    FAILED_FILE="$1"
    if [ ! -f "${FAILED_FILE}" ]; then
        echo "[ERROR] File not found: ${FAILED_FILE}"
        exit 1
    fi
    
    echo "Checking episodes from: ${FAILED_FILE}"
    echo ""
    python3 "${CHECK_SCRIPT}" \
        --episodes_file "${FAILED_FILE}" \
        --local_roots "${LOCAL_DATA}" \
        --gcs_bucket "${GCS_BUCKET}" \
        --output "${FAILED_FILE%.txt}_availability.csv" \
        --output_missing "${FAILED_FILE%.txt}_truly_missing.txt"
        
else
    # Find latest log directory
    LATEST_LOG=$(ls -td /data/logs/pipeline_huggingface_* 2>/dev/null | head -1)
    
    if [ -z "${LATEST_LOG}" ]; then
        echo "[ERROR] No pipeline log directories found in /data/logs/"
        echo ""
        echo "Usage:"
        echo "  $0                           # Check failed episodes from latest log"
        echo "  $0 /path/to/failed.txt       # Check episodes from file"
        echo "  $0 --single \"ILIAD+xxx\"      # Check single episode"
        exit 1
    fi
    
    FAILED_FILE="${LATEST_LOG}/failed_episodes.txt"
    
    if [ ! -f "${FAILED_FILE}" ]; then
        echo "[INFO] No failed_episodes.txt found in ${LATEST_LOG}"
        echo "[INFO] Listing available files:"
        ls -la "${LATEST_LOG}/"
        exit 0
    fi
    
    FAILED_COUNT=$(wc -l < "${FAILED_FILE}")
    echo "Found ${FAILED_COUNT} failed episodes in: ${FAILED_FILE}"
    echo ""
    
    # Show breakdown by failure step
    echo "=== Failures by Step ==="
    awk -F',' '{print $2}' "${FAILED_FILE}" | sort | uniq -c | sort -rn
    echo ""
    
    # Extract just episode IDs (first column)
    EPISODES_ONLY="${LATEST_LOG}/failed_episode_ids.txt"
    awk -F',' '{print $1}' "${FAILED_FILE}" | sort -u > "${EPISODES_ONLY}"
    UNIQUE_COUNT=$(wc -l < "${EPISODES_ONLY}")
    
    echo "Checking ${UNIQUE_COUNT} unique failed episodes..."
    echo ""
    
    python3 "${CHECK_SCRIPT}" \
        --episodes_file "${EPISODES_ONLY}" \
        --local_roots "${LOCAL_DATA}" \
        --gcs_bucket "${GCS_BUCKET}" \
        --output "${LATEST_LOG}/episode_availability.csv" \
        --output_missing "${LATEST_LOG}/truly_missing_episodes.txt"
    
    echo ""
    echo "=== Output Files ==="
    echo "Full availability report: ${LATEST_LOG}/episode_availability.csv"
    echo "Truly missing episodes:   ${LATEST_LOG}/truly_missing_episodes.txt"
fi
