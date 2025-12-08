#!/bin/bash

# This script processes DROID episodes by first extracting RGB+Depth,
# and then generating tracks and metadata.
# It uses a helper python script to select and sort episodes based on quality.
#
# As a starting point, this script is kept simple. Error handling and
# command-line argument parsing can be added later.

set -e

# --- Configuration ---
# Number of episodes per lab to process. Set to -1 to process all.
LIMIT=-1 

# Path to the JSON file containing episode quality information.
JSON_PATH="/data/droid/calib_and_annot/droid/cam2base_extrinsic_superset.json"

# Root directory of the DROID data.
DROID_DATA_ROOT="/data/droid"

# --- Script ---

echo "Fetching and sorting episodes..."

# Get the list of episode directories from the helper script.
# NOTE: This simple 'for' loop will not work correctly if paths have spaces.
# It is used for simplicity as requested.
EPISODE_DIRS=$(python3 conversions/droid/process_episodes.py --json_path "$JSON_PATH" --data_root "$DROID_DATA_ROOT" --limit "$LIMIT")

if [ -z "$EPISODE_DIRS" ]; then
    echo "No episodes found to process. Exiting."
    exit 0
fi

TOTAL_TIME=0
EPISODE_COUNT=0

echo "Starting processing..."

for EPISODE_DIR in $EPISODE_DIRS; do
    echo "================================================================="
    echo "Processing episode: $EPISODE_DIR"
    echo "================================================================="
    
    START_TIME=$SECONDS

    # Step 1: Extract RGB and Depth
    echo "--- Running extract_rgb_depth.py ---"
    python3 conversions/droid/training_data/extract_rgb_depth.py --episode_dir "$EPISODE_DIR"
    
    # Step 2: Generate Tracks and Metadata
    echo "--- Running generate_tracks_and_metadata.py ---"
    python3 conversions/droid/training_data/generate_tracks_and_metadata.py --episode_dir "$EPISODE_DIR"

    END_TIME=$SECONDS
    DURATION=$((END_TIME - START_TIME))
    
    TOTAL_TIME=$((TOTAL_TIME + DURATION))
    EPISODE_COUNT=$((EPISODE_COUNT + 1))
    
    # Avoid division by zero if a script runs in less than a second
    if [ "$EPISODE_COUNT" -gt 0 ]; then
        AVG_TIME=$((TOTAL_TIME / EPISODE_COUNT))
    else
        AVG_TIME=0
    fi

    echo "-----------------------------------------------------------------"
    echo "Episode finished in $DURATION seconds."
    echo "Processed $EPISODE_COUNT episodes so far."
    echo "Total time: $TOTAL_TIME seconds."
    echo "Running average: $AVG_TIME seconds per episode."
    echo "================================================================="
    echo ""
done

echo "All episodes processed."

