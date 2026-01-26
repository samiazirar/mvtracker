#!/bin/bash
# Quick start script to generate all visualizations
# Usage: bash run_all_visualizations.sh

set -e

echo "=============================================="
echo "  MVTracker Visualization Generator"
echo "=============================================="
echo ""

# Create output directories
mkdir -p visualizations output_videos

# Check if running in container
if [ -f /.dockerenv ]; then
    echo "[INFO] Running inside Docker container"
    WORKSPACE=/workspace
else
    echo "[INFO] Running on host - may need to use docker exec"
    WORKSPACE=$(pwd)
fi

cd $WORKSPACE

echo ""
echo "Step 1: Generate contact flow from training shards..."
echo "-----------------------------------------------"

for i in $(seq 0 6); do
    SHARD="/data/droid/training_shards/shard_000${i}.tar"
    if [ -f "$SHARD" ]; then
        echo "Processing shard_000${i}..."
        python visualize_training_shards.py \
            --shard_path "$SHARD" \
            --num_samples 10 \
            --output_dir visualizations
    fi
done

echo ""
echo "Step 2: Generate combined episodes visualization..."
echo "-----------------------------------------------"
python visualize_combined_episodes.py \
    --max_shards 5 \
    --samples_per_shard 3 \
    --output visualizations/combined_all.rrd

echo ""
echo "Step 3: Generate video outputs with 2D tracks..."
echo "-----------------------------------------------"
for i in $(seq 0 3); do
    SHARD="/data/droid/training_shards/shard_000${i}.tar"
    if [ -f "$SHARD" ]; then
        echo "Processing videos from shard_000${i}..."
        python visualize_videos_with_tracks.py \
            --shard_path "$SHARD" \
            --num_samples 3 \
            --output_dir output_videos
    fi
done

echo ""
echo "Step 4: Show summary..."
echo "-----------------------------------------------"
python show_visualization_summary.py

echo ""
echo "=============================================="
echo "  All visualizations generated!"
echo "=============================================="
echo ""
echo "View with:"
echo "  rerun visualizations/<filename>.rrd"
echo "  ffplay output_videos/<filename>.mp4"
