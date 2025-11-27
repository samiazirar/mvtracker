#!/usr/bin/env bash
set -e

SCRIPTS_DIR="conversions/droid"
OUT_DIR="point_clouds"

echo "[1] Cleaning ${OUT_DIR}/ ..."
rm -rf "${OUT_DIR:?}/"*
mkdir -p "${OUT_DIR}"

echo "[3] Running scripts in parallel..."

# 1. generate_pointcloud_from_droid.py
echo "Launching: conversions/droid/generate_pointcloud_from_droid.py"
python conversions/droid/generate_pointcloud_from_droid.py > "logs_generate_pointcloud_from_droid.out" 2>&1 &
pid1=$!

# 2. generate_pointcloud_from_droid_no_optimization.py
echo "Launching: conversions/droid/generate_pointcloud_from_droid_no_optimization.py"
python conversions/droid/generate_pointcloud_from_droid_no_optimization.py > "logs_generate_pointcloud_from_droid_no_optimization.out" 2>&1 &
pid2=$!

# 3. generate_pointcloud_from_droid_rerun_transforms.py
echo "Launching: conversions/droid/generate_pointcloud_from_droid_rerun_transforms.py"
python conversions/droid/generate_pointcloud_from_droid_rerun_transforms.py > "logs_generate_pointcloud_from_droid_rerun_transforms.out" 2>&1 &
pid3=$!

# # 4. generate_pointcloud_from_droid_tracking.py
# echo "Launching: conversions/droid/generate_pointcloud_from_droid_tracking.py"
# python conversions/droid/generate_pointcloud_from_droid_tracking.py > "logs_generate_pointcloud_from_droid_tracking.out" 2>&1 &
# pid4=$!

# 5. generate_pointcloud_from_droid_with_video.py
echo "Launching: conversions/droid/generate_pointcloud_from_droid_with_video.py"
python conversions/droid/generate_pointcloud_from_droid_with_video.py > "logs_generate_pointcloud_from_droid_with_video.out" 2>&1 &
pid5=$!

echo ""
echo "[4] Waiting for all jobs..."

wait $pid1
echo "✔ generate_pointcloud_from_droid finished"

wait $pid2
echo "✔ generate_pointcloud_from_droid_no_optimization finished"

wait $pid3
echo "✔ generate_pointcloud_from_droid_rerun_transforms finished"

wait $pid4
echo "✔ generate_pointcloud_from_droid_tracking finished"

wait $pid5
echo "✔ generate_pointcloud_from_droid_with_video finished"

echo ""
echo "[DONE] All scripts completed. Output in ${OUT_DIR}/"