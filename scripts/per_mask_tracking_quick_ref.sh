#!/usr/bin/env bash
# Quick reference for per-mask tracking commands

# ============================================================================
# QUICK START: Run per-mask tracking on your data
# ============================================================================

# 1. Set your paths
TASK_FOLDER="task_0034_user_0014_scene_0004_cfg_0006_human"
INPUT_NPZ="third_party/HOISTFormer/sam2_tracking_output/${TASK_FOLDER}_processed_hand_tracked_hoist_sam2.npz"

# 2. Run per-mask tracking (all-in-one)
python process_masks_independently.py \
  --npz "${INPUT_NPZ}" \
  --mask-key sam2_masks \
  --tracker mvtracker \
  --output-dir "./tracking_per_mask/${TASK_FOLDER}" \
  --base-name "${TASK_FOLDER}" \
  --use-first-frame

# 3. View results
bash "./tracking_per_mask/${TASK_FOLDER}/view_${TASK_FOLDER}_mvtracker_combined.sh"


# ============================================================================
# MANUAL STEPS: Run each step separately
# ============================================================================

# Step 1: Split NPZ by mask instances
python -c "
from pathlib import Path
from utils.mask_instance_utils import split_npz_by_instances

split_npz_by_instances(
    input_npz_path=Path('${INPUT_NPZ}'),
    output_dir=Path('./instances'),
    mask_key='sam2_masks',
)
"

# Step 2: Create query points for each instance
for instance_npz in ./instances/*_instance_*.npz; do
  python create_query_points_from_masks.py \
    --npz "${instance_npz}" \
    --key sam2_masks \
    --use-first-frame \
    --frames-before 3 \
    --frames-after 10
done

# Step 3: Run tracking for each query NPZ
for query_npz in ./instances/*_query.npz; do
  instance_name=$(basename "${query_npz}" _query.npz)
  python demo.py \
    --sample-path "${query_npz}" \
    --tracker mvtracker \
    --depth_estimator gt \
    --rerun save \
    --rrd "${instance_name}_tracking.rrd"
done

# Step 4: View all together
rerun *_tracking.rrd


# ============================================================================
# COMMON USE CASES
# ============================================================================

# Compare two trackers
python process_masks_independently.py --npz "${INPUT_NPZ}" --tracker mvtracker --output-dir ./results_mvt
python process_masks_independently.py --npz "${INPUT_NPZ}" --tracker spatialtrackerv2 --output-dir ./results_spat
rerun ./results_mvt/*.rrd ./results_spat/*.rrd

# Process only specific instance (manual)
python -c "
from pathlib import Path
from utils.mask_instance_utils import create_single_instance_npz

create_single_instance_npz(
    input_npz_path=Path('${INPUT_NPZ}'),
    instance_id='instance_0',
    output_npz_path=Path('mask_0_only.npz'),
    mask_key='sam2_masks',
)
"

# Check what instances are in your NPZ
python -c "
from pathlib import Path
from utils.mask_instance_utils import get_mask_instance_ids

instance_ids = get_mask_instance_ids(Path('${INPUT_NPZ}'), mask_key='sam2_masks')
print(f'Found {len(instance_ids)} instances:', instance_ids)
"


# ============================================================================
# TROUBLESHOOTING
# ============================================================================

# Inspect NPZ structure
python -c "
import numpy as np
data = np.load('${INPUT_NPZ}', allow_pickle=True)
print('Keys:', list(data.keys()))
if 'sam2_masks' in data:
    masks = data['sam2_masks'].item()
    print(f'Instances: {list(masks.keys())}')
    for inst, arr in masks.items():
        print(f'  {inst}: shape {arr.shape}')
"

# Test single instance tracking
python create_query_points_from_masks.py \
  --npz ./instances/*_instance_0.npz \
  --key sam2_masks \
  --use-first-frame

python demo.py \
  --sample-path ./instances/*_instance_0_query.npz \
  --tracker mvtracker \
  --rerun save \
  --rrd test_single_instance.rrd

rerun test_single_instance.rrd


# ============================================================================
# VIEWING OPTIONS
# ============================================================================

# View all masks from one tracker
rerun ./tracking_per_mask/${TASK_FOLDER}/*.rrd

# View specific masks
rerun ./tracking_per_mask/${TASK_FOLDER}/*_instance_0.rrd \
     ./tracking_per_mask/${TASK_FOLDER}/*_instance_1.rrd

# Use generated script
bash ./tracking_per_mask/${TASK_FOLDER}/view_*_combined.sh

# View with web viewer
rerun ./tracking_per_mask/${TASK_FOLDER}/*.rrd --web-viewer
