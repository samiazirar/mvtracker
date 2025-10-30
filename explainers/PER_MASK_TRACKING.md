# Per-Mask Independent Tracking

This directory contains utilities for processing and tracking each mask instance independently, generating separate `.rrd` files per mask that can be viewed together.

## Overview

The per-mask tracking workflow splits mask instances, processes each one independently, and creates combined visualizations. This is useful for:
- Tracking different objects separately
- Comparing tracking quality per object
- Debugging individual mask/object tracking issues
- Analyzing object-specific trajectories

## Files

### Core Utilities (in `/utils`)

- **`mask_instance_utils.py`**: Split NPZ files by mask instances
  - `get_mask_instance_ids()`: Get list of instance IDs from NPZ
  - `split_npz_by_instances()`: Create separate NPZ per instance
  - `create_single_instance_npz()`: Extract single instance to new NPZ
  - `verify_instance_npz()`: Verify instance NPZ is valid

- **`rerun_combine_utils.py`**: Combine multiple RRD files
  - `combine_rrd_files_python()`: Merge RRDs (metadata only)
  - `create_viewing_script()`: Generate shell script to view all RRDs
  - `combine_tracking_results()`: Full combination workflow

### Orchestration Scripts

- **`process_masks_independently.py`**: Main pipeline orchestrator
  - Splits NPZ by instances
  - Creates query points per instance
  - Runs tracking per instance
  - Combines results for visualization

- **`scripts/run_per_mask_tracking.sh`**: Example bash wrapper
  - Shows how to integrate into existing workflows
  - Configurable parameters
  - Example for your RH20T human data

## Usage

### Basic Usage

```bash
# Process all masks independently
python process_masks_independently.py \
  --npz path/to/masks.npz \
  --mask-key sam2_masks \
  --tracker mvtracker \
  --output-dir ./tracking_results \
  --base-name scene_tracking \
  --use-first-frame
```

### With Your Existing Pipeline

Add to `scripts/run_human_example.sh` after SAM2 tracking:

```bash
# ... existing pipeline creates sam2 masks ...

SAMPLE_PATH_HAND_TRACKED="third_party/HOISTFormer/sam2_tracking_output/${TASK_FOLDER}_processed_hand_tracked_hoist_sam2.npz"

# Run per-mask tracking
echo "Running per-mask independent tracking"
python process_masks_independently.py \
  --npz "$SAMPLE_PATH_HAND_TRACKED" \
  --mask-key sam2_masks \
  --tracker mvtracker \
  --output-dir "./tracking_per_mask/${TASK_FOLDER}" \
  --base-name "${TASK_FOLDER}" \
  --frames-before 3 \
  --frames-after 10 \
  --use-first-frame

# View results
bash "./tracking_per_mask/${TASK_FOLDER}/view_${TASK_FOLDER}_mvtracker_combined.sh"
```

### Command-Line Options

```
--npz PATH                Input NPZ with multiple mask instances [required]
--mask-key KEY            Mask key in NPZ (default: sam2_masks)
--tracker NAME            Tracker: mvtracker, spatialtrackerv2, cotracker3_offline
--output-dir PATH         Output directory for all results [required]
--base-name NAME          Base name for output files (default: tracking)

# Query Point Generation
--frames-before N         Frames before contact (default: 3)
--frames-after N          Frames after contact (default: 10)
--use-first-frame         Use first frame for all cameras (recommended)

# Tracking Parameters
--temporal-stride N       Temporal stride (default: 1)
--spatial-downsample N    Spatial downsampling (default: 1)
--depth-estimator NAME    Depth estimator (default: gt)
--depth-cache-dir PATH    Depth cache directory

# Advanced
--skip-split              Skip splitting if instances already exist
--instances-dir PATH      Use pre-split instance NPZ files
```

## Workflow Steps

The pipeline executes these steps automatically:

### 1. Split NPZ by Instances
```python
# Input:  masks.npz with {'instance_0': [...], 'instance_1': [...]}
# Output: masks_instance_0.npz, masks_instance_1.npz, ...
```

### 2. Create Query Points per Instance
```python
# For each instance NPZ:
#   Input:  masks_instance_0.npz
#   Output: masks_instance_0_query.npz (with query_points array)
```

### 3. Run Tracking per Instance
```python
# For each query NPZ:
#   Input:  masks_instance_0_query.npz
#   Output: tracking_mvtracker_instance_0.rrd
```

### 4. Combine Results
```python
# Creates:
#   - tracking_mvtracker_combined.rrd (metadata)
#   - view_tracking_mvtracker_combined.sh (viewing script)
```

## Output Structure

```
tracking_results/
├── instances/
│   ├── scene_instance_0.npz
│   ├── scene_instance_0_query.npz
│   ├── scene_instance_1.npz
│   └── scene_instance_1_query.npz
├── scene_mvtracker_instance_0.rrd
├── scene_mvtracker_instance_1.rrd
├── scene_mvtracker_combined.rrd (metadata only)
└── view_scene_mvtracker_combined.sh (executable)
```

## Viewing Results

### View All Masks Together
```bash
# Use the generated viewing script
bash tracking_results/view_scene_mvtracker_combined.sh

# Or directly with rerun CLI
rerun tracking_results/scene_mvtracker_instance_*.rrd
```

### View Individual Masks
```bash
# View single mask
rerun tracking_results/scene_mvtracker_instance_0.rrd

# View specific masks
rerun tracking_results/scene_mvtracker_instance_0.rrd \
      tracking_results/scene_mvtracker_instance_1.rrd
```

## Integration with Existing Code

### Option 1: Replace Existing Tracking Section

Replace these lines in `run_human_example.sh`:
```bash
# OLD: Single combined tracking
python demo.py --sample-path "$SAMPLE_PATH_HAND_TRACKED" --tracker mvtracker
```

With:
```bash
# NEW: Per-mask independent tracking
python process_masks_independently.py \
  --npz "$SAMPLE_PATH_HAND_TRACKED" \
  --tracker mvtracker \
  --output-dir "./tracking_per_mask/${TASK_FOLDER}"
```

### Option 2: Add as Additional Analysis

Keep existing tracking and add per-mask as extra analysis:
```bash
# Existing combined tracking
python demo.py --sample-path "$SAMPLE_PATH_HAND_TRACKED" --tracker mvtracker \
  --rrd "./mvtracker_demo_combined.rrd"

# Additional per-mask analysis
echo "Running per-mask analysis..."
python process_masks_independently.py \
  --npz "$SAMPLE_PATH_HAND_TRACKED" \
  --tracker mvtracker \
  --output-dir "./tracking_per_mask/${TASK_FOLDER}"
```

## Advanced Usage

### Compare Multiple Trackers per Mask
```bash
# Run mvtracker per mask
python process_masks_independently.py \
  --npz masks.npz --tracker mvtracker --output-dir ./results_mvt

# Run spatialtrackerv2 per mask
python process_masks_independently.py \
  --npz masks.npz --tracker spatialtrackerv2 --output-dir ./results_spat

# Compare both
rerun ./results_mvt/*.rrd ./results_spat/*.rrd
```

### Reuse Pre-Split Instances
```bash
# First run: splits and tracks
python process_masks_independently.py \
  --npz masks.npz --output-dir ./results --tracker mvtracker

# Later: reuse split files with different tracker
python process_masks_independently.py \
  --npz masks.npz --output-dir ./results_v2 --tracker spatialtrackerv2 \
  --skip-split --instances-dir ./results/instances
```

### Process Specific Instances Only

Manually create instance NPZ files for specific masks:
```python
from utils.mask_instance_utils import create_single_instance_npz

# Extract only instance_0
create_single_instance_npz(
    input_npz_path=Path("masks.npz"),
    instance_id="instance_0",
    output_npz_path=Path("mask_0_only.npz"),
)

# Then run normal pipeline on this single file
```

## Limitations & Notes

### RRD Combination
- Currently, Rerun SDK doesn't provide full programmatic RRD merging
- The "combined" RRD contains only metadata
- Use the generated viewing script or `rerun` CLI to view multiple files
- Future SDK versions may support true merging

### Performance
- Processing N masks takes ~N times longer than combined processing
- Each mask runs full tracking independently
- Use `--skip-split` to avoid re-splitting when experimenting

### Memory
- Each tracking instance loads full scene data
- Consider processing masks sequentially for large datasets
- The pipeline already does this automatically

## Troubleshooting

### "No mask instances found"
- Check `--mask-key` parameter matches your NPZ
- Verify masks are stored as dict: `{instance_id: array}`
- Use `python -c "import numpy as np; print(list(np.load('file.npz', allow_pickle=True).keys()))"` to inspect

### "demo.py failed"
- Check individual instance NPZ files in `output_dir/instances/`
- Verify query points were generated: `*_query.npz` files
- Run demo.py manually on one instance to debug

### "Combined RRD is empty"
- This is expected - it's metadata only
- Use the viewing script instead
- Or: `rerun output_dir/*.rrd`

## Example: Full Integration

```bash
#!/usr/bin/env bash
# Full pipeline with per-mask tracking

TASK="task_0034_user_0014_scene_0004_cfg_0006_human"
OUT_DIR="./data/human_high_res_filtered"

# Step 1: Your existing pipeline
python create_sparse_depth_map.py --task-folder "$TASK" --out-dir "$OUT_DIR"
cd third_party/hamer && python add_hand_mask_from_sam_to_rh20t.py && cd ../..
cd third_party/HOISTFormer && python demo_npz.py && python demo_sam2_object_tracking_debug.py && cd ../..

# Step 2: Create query points (original way - creates combined points)
SAMPLE_PATH="third_party/HOISTFormer/sam2_tracking_output/${TASK}_processed_hand_tracked_hoist_sam2.npz"
python create_query_points_from_masks.py --npz "$SAMPLE_PATH" --use-first-frame

# Step 3A: Original combined tracking
python demo.py --sample-path "$SAMPLE_PATH" --tracker mvtracker \
  --rrd "mvtracker_demo_combined.rrd"

# Step 3B: NEW - Per-mask independent tracking
python process_masks_independently.py \
  --npz "$SAMPLE_PATH" \
  --mask-key sam2_masks \
  --tracker mvtracker \
  --output-dir "./tracking_per_mask/${TASK}" \
  --base-name "${TASK}" \
  --use-first-frame

# View results
echo "View combined: rerun mvtracker_demo_combined.rrd"
echo "View per-mask: bash ./tracking_per_mask/${TASK}/view_${TASK}_mvtracker_combined.sh"
```
