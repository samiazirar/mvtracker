# Summary: Per-Mask Independent Tracking Implementation

## What Was Created

Three new utility files and documentation to enable per-mask tracking without modifying `demo.py`:

### 1. `/workspace/utils/mask_instance_utils.py`
**Purpose:** Split NPZ files by mask instances

**Key Functions:**
- `get_mask_instance_ids()` - Extract list of instance IDs from NPZ
- `split_npz_by_instances()` - Create separate NPZ file for each mask instance
- `create_single_instance_npz()` - Extract single instance to new NPZ
- `verify_instance_npz()` - Validate instance NPZ structure

### 2. `/workspace/utils/rerun_combine_utils.py`
**Purpose:** Combine multiple RRD files for joint visualization

**Key Functions:**
- `combine_rrd_files_python()` - Merge RRDs (creates metadata RRD)
- `create_viewing_script()` - Generate bash script to view all RRDs together
- `combine_tracking_results()` - Complete combination workflow

**Note:** Rerun SDK doesn't yet support full programmatic RRD merging, so the utility creates:
- A metadata RRD file
- A viewing script that uses `rerun` CLI to view all files together

### 3. `/workspace/process_masks_independently.py`
**Purpose:** Main orchestration script for per-mask workflow

**Workflow:**
1. Load NPZ with multiple mask instances
2. Split into separate NPZ files (one per instance)
3. For each instance:
   - Create query points using existing `create_query_points_from_masks.py`
   - Run tracking using existing `demo.py`
   - Generate individual `.rrd` file
4. Combine all `.rrd` files for joint visualization

**Usage:**
```bash
python process_masks_independently.py \
  --npz path/to/masks.npz \
  --mask-key sam2_masks \
  --tracker mvtracker \
  --output-dir ./tracking_results \
  --base-name scene_tracking
```

### 4. `/workspace/scripts/run_per_mask_tracking.sh`
**Purpose:** Example bash wrapper showing integration

Shows how to integrate into your existing pipeline with proper paths and parameters for RH20T human data.

### 5. `/workspace/docs/PER_MASK_TRACKING.md`
**Purpose:** Complete documentation

Covers usage, integration, examples, troubleshooting, and workflow explanations.

## How It Works

### Data Flow

```
Input: masks.npz
  ├─ sam2_masks: {
  │    'instance_0': [C, T, H, W],
  │    'instance_1': [C, T, H, W]
  │  }
  └─ (other data: rgbs, depths, intrs, extrs, etc.)

Step 1: Split by instances
  ├─ masks_instance_0.npz (contains only instance_0)
  └─ masks_instance_1.npz (contains only instance_1)

Step 2: Create query points (uses existing script)
  ├─ masks_instance_0_query.npz (adds query_points array)
  └─ masks_instance_1_query.npz (adds query_points array)

Step 3: Run tracking (uses existing demo.py)
  ├─ tracking_mvtracker_instance_0.rrd
  └─ tracking_mvtracker_instance_1.rrd

Step 4: Combine for visualization
  ├─ tracking_mvtracker_combined.rrd (metadata)
  └─ view_tracking_mvtracker_combined.sh (viewing script)
```

## Integration Options

### Option 1: Add to Existing Script

Add to `scripts/run_human_example.sh` after SAM2 tracking:

```bash
# After this line:
SAMPLE_PATH_HAND_TRACKED="third_party/HOISTFormer/sam2_tracking_output/${TASK_FOLDER}_processed_hand_tracked_hoist_sam2.npz"

# Add per-mask tracking:
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

### Option 2: Use Standalone Script

```bash
bash scripts/run_per_mask_tracking.sh
```

Edit the script to set your task folder and parameters.

## Output Structure

```
tracking_per_mask/task_0034.../
├── instances/
│   ├── task_0034..._instance_0.npz
│   ├── task_0034..._instance_0_query.npz
│   ├── task_0034..._instance_1.npz
│   └── task_0034..._instance_1_query.npz
├── task_0034..._mvtracker_instance_0.rrd
├── task_0034..._mvtracker_instance_1.rrd
├── task_0034..._mvtracker_combined.rrd
└── view_task_0034..._mvtracker_combined.sh
```

## Viewing Results

### All masks together:
```bash
bash tracking_per_mask/task_0034.../view_task_0034..._mvtracker_combined.sh
```

Or directly:
```bash
rerun tracking_per_mask/task_0034.../*.rrd
```

### Individual masks:
```bash
rerun tracking_per_mask/task_0034.../task_0034..._mvtracker_instance_0.rrd
```

## Key Design Decisions

### Why Not Modify demo.py?
- Per your requirement: "i do not want to modify demo.py"
- Keeps your core tracking code unchanged
- Uses composition instead of modification
- Easier to maintain and update

### Why Split NPZ Files?
- `demo.py` expects single-instance query points
- Splitting creates clean per-instance inputs
- Each instance tracks independently
- No risk of inter-instance interference

### Why Shell Script for Viewing?
- Rerun SDK doesn't yet support full RRD merging programmatically
- Rerun CLI can view multiple files natively: `rerun file1.rrd file2.rrd`
- Shell script provides convenient wrapper
- When SDK adds merging, easy to update utilities

## Testing Your Setup

1. **Verify mask structure:**
```bash
python -c "
import numpy as np
data = np.load('your_file.npz', allow_pickle=True)
masks = data['sam2_masks'].item()
print(f'Found {len(masks)} instances: {list(masks.keys())}')
"
```

2. **Test splitting:**
```bash
python utils/mask_instance_utils.py \
  --npz your_file.npz \
  --output-dir ./test_split \
  --mask-key sam2_masks
```

3. **Run full pipeline:**
```bash
python process_masks_independently.py \
  --npz your_file.npz \
  --mask-key sam2_masks \
  --tracker mvtracker \
  --output-dir ./test_tracking \
  --base-name test
```

## What You Need to Change

**In your workflow:**
1. Identify where SAM2 tracking creates the final NPZ
2. Add call to `process_masks_independently.py` after that point
3. Optionally keep existing combined tracking for comparison

**Example for your `run_human_example.sh`:**
```bash
# Around line 124, after creating query points:
SAMPLE_PATH_HAND_TRACKED="third_party/HOISTFormer/sam2_tracking_output/${TASK_FOLDER}_processed_hand_tracked_hoist_sam2.npz"

# Add this:
python process_masks_independently.py \
  --npz "$SAMPLE_PATH_HAND_TRACKED" \
  --mask-key sam2_masks \
  --tracker mvtracker \
  --output-dir "./tracking_per_mask/${TASK_FOLDER}" \
  --base-name "${TASK_FOLDER}" \
  --use-first-frame
```

## Next Steps

1. **Test with your data:**
   ```bash
   bash scripts/run_per_mask_tracking.sh
   ```

2. **Verify masks separate correctly:**
   - Check Rerun visualization
   - Verify each instance tracks independently
   - Ensure mask IDs are consistent across frames

3. **Integrate into your pipeline:**
   - Add to `run_human_example.sh`
   - Or create separate analysis script

4. **Compare tracking methods:**
   - Run with different trackers (mvtracker, spatialtrackerv2)
   - View results side-by-side in Rerun

## Files You Can Delete (if needed)

All new files are self-contained:
- `/workspace/utils/mask_instance_utils.py`
- `/workspace/utils/rerun_combine_utils.py`
- `/workspace/process_masks_independently.py`
- `/workspace/scripts/run_per_mask_tracking.sh`
- `/workspace/docs/PER_MASK_TRACKING.md`

No modifications to existing code, so easy to remove if not needed.
