# Creating Query Points from HOISTFormer Masks

This script (`create_query_points_from_masks.py`) creates query points for MVTracker by lifting HOISTFormer mask detections to 3D around contact frames.

## Overview

The script processes NPZ files produced by `demo_npz.py` (HOISTFormer) that contain:
- `hoist_masks`: Detected object/hand instance masks
- `hoist_contact_frames`: Frame numbers where hand-object contact was detected
- Standard data: `depths`, `intrs`, `extrs`, `camera_ids`

It generates `query_points` in the format required by `demo.py`: a `[N, 4]` array with columns `[frame_idx, x, y, z]`.

## Usage

### Basic Usage

```bash
python create_query_points_from_masks.py \
    --npz path/to/file_hoist.npz \
    --frames-before 3 \
    --frames-after 1
```

This will:
1. Load the HOISTFormer output NPZ
2. For each detected instance, find the contact frame(s)
3. Select frames: `[contact - 3, ..., contact, ..., contact + 1]`
4. Lift all mask pixels to 3D world coordinates
5. Save a new NPZ with `query_points` field

### Example

```bash
# Process HOISTFormer output with default settings
python create_query_points_from_masks.py \
    --npz third_party/HOISTFormer/hoist_output/task_0045_hoist.npz

# Custom frame range: 5 frames before, 2 after contact
python create_query_points_from_masks.py \
    --npz third_party/HOISTFormer/hoist_output/task_0045_hoist.npz \
    --frames-before 5 \
    --frames-after 2

# Limit points per frame to reduce memory
python create_query_points_from_masks.py \
    --npz third_party/HOISTFormer/hoist_output/task_0045_hoist.npz \
    --frames-before 3 \
    --frames-after 1 \
    --max-points-per-frame 50
```

## Arguments

- `--npz`: Path to input NPZ file (required)
  - Must contain `hoist_masks` and `hoist_contact_frames`
  
- `--frames-before`: Number of frames before contact to include (default: 3)
  - If contact is at frame 22, with `--frames-before 3`, includes frames 19, 20, 21
  
- `--frames-after`: Number of frames after contact to include (default: 1)
  - If contact is at frame 22, with `--frames-after 1`, includes frame 23
  
- `--min-depth`: Minimum valid depth in meters (default: 1e-6)
  - Filters out zero/invalid depths for sparse depth data
  
- `--max-depth`: Maximum valid depth in meters (default: 10.0)
  
- `--max-points-per-frame`: Limit on points per frame (default: no limit)
  - If exceeded, randomly samples this many points
  - Useful for controlling memory usage
  
- `--output`: Output NPZ path (default: input path with `_query` suffix)

## Output Format

The output NPZ contains:
- All original data from the input file
- `query_points`: `[N, 4]` array with columns `[frame_idx, x, y, z]`
  - `frame_idx`: Frame number (0-indexed)
  - `x, y, z`: 3D coordinates in world space (meters)
- `query_generation_config`: Metadata about generation parameters

## Frame Selection Logic

For contact frame 22 with `--frames-before 3` and `--frames-after 1`:
```
Selected frames: [19, 20, 21, 22, 23]
                  └──────┬─────┘ │  └┬┘
                   before      contact after
```

The script will:
1. Extract masks for the instance at these frames
2. Lift non-zero mask pixels to 3D using sparse depth
3. Combine all points with their frame indices

## Sparse Depth Handling

The script is designed for **sparse depth** data:
- Only pixels with depth > `min_depth` are used
- Zero or invalid depths are automatically filtered out
- This is normal for sparse depth from methods like COLMAP

If you see "No valid 3D points" messages, this usually means:
- The mask pixels don't have depth values (they're in holes of the sparse depth)
- The object/mask appeared in earlier/later frames with better depth coverage

## Integration with demo.py

Use the output NPZ directly with MVTracker:

```bash
python demo.py \
    --sample-path third_party/HOISTFormer/hoist_output/task_0045_hoist_query.npz \
    --tracker mvtracker \
    --rerun save
```

The `query_points` field will be automatically used for tracking.

## Pipeline Example

Complete workflow from raw data to tracking:

```bash
# 1. Run HOISTFormer to detect hand-object interactions
cd third_party/HOISTFormer
bash run_demo_npz.sh  # Produces *_hoist.npz

# 2. Create query points from detections
cd /workspace
python create_query_points_from_masks.py \
    --npz third_party/HOISTFormer/hoist_output/task_0045_hoist.npz \
    --frames-before 3 \
    --frames-after 1

# 3. Run MVTracker with the query points
python demo.py \
    --sample-path third_party/HOISTFormer/hoist_output/task_0045_hoist_query.npz \
    --tracker mvtracker \
    --rerun save
```

## Notes

- The script handles multiple instances automatically
- Each instance gets its own set of query points
- Contact frames can differ across cameras for the same instance
- Points are generated for all cameras where contact was detected
- Frame indices are clipped to valid range [0, T-1]
