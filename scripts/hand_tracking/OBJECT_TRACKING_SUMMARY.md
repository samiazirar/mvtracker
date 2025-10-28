# Summary: Object Tracking Scripts for Grasped Objects

## What Was Created

I've created a simple and understandable pipeline to segment and track objects being grasped by hands using SAM2.1.

### Files Created

1. **`scripts/filter_grasped_object_with_sam.py`** (423 lines)
   - Main Python script for object tracking
   - Clear structure with sections for constants, helpers, and main processing
   - Well-documented with docstrings and comments

2. **`scripts/run_object_tracking_example.sh`** (58 lines)
   - Example bash script with hardcoded paths (same style as `run_human_example.sh`)
   - Easy to configure for different tasks
   - Includes helpful echo messages

3. **`scripts/identify_grasp_timestamps.py`** (283 lines)
   - Helper script to automatically identify grasp frames
   - Analyzes finger distances from hand keypoints
   - Suggests timestamps and creates visualization plots

4. **`scripts/OBJECT_TRACKING_README.md`** (258 lines)
   - Comprehensive documentation
   - Quick start guide
   - Detailed explanations of how everything works
   - Troubleshooting section

## How It Works

### Input
- Takes a `*_hand_tracked.npz` file (from the existing hand tracking pipeline)
- Requires a list of frame timestamps where gripping occurs

### Processing
1. **Loads hand tracking data**: RGB images, depths, hand masks, and 3D keypoints
2. **Identifies gripper position**: Projects finger keypoints to 2D at grasp frames
3. **Segments object with SAM2.1**: Uses finger positions as prompts, tracks across frames
4. **Generates outputs**: New NPZ with object masks + overlay videos

### Output
- **NPZ file** with new fields:
  - `sam_object_masks`: [C, T, H, W] - Binary masks of grasped object
  - `sam_object_scores`: [C, T, H, W] - Confidence scores
  - `grasp_timestamps`: [K] - Frame indices used for prompting
  
- **Videos**: One per camera showing:
  - Object mask overlay in cyan
  - Prompt points (red dots) on grasp frame

## Code Structure (Easy to Understand)

### Main Functions

```python
# 1. Load data
data = load_hand_tracked_npz(npz_path)

# 2. Get prompt points from finger positions
prompt_points = get_gripper_prompt_points(
    query_points, frame_idx, intr, extr, img_shape
)

# 3. Segment and track with SAM2.1
masks, scores = segment_object_with_sam2(
    video_frames, grasp_frame_idx, prompt_points, sam_predictor
)

# 4. Save results
np.savez_compressed(output_path, **payload)
```

### Key Features

✅ **Simple to understand**: Clear variable names, extensive comments
✅ **Modular**: Each function does one thing
✅ **Type hints**: All functions have clear input/output types
✅ **Error handling**: Checks for missing data, out-of-bounds, etc.
✅ **Progress bars**: Uses tqdm for visual feedback
✅ **Flexible**: Works with static or time-varying intrinsics/extrinsics

## Example Usage

### Quick Start (Hardcoded Paths)

```bash
# 1. Identify grasp timestamps automatically (NEW!)
python scripts/identify_grasp_timestamps.py \
  --npz data/human_high_res_filtered/task_*_hand_tracked.npz \
  --threshold 0.05 \
  --plot grasp_analysis.png

# This will output suggested timestamps, e.g.: "10 15 20 25"

# 2. Edit the timestamps in run_object_tracking_example.sh
vim scripts/run_object_tracking_example.sh
# Set: GRASP_TIMESTAMPS="10 15 20 25"

# 3. Run
bash scripts/run_object_tracking_example.sh
```

### Custom Usage

```bash
python scripts/filter_grasped_object_with_sam.py \
  --npz data/human_high_res_filtered/task_*_hand_tracked.npz \
  --grasp-timestamps 10 15 20 25 \
  --output-dir output/object_tracking \
  --device cuda
```

## Integration with Existing Pipeline

The scripts integrate seamlessly with the existing workflow:

```
create_sparse_depth_map.py
  ↓ creates *_processed.npz
third_party/hamer/add_hand_mask_from_sam_to_rh20t.py
  ↓ creates *_hand_tracked.npz
scripts/filter_grasped_object_with_sam.py  ← NEW!
  ↓ creates *_with_object_masks.npz
demo.py (MVTracker)
  ↓ tracks object in 3D
```

## What Makes It "Simple and Understandable"

1. **Clear Comments**: Every section is explained
2. **Descriptive Names**: `find_gripper_center_in_image()` not `find_gc()`
3. **Logical Flow**: Step-by-step processing, no jumping around
4. **Constants Section**: All magic numbers at the top with explanations
5. **Helper Functions**: Small, focused functions that do one thing
6. **Comprehensive Docs**: README explains every detail

## Example Output Structure

```
data/human_high_res_filtered/object_tracking/
├── task_0045_user_0020_scene_0004_cfg_0006_human_processed_hand_tracked_with_object_masks.npz
├── task_0045_user_0020_scene_0004_cfg_0006_human_processed_hand_tracked_cam400210_object_tracked.mp4
├── task_0045_user_0020_scene_0004_cfg_0006_human_processed_hand_tracked_cam400212_object_tracked.mp4
└── ...
```

## Key Differences from Hand Tracking

| Feature | Hand Tracking | Object Tracking (NEW) |
|---------|---------------|---------------------|
| Target | Hands | Objects being grasped |
| Mask Color | Magenta | Cyan |
| Prompts | Hand bounding boxes | Finger keypoints |
| Temporal | Frame-by-frame | Video-based (SAM2.1) |
| Output Field | `sam_hand_masks` | `sam_object_masks` |

## Next Steps

After running the pipeline, you can:

1. **Visualize**: Watch the videos to verify object segmentation
2. **Track in 3D**: Use MVTracker with the new NPZ
3. **Refine**: Adjust grasp timestamps if segmentation is poor
4. **Integrate**: Use object masks for downstream tasks

## Technical Details

- **SAM2.1 Video Predictor**: Ensures temporal consistency
- **Multi-Camera Support**: Processes each camera independently
- **Depth Integration**: Can lift masks to 3D using depth maps
- **Robust Projection**: Handles static/dynamic camera parameters

## Questions?

See `scripts/OBJECT_TRACKING_README.md` for:
- Detailed workflow examples
- Troubleshooting tips
- Parameter explanations
- Integration guides
