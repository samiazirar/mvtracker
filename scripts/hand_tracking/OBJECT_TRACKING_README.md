# Object Tracking with SAM2.1 for Grasped Objects

This directory contains scripts to segment and track objects being grasped by hands using SAM2.1.

## Overview

The pipeline consists of:

1. **Hand tracking** (existing): Detects hands and generates SAM masks for hands
2. **Object tracking** (new): Uses finger positions during grasping to segment and track the grasped object

## Files

- `filter_grasped_object_with_sam.py` - Main Python script for object tracking
- `run_object_tracking_example.sh` - Example bash script with hardcoded paths
- `run_human_example.sh` - Existing script that generates hand-tracked data

## Quick Start

### Step 1: Generate Hand-Tracked Data

First, process your human demonstration to get hand tracking data:

```bash
bash scripts/run_human_example.sh
```

This creates:
- `data/human_high_res_filtered/task_*_processed_hand_tracked.npz`
- Videos with hand masks overlay

### Step 2: Track Grasped Object

Edit `scripts/run_object_tracking_example.sh` to specify:
- **TASK_FOLDER**: The task identifier
- **GRASP_TIMESTAMPS**: Frame indices where grasping occurs (e.g., "10 15 20 25")

Then run:

```bash
bash scripts/run_object_tracking_example.sh
```

This creates:
- `data/human_high_res_filtered/object_tracking/*_with_object_masks.npz` - NPZ with object masks
- `data/human_high_res_filtered/object_tracking/*_object_tracked.mp4` - Videos with object overlay

## How to Determine Grasp Timestamps

You need to identify frame indices where the hand is actively grasping an object. Several methods:

### Method 1: Manual Inspection
1. Watch the hand-tracking videos: `data/human_high_res_filtered/*_cam*_sam_hand.mp4`
2. Note frame numbers where fingers are closed around an object
3. Convert time to frame index: `frame = time_in_seconds * fps` (fps is typically 12)

### Method 2: Using Rerun Visualization
```bash
rerun data/human_high_res_filtered/*_visualization.rrd
```
- Scrub through timeline
- Identify frames where hand keypoints are close together
- Note the frame indices

### Method 3: Automatic Detection (RECOMMENDED)

Use the provided helper script to automatically identify grasp frames:

```bash
python scripts/hand_tracking/identify_grasp_timestamps.py \
  --npz data/human_high_res_filtered/task_*_hand_tracked.npz \
  --threshold 0.05 \
  --min-duration 3 \
  --plot grasp_analysis.png
```

This will:
- Compute finger distances across all frames
- Identify periods where fingers are close (grasping)
- Suggest specific timestamps to use
- Create a visualization plot

**Example output:**
```
[INFO] Found 2 grasp period(s):
  1. Frames 10-18 (duration: 9, avg dist: 0.0387m)
  2. Frames 25-32 (duration: 8, avg dist: 0.0412m)

[INFO] Suggested grasp timestamps for SAM prompting:
  10 14 18 25 28 32

[COMMAND] Use these timestamps with:
  bash scripts/run_object_tracking_example.sh
  # Set: GRASP_TIMESTAMPS="10 14 18 25 28 32"
```

**Parameters:**
- `--threshold`: Max finger distance to consider grasping (default: 0.05m)
- `--min-duration`: Min consecutive frames (default: 3)
- `--num-samples`: Samples per grasp period (default: 3)
- `--plot`: Save visualization to file (optional)

## Output Format

The output NPZ file contains all original fields plus:

```python
{
    # Original fields from hand_tracked.npz:
    "rgbs": [C, T, 3, H, W],           # RGB images
    "depths": [C, T, 1, H, W],         # Depth maps
    "sam_hand_masks": [C, T, H, W],    # Hand masks
    "query_points": [N, 4],            # Hand keypoints (t,x,y,z)
    
    # New fields:
    "sam_object_masks": [C, T, H, W],  # Object masks (UINT8)
    "sam_object_scores": [C, T, H, W], # Confidence scores (FLOAT32)
    "grasp_timestamps": [K],           # Frame indices used for prompting
}
```

## Advanced Usage

### Custom Script Call

```bash
python scripts/filter_grasped_object_with_sam.py \
  --npz path/to/task_*_hand_tracked.npz \
  --grasp-timestamps 10 15 20 25 \
  --output-dir ./output \
  --device cuda
```

### Parameters

- `--npz` (required): Path to hand-tracked NPZ file
- `--grasp-timestamps` (required): Space-separated frame indices where grasping occurs
- `--output-dir`: Output directory (default: "object_tracking_output")
- `--sam-config`: SAM2 config file (default: auto-detected)
- `--sam-checkpoint`: SAM2 checkpoint path (default: auto-detected)
- `--device`: cuda or cpu (default: cuda if available)

## How It Works

### 1. Load Hand Tracking Data
The script loads the `*_hand_tracked.npz` file which contains:
- RGB images for all cameras
- Depth maps
- Hand masks from SAM
- 3D query points derived from hand keypoints

### 2. Identify Gripper Position
At the specified grasp frames:
- Filters query points (hand keypoints) for that frame
- Projects them to 2D image coordinates
- Uses them as prompt points for SAM2.1

### 3. Segment Object with SAM2.1
- Initializes SAM2.1 video predictor
- Provides positive prompts at the grasp frame (finger positions)
- Propagates segmentation across all frames using temporal consistency

### 4. Generate Outputs
- Saves NPZ with object masks and scores
- Creates overlay videos showing:
  - Object mask in cyan
  - Prompt points (red dots) on grasp frame

## Visualization

The output videos show:
- **Cyan overlay**: Segmented object being grasped
- **Red dots** (on grasp frame): Prompt points used for SAM2.1

## Troubleshooting

### No valid prompt points found
- Check that grasp_timestamps are valid (0 <= t < num_frames)
- Ensure hand tracking detected hands at those frames
- Try different grasp timestamps

### Poor segmentation quality
- Ensure grasp frames show clear grasping (fingers around object)
- Try multiple grasp frames for better prompts
- Check that object is visible and has sufficient contrast

### CUDA out of memory
- Reduce video resolution
- Use fewer frames
- Switch to CPU: `--device cpu`

## Integration with MVTracker

The output NPZ can be used with MVTracker for 3D tracking:

```bash
python demo.py \
  --temporal_stride 1 \
  --spatial_downsample 1 \
  --depth_estimator gt \
  --sample-path data/human_high_res_filtered/object_tracking/*_with_object_masks.npz \
  --tracker cotracker3_offline
```

## Example Workflow

```bash
# 1. Process human demonstration
bash scripts/run_human_example.sh

# 2. Watch the hand-tracking video to identify grasp frames
# Example: frames 10-25 show clear grasping

# 3. Edit run_object_tracking_example.sh
#    Set GRASP_TIMESTAMPS="10 15 20 25"

# 4. Run object tracking
bash scripts/run_object_tracking_example.sh

# 5. Review results
ls data/human_high_res_filtered/object_tracking/
# - task_*_with_object_masks.npz
# - task_*_cam*_object_tracked.mp4

# 6. Visualize with Rerun (if available)
rerun data/human_high_res_filtered/object_tracking/*.rrd
```

## Notes

- The script uses the **middle grasp timestamp** as the primary prompt frame
- SAM2.1 provides temporal consistency across frames
- Object masks are saved per-camera for multi-view tracking
- The approach works best when:
  - Object is clearly visible during grasping
  - Hand is closed around object (not just touching)
  - Multiple grasp frames are provided for robustness

## Future Improvements

- [ ] Automatic grasp detection using finger distance
- [ ] Multi-object tracking
- [ ] Background/negative prompts for better segmentation
- [ ] Integration with HaMeR for refined finger positions
- [ ] Rerun visualization integration
