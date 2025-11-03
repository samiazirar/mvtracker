# Quick Start: Track Refinement Pipeline

## Overview

The track refinement pipeline is now **fully compatible** with per-mask tracking output!

## What Changed?

✅ `demo.py` now exports tracking results as NPZ files (with `--save-npz`)  
✅ `process_masks_independently.py` automatically creates NPZ exports  
✅ `run_track_refinement_pipeline.sh` now works with per-mask tracking output  

## Complete Workflow

### 1. Run Per-Mask Tracking

```bash
bash scripts/run_per_mask_tracking.sh
```

**Output**:
```
tracking_per_mask/task_name/
├── task_name_spatialtrackerv2_instance_0.rrd          # Visualization
├── task_name_spatialtrackerv2_instance_0_tracked.npz  # Tracking results ← NEW!
├── task_name_spatialtrackerv2_instance_1.rrd
├── task_name_spatialtrackerv2_instance_1_tracked.npz  # Tracking results ← NEW!
└── ...
```

### 2. View Tracking Results

```bash
# View combined visualization
bash tracking_per_mask/task_name/view_*_combined.sh

# Or individual instance
rerun tracking_per_mask/task_name/*_instance_0.rrd
```

### 3. Refine Tracks and Export Videos (NEW!)

```bash
bash scripts/run_track_refinement_pipeline.sh
```

**What it does**:
1. Loads `*_tracked.npz` files
2. Filters out static tracks (motion < threshold)
3. Removes tracks with no visible points
4. Saves refined NPZ + statistics JSON
5. Exports per-camera visualization videos

**Output**:
```
refined_tracks/task_name/
├── task_name_instance_0_refined.npz
├── task_name_instance_0_refined_stats.json
├── task_name_instance_1_refined.npz
└── task_name_instance_1_refined_stats.json

track_videos/task_name/
├── task_name_instance_0/
│   ├── cam_00.mp4
│   ├── cam_01.mp4
│   └── ...
└── task_name_instance_1/
    ├── cam_00.mp4
    └── ...
```

### 4. View Results

```bash
# Check statistics
cat refined_tracks/task_name/*_instance_0_refined_stats.json

# Play videos
vlc track_videos/task_name/task_*_instance_0/cam_00.mp4

# Or use file manager
nautilus track_videos/task_name/
```

## Configuration

Edit `scripts/run_track_refinement_pipeline.sh` to customize:

```bash
# Which task to process
TASK_FOLDER="task_0035_user_0020_scene_0006_cfg_0006_human"

# Which tracker was used
TRACKER="spatialtrackerv2"  # Must match run_per_mask_tracking.sh

# Refinement parameters
MOTION_THRESHOLD="0.01"      # Minimum motion in meters
MOTION_METHOD="total_displacement"

# Video parameters
FPS="30.0"
TRAIL_LENGTH="10"            # Frames of trail to show
```

## Troubleshooting

### Error: "No tracking NPZ files found"

**Cause**: Per-mask tracking was run with older version without NPZ export.

**Solution**: Re-run the tracking:
```bash
bash scripts/run_per_mask_tracking.sh
```

### Error: "tracks_3d not found in NPZ"

**Cause**: Using wrong NPZ file (input masks instead of tracking results).

**Solution**: Ensure you're using files with `*_tracked.npz` suffix, not the instance NPZ files in `instances/` directory.

### Videos are too fast/slow

**Solution**: Adjust `FPS` variable in the refinement script:
```bash
FPS="10.0"  # Slower
FPS="60.0"  # Faster
```

### Too many/few tracks in videos

**Solution**: Adjust `MOTION_THRESHOLD`:
```bash
MOTION_THRESHOLD="0.001"  # Keep more tracks (less filtering)
MOTION_THRESHOLD="0.1"    # Keep fewer tracks (more filtering)
```

## Example: Process Different Task

```bash
# 1. Edit run_per_mask_tracking.sh
vim scripts/run_per_mask_tracking.sh
# Change: TASK_FOLDER="task_0034_..."

# 2. Run tracking
bash scripts/run_per_mask_tracking.sh

# 3. Edit run_track_refinement_pipeline.sh
vim scripts/run_track_refinement_pipeline.sh
# Change: TASK_FOLDER="task_0034_..." (same as above)

# 4. Run refinement
bash scripts/run_track_refinement_pipeline.sh
```

## Next Steps

- Analyze statistics JSON to understand track quality
- Compare different trackers (mvtracker vs spatialtrackerv2)
- Use refined NPZ files for further analysis
- Share videos for visual inspection
