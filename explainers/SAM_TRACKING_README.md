# SAM Mask Tracking Pipeline with SpatialTrackerV2

This pipeline tracks all points within SAM2 masks across time using SpatialTrackerV2, then lifts them to 3D and visualizes in Rerun.

## Overview

The pipeline consists of 3 main components:

1. **`track_sam_masks_per_camera.py`** - Track SAM mask points per camera (2D tracking)
2. **`lift_and_visualize_tracks.py`** - Lift 2D tracks to 3D and create Rerun visualization
3. **Bash scripts** - Convenience wrappers to run the full pipeline

## Quick Start

### Option 1: Quick and Dirty (Recommended for Testing)

Edit the NPZ file path in `run_quick_sam_tracking.sh` and run:

```bash
bash run_quick_sam_tracking.sh
```

### Option 2: Full Pipeline Script

Edit configuration in `run_sam_mask_tracking_pipeline.sh` and run:

```bash
bash run_sam_mask_tracking_pipeline.sh
```

### Option 3: Manual Step-by-Step

```bash
# Step 1: Track mask points per camera (2D tracking)
python track_sam_masks_per_camera.py \
    --npz third_party/HOISTFormer/sam2_tracking_output/task_0045_sam2.npz \
    --mask-key sam2_masks \
    --track-mode offline \
    --max-points-per-mask 500 \
    --output sam_tracks_per_camera.npz

# Step 2: Lift to 3D and visualize
python lift_and_visualize_tracks.py \
    --npz sam_tracks_per_camera.npz \
    --output sam_tracks_3d.rrd \
    --fps 10 \
    --spawn

# View the result
rerun sam_tracks_3d.rrd --web-viewer
```

## Pipeline Details

### Step 1: Track Mask Points Per Camera

**Script**: `track_sam_masks_per_camera.py`

**What it does**:
- Loads SAM2 masks from NPZ file
- For each camera independently:
  - Extracts all pixel coordinates from masks in the first frame
  - Uses SpatialTrackerV2 to track those points through time (2D tracking)
  - Returns tracks with visibility flags

**Key parameters**:
- `--npz`: Input NPZ file with SAM2 masks and camera data
- `--mask-key`: Key for masks in NPZ (default: "sam2_masks")
- `--track-mode`: "offline" (more accurate) or "online" (faster)
- `--max-points-per-mask`: Limit points per mask to avoid memory issues
  - Set to 500-1000 for testing
  - Remove or set high (5000+) to track all points
- `--fps`: Frame rate (should match your data)
- `--track-num`: Max VO points for tracker (default: 756)

**Output**:
- NPZ file with suffix `_tracks_per_camera.npz` containing:
  - `tracks_2d`: Dict {camera_id: [T, N, 2]} - 2D pixel coordinates
  - `visibility`: Dict {camera_id: [T, N]} - boolean visibility flags
  - `track_instance_ids`: Dict {camera_id: [N]} - instance ID for each track
  - All original data from input

### Step 2: Lift to 3D and Visualize

**Script**: `lift_and_visualize_tracks.py`

**What it does**:
- Loads 2D tracks from Step 1
- Lifts each track to 3D world coordinates using:
  - Depth maps (from sparse depth or original depth)
  - Camera intrinsics and extrinsics
- Combines tracks from all cameras
- Creates Rerun visualization with:
  - RGB point clouds (togglable)
  - Camera frustums
  - Track trajectories as colored lines
  - Track points with temporal animation

**Key parameters**:
- `--npz`: Input NPZ file with 2D tracks
- `--output`: Output .rrd file path
- `--fps`: Frame rate for visualization (default: 10)
- `--max-frames`: Limit frames for faster preview
- `--spawn`: Automatically open Rerun viewer

**Output**:
- .rrd file for viewing in Rerun
- Tracks colored by camera
- Temporal animation showing track evolution

## Input Requirements

Your NPZ file must contain:
- `sam2_masks` (or specified key): Dict of masks {instance_name: [C, T, H, W]}
- `rgbs`: RGB images [C, T, H, W, 3] or [C, T, 3, H, W]
- `depths`: Depth maps [C, T, H, W]
- `intrs`: Camera intrinsics [C, T, 3, 3] or [C, 3, 3]
- `extrs`: Camera extrinsics [C, T, 3, 4] or [C, 3, 4]
- `camera_ids`: Camera ID strings [C]

This format matches the output from:
- `demo_sam2_object_tracking_debug.py` (sam2_masks)
- `create_sparse_depth_map.py` with SAM hand masks

## Tracking All Points

To track **all points** in the masks (not just a subset):

1. Remove or increase `--max-points-per-mask`:
   ```bash
   # Track all points (no limit)
   python track_sam_masks_per_camera.py \
       --npz "$NPZ_FILE" \
       --mask-key sam2_masks \
       --track-mode offline
   ```

2. Or set a high limit:
   ```bash
   python track_sam_masks_per_camera.py \
       --npz "$NPZ_FILE" \
       --mask-key sam2_masks \
       --track-mode offline \
       --max-points-per-mask 10000
   ```

**Note**: Tracking many points (>5000) may require significant GPU memory. If you encounter OOM errors:
- Reduce `--max-points-per-mask`
- Use `--track-mode online` (less memory intensive)
- Process fewer cameras at once

## Visualization Tips

1. **In Rerun**:
   - Toggle layers on/off (RGB point clouds, tracks, cameras)
   - Use timeline to scrub through time
   - Tracks are colored by camera for easy identification
   - Trajectory lines show full path of each track

2. **Performance**:
   - Use `--max-frames` for quick preview
   - Full visualization with many tracks may load slowly

3. **View Options**:
   ```bash
   # Web viewer (lighter weight)
   rerun file.rrd --web-viewer
   
   # Desktop viewer (more features)
   rerun file.rrd
   ```

## Troubleshooting

### "No query points generated"
- Check that masks exist in first frame
- Verify mask key is correct (`--mask-key`)
- Check mask format (should be [C, T, H, W])

### "Out of memory" during tracking
- Reduce `--max-points-per-mask` (try 100-500)
- Use `--track-mode online`
- Process one camera at a time

### "No valid 3D points" during lifting
- Check depth maps (should have valid non-zero values)
- Verify depth format ([C, T, H, W])
- Check intrinsics/extrinsics are correct

### Tracks look wrong
- Verify camera extrinsics (world-to-camera transform)
- Check that fps matches your data
- Try `--track-mode offline` for more accuracy

## Example Workflow

Here's a complete example from SAM2 output to 3D visualization:

```bash
# 1. Run SAM2 tracking (if not done already)
cd third_party/HOISTFormer
bash run_sam2_tracking.sh

# 2. Track mask points with SpatialTrackerV2
cd /workspace
python track_sam_masks_per_camera.py \
    --npz third_party/HOISTFormer/sam2_tracking_output/task_0045_sam2.npz \
    --mask-key sam2_masks \
    --track-mode offline \
    --max-points-per-mask 500

# 3. Lift to 3D and visualize
python lift_and_visualize_tracks.py \
    --npz third_party/HOISTFormer/sam2_tracking_output/task_0045_sam2_tracks_per_camera.npz \
    --spawn \
    --fps 10

# 4. View result
rerun third_party/HOISTFormer/sam2_tracking_output/task_0045_sam2_tracks_per_camera_tracks_3d.rrd
```

## Output Files

After running the pipeline, you'll have:

1. **`*_tracks_per_camera.npz`**: 2D tracks per camera
   - Use this for further processing
   - Contains all tracking data

2. **`*_tracks_3d.rrd`**: Rerun visualization
   - Open with `rerun` command
   - Interactive 3D view of all tracks

## Next Steps

After generating tracks, you can:
- Use tracks for motion analysis
- Extract track statistics (velocity, acceleration)
- Filter tracks by visibility or instance
- Combine with other tracking methods
- Export tracks to other formats

## Related Scripts

- `create_query_points_from_masks.py` - Alternative: create query points for MVTracker
- `lift_and_visualize_masks.py` - Visualize masks in 3D (no tracking)
- `demo.py` - MVTracker demo (different tracking method)
