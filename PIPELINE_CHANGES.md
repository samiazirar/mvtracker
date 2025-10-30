# Pipeline Changes: Track Refinement Integration

## Summary

Made the track refinement pipeline compatible with per-mask tracking output by adding NPZ export functionality.

## Changes Made

### 1. Added NPZ Export to `demo.py`

**New argument**: `--save-npz`
- Saves tracking results (tracks_3d, visibilities, query_points) to NPZ file
- Compatible with `refine_tracks.py` expected format
- Includes camera data (rgbs, depths, intrinsics, extrinsics) for video export

**Keys saved in NPZ**:
- `tracks_3d`: [T, N, 3] - 3D trajectories
- `visibilities`: [T, N] - Visibility mask
- `query_points`: [N, 4] - Query points (t, x, y, z)
- `rgbs`, `depths`, `intrs`, `extrs`: Camera data
- `camera_ids`: Camera identifiers
- Metadata: `tracker`, `temporal_stride`, `spatial_downsample`

### 2. Updated `process_masks_independently.py`

**Function signature change**: `run_tracking_for_instance()`
- Added `output_npz` parameter (optional)
- Passes `--save-npz` to demo.py when specified

**All tracking calls updated** to create NPZ files:
- Single instance tracking: Creates `*_tracked.npz`
- Batched tracking: Creates `*_batch_N_tracked.npz`

### 3. Updated `scripts/run_track_refinement_pipeline.sh`

**Now fully functional**:
- Removed "TEMPLATE ONLY" warning
- Looks for `*_tracked.npz` files in tracking output directory
- Processes each tracked NPZ file through refinement pipeline
- Creates refined NPZ + statistics JSON + per-camera videos

**Input**: `./tracking_per_mask/${TASK_FOLDER}/*_tracked.npz`
**Output**: 
- `./refined_tracks/${TASK_FOLDER}/*_refined.npz`
- `./refined_tracks/${TASK_FOLDER}/*_refined_stats.json`
- `./track_videos/${TASK_FOLDER}/*/cam_*.mp4`

### 4. Updated `scripts/run_per_mask_tracking.sh`

**Documentation updated** to reflect new outputs:
- Now creates both RRD (visualization) AND NPZ (tracking results)
- Added step 3 in next steps: run refinement pipeline

## Complete Workflow

### Step 1: Per-Mask Tracking
```bash
bash scripts/run_per_mask_tracking.sh
```
**Creates**:
- `tracking_per_mask/task_name/task_name_tracker_instance_N.rrd` (visualization)
- `tracking_per_mask/task_name/task_name_tracker_instance_N_tracked.npz` (NEW!)

### Step 2: View Results
```bash
# View combined visualization
bash tracking_per_mask/task_name/view_task_name_tracker_combined.sh

# Or view individual
rerun tracking_per_mask/task_name/task_name_tracker_instance_0.rrd
```

### Step 3: Refine and Export Videos (NEW!)
```bash
bash scripts/run_track_refinement_pipeline.sh
```
**Creates**:
- Refined tracking NPZ files
- Statistics JSON (tracks filtered, motion stats, etc.)
- Per-camera visualization videos with trails

## Testing

To test the complete pipeline:

```bash
# 1. Run per-mask tracking (will now create NPZ files)
bash scripts/run_per_mask_tracking.sh

# 2. Run refinement (now compatible!)
bash scripts/run_track_refinement_pipeline.sh

# 3. View results
cat refined_tracks/task_name/*_refined_stats.json
vlc track_videos/task_name/instance_0/cam_00.mp4
```

## Backward Compatibility

- Scripts work with existing RRD files for visualization
- NPZ export is automatic (no configuration needed)
- Old workflow (just RRD viewing) still works
- New workflow (refinement + videos) now available

## File Size Considerations

NPZ files will increase disk usage:
- Each `*_tracked.npz` contains full tracking results + camera data
- For large datasets, consider cleaning up intermediate files after refinement
- Compressed NPZ format used to minimize size
