# Fixes Summary - Tracking and Timing Issues

## Issues Fixed

### 1. ✅ Timing Synchronization (Point Cloud Speed Mismatch)
**Problem**: The sensor data point cloud appeared to move much faster than the bounding boxes and tracking overlays in videos and Rerun visualization.

**Root Cause**: 
- Robot points, bboxes, and tracking were all logged using different time coordinate systems
- `rr.set_time_seconds("frame", idx / fps)` calculations were inconsistent
- FPS calculations (`clip_fps`, `effective_fps`, `default_fps`) varied and caused desync

**Solution**:
- Changed ALL Rerun logging to use `rr.set_time_sequence("frame", idx)` with direct frame indices
- Removed FPS-based time calculations for Rerun timeline
- Now robot, bboxes, and tracks all use the same frame index (0, 1, 2, 3...) for perfect synchronization

**Changed Files**: `create_sparse_depth_map.py`
- Robot point cloud logging
- Gripper bbox logging (contact, body, fingertip)
- Gripper pad points logging
- TCP points logging
- Object points logging
- Track visualization logging
- SAM segmentation logging

**Commit**: `17d5dbe` - "Fix timing synchronization: use frame sequence instead of time_seconds for all Rerun logging"

---

### 2. ✅ Color Gradient Timing  
**Problem**: Color gradient (turbo colormap) was not correctly synchronized with frames.

**Root Cause**:
- Color gradient was using `time_fps` calculations that didn't match actual frame timing
- `rr.set_time_seconds()` with FPS division caused color mismatch

**Solution**:
- Color gradient now directly maps to frame index: `time_colors_rgb[t]` for frame `t`
- Changed from `rr.set_time_seconds("frame", t / time_fps)` to `rr.set_time_sequence("frame", t)`
- Each frame gets its correct color from the turbo colormap

**Commit**: Same as #1 (`17d5dbe`)

---

### 3. ✅ CoTracker 2D-Only Mode
**Problem**: User requested ability to run pure 2D CoTracker on RGB frames without any 3D tracking or MVTracker overhead.

**Solution**: Added new `--tracker=cotracker2d_only` mode

**New Features**:
- `--tracker=cotracker2d_only` - Run CoTracker in 2D-only mode
- `--cotracker-grid-size N` - Control track density (default: 20)
- No 3D reconstruction, no depth maps, no MVTracker
- Creates per-camera 2D tracking videos with dense optical flow visualization
- Uses grid sampling (no manual query points needed)
- Skips bounding box computation entirely

**New Function**: `_track_cotracker_2d_only()`
- Runs CoTrackerOfflineWrapper on each camera view independently
- Returns per-view 2D track results
- Compatible with existing video export pipeline

**Commit**: `8611797` - "Add CoTracker 2D-only mode for pure 2D RGB tracking"

---

## Testing Commands

### Test 1: Verify Timing Synchronization (with robot and bboxes)
```bash
python create_sparse_depth_map.py \
  --task-folder /data/rh20t_api/data/test_data_full_rgb_upscaled_depth/uncompressed_low_res_data/task_0065_user_0010_scene_0009_cfg_0004 \
  --high-res-folder /data/rh20t_api/data/test_data_full_rgb_upscaled_depth/rgb_data/RH20T_cfg4/task_0065_user_0010_scene_0009_cfg_0004 \
  --out-dir ./data/test_timing_fix \
  --max-frames 50 \
  --add-robot \
  --gripper-bbox \
  --gripper-body-bbox \
  --gripper-fingertip-bbox \
  --track-gripper \
  --tracker mvtracker \
  --export-track-video \
  --no-sharpen-edges-with-mesh
```

**What to Check**:
- Open the `.rrd` file in Rerun viewer
- Verify robot and bboxes move in sync (no speed mismatch)
- Check that time-colored tracks have correct gradient progression
- Orange (gripper), red (body), and blue (fingertip) bboxes should track together

---

### Test 2: CoTracker 2D-Only Mode (no 3D, just 2D tracking)
```bash
python create_sparse_depth_map.py \
  --task-folder /data/rh20t_api/data/test_data_full_rgb_upscaled_depth/uncompressed_low_res_data/task_0065_user_0010_scene_0009_cfg_0004 \
  --high-res-folder /data/rh20t_api/data/test_data_full_rgb_upscaled_depth/rgb_data/RH20T_cfg4/task_0065_user_0010_scene_0009_cfg_0004 \
  --out-dir ./data/test_cotracker2d \
  --max-frames 50 \
  --tracker cotracker2d_only \
  --cotracker-grid-size 30 \
  --export-track-video \
  --no-add-robot \
  --no-gripper-bbox \
  --no-sharpen-edges-with-mesh
```

**What to Check**:
- Look for videos in `./data/test_cotracker2d/track_videos/cotracker2d_only/`
- Should see dense 2D tracking on RGB frames
- Tracks should follow scene motion (objects, gripper, etc.)
- No 3D bounding boxes or robot visualization (since we disabled them)

---

### Test 3: Compare Before vs After (Timing Fix)
Run the same command twice - once on old commit, once on new:

**Old (before fix):**
```bash
git checkout 1189095  # Before timing fix
python create_sparse_depth_map.py [same args as Test 1]
# Save output to ./data/before_fix/
```

**New (after fix):**
```bash
git checkout 8611797  # Latest commit
python create_sparse_depth_map.py [same args as Test 1]
# Save output to ./data/after_fix/
```

**Compare**:
- Open both `.rrd` files side by side in Rerun
- OLD: Point cloud should move faster than bboxes
- NEW: Everything should move in perfect sync

---

## Key Files Changed

1. **`create_sparse_depth_map.py`**:
   - All `rr.set_time_seconds()` → `rr.set_time_sequence()`
   - Added `_track_cotracker_2d_only()` function
   - Added CoTracker 2D-only mode support in tracking pipeline
   - Fixed color gradient timing
   - Added `--cotracker-grid-size` argument

---

## Commit History

```bash
git log --oneline feature/improve-bbox

8611797 Add CoTracker 2D-only mode for pure 2D RGB tracking
17d5dbe Fix timing synchronization: use frame sequence instead of time_seconds for all Rerun logging
e69b8cd Save current state before fixes - timing and tracking issues identified
```

---

## Expected Outputs

### Rerun Visualization (`.rrd` file)
- **Timeline**: Single frame sequence (0, 1, 2, ..., N)
- **Robot**: Gray point cloud moving smoothly with correct timing
- **Bboxes**: Orange (contact), red (body), blue (fingertip) in sync
- **Tracks**: Time-colored gradient (turbo) matching frame progression
- **All entities**: Perfectly synchronized, no speed mismatches

### Tracking Videos (`.mp4` files)
- **Location**: `<out-dir>/track_videos/`
- **CoTracker 2D-only**: Dense 2D tracks on RGB, per camera
- **MVTracker**: 3D tracks projected to 2D, with visibility handling
- **Frame rate**: Consistent across all videos

---

## Notes

- **FPS is now only used for video export**, not for Rerun timeline
- **Frame indices are the source of truth** for synchronization
- **CoTracker 2D-only mode** is faster and simpler if you only need 2D tracking
- **All tracking modes** now use the same visualization pipeline
