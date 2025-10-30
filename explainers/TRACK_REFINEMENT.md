# Track Refinement and Video Export

This module provides tools to refine tracking results by filtering static tracks and exporting visualizations to per-camera videos.

## Overview

The workflow consists of three main steps:

1. **Track Refinement** - Filter out static tracks and remove invalid points
2. **Video Export** - Project 3D tracks to 2D camera views and create videos
3. **Analysis** - Review statistics and visualizations

## Components

### Utilities (in `utils/`)

- **`track_refinement_utils.py`** - Core refinement functions
  - Calculate track motion metrics
  - Filter static tracks based on motion threshold
  - Align tracks to query points (remove invalid tracks)
  - Compute comprehensive track statistics

- **`video_export_utils.py`** - Video generation functions
  - Project 3D tracks to 2D camera views
  - Draw tracks on video frames with trails and colors
  - Export videos per camera with track overlays

### Scripts (in root directory)

- **`refine_tracks.py`** - Main refinement script
  - Load tracking results from NPZ files
  - Apply motion filtering and alignment
  - Save refined NPZ with only moving, valid tracks
  - Generate detailed statistics

- **`export_tracks_to_videos.py`** - Video export script
  - Load refined tracking results
  - Project tracks to each camera view
  - Export videos per camera with track visualizations

- **`scripts/run_track_refinement_pipeline.sh`** - Complete workflow example

## Usage

### 1. Refine Tracks

Filter out static tracks and remove invalid points:

```bash
python refine_tracks.py \
  --input tracking_results.npz \
  --output refined_tracks.npz \
  --motion-threshold 0.01 \
  --motion-method total_displacement
```

**Parameters:**
- `--motion-threshold`: Minimum motion to keep a track (in world units, e.g., meters)
- `--motion-method`: Method to compute motion
  - `total_displacement` (default): Sum of all frame-to-frame distances
  - `max_displacement`: Maximum distance from starting point
  - `endpoint_distance`: Distance between first and last visible points
  - `std_displacement`: Standard deviation of positions

**Output:**
- Refined NPZ file with filtered tracks
- JSON file with refinement statistics

### 2. Export to Videos

Create per-camera videos with track overlays:

```bash
python export_tracks_to_videos.py \
  --input refined_tracks.npz \
  --output-dir ./videos \
  --fps 30 \
  --trail-length 10
```

**Parameters:**
- `--fps`: Frames per second (default: 30)
- `--trail-length`: Number of previous frames to show in trail
  - `0`: No trail, only current points
  - Positive number: Show last N frames
  - `-1`: Show entire trajectory

**Output:**
- One MP4 video per camera view
- Tracks colored by trajectory endpoint (rainbow colormap)
- Query frame highlighted with larger circle

### 3. Batch Processing

Process multiple NPZ files at once:

```bash
# Refine multiple files
python refine_tracks.py \
  --input-dir ./tracking_per_mask \
  --output-dir ./refined_tracks \
  --motion-threshold 0.01

# Export videos for all refined files
python export_tracks_to_videos.py \
  --input-dir ./refined_tracks \
  --output-dir ./videos \
  --fps 30
```

### 4. Complete Pipeline

Run the entire workflow with the example script:

```bash
bash scripts/run_track_refinement_pipeline.sh
```

This script:
1. Refines tracks from tracking results
2. Exports per-camera videos
3. Prints summary and output locations

## Data Format

### Input NPZ Requirements

The input NPZ file should contain:

**Required keys:**
- `tracks_3d` or `traj_e`: 3D trajectories `[T, N, 3]` in world coordinates
- `visibilities` or `vis_e`: Visibility mask `[T, N]` (boolean)
- `rgbs`: RGB frames `[V, T, 3, H, W]` (for video export)
- `intrinsics`: Camera intrinsics `[V, 3, 3]` (for video export)
- `extrinsics`: Camera extrinsics `[V, 4, 4]` (for video export)

**Optional keys:**
- `query_points`: Query points `[N, 4]` where cols=`[t, x, y, z]`
- `camera_ids`: Camera identifiers (list/array of strings)

### Output NPZ Format

The refined NPZ contains:
- `tracks_3d`: Refined 3D trajectories `[T, N_refined, 3]`
- `visibilities`: Refined visibility mask `[T, N_refined]`
- `query_points`: Refined query points `[N_refined, 4]` (if provided)
- `refinement_stats`: JSON string with statistics
- All original keys (rgbs, intrinsics, extrinsics, etc.)

## Examples

### Example 1: Basic Refinement

```bash
# Refine tracks with default settings
python refine_tracks.py \
  --input mvtracker_demo_hands.npz \
  --output refined_tracks.npz

# Export videos
python export_tracks_to_videos.py \
  --input refined_tracks.npz \
  --output-dir ./videos
```

### Example 2: Custom Motion Threshold

```bash
# More aggressive filtering (higher threshold)
python refine_tracks.py \
  --input tracking_results.npz \
  --output refined_aggressive.npz \
  --motion-threshold 0.05 \
  --motion-method max_displacement
```

### Example 3: Per-Mask Tracking Results

```bash
# Refine each mask's tracks separately
python refine_tracks.py \
  --input-dir ./tracking_per_mask/task_0034 \
  --output-dir ./refined_tracks/task_0034 \
  --motion-threshold 0.01

# Export videos for each mask
python export_tracks_to_videos.py \
  --input-dir ./refined_tracks/task_0034 \
  --output-dir ./videos/task_0034 \
  --trail-length 20
```

### Example 4: Long Trails for Analysis

```bash
# Export with full trajectory trails
python export_tracks_to_videos.py \
  --input refined_tracks.npz \
  --output-dir ./videos_full_trails \
  --trail-length -1 \
  --fps 15
```

## Refinement Statistics

The refinement process generates detailed statistics saved in JSON format:

```json
{
  "n_original": 1000,
  "alignment": {
    "n_original": 1000,
    "n_valid": 950,
    "n_invalid": 50,
    "fraction_valid": 0.95
  },
  "motion_filtering": {
    "n_original": 950,
    "n_moving": 300,
    "n_static": 650,
    "fraction_moving": 0.316,
    "motion_threshold": 0.01,
    "motion_min": 0.0001,
    "motion_max": 2.5,
    "motion_mean": 0.15,
    "motion_median": 0.008
  },
  "n_refined": 300,
  "total_reduction_factor": 3.33
}
```

## Visualization Features

Videos include:
- **Colored tracks**: Rainbow colormap based on trajectory endpoint
- **Track trails**: Showing recent history with fading effect
- **Query frame highlighting**: Larger circle at query frame
- **Visibility filtering**: Only show tracks when visible and in frame

## Tips

1. **Choosing Motion Threshold**:
   - Start with `0.01` (1 cm for meter-based coordinates)
   - Check statistics to see motion distribution
   - Adjust based on `motion_median` and `motion_mean`

2. **Motion Methods**:
   - Use `total_displacement` for cumulative motion
   - Use `max_displacement` for maximum deviation
   - Use `endpoint_distance` for start-to-end motion

3. **Trail Length**:
   - Short trails (5-10): Good for dense tracking
   - Medium trails (20-30): Balance between visibility and clutter
   - Full trails (-1): Best for analysis and understanding motion

4. **Performance**:
   - Batch processing is more efficient for multiple files
   - Use `--quiet` flag to reduce console output
   - Videos are compressed with H.264 codec

## Integration with Existing Pipeline

This module integrates with the existing tracking pipeline:

```bash
# 1. Run tracking (existing pipeline)
bash scripts/run_human_example.sh

# 2. Refine tracks (new)
python refine_tracks.py \
  --input data/human_high_res_filtered/task_0034_processed_hand_tracked_hoist_sam2.npz \
  --output refined_tracks/task_0034_refined.npz

# 3. Export videos (new)
python export_tracks_to_videos.py \
  --input refined_tracks/task_0034_refined.npz \
  --output-dir videos/task_0034
```

## Troubleshooting

**Error: "No tracks found in NPZ"**
- Ensure NPZ has `tracks_3d` or `traj_e` key
- Check that NPZ is from tracking output (not preprocessing)

**Error: "Missing required keys"**
- For refinement: Only needs tracks and visibilities
- For video export: Needs tracks, visibilities, rgbs, intrinsics, extrinsics

**Videos look cluttered**
- Reduce `--trail-length`
- Increase `--motion-threshold` to filter more tracks

**No tracks remaining after refinement**
- Lower `--motion-threshold`
- Check if tracks are in correct coordinate system (world space)
- Review statistics to see motion distribution
