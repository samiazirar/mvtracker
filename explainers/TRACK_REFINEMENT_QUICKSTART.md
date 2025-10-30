# Track Refinement Quick Start

## What was implemented

A complete pipeline to refine tracking results and export them to per-camera videos:

1. **Utilities (isolated in `utils/` folder)**:
   - `utils/track_refinement_utils.py` - Motion filtering and track alignment
   - `utils/video_export_utils.py` - Video generation with track overlays

2. **Main Scripts**:
   - `refine_tracks.py` - Filter static tracks and align to query points
   - `export_tracks_to_videos.py` - Export tracks to per-camera videos

3. **Documentation**:
   - `docs/TRACK_REFINEMENT.md` - Complete documentation
   - `scripts/run_track_refinement_pipeline.sh` - Example workflow

## Quick Usage

### Single file workflow

```bash
# 1. Refine tracks (filter static, remove invalid)
python refine_tracks.py \
  --input tracking_results.npz \
  --output refined_tracks.npz \
  --motion-threshold 0.01

# 2. Export to videos
python export_tracks_to_videos.py \
  --input refined_tracks.npz \
  --output-dir ./videos \
  --fps 30 \
  --trail-length 10
```

### Complete pipeline with example script

```bash
bash scripts/run_track_refinement_pipeline.sh
```

### Batch processing

```bash
# Refine multiple NPZ files
python refine_tracks.py \
  --input-dir ./tracking_per_mask/task_0034 \
  --output-dir ./refined_tracks/task_0034 \
  --motion-threshold 0.01

# Export all to videos
python export_tracks_to_videos.py \
  --input-dir ./refined_tracks/task_0034 \
  --output-dir ./videos/task_0034
```

## What it does

### Track Refinement (`refine_tracks.py`)
1. **Removes invalid tracks** - Tracks with no visible points
2. **Filters static tracks** - Based on motion threshold
3. **Outputs**:
   - Refined NPZ with only moving tracks
   - JSON statistics (reduction factor, motion metrics, etc.)

### Video Export (`export_tracks_to_videos.py`)
1. **Projects 3D tracks to 2D** - For each camera view
2. **Draws tracks on frames** - With trails and colors
3. **Exports videos** - One per camera with track overlays

## Key Features

- **Isolated utilities** - All core functions in `utils/` folder
- **Motion-based filtering** - Multiple methods (total displacement, max displacement, etc.)
- **Track alignment** - Ensures tracks match query points
- **Per-camera videos** - Separate video for each camera view
- **Customizable visualization** - Trail length, colors, FPS
- **Batch processing** - Handle multiple NPZ files at once
- **Statistics tracking** - Detailed JSON output

## Output Examples

### Refinement Statistics
```json
{
  "n_original": 1000,
  "n_refined": 300,
  "total_reduction_factor": 3.33,
  "motion_filtering": {
    "motion_threshold": 0.01,
    "n_moving": 300,
    "n_static": 650
  }
}
```

### Video Output
- `videos/task_0034_cam_00.mp4`
- `videos/task_0034_cam_01.mp4`
- ...one per camera

## Git Commits

All changes committed in 5 separate commits:
1. `8c0e6be` - Track refinement utilities
2. `1f637bc` - Video export utilities
3. `71540e7` - Main refinement script
4. `2214ece` - Video export script
5. `0122d29` - Documentation and example

## Next Steps

For detailed documentation, see `docs/TRACK_REFINEMENT.md`
