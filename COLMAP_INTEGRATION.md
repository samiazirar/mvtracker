# COLMAP Integration for RH20T Processing

This document describes the COLMAP integration features added to `create_sparse_depth_map.py`.

## Features

### COLMAP Refinement (`--refine-colmap`)

Uses pycolmap's structure-from-motion pipeline to:
- Extract and match features across all camera views
- Reconstruct the 3D scene and camera poses
- Evaluate geometric consistency of each camera
- Filter to the best N cameras based on geometric quality

**How it works:**
1. Sets up a COLMAP workspace with all images and camera calibration
2. Runs pycolmap feature extraction
3. Runs pycolmap exhaustive feature matching
4. Runs pycolmap mapper to reconstruct the scene
5. Evaluates each camera based on the number of 3D points it observes
6. If `--limit-num-cameras` is set, keeps only the best N cameras
7. **Critically: Updates all NPZ output arrays to contain only the selected cameras**

**Arguments:**
- `--refine-colmap`: Enable COLMAP refinement (default: False)
- `--limit-num-cameras <int>`: Keep only N best cameras (optional)

**Example:**
```bash
python create_sparse_depth_map.py \
  --task-folder /path/to/task \
  --out-dir ./output \
  --refine-colmap \
  --limit-num-cameras 4
```

## Complete Example

Process a task with COLMAP camera filtering:

```bash
python create_sparse_depth_map.py \
  --task-folder /path/to/task_folder \
  --high-res-folder /path/to/high_res_folder \
  --out-dir ./output \
  --max-frames 50 \
  --refine-colmap \
  --limit-num-cameras 2
```

This will:
1. Process all available cameras
2. Run COLMAP to evaluate geometric consistency
3. Select the best 2 cameras
4. **Output NPZ file will contain only 2 cameras**

## Installation Requirements

pycolmap must be installed:
```bash
pip install pycolmap
```

## Implementation Details

### Camera Quality Scoring

Cameras are scored based on the number of 3D points they observe in the COLMAP reconstruction. Cameras that observe more points are considered more reliable for multi-view geometry.

### NPZ Output

**Important**: When `--limit-num-cameras` is used, the output NPZ file will contain data for ONLY the selected cameras. This means:
- `rgbs`: Shape will be (N, T, H, W, 3) where N is the number of selected cameras
- `depths`: Shape will be (N, T, H, W)
- `intrs`: Shape will be (N, T, 3, 3)
- `extrs`: Shape will be (N, T, 3, 4)
- `camera_ids`: Will contain only the selected camera IDs

This ensures downstream processing always works with the best cameras.

### Coordinate Systems

- Input: Uses the RH20T coordinate system (right-hand, Z-up)
- COLMAP: Internally uses its own coordinate system
- Output: All results are converted back to RH20T coordinate system

### Performance Considerations

- **Feature extraction**: ~1-2 seconds per image (GPU accelerated if available)
- **Feature matching**: ~O(N²) where N is number of images
- **Mapping**: ~10-30 seconds for typical scenes

## Troubleshooting

### "pycolmap not found"
Install pycolmap: `pip install pycolmap`

### "Feature extraction failed"
- Check that images are valid and not corrupted
- Ensure sufficient disk space for the database

### "Mapping failed"
- This can happen if cameras don't have sufficient overlap
- Try with more frames (`--max-frames`)
- Check that camera calibration is correct

### "No cameras selected"
- This means COLMAP couldn't reconstruct the scene
- Check input data quality

## Git Commits

The COLMAP integration was added in the following commits:

1. `f83b48a` - Add COLMAP helper functions and argument parsing
2. `773f9d1` - Integrate COLMAP processing into main workflow
3. `8eac31f` - Remove SAM2/crop features, switch to pycolmap API
4. `c012559` - Fix camera filtering to properly update NPZ arrays

## Changes from Original Design

**Removed Features:**
- Camera cropping (`--crop-camera`) - Not reliable
- SAM2 tracking integration - Moved to separate pipeline
- COLMAP densification - Too slow, use separate tools

**Simplified:**
- Using pycolmap Python API instead of subprocess calls
- Automatic temporary workspace management
- Focus on camera selection only
