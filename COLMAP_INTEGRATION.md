# COLMAP Integration for RH20T Processing

This document describes the COLMAP integration features added to `create_sparse_depth_map.py`.

## Features

### 1. Camera Cropping (`--crop-camera`)

Crops all camera views to a region of interest (ROI) before processing. This reduces computational overhead and focuses on the relevant parts of the scene.

**How it works:**
- Projects a 3D bounding box (typically the gripper bbox) to each camera's image plane
- Computes a 2D bounding box that encompasses all projected points
- Adds a configurable margin around the bbox
- Crops all RGB and depth images accordingly
- Updates camera intrinsics to reflect the new image coordinates

**Arguments:**
- `--crop-camera`: Enable camera cropping (default: False)
- `--crop-margin <float>`: Margin ratio to add around bbox (default: 0.1, i.e., 10%)

**Example:**
```bash
python create_sparse_depth_map.py \
  --task-folder /path/to/task \
  --out-dir ./output \
  --crop-camera \
  --crop-margin 0.15 \
  --add-robot \
  --gripper-body-bbox
```

### 2. COLMAP Refinement (`--refine-colmap`)

Uses COLMAP's structure-from-motion pipeline to:
- Extract and match features across all camera views
- Reconstruct the 3D scene and camera poses
- Evaluate geometric consistency of each camera
- Optionally filter to the best N cameras

**How it works:**
1. Sets up a COLMAP workspace with all images and camera calibration
2. Runs COLMAP feature extraction
3. Runs COLMAP exhaustive feature matching
4. Runs COLMAP mapper to reconstruct the scene
5. Evaluates each camera based on the number of 3D points it observes
6. If `--limit-num-cameras` is set, keeps only the best N cameras

**Arguments:**
- `--refine-colmap`: Enable COLMAP refinement (default: False)
- `--limit-num-cameras <int>`: Keep only N best cameras (optional)
- `--colmap-workspace <path>`: Path to COLMAP workspace (optional, temp dir by default)

**Example:**
```bash
python create_sparse_depth_map.py \
  --task-folder /path/to/task \
  --out-dir ./output \
  --refine-colmap \
  --limit-num-cameras 4 \
  --colmap-workspace ./colmap_work
```

### 3. COLMAP Densification (`--colmap-densification`)

Uses COLMAP's multi-view stereo (MVS) pipeline to create a denser point cloud.

**How it works:**
1. Undistorts images using COLMAP's reconstruction
2. Runs patch-match stereo to compute dense depth maps
3. Fuses all depth maps into a single dense point cloud
4. Logs the result to Rerun for visualization

**Arguments:**
- `--colmap-densification`: Enable COLMAP densification (default: False)
- `--colmap-workspace <path>`: Path to COLMAP workspace (optional)

**Example:**
```bash
python create_sparse_depth_map.py \
  --task-folder /path/to/task \
  --out-dir ./output \
  --refine-colmap \
  --colmap-densification
```

## Complete Example

Process a task with all COLMAP features enabled:

```bash
python create_sparse_depth_map.py \
  --task-folder /path/to/task_folder \
  --high-res-folder /path/to/high_res_folder \
  --out-dir ./output \
  --max-frames 50 \
  --add-robot \
  --gripper-body-bbox \
  --crop-camera \
  --crop-margin 0.2 \
  --refine-colmap \
  --limit-num-cameras 6 \
  --colmap-densification \
  --colmap-workspace ./colmap_workspace
```

## Installation Requirements

COLMAP must be installed and available in your PATH. Installation instructions:
- Official website: https://colmap.github.io/install.html
- Ubuntu: `sudo apt install colmap`
- macOS: `brew install colmap`
- From source: See COLMAP documentation

## Implementation Details

### Camera Quality Scoring

Cameras are scored based on the number of 3D points they observe in the COLMAP reconstruction. Cameras that observe more points are considered more reliable for multi-view geometry.

### Coordinate Systems

- Input: Uses the RH20T coordinate system (right-hand, Z-up)
- COLMAP: Internally uses its own coordinate system
- Output: All results are converted back to RH20T coordinate system

### Performance Considerations

- **Feature extraction**: ~1-2 seconds per image (GPU accelerated if available)
- **Feature matching**: ~O(N²) where N is number of images
- **Mapping**: ~10-30 seconds for typical scenes
- **Densification**: ~5-10 minutes for typical scenes (most expensive step)

**Tip**: Use `--crop-camera` to reduce image resolution and speed up processing.

### Error Handling

The implementation is designed to be robust:
- If COLMAP is not installed, a helpful error message is shown
- If any COLMAP step fails, the pipeline continues with a warning
- Temporary workspaces are cleaned up automatically (unless `--colmap-workspace` is specified)

## Troubleshooting

### "COLMAP is not installed"
Install COLMAP following the official instructions: https://colmap.github.io/install.html

### "Feature extraction failed"
- Check that images are valid and not corrupted
- Ensure sufficient disk space for the database
- Try reducing image resolution with `--crop-camera`

### "Mapping failed"
- This can happen if cameras don't have sufficient overlap
- Try with more frames (`--max-frames`)
- Check that camera calibration is correct

### "No cameras selected"
- This means COLMAP couldn't reconstruct the scene
- Check input data quality
- Try without `--limit-num-cameras` first

## Git Commits

The COLMAP integration was added in the following commits:

1. `f83b48a` - Add COLMAP helper functions and argument parsing
2. `773f9d1` - Integrate COLMAP processing into main workflow
3. `dff40a3` - Implement camera cropping based on 3D bbox projection
4. `d74e6eb` - Fix newline characters in COLMAP file writing
5. `3f14f30` - Add COLMAP availability check with helpful error message

## Future Improvements

Potential enhancements for future work:
- [ ] Support for custom COLMAP parameters
- [ ] Parallel processing of multiple frames
- [ ] Integration with neural rendering methods
- [ ] Camera pose refinement using COLMAP's bundle adjustment
- [ ] Support for radial distortion models
