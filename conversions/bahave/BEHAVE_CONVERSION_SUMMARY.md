# BEHAVE Dataset Conversion - Summary

This document summarizes the BEHAVE dataset conversion scripts created for the mvtracker project.

## Overview

Created a complete pipeline to convert the BEHAVE dataset (human-object interaction multi-view RGBD data) into `.npz` format suitable for point tracking and 3D reconstruction tasks.

## Files Created

### 1. `conversions/behave_to_npz.py` (Main Conversion Script)

**Purpose:** Convert a single BEHAVE scene to `.npz` format

**Key Features:**
- Loads multi-view RGB-D data from 4 Kinect cameras
- Extracts camera intrinsics and extrinsics from calibration files
- Processes person/hand masks
- Extracts query points from masks for tracking
- Supports image downscaling to reduce file size
- Optional frame limiting for faster testing
- Optional rerun visualization output

**Usage:**
```bash
python conversions/behave_to_npz.py --scene Date01_Sub01_backpack_back --max_frames 50
```

### 2. `conversions/inspect_behave_npz.py` (Inspection Tool)

**Purpose:** Visualize and inspect converted `.npz` files

**Key Features:**
- Prints detailed scene information
- Shows camera parameters
- Displays data statistics
- Generates visualization with RGB, masks, and query points
- Saves visualization as PNG

**Usage:**
```bash
python conversions/inspect_behave_npz.py conversions/behave_converted/Date01_Sub01_backpack_back.npz
```

### 3. `conversions/batch_convert_behave.py` (Batch Processor)

**Purpose:** Convert multiple scenes in batch

**Key Features:**
- Auto-discover scenes in BEHAVE dataset
- Filter scenes by pattern
- Convert multiple scenes sequentially
- Optional inspection after each conversion
- Progress tracking and summary statistics

**Usage:**
```bash
# List all scenes
python conversions/batch_convert_behave.py --list_scenes

# Convert all basketball scenes
python conversions/batch_convert_behave.py --pattern "basketball" --max_frames 100 --inspect
```

### 4. `conversions/batch_convert_behave.sh` (Bash Alternative)

Bash script version of the batch converter for simple use cases.

### 5. `conversions/README.md` (Documentation)

Complete documentation with:
- Dataset structure explanation
- Script usage examples
- Output format specification
- Camera calibration details
- Integration guide

## Output Format

The converted `.npz` files contain:

```
rgbs:        (C, T, 3, H, W)  - RGB images
depths:      (C, T, H, W)     - Depth maps (meters)
masks:       (C, T, H, W)     - Person/hand masks
intrinsics:  (C, 3, 3)        - Camera intrinsics
extrinsics:  (C, 3, 4)        - Camera extrinsics (cam-to-world)
query_points: (T, C, N, 2)    - Query points [x, y] from masks

Where: C=4 cameras, T=frames, H×W=768×1024 (default), N=256 points
```

## Key Design Decisions

1. **Camera-to-World Extrinsics:** Store extrinsics in camera-to-world format to match BEHAVE's native format

2. **Downscaling:** Default 2x downscaling (2048×1536 → 1024×768) balances quality and file size

3. **Query Points:** Extract 256 query points per mask using random sampling from foreground pixels

4. **Depth Units:** Convert from millimeters (BEHAVE format) to meters for consistency

5. **Mask Type:** Support both person and hand masks, with person as default

## Testing

Successfully tested on `Date01_Sub01_backpack_back`:
- Converted 10 frames (test run)
- Output size: ~55 MB (10 frames, 4 cameras, downscale=2)
- Generated visualization confirms correct data loading
- Query points properly extracted from masks

## Integration with MVTracker

The output format is designed to be compatible with existing tracking code:

```python
import numpy as np

data = np.load('scene.npz')
rgbs = data['rgbs']           # Ready for tracking input
query_points = data['query_points']  # Initial tracking points
intrinsics = data['intrinsics']     # For projection/unprojection
extrinsics = data['extrinsics']     # For multi-view geometry
```

## Next Steps

Potential enhancements:
1. Add hand detection/segmentation for better hand-specific query points
2. Support for object masks and object-specific tracking
3. 3D point cloud generation from RGBD data
4. Camera trajectory visualization
5. Dataset statistics and quality metrics
6. Support for other BEHAVE-like datasets

## File Sizes (Estimated)

For a typical scene with 100 frames, 4 cameras, and 2x downscaling:
- RGB data: ~300 MB
- Depth data: ~300 MB
- Masks: ~75 MB
- Metadata: <1 MB
- **Total: ~550-600 MB per scene**

For reference, the full BEHAVE dataset has ~300 scenes.

## Performance

Conversion speed (approximate):
- ~8-10 frames/second
- 100 frames takes ~10-15 seconds
- Full scene (200+ frames) takes ~30-40 seconds

The bottleneck is disk I/O reading JPEG/PNG files.
