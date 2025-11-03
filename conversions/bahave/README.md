# BEHAVE Dataset Conversion to NPZ

This directory contains scripts to convert the BEHAVE dataset into `.npz` format that can be used for tracking tasks with `demo.py` and other tracking frameworks.

## Quick Start (One Command)

Convert a BEHAVE scene and make it ready for `demo.py`:

```bash
python conversions/behave_to_demo.py --scene Date01_Sub01_backpack_back
```

This will create a `_demo.npz` file ready to use:

```bash
python demo.py --sample-path conversions/behave_converted/Date01_Sub01_backpack_back_demo.npz \
    --tracker mvtracker --depth_estimator gt --rerun save
```

## BEHAVE Dataset Structure

The BEHAVE dataset contains multi-view RGB-D sequences of human-object interactions. Each scene has:
- 4 Kinect cameras (k0-k3)
- Multiple timestamps (t####.000)
- RGB images, depth maps, person masks, and object masks
- Camera calibration data (intrinsics and extrinsics)

## Scripts

### Core Scripts

1. **`behave_to_demo.py`** - **ONE-STEP CONVERTER (Recommended)**
   - Converts BEHAVE scene directly to demo.py-compatible format
   - Combines conversion + adaptation in one command
   - Ready to use immediately with tracking

2. **`behave_to_npz.py`** - Low-level converter
   - Converts BEHAVE scene to intermediate `.npz` format
   - Useful for custom processing or inspection

3. **`adapt_behave_for_demo.py`** - Format adapter
   - Adapts intermediate `.npz` to demo.py format
   - Broadcasts camera parameters to temporal dimension
   - Unprojects 2D query points to 3D

### Utility Scripts

4. **`inspect_behave_npz.py`** - Visualization tool
   - Inspects and visualizes converted `.npz` files
   - Shows statistics and generates preview images

5. **`batch_convert_behave.py`** - Batch processor
   - Convert multiple scenes automatically
   - Scene discovery and filtering

6. **`example_usage.py`** - Usage examples
   - Shows how to load and use converted data
   - Examples for single/multi-view tracking

## Usage

### Convert a single scene:

```bash
python conversions/behave_to_npz.py \
    --scene Date01_Sub01_backpack_back \
    --max_frames 50 \
    --downscale_factor 2
```

**Arguments:**
- `--behave_root`: Path to BEHAVE dataset root (default: `/data/behave-dataset/behave_all`)
- `--scene`: Scene name (e.g., `Date01_Sub01_backpack_back`) **[Required]**
- `--output_dir`: Output directory for `.npz` files (default: `./conversions/behave_converted`)
- `--mask_type`: Type of mask to use (`person` or `hand`, default: `person`)
- `--num_cameras`: Number of cameras to process (default: 4)
- `--num_query_points`: Number of query points to extract per mask (default: 256)
- `--downscale_factor`: Factor to downscale images (default: 2, i.e., 2048×1536 → 1024×768)
- `--max_frames`: Maximum number of frames to process (default: None = all frames)
- `--save_rerun`: Save rerun visualization

### Inspect converted file:

```bash
python conversions/inspect_behave_npz.py \
    conversions/behave_converted/Date01_Sub01_backpack_back.npz
```

This will:
- Print scene info and statistics
- Display camera parameters
- Generate a visualization showing RGB, masks, and query points
- Save the visualization as a `.png` file

## Output Format

The `.npz` file contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `rgbs` | (C, T, 3, H, W) | RGB images (uint8) |
| `depths` | (C, T, H, W) | Depth maps in meters (float32) |
| `masks` | (C, T, H, W) | Person/hand masks (uint8) |
| `intrinsics` | (C, 3, 3) | Camera intrinsics matrices (float32) |
| `extrinsics` | (C, 3, 4) | Camera extrinsics [R\|t] cam-to-world (float32) |
| `query_points` | (T, C, N, 2) | Query points [x, y] from masks (float32) |
| `scene_name` | scalar | Scene name (str) |
| `mask_type` | scalar | Mask type used (str) |
| `num_frames` | scalar | Number of frames (int) |
| `num_cameras` | scalar | Number of cameras (int) |
| `image_height` | scalar | Image height (int) |
| `image_width` | scalar | Image width (int) |
| `downscale_factor` | scalar | Downscaling applied (int) |

Where:
- C = number of cameras (default: 4)
- T = number of timestamps/frames
- H, W = image height and width
- N = number of query points (default: 256)

## Query Points

Query points are extracted from the person/hand masks using one of two methods:

1. **Random sampling** (default): Randomly sample N points from foreground pixels
2. **Grid sampling**: Sample points on a grid within the mask bounding box

These query points can be used as initial tracking points for point tracking algorithms.

## Example Scenes

List available scenes:
```bash
ls /data/behave-dataset/behave_all/ | grep "Date01"
```

Some example scenes:
- `Date01_Sub01_backpack_back`
- `Date01_Sub01_backpack_hand`
- `Date01_Sub01_basketball`
- `Date01_Sub01_chairblack_sit`
- etc.

## Batch Conversion

### Option 1: Using Python batch script (recommended)

List available scenes:
```bash
python conversions/batch_convert_behave.py --list_scenes
```

Filter scenes by pattern:
```bash
python conversions/batch_convert_behave.py --list_scenes --pattern "Date01_Sub01"
```

Convert specific scenes:
```bash
python conversions/batch_convert_behave.py \
    --scenes Date01_Sub01_backpack_back Date01_Sub01_basketball \
    --max_frames 100 \
    --inspect
```

Convert all scenes matching a pattern:
```bash
python conversions/batch_convert_behave.py \
    --pattern "basketball" \
    --max_frames 50 \
    --downscale_factor 2 \
    --inspect
```

Convert all scenes in the dataset:
```bash
python conversions/batch_convert_behave.py \
    --max_frames 100 \
    --downscale_factor 2
```

### Option 2: Using bash script

Edit `batch_convert_behave.sh` to specify scenes, then run:
```bash
./conversions/batch_convert_behave.sh
```

## Notes

- Default downscale factor is 2, reducing 2048×1536 images to 1024×768
- Depth values are converted from millimeters to meters
- Camera extrinsics are in camera-to-world format (not world-to-camera)
- The first camera (k0) is typically NOT at the origin; each camera has its own extrinsics
- Query points with coordinates (0, 0) indicate no valid mask pixels

## Camera Calibration

The BEHAVE dataset provides:

**Intrinsics** (per camera):
- Stored in `/behave_all/calibs/intrinsics/{camera_id}/calibration.json`
- Format: fx, fy, cx, cy for color images (2048×1536)

**Extrinsics** (per camera per date/location):
- Stored in `/behave_all/calibs/{date}/config/{camera_id}/config.json`
- Format: 3×3 rotation matrix + 3×1 translation vector (camera-to-world)

## Integration with Tracking

The output `.npz` format is designed to be compatible with existing tracking frameworks. The query points can be used as:

1. Initial points for point tracking (e.g., TAP-Vid, CoTracker)
2. Mask-based region tracking
3. Multi-view 3D reconstruction seeds

Example usage in tracking code:
```python
import numpy as np

data = np.load('conversions/behave_converted/Date01_Sub01_backpack_back.npz')

rgbs = data['rgbs']  # (4, T, 3, H, W)
query_points = data['query_points']  # (T, 4, 256, 2)
intrinsics = data['intrinsics']  # (4, 3, 3)
extrinsics = data['extrinsics']  # (4, 3, 4)

# Use for tracking...
```
