# Quick Start Guide: BEHAVE Dataset Conversion

## Installation & Setup

No additional dependencies needed beyond the main project requirements.

## Step 1: Convert a Scene

Convert a single scene with limited frames (for testing):

```bash
python conversions/behave_to_npz.py \
    --scene Date01_Sub01_backpack_back \
    --max_frames 50 \
    --downscale_factor 2
```

Output: `conversions/behave_converted/Date01_Sub01_backpack_back.npz`

## Step 2: Inspect the Conversion

Visualize the converted data:

```bash
python conversions/inspect_behave_npz.py \
    conversions/behave_converted/Date01_Sub01_backpack_back.npz
```

This generates:
- Terminal output with statistics
- `Date01_Sub01_backpack_back.png` visualization

## Step 3: Use the Data

Run example usage:

```bash
python conversions/example_usage.py \
    conversions/behave_converted/Date01_Sub01_backpack_back.npz
```

Or load in your own code:

```python
import numpy as np

# Load data
data = np.load('conversions/behave_converted/Date01_Sub01_backpack_back.npz')

# Access data
rgbs = data['rgbs']              # (4, T, 3, H, W)
query_points = data['query_points']  # (T, 4, 256, 2)
intrinsics = data['intrinsics']      # (4, 3, 3)
extrinsics = data['extrinsics']      # (4, 3, 4)

# For single-camera tracking
camera_0_rgbs = rgbs[0]          # (T, 3, H, W)
camera_0_queries = query_points[0, 0]  # (256, 2) from first frame
```

## Step 4: Batch Convert Multiple Scenes

List available scenes:

```bash
python conversions/batch_convert_behave.py --list_scenes
```

Convert multiple scenes:

```bash
python conversions/batch_convert_behave.py \
    --pattern "Date01_Sub01" \
    --max_frames 100 \
    --inspect
```

## Common Use Cases

### Use Case 1: Single-Camera Point Tracking

```python
import numpy as np
import torch

# Load scene
data = np.load('scene.npz')
camera_id = 0

# Get single camera data
rgbs = torch.from_numpy(data['rgbs'][camera_id]).float() / 255.0  # (T, 3, H, W)
queries = torch.from_numpy(data['query_points'][0, camera_id])     # (N, 2)

# Use with your tracking model
# tracks = model(rgbs, queries)
```

### Use Case 2: Multi-View Tracking

```python
import numpy as np

# Load scene
data = np.load('scene.npz')

# Get all camera data
rgbs = data['rgbs']              # (4, T, 3, H, W)
intrinsics = data['intrinsics']  # (4, 3, 3)
extrinsics = data['extrinsics']  # (4, 3, 4)

# Use with multi-view tracking
# tracks_3d = multi_view_tracker(rgbs, intrinsics, extrinsics)
```

### Use Case 3: 3D Reconstruction

```python
import numpy as np

# Load scene
data = np.load('scene.npz')

# Get depth and camera params
depths = data['depths']          # (4, T, H, W)
intrinsics = data['intrinsics']  # (4, 3, 3)
extrinsics = data['extrinsics']  # (4, 3, 4)

# Unproject to 3D
# point_cloud = unproject_depth(depths, intrinsics, extrinsics)
```

## Tips & Best Practices

1. **Start Small**: Use `--max_frames 50` for testing before processing full scenes

2. **Downscaling**: Default `--downscale_factor 2` is recommended (balances quality and size)

3. **Memory**: Processing ~100 frames with 4 cameras needs ~2-3 GB RAM

4. **Storage**: Each scene (~100 frames) produces ~550 MB `.npz` file

5. **Query Points**: Automatically extracted from masks, ready for tracking

6. **Inspect Always**: Run `inspect_behave_npz.py` to verify conversions

## Troubleshooting

**Problem:** Scene not found

```bash
# Check available scenes
ls /data/behave-dataset/behave_all/
```

**Problem:** Out of memory

```bash
# Reduce frames or increase downscaling
python conversions/behave_to_npz.py \
    --scene SCENE_NAME \
    --max_frames 50 \
    --downscale_factor 4
```

**Problem:** Invalid query points

Check the mask visualization - if masks are empty, query points will be zeros.

## Next Steps

1. Process more scenes for your experiments
2. Integrate with your tracking pipeline
3. Experiment with different `num_query_points` values
4. Use hand masks if needed (`--mask_type hand`)

## Full Documentation

See `conversions/README.md` for complete documentation.
