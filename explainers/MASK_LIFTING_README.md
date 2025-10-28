# Mask Lifting and 3D Visualization

This module provides utilities for lifting 2D segmentation masks to 3D point clouds and visualizing them in Rerun.

## Features

- **Lift 2D masks to 3D**: Convert binary segmentation masks to 3D point clouds using depth maps
- **Batch processing**: Handle multiple masks across cameras and frames efficiently
- **Rerun visualization**: Beautiful 3D visualization with customizable colors and styling
- **Flexible input**: Support for various data formats and camera configurations

## Core Functions

### `lift_mask_to_3d()`

Lift a single 2D mask to 3D world coordinates.

```python
from utils.pointcloud_utils import lift_mask_to_3d
import numpy as np

# Create a simple mask
mask = np.zeros((480, 640), dtype=bool)
mask[100:200, 150:250] = True

# Depth map and camera parameters
depth = np.random.rand(480, 640) * 5.0
K = np.eye(3)
K[0, 0] = K[1, 1] = 500.0  # Focal length
K[0, 2], K[1, 2] = 320, 240  # Principal point
E = np.eye(4)[:3, :]  # Identity transform

# Lift to 3D
points_3d, colors = lift_mask_to_3d(mask, depth, K, E)
print(f"Generated {len(points_3d)} 3D points")
```

### `lift_mask_to_3d_batch()`

Batch version for processing multiple masks.

```python
from utils.pointcloud_utils import lift_mask_to_3d_batch

# Multi-camera, multi-frame data
masks = np.random.rand(3, 50, 480, 640) > 0.5  # [C, T, H, W]
depths = np.random.rand(3, 50, 480, 640) * 5.0
intrs = np.tile(np.eye(3)[None, None], (3, 50, 1, 1))
extrs = np.tile(np.eye(4)[:3, :][None, None], (3, 50, 1, 1))

# Lift all masks
points_list, colors_list = lift_mask_to_3d_batch(
    masks, depths, intrs, extrs
)
print(f"Processed {len(points_list)} masks")
```

### `visualize_mask_3d()`

Visualize a single mask in Rerun.

```python
from utils.visualization_utils import visualize_mask_3d
import rerun as rr

# Initialize Rerun
rr.init("mask_viz", spawn=True)

# Visualize mask
num_points = visualize_mask_3d(
    mask=mask,
    depth=depth,
    intr=K,
    extr=E,
    entity_path="world/my_mask",
    color=np.array([255, 0, 255], dtype=np.uint8),  # Magenta
    radius=0.01,
)
print(f"Visualized {num_points} points")
```

### `visualize_masks_batch()`

Batch visualization for multiple masks.

```python
from utils.visualization_utils import visualize_masks_batch
import rerun as rr

# Initialize Rerun
rr.init("batch_masks", spawn=True)

# Visualize all masks
stats = visualize_masks_batch(
    masks=masks,
    depths=depths,
    intrs=intrs,
    extrs=extrs,
    entity_base_path="world/masks",
    camera_ids=["cam0", "cam1", "cam2"],
    color=np.array([0, 255, 255], dtype=np.uint8),  # Cyan
    fps=30.0,
)

print(f"Total points visualized: {stats['total_points']}")
```

## Command-Line Tools

### `lift_and_visualize_masks.py`

Standalone script for visualizing masks from NPZ files.

```bash
# Basic usage
python lift_and_visualize_masks.py \
    --npz data/hand_tracked.npz \
    --mask-key sam_hand_masks \
    --output masks_3d.rrd

# With custom magenta color for hands
python lift_and_visualize_masks.py \
    --npz data/hand_tracked.npz \
    --mask-key sam_hand_masks \
    --color 255 0 255 \
    --max-frames 50

# Using RGB colors from images
python lift_and_visualize_masks.py \
    --npz data/processed.npz \
    --use-rgb-colors \
    --spawn  # Automatically open Rerun viewer
```

### `demo_mask_lifting.py`

Interactive demos showing the functionality.

```bash
# Run all demos
python demo_mask_lifting.py

# Run specific demo
python demo_mask_lifting.py --demo 1  # Single mask
python demo_mask_lifting.py --demo 2  # Batch processing
python demo_mask_lifting.py --demo 3  # Generate test NPZ
```

## NPZ Data Format

Expected format for NPZ files:

```python
{
    # Required
    "masks": np.ndarray,        # [C, T, H, W] bool - Binary segmentation masks
    "depths": np.ndarray,       # [C, T, H, W] float32 - Depth maps in meters
    "intrs": np.ndarray,        # [C, T, 3, 3] or [C, 3, 3] - Camera intrinsics
    "extrs": np.ndarray,        # [C, T, 3, 4] or [C, 3, 4] - Camera extrinsics
    
    # Optional
    "rgbs": np.ndarray,         # [C, T, H, W, 3] or [C, T, 3, H, W] uint8 - RGB images
    "camera_ids": np.ndarray,   # [C] object - Camera identifier strings
}
```

Where:
- `C` = number of cameras
- `T` = number of frames/timesteps
- `H` = image height
- `W` = image width

## Examples

### Example 1: Hand Tracking Visualization

```python
import numpy as np
import rerun as rr
from utils.visualization_utils import visualize_masks_batch

# Load data
data = np.load("hand_tracked.npz")

# Initialize Rerun
rr.init("hand_tracking", spawn=True)
rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

# Visualize hand masks in magenta
stats = visualize_masks_batch(
    masks=data["sam_hand_masks"],
    depths=data["depths"][:, :, 0],  # Remove channel dim if needed
    intrs=data["intrs"],
    extrs=data["extrs"],
    entity_base_path="world/hands",
    camera_ids=data["camera_ids"],
    color=np.array([255, 0, 255], dtype=np.uint8),
    fps=30.0,
)

print(f"Visualized {stats['total_points']} hand points")
```

### Example 2: Object Segmentation

```python
# Visualize multiple object masks with different colors
objects = {
    "cup": {"mask_key": "cup_masks", "color": [255, 0, 0]},      # Red
    "plate": {"mask_key": "plate_masks", "color": [0, 255, 0]},  # Green
    "fork": {"mask_key": "fork_masks", "color": [0, 0, 255]},    # Blue
}

for obj_name, obj_info in objects.items():
    visualize_masks_batch(
        masks=data[obj_info["mask_key"]],
        depths=data["depths"],
        intrs=data["intrs"],
        extrs=data["extrs"],
        entity_base_path=f"world/objects/{obj_name}",
        color=np.array(obj_info["color"], dtype=np.uint8),
        fps=30.0,
    )
```

### Example 3: Depth Filtering

```python
# Visualize only nearby objects (depth < 2 meters)
visualize_masks_batch(
    masks=data["object_masks"],
    depths=data["depths"],
    intrs=data["intrs"],
    extrs=data["extrs"],
    entity_base_path="world/nearby_objects",
    min_depth=0.1,   # Ignore invalid/too-close points
    max_depth=2.0,   # Only show objects within 2 meters
    color=np.array([255, 255, 0], dtype=np.uint8),  # Yellow
)
```

## Performance Tips

1. **Batch processing**: Use `*_batch()` functions for multiple masks - much faster than loops
2. **Depth filtering**: Set appropriate `min_depth`/`max_depth` to reduce point count
3. **Frame limiting**: Use `max_frames` parameter to process subset of frames during development
4. **Point radius**: Adjust `radius` parameter for visualization - smaller values = better performance

## Coordinate Systems

The functions assume:
- **Camera coordinates**: Right-hand coordinate system, Z forward, Y down, X right
- **World coordinates**: Configurable via Rerun's `ViewCoordinates`
- **Extrinsics**: World-to-camera transformation matrix

## Troubleshooting

### "No points visualized"

Check:
1. Are masks actually True anywhere? `print(mask.sum())`
2. Is depth valid (> 0) at mask locations?
3. Are depth values within `min_depth` to `max_depth` range?

### "Points appear at wrong location"

Check:
1. Extrinsics format: Should be world-to-camera, not camera-to-world
2. Intrinsics: Verify focal length and principal point values
3. Depth units: Should be in meters

### "Masks key not found in NPZ"

Use `--mask-key` to specify the correct key:
```bash
python lift_and_visualize_masks.py --npz data.npz --mask-key your_mask_key
```

## Integration with Existing Code

These functions integrate seamlessly with existing RH20T processing:

```python
# In create_sparse_depth_map.py or similar
from utils.pointcloud_utils import lift_mask_to_3d_batch
from utils.visualization_utils import visualize_masks_batch

# After processing masks...
if args.visualize_masks:
    visualize_masks_batch(
        masks=sam_masks,
        depths=depths,
        intrs=intrs,
        extrs=extrs,
        entity_base_path="world/segmentation",
        camera_ids=camera_ids,
        color=np.array([255, 100, 200], dtype=np.uint8),
    )
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{mask_lifting_utils,
  title = {Mask Lifting and 3D Visualization Utilities},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/yourrepo}
}
```

## License

This code is released under the same license as the parent project.
