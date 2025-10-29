# Global Mask ID Assignment for Multi-Camera Tracking

## Problem Statement

When tracking objects across multiple cameras, each camera's tracking algorithm assigns its own local IDs to detected objects. For example:
- **Camera 0** might detect 3 objects: IDs `[0, 1, 2]`
- **Camera 1** might detect 2 objects: IDs `[0, 1]`

However, these local IDs don't correspond across cameras. Camera 1's object with ID `0` might actually be the same physical object as Camera 0's object with ID `2`.

## Solution: Global ID Assignment

The `assign_global_mask_ids()` function solves this by:

1. **Computing 3D centroids** for each mask in each camera
2. **Matching masks across cameras** using 3D spatial proximity
3. **Assigning consistent global IDs** so the same object has the same ID across all cameras

### How It Works

```
Camera 0 (3 masks):           Camera 1 (2 masks):
  Local ID 0 @ (1.0, 0.5, 0.2)   Local ID 0 @ (1.05, 0.48, 0.21)  <- Close!
  Local ID 1 @ (2.0, 1.0, 0.5)   Local ID 1 @ (3.5, 2.0, 0.8)     <- Close!
  Local ID 2 @ (3.5, 2.1, 0.75)

Global ID Assignment:
  Camera 0: {0: 0, 1: 1, 2: 2}  (local_id: global_id)
  Camera 1: {0: 0, 1: 2}        (local_id: global_id)
  
Result: 
  - Cam0's ID 0 and Cam1's ID 0 -> Global ID 0 (matched, distance < threshold)
  - Cam0's ID 1 -> Global ID 1 (no match in Cam1)
  - Cam0's ID 2 and Cam1's ID 1 -> Global ID 2 (matched, distance < threshold)
```

### Distance Threshold

The `distance_threshold` parameter (default: 0.15 meters = 15cm) determines when two masks are considered the same object:
- **< threshold**: Masks are matched (same object)
- **≥ threshold**: Masks are separate objects (assigned different global IDs)

**Recommended values:**
- Small objects (hands, tools): `0.10 - 0.15m`
- Medium objects (boxes, bottles): `0.15 - 0.25m`
- Large objects (furniture): `0.25 - 0.50m`

## Usage

### Command Line

```bash
# With global ID assignment
python lift_and_visualize_masks.py \
    --npz path/to/masks.npz \
    --mask-key hoist_masks \
    --use-global-ids \
    --distance-threshold 0.15 \
    --frame-for-matching 0 \
    --max-frames 100

# Without global IDs (original per-camera mode)
python lift_and_visualize_masks.py \
    --npz path/to/masks.npz \
    --mask-key hoist_masks \
    --max-frames 100
```

### Python API

```python
from utils.mask_lifting_utils import (
    assign_global_mask_ids,
    visualize_masks_with_global_ids
)

# Load your data
masks_dict = {"hand": hand_masks}  # [C, T, H, W] with integer IDs
depths = ...  # [C, T, H, W]
intrs = ...   # [C, 3, 3] or [C, T, 3, 3]
extrs = ...   # [C, 3, 4] or [C, T, 3, 4]

# Method 1: Just get the ID mapping
global_id_mapping = assign_global_mask_ids(
    masks_dict=masks_dict,
    depths=depths,
    intrs=intrs,
    extrs=extrs,
    distance_threshold=0.15,
    frame_index=0,
)

# Result structure:
# {
#     'hand': {
#         0: {0: 0, 1: 1, 2: 2},  # Camera 0: local -> global
#         1: {0: 0, 1: 2},        # Camera 1: local -> global
#         2: {0: 1, 1: 3},        # Camera 2: local -> global
#     }
# }

# Method 2: Visualize with global IDs in Rerun
stats = visualize_masks_with_global_ids(
    masks_dict=masks_dict,
    depths=depths,
    intrs=intrs,
    extrs=extrs,
    entity_base_path="world/hands",
    distance_threshold=0.15,
    frame_for_matching=0,
)

print(f"Total points: {stats['total_points']}")
print(f"Global ID mapping: {stats['global_id_mapping']}")
```

## Visualization in Rerun

### Without Global IDs (Original)
```
world/masks/
  hand/
    camera_cam0/  <- Toggle camera 0's hand masks
    camera_cam1/  <- Toggle camera 1's hand masks
    camera_cam2/  <- Toggle camera 2's hand masks
```

Each camera is a separate layer. You can't easily track the same object across cameras.

### With Global IDs (New)
```
world/masks/
  hand/
    global_id_0/  <- Same object across all cameras (Red)
    global_id_1/  <- Another object (Green)
    global_id_2/  <- Another object (Blue)
```

Each global ID is a separate layer showing the same object from all cameras. Each global ID gets a unique color.

## Technical Details

### Input Data Format

Masks should be **integer-valued arrays** (not binary):
```python
# Good: Integer IDs
mask_frame = np.array([
    [0, 0, 1, 1],
    [0, 0, 1, 2],
    [0, 0, 2, 2],
])  # Background=0, Object1=1, Object2=2

# Bad: Binary mask (can't distinguish multiple objects)
mask_frame = np.array([
    [0, 0, 1, 1],
    [0, 0, 1, 1],
    [0, 0, 1, 1],
])  # All objects have same ID
```

### Matching Algorithm

1. **Extract masks per camera**: For each camera, extract all unique mask IDs at the matching frame
2. **Compute 3D centroids**: Lift each mask to 3D and compute its centroid in world coordinates
3. **Initialize with Camera 0**: Assign sequential global IDs to Camera 0's masks
4. **Match subsequent cameras**:
   - For each mask in Camera N, compute distances to all existing global centroids
   - If minimum distance < threshold: Assign existing global ID
   - If minimum distance ≥ threshold: Create new global ID
5. **Greedy assignment**: Use greedy matching to avoid duplicates (each global ID matched at most once per camera)

### Centroid Computation

```python
# For a mask, centroid is the mean of all 3D points
points_3d = lift_mask_to_3d(mask, depth, intr, extr)
centroid = np.mean(points_3d, axis=0)  # [3] array: (x, y, z)
```

## Isolated Functions

The implementation is designed to be **non-breaking**:
- ✅ `visualize_masks_batch()` - Original function, unchanged behavior
- ✅ `lift_mask_to_3d()` - Original function, unchanged behavior
- ✅ `assign_global_mask_ids()` - **New**, isolated function
- ✅ `visualize_masks_with_global_ids()` - **New**, isolated function

You can use the new global ID features without affecting existing code.

## Examples

### Example 1: Hand Tracking with HOIST

```bash
python lift_and_visualize_masks.py \
    --npz third_party/HOISTFormer/hoist_output/task_0045_user_0020_scene_0004_cfg_0006_human_processed_hand_tracked_hoist.npz \
    --mask-key hoist_masks \
    --use-global-ids \
    --distance-threshold 0.12 \
    --max-frames 111 \
    --spawn
```

### Example 2: SAM2 Object Tracking

```bash
python lift_and_visualize_masks.py \
    --npz third_party/HOISTFormer/sam2_tracking_output/task_0045_user_0020_scene_0004_cfg_0006_human_processed_hand_tracked_hoist_sam2.npz \
    --mask-key sam2_predictions \
    --use-global-ids \
    --distance-threshold 0.20 \
    --max-frames 111 \
    --spawn
```

### Example 3: Custom Python Script

```python
import numpy as np
import rerun as rr
from utils.mask_lifting_utils import visualize_masks_with_global_ids

# Initialize Rerun
rr.init("my_tracking_viz", spawn=True)
rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

# Load your multi-camera tracking data
data = np.load("tracking_output.npz")
masks_dict = {"objects": data["masks"]}  # [C, T, H, W]
depths = data["depths"]
intrs = data["intrs"]
extrs = data["extrs"]

# Visualize with global IDs
stats = visualize_masks_with_global_ids(
    masks_dict=masks_dict,
    depths=depths,
    intrs=intrs,
    extrs=extrs,
    entity_base_path="world/tracked_objects",
    distance_threshold=0.15,
    camera_ids=["cam_left", "cam_right", "cam_top"],
    fps=30.0,
)

print(f"Tracked {stats['mask_stats']['objects']['num_global_ids']} unique objects")
print("Global ID mapping:", stats['global_id_mapping'])

rr.save("tracking_result.rrd")
```

## Troubleshooting

### Issue: All objects get separate global IDs (no matching)

**Cause**: Distance threshold is too small or coordinate systems are misaligned

**Solution**:
- Increase `--distance-threshold` (try 0.2, 0.3, 0.5)
- Check that extrinsics are correct (same world coordinate system)
- Verify intrinsics are calibrated properly

### Issue: Multiple objects get the same global ID

**Cause**: Distance threshold is too large

**Solution**:
- Decrease `--distance-threshold` (try 0.1, 0.08, 0.05)
- Choose a frame where objects are well-separated for matching

### Issue: ImportError: No module named 'scipy'

**Cause**: scipy not installed

**Solution**:
```bash
pip install scipy
# or
pip install scikit-learn  # includes scipy as dependency
```

## Performance

- **Time complexity**: O(C × M²) where C is cameras, M is masks per camera
- **Memory**: Minimal overhead (only stores centroids, not full point clouds)
- **Typical performance**: 3 cameras × 5 masks each → ~1-2 seconds

## Future Improvements

Possible enhancements:
- [ ] Temporal consistency: Track global IDs across frames (not just single frame)
- [ ] Hungarian algorithm for optimal bipartite matching
- [ ] Confidence scores based on overlap/distance
- [ ] Multi-frame consensus for robust matching
- [ ] Support for mask similarity metrics (IoU in 3D)
