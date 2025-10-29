# Quick Start: Global Mask ID Assignment

## Test the Implementation

Run the test script to verify everything works:

```bash
python test_global_ids.py
```

Expected output:
```
✓ SUCCESS: 2 object(s) matched across cameras!
All assertions passed!
```

## Basic Usage

### 1. Visualize with Global IDs

```bash
# Your tracking output with integer mask IDs
python lift_and_visualize_masks.py \
    --npz third_party/HOISTFormer/hoist_output/task_0045_user_0020_scene_0004_cfg_0006_human_processed_hand_tracked_hoist.npz \
    --mask-key hoist_masks \
    --use-global-ids \
    --distance-threshold 0.15 \
    --max-frames 111 \
    --spawn
```

### 2. Adjust Distance Threshold

If objects aren't matching:
```bash
# Increase threshold (objects further apart)
--distance-threshold 0.25
```

If wrong objects are matching:
```bash
# Decrease threshold (objects closer together)
--distance-threshold 0.10
```

### 3. Choose Matching Frame

Use a frame where objects are well-separated and clearly visible:

```bash
# Use frame 10 for computing the global ID assignments
--frame-for-matching 10
```

## Understanding the Output

### Rerun Viewer

Open the generated `.rrd` file in Rerun:

```bash
rerun your_output_masks_3d.rrd --web-viewer
```

**Without `--use-global-ids`:**
```
world/masks/
  hand/
    camera_cam0/  ← Can't track same hand across cameras
    camera_cam1/
    camera_cam2/
```

**With `--use-global-ids`:**
```
world/masks/
  hand/
    global_id_0/  ← Red (left hand across all cameras)
    global_id_1/  ← Green (right hand across all cameras)
```

### Terminal Output

```
[INFO] Assigning global IDs for mask type: 'hand'
[INFO]   Camera 0: Found 2 masks
[INFO]   Camera 1: Found 2 masks
[INFO]   Camera 2: Found 2 masks
[INFO]   Camera 1, local ID 0 -> global ID 0 (dist: 0.082m)  ← Matched!
[INFO]   Camera 1, local ID 1 -> global ID 1 (dist: 0.091m)  ← Matched!
[INFO]   Camera 2, local ID 0 -> global ID 0 (dist: 0.105m)  ← Matched!
[INFO]   Camera 2, local ID 1 -> global ID 1 (dist: 0.098m)  ← Matched!
[INFO] Assigned 2 global IDs for 'hand'

[INFO] Global ID Mapping:
[INFO]   hand:
[INFO]     Camera 0: {0: 0, 1: 1}      ← local ID : global ID
[INFO]     Camera 1: {0: 0, 1: 1}
[INFO]     Camera 2: {0: 0, 1: 1}
```

## Python API Example

```python
import numpy as np
import rerun as rr
from utils.mask_lifting_utils import visualize_masks_with_global_ids

# Load your data
data = np.load("tracking_output.npz")

# Initialize Rerun
rr.init("tracking_viz", spawn=True)
rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

# Visualize with global IDs
stats = visualize_masks_with_global_ids(
    masks_dict={"hands": data["masks"]},  # Must have integer IDs
    depths=data["depths"],
    intrs=data["intrs"],
    extrs=data["extrs"],
    entity_base_path="world/hands",
    distance_threshold=0.15,  # 15cm threshold
    camera_ids=["cam0", "cam1", "cam2"],
)

print(f"Matched {stats['mask_stats']['hands']['num_global_ids']} unique hands")
print("Mapping:", stats['global_id_mapping'])

# Save
rr.save("tracking_result.rrd")
```

## Troubleshooting

### "Masks should contain integer IDs"

Your masks need to be like:
```python
mask[100:200, 100:200] = 1  # Object 1
mask[250:350, 250:350] = 2  # Object 2
```

Not binary:
```python
mask[100:200, 100:200] = True  # ❌ Can't distinguish objects
```

### All objects get different global IDs

- **Check**: Are camera extrinsics in the same world coordinate system?
- **Try**: Increase `--distance-threshold` to 0.3 or 0.5
- **Verify**: Run `test_global_ids.py` to check basic functionality

### Multiple objects get same global ID

- **Try**: Decrease `--distance-threshold` to 0.08 or 0.10
- **Use**: Different `--frame-for-matching` where objects are more separated

### ImportError: scipy

```bash
pip install scipy
```

## Next Steps

- See [GLOBAL_MASK_ID_README.md](GLOBAL_MASK_ID_README.md) for detailed documentation
- Check `utils/mask_lifting_utils.py` for API reference
- Try different distance thresholds for your use case
