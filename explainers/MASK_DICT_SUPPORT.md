# Dictionary of Masks Support

## Overview

The mask lifting visualization system now supports both:
1. **Single mask arrays**: `[C, T, H, W]` - A single type of mask across cameras and frames
2. **Dictionary of masks**: `{name: [C, T, H, W]}` - Multiple mask types, each with a unique name

## Use Cases

### Single Mask Array (Original)
Used when you have one type of segmentation mask:
```python
masks = hand_masks  # [3, 50, 480, 640]
np.savez("data.npz", masks=masks, depths=depths, ...)
```

### Dictionary of Masks (New)
Used when you have multiple mask types that you want to visualize separately:
```python
masks_dict = {
    "hand": hand_masks,      # [3, 50, 480, 640]
    "object": object_masks,  # [3, 50, 480, 640]
    "table": table_masks,    # [3, 50, 480, 640]
}
np.savez("data.npz", masks=masks_dict, depths=depths, ...)
```

## Visualization Structure in Rerun

### Single Mask
```
world/
  ├─ masks/
  │   ├─ camera_cam0/  (toggleable)
  │   ├─ camera_cam1/  (toggleable)
  │   └─ camera_cam2/  (toggleable)
```

### Dictionary of Masks
```
world/
  ├─ masks/
  │   ├─ hand/
  │   │   ├─ camera_cam0/  (toggleable)
  │   │   ├─ camera_cam1/  (toggleable)
  │   │   └─ camera_cam2/  (toggleable)
  │   ├─ object/
  │   │   ├─ camera_cam0/  (toggleable)
  │   │   ├─ camera_cam1/  (toggleable)
  │   │   └─ camera_cam2/  (toggleable)
  │   └─ table/
  │       ├─ camera_cam0/  (toggleable)
  │       ├─ camera_cam1/  (toggleable)
  │       └─ camera_cam2/  (toggleable)
```

Each mask type and camera becomes a **separate toggleable layer** in the Rerun viewer!

## Example: HOISTFormer Output

HOISTFormer detects multiple instances (hands, objects). You can save them as:

```python
# Process HOISTFormer output
hoist_predictions = {
    "hand_left": left_hand_masks,   # [C, T, H, W]
    "hand_right": right_hand_masks, # [C, T, H, W]
    "object_1": object1_masks,      # [C, T, H, W]
    "object_2": object2_masks,      # [C, T, H, W]
}

np.savez(
    "output.npz",
    hoist_predictions=hoist_predictions,  # Save as dict!
    depths=depths,
    intrs=intrs,
    extrs=extrs,
    rgbs=rgbs,
    camera_ids=camera_ids
)
```

Then visualize with:
```bash
python lift_and_visualize_masks.py \
    --npz output.npz \
    --mask-key hoist_predictions \
    --color 255 0 255 \
    --spawn
```

## Benefits

1. **Independent Toggling**: Turn each mask type on/off separately in Rerun viewer
2. **Color Coding**: Assign different colors to different mask types
3. **Organized Hierarchy**: Clear structure in the entity tree
4. **Flexible Analysis**: Compare different segmentation results side-by-side

## Implementation Details

### Data Loading (`lift_and_visualize_masks.py`)

The loader detects whether the mask data is:
- A numpy array → wraps it as `{"default": masks}`
- A dictionary → uses it directly

```python
if isinstance(masks_raw, dict):
    # Dictionary of masks - use as-is
    masks = masks_raw
else:
    # Single array - wrap in default dict
    masks = {"default": masks_raw}
```

### Visualization Loop

Each mask type gets its own entity path:
```python
for mask_name, mask_array in masks_dict.items():
    entity_path = f"{base_path}/{mask_name}"
    visualize_masks_batch(
        masks=mask_array,
        entity_base_path=entity_path,
        ...
    )
```

Within each mask type, cameras are further separated:
```python
for camera_id in camera_ids:
    camera_entity = f"{entity_path}/camera_{camera_id}"
    # Log points for this camera at this entity
```

## Testing

Run the test script to see it in action:
```bash
python test_dict_masks.py
python lift_and_visualize_masks.py --npz test_dict_masks.npz --mask-key masks --spawn
```

This creates synthetic data with 3 mask types (hand, object, background) and visualizes them with separate toggleable layers.
