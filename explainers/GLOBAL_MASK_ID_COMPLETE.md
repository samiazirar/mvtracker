# Global Mask ID Assignment - Complete Solution

## 🎯 Problem Solved

**Before:** Each camera assigns its own local IDs to tracked objects, making cross-camera tracking impossible.

**After:** Objects get consistent global IDs across all cameras based on their 3D positions.

## 📦 What Was Delivered

### Core Implementation

✅ **3 New Functions in `utils/mask_lifting_utils.py`:**
1. `compute_mask_centroid_3d()` - Compute 3D centroid of a mask
2. `assign_global_mask_ids()` - Match masks across cameras and assign global IDs
3. `visualize_masks_with_global_ids()` - Visualize with global IDs in Rerun

✅ **Enhanced `lift_and_visualize_masks.py`:**
- Added `--use-global-ids` flag
- Added `--distance-threshold` parameter (default: 0.15m)
- Added `--frame-for-matching` parameter (default: 0)
- Maintains backwards compatibility (old behavior unchanged)

### Documentation

✅ **Complete Documentation:**
1. `GLOBAL_MASK_ID_README.md` - Full technical documentation
2. `GLOBAL_MASK_ID_QUICKSTART.md` - Quick start guide
3. `IMPLEMENTATION_SUMMARY.md` - Implementation details
4. `test_global_ids.py` - Working test with synthetic data

### Key Features

✅ **Isolated & Non-Breaking:**
- All new functions are separate
- Existing functions unchanged
- No breaking changes to existing workflows

✅ **Spatial Matching:**
- Uses 3D centroids in world coordinates
- Configurable distance threshold
- Greedy matching algorithm

✅ **Visualization:**
- Each global ID = separate toggleable layer in Rerun
- Unique color per global ID
- Clear logging of matches

## 🚀 Quick Start

### 1. Test It
```bash
python test_global_ids.py
```

### 2. Use It
```bash
# With your tracking output
python lift_and_visualize_masks.py \
    --npz your_tracking_output.npz \
    --mask-key your_mask_key \
    --use-global-ids \
    --distance-threshold 0.15 \
    --spawn
```

### 3. Adjust Threshold

**Objects not matching?** Increase threshold:
```bash
--distance-threshold 0.25  # 25cm
```

**Wrong objects matching?** Decrease threshold:
```bash
--distance-threshold 0.10  # 10cm
```

## 📊 Example Output

```
[INFO] Assigning global IDs for mask type: 'hand'
[INFO]   Camera 0: Found 2 masks
[INFO]   Camera 1: Found 2 masks
[INFO]   Camera 1, local ID 0 -> global ID 0 (dist: 0.089m)  ✓ Matched!
[INFO]   Camera 1, local ID 1 -> global ID 1 (dist: 0.126m)  ✓ Matched!
[INFO] Assigned 2 global IDs for 'hand'

[INFO] Global ID Mapping:
[INFO]   hand:
[INFO]     Camera 0: {0: 0, 1: 1}
[INFO]     Camera 1: {0: 0, 1: 1}  ← Same global IDs!
```

### Rerun Visualization

**Toggle each object independently:**
```
world/masks/
  hand/
    global_id_0/  ← Left hand (Red) across all cameras
    global_id_1/  ← Right hand (Green) across all cameras
```

## 🔧 How It Works

1. **Extract masks** from each camera at matching frame
2. **Compute 3D centroids** using depth + camera calibration
3. **Match across cameras** using Euclidean distance
4. **Assign global IDs** based on spatial proximity
5. **Visualize** each global ID as separate layer

### Distance-Based Matching

```python
# Camera 0: Object at position (1.0, 0.5, 0.2)
# Camera 1: Object at position (1.05, 0.48, 0.21)
# Distance = 0.073m < threshold (0.15m)
# Result: MATCHED → same global ID ✓

# Camera 0: Object at position (1.0, 0.5, 0.2)  
# Camera 2: Object at position (2.5, 1.8, 0.9)
# Distance = 1.876m > threshold (0.15m)
# Result: NOT MATCHED → different global IDs
```

## 💡 Use Cases

### Hand Tracking
```bash
python lift_and_visualize_masks.py \
    --npz hoist_output.npz \
    --mask-key hoist_masks \
    --use-global-ids \
    --distance-threshold 0.12  # Hands are ~12cm apart
```

### Object Tracking
```bash
python lift_and_visualize_masks.py \
    --npz sam2_output.npz \
    --mask-key sam2_predictions \
    --use-global-ids \
    --distance-threshold 0.20  # Objects 20cm apart
```

### Multi-Camera People Tracking
```bash
python lift_and_visualize_masks.py \
    --npz person_masks.npz \
    --mask-key person_masks \
    --use-global-ids \
    --distance-threshold 0.50  # People 50cm apart
```

## 📚 Documentation

| File | Purpose |
|------|---------|
| `GLOBAL_MASK_ID_README.md` | Complete technical documentation |
| `GLOBAL_MASK_ID_QUICKSTART.md` | Quick start & examples |
| `IMPLEMENTATION_SUMMARY.md` | Implementation details |
| `test_global_ids.py` | Test script with synthetic data |

## ✅ Testing

**Run the test:**
```bash
python test_global_ids.py
```

**Expected output:**
```
✓ SUCCESS: 2 object(s) matched across cameras!
All assertions passed!
```

**Test validates:**
- Centroid computation works
- Distance-based matching works
- Global ID assignment is correct
- Multiple cameras are handled

## 🔍 Troubleshooting

### All objects get different global IDs

**Check:**
1. Are extrinsics in the same world coordinate system?
2. Is the distance threshold too small?

**Try:**
```bash
--distance-threshold 0.30  # Increase threshold
--frame-for-matching 10    # Try different frame
```

### Wrong objects are matched

**Try:**
```bash
--distance-threshold 0.08  # Decrease threshold
```

### No masks found

**Check:**
- Masks contain integer IDs (not binary)
- Mask key is correct: `--mask-key your_key`
- Masks are not all zero at matching frame

## 🎨 Visualization Examples

### Without Global IDs (Original)
Each camera is separate - can't track same object across cameras:
```
world/masks/hand/camera_cam0/
world/masks/hand/camera_cam1/
world/masks/hand/camera_cam2/
```

### With Global IDs (New)
Each object is separate - can track across all cameras:
```
world/masks/hand/global_id_0/  (Red - left hand)
world/masks/hand/global_id_1/  (Green - right hand)
```

Toggle each object on/off in Rerun to see it from all camera angles!

## 🚀 Next Steps

1. **Test with your data:**
   ```bash
   python lift_and_visualize_masks.py --npz YOUR_FILE.npz --use-global-ids --spawn
   ```

2. **Adjust threshold** based on your object sizes

3. **Choose good matching frame** where objects are clearly visible and separated

4. **View in Rerun** and toggle global IDs on/off

## 📖 API Reference

### Python API

```python
from utils.mask_lifting_utils import (
    assign_global_mask_ids,
    visualize_masks_with_global_ids,
)

# Get mapping only
mapping = assign_global_mask_ids(
    masks_dict={"hand": masks},
    depths=depths,
    intrs=intrs,
    extrs=extrs,
    distance_threshold=0.15,
)

# Visualize with mapping
stats = visualize_masks_with_global_ids(
    masks_dict={"hand": masks},
    depths=depths,
    intrs=intrs,
    extrs=extrs,
    entity_base_path="world/hands",
    distance_threshold=0.15,
)
```

## ✨ Summary

You now have:
- ✅ Global ID assignment across cameras
- ✅ Distance-based spatial matching
- ✅ Configurable threshold
- ✅ Non-breaking implementation
- ✅ Complete documentation
- ✅ Working test
- ✅ Easy command-line usage

**Try it now:**
```bash
python test_global_ids.py && echo "Ready to use!"
```
