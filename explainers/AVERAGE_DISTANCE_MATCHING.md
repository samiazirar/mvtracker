# Average Distance Global ID Matching

## Overview

This document describes the **simplified average distance matching** approach for assigning global IDs to masks across multiple cameras.

## Key Principle

**Each local mask ID represents ONE object across all time.**

- Camera A with local IDs `[0, 1]` = 2 objects that persist across time
- Camera B with local IDs `[0, 1, 3]` = 3 objects that persist across time
- Objects may appear/disappear at different frames (temporal gaps are OK)

## Algorithm

### Step 1: Extract Local Mask IDs Per Camera

For each camera, extract all unique local IDs that appear across **any** frame:

```
Camera 0: local_ids = [1, 2]       # IDs in mask array (0 is background)
Camera 1: local_ids = [1, 2, 4]    # IDs in mask array
```

For each local ID, compute its 3D centroid at **every frame where it exists**:

```
Camera 0:
  local_id=1: {frame_0: [x,y,z], frame_1: [x,y,z], ..., frame_9: [x,y,z]}
  local_id=2: {frame_0: [x,y,z], frame_1: [x,y,z], ..., frame_9: [x,y,z]}

Camera 1:
  local_id=1: {frame_2: [x,y,z], frame_3: [x,y,z], ..., frame_9: [x,y,z]}  # Missing frames 0-1!
  local_id=2: {frame_0: [x,y,z], frame_1: [x,y,z], ..., frame_9: [x,y,z]}
  local_id=4: {frame_5: [x,y,z], frame_6: [x,y,z], ..., frame_9: [x,y,z]}  # Only 5-9
```

### Step 2: Compute Average Distances

For every pair of `(camera_i, local_id_i)` and `(camera_j, local_id_j)`:

1. **Find common frames** where both masks exist
2. **Compute distance** at each common frame: `dist = ||centroid_i - centroid_j||`
3. **Average** the distances across all common frames

Example:
```
Camera 0, local_id=1 vs Camera 1, local_id=1:
  Common frames: [2, 3, 4, 5, 6, 7, 8, 9]  (8 frames)
  Distances: [0.03, 0.035, 0.04, 0.032, 0.038, 0.034, 0.031, 0.036]
  Average: 0.035m
  
Camera 0, local_id=1 vs Camera 1, local_id=4:
  Common frames: [5, 6, 7, 8, 9]  (5 frames)
  Distances: [0.28, 0.29, 0.27, 0.30, 0.28]
  Average: 0.283m
```

### Step 3: Greedy Matching Across Cameras

Start with Camera 0 as reference:
```
Camera 0, local_id=1 -> global_id=0
Camera 0, local_id=2 -> global_id=1
```

For each subsequent camera, match each local ID to the **closest** already-assigned global ID:

```
Camera 1, local_id=1:
  Best match: global_id=0 (avg_dist=0.035m < threshold=0.15m)
  ✓ Assign global_id=0

Camera 1, local_id=2:
  Best match: global_id=1 (avg_dist=0.028m < threshold=0.15m)
  ✓ Assign global_id=1

Camera 1, local_id=4:
  Best match: global_id=0 (avg_dist=0.283m > threshold=0.15m)
  ✗ Too far! Create NEW global_id=2
```

## Final Mapping

```
Camera 0:
  local_id=1 -> global_id=0
  local_id=2 -> global_id=1

Camera 1:
  local_id=1 -> global_id=0  ← Same object as Camera 0, local_id=1
  local_id=2 -> global_id=1  ← Same object as Camera 0, local_id=2
  local_id=4 -> global_id=2  ← New unique object
```

**Result:** 3 global objects total

## Usage

```bash
python lift_and_visualize_masks.py \
    --npz YOUR_DATA.npz \
    --mask-key your_mask_key \
    --use-temporal-global-ids \
    --distance-threshold 0.15 \
    --spawn
```

### Parameters

- `--distance-threshold`: Maximum **average distance** (meters) to match masks across cameras
  - Default: `0.15m` (15cm)
  - Lower = stricter matching (fewer false positives)
  - Higher = more lenient matching (may merge distinct objects)

### Tuning Guidelines

| Object Motion | Recommended Threshold |
|--------------|----------------------|
| Stationary objects | 0.05 - 0.10m |
| Slow moving (hands, tools) | 0.10 - 0.20m |
| Fast moving objects | 0.20 - 0.30m |
| Highly dynamic scene | 0.30 - 0.50m |

## Advantages

✅ **Simple:** Each local ID = one object (no complex temporal tracking)  
✅ **Robust:** Handles temporal gaps naturally (uses only common frames)  
✅ **Accurate:** Average distance reduces impact of single-frame outliers  
✅ **Efficient:** O(C×L²×T) where C=cameras, L=local IDs per camera, T=frames  

## Comparison with Previous Approach

| Aspect | Old (Temporal Tracking) | New (Average Distance) |
|--------|------------------------|------------------------|
| **Assumption** | Masks may split/merge | Each ID = persistent object |
| **Complexity** | Frame-to-frame linking + matching | Direct average distance |
| **Temporal gaps** | Creates separate tracks | Naturally handled |
| **Matching** | Track-to-track | Local ID-to-local ID |
| **Best for** | Complex ID reassignment | Consistent ID per object |

## Example Test

See `test_average_distance_global_ids.py` for a complete example:

```python
# Camera 0: 2 objects across 10 frames
# Camera 1: 3 objects (2 matched, 1 unique) with temporal gaps
python test_average_distance_global_ids.py
```

Expected output:
```
✅ ALL TESTS PASSED!

Key behaviors verified:
  ✓ Each local mask ID represents ONE object across time
  ✓ Temporal gaps in mask appearance are handled correctly
  ✓ Average distance computed only over frames where both masks exist
  ✓ Objects matched correctly based on average distance threshold
  ✓ Far objects get unique global IDs
```

## Implementation Files

- **`utils/mask_lifting_utils.py`**:
  - `assign_global_ids_temporal()` - Core matching algorithm
  - `visualize_masks_with_temporal_global_ids()` - Visualization wrapper

- **`lift_and_visualize_masks.py`**:
  - CLI with `--use-temporal-global-ids` flag

- **`test_average_distance_global_ids.py`**:
  - Comprehensive test with temporal gaps
