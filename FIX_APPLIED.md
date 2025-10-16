# Fix Applied: Gripper Bounding Box Support for WSG-50 (Config 3)

## Commits
- **Before fix 1**: `Previous commits` - Analysis complete
- **After fix 1**: `ba5cb60` - Fix: Add ee_link fallback for grippers without finger pad links (WSG-50)
- **After fix 1 docs**: `dd173a1` - Add documentation for WSG-50 gripper bbox fix
- **After fix 2**: `e13bc1e` - Fix: Compute gripper bbox when any bbox flag is set, not just --gripper-bbox

## Problem 1: WSG-50 Gripper Bbox Not Computing (FIXED ✅)

### Summary
The gripper bounding box computation was failing for Config 3 (WSG-50 gripper) but working for Config 4 (Robotiq 2F-85 gripper).

### Root Cause
The gripper bounding box computation was failing for Config 3 (WSG-50 gripper) but working for Config 4 (Robotiq 2F-85 gripper).

## Root Cause
The `_compute_gripper_bbox()` function searches for gripper links using keywords like:
- "left"/"right" + "finger"  
- "left"/"right" + "pad"
- "left"/"right" + "knuckle"

**Config 4** (Robotiq 2F-85): The URDF (`ur5_robotiq_85.urdf`) includes detailed finger links:
```
left_inner_finger_pad
right_inner_finger_pad
```
✅ Search succeeds, bbox computed correctly

**Config 3** (WSG-50): The URDF (`ur5.urdf`) contains NO finger-specific links:
```xml
<link name="base_link">
<link name="shoulder_link">
<link name="upper_arm_link">
<link name="forearm_link">
<link name="wrist_1_link">
<link name="wrist_2_joint">
<link name="wrist_3_link">
<link name="ee_link">  <!-- End-effector only, no finger details -->
```
❌ Search fails → returns `None` → no bbox

## Solution Implemented

### Changes to `create_sparse_depth_map.py`

#### 1. Modified `_compute_gripper_bbox()` (lines 435-447)
Added fallback to use `ee_link` when no finger pad links are found:

```python
# Fetch homogeneous transforms for both finger pads if available.
left_tf = fk_map[left_link].matrix().astype(np.float32) if left_link and left_link in fk_map else None
right_tf = fk_map[right_link].matrix().astype(np.float32) if right_link and right_link in fk_map else None

# FALLBACK: If no finger links found, try using end-effector link (e.g., for WSG-50)
ee_tf = None
if left_tf is None and right_tf is None and ee_link and ee_link in fk_map:
    print(f"[Info] No finger pad links found; using end-effector link '{ee_link}' as gripper frame.")
    ee_tf = fk_map[ee_link].matrix().astype(np.float32)
```

#### 2. Added `ee_tf` handling case (lines 513-522)
Added a new elif branch to handle the end-effector-only case:

```python
elif ee_tf is not None:
    print("[Info] Using end-effector link as gripper frame (no finger pads available).")
    # Fallback: use end-effector link when no finger links are found
    # This is common for grippers like WSG-50 where the URDF doesn't include finger details
    pad_midpoint = ee_tf[:3, 3].astype(np.float32)
    # Use standard EE frame convention: X = width, Y = height, Z = approach
    width_axis = ee_tf[:3, 0].astype(np.float32)  # X-axis of EE frame
    approach_axis = ee_tf[:3, 2].astype(np.float32)  # Z-axis of EE frame
    height_axis = ee_tf[:3, 1].astype(np.float32)  # Y-axis of EE frame
    measured_width = None
```

#### 3. Modified `_compute_gripper_pad_points()` (lines 717-727)
Added similar fallback for pad point computation:

```python
# FALLBACK: If no finger pads found, use end-effector link
if not points:
    robot_type = getattr(robot_conf, "robot", None)
    ee_link = ROBOT_EE_LINK_MAP.get(robot_type)
    if ee_link and ee_link in fk_map:
        T = fk_map[ee_link].matrix()
        p = T[:3, 3].astype(np.float32)
        points.append(p)
        print(f"[Info] Using end-effector link '{ee_link}' for gripper pad point.")
```

## Test Results

### Config 3 (WSG-50) - Now Works ✅
```bash
python create_sparse_depth_map.py \
  --task-folder /data/rh20t_api/data/low_res_data/RH20T_cfg3/task_0024_user_0010_scene_0005_cfg_0003 \
  --high-res-folder /data/rh20t_api/data/RH20T/RH20T_cfg3/task_0024_user_0010_scene_0005_cfg_0003 \
  --out-dir /tmp/test_cfg3_fix \
  --max-frames 3 \
  --add-robot \
  --gripper-body-bbox
```

**Output:**
```
[Info] No finger pad links found; using end-effector link 'ee_link' as gripper frame.
[Info] Using end-effector link as gripper frame (no finger pads available).
[INFO] Added robot with 70000 points for frame 0 (gripper width: 6.36 mm)
✅ [OK] Wrote NPZ file to: /tmp/test_cfg3_fix/task_0024_user_0010_scene_0005_cfg_0003_processed.npz
✅ [OK] Saved Rerun visualization to: /tmp/test_cfg3_fix/task_0024_user_0010_scene_0005_cfg_0003_reprojected.rrd
```

### Config 4 (Robotiq) - Still Works ✅
```bash
python create_sparse_depth_map.py \
  --task-folder /data/rh20t_api/data/.../task_0065_user_0010_scene_0009_cfg_0004 \
  --high-res-folder /data/rh20t_api/data/.../task_0065_user_0010_scene_0009_cfg_0004 \
  --out-dir /tmp/test_cfg4_fix \
  --max-frames 2 \
  --add-robot \
  --gripper-body-bbox
```

**Output:**
```
[Info] Both gripper pad transforms available; computing gripper frame from finger pads.
[INFO] Added robot with 70000 points for frame 0 (gripper width: 13.33 mm)
```

## Known Limitation

**Gripper boxes are not saved to NPZ**: The `save_data_to_npz()` function currently only saves:
- rgbs, depths, intrs, extrs
- timestamps, per_camera_timestamps, camera_ids  
- query_points (if available)

The computed gripper boxes (`robot_gripper_boxes`, `robot_gripper_body_boxes`, etc.) are logged to Rerun but **not included in the NPZ file**. This is a pre-existing issue unrelated to the WSG-50 fix.

---

## Problem 2: Only Orange Bbox Visible (FIXED ✅)

### Summary
When using `--gripper-body-bbox` or `--gripper-fingertip-bbox` flags alone, no bounding boxes were computed at all. Only when using `--gripper-bbox` (contact bbox) would ANY bbox appear.

### Root Cause
The gripper bbox computation code was gated behind a single condition:
```python
if robot_gripper_boxes is not None:
    # ALL bbox computation happens here
```

This meant:
- ✅ `--gripper-bbox`: Sets `robot_gripper_boxes != None` → computation runs → orange contact bbox appears
- ❌ `--gripper-body-bbox` only: `robot_gripper_boxes = None` → computation SKIPPED → no bbox at all
- ❌ `--gripper-fingertip-bbox` only: `robot_gripper_boxes = None` → computation SKIPPED → no bbox at all

### Solution (Commit e13bc1e)
Changed the condition to run bbox computation if **ANY** bbox output list is requested:

```python
# Compute gripper bbox if ANY bbox output is requested
if (robot_gripper_boxes is not None or 
    robot_gripper_body_boxes is not None or 
    robot_gripper_fingertip_boxes is not None):
    # Bbox computation happens here
```

### Test Results
Now all combinations work correctly:

```bash
# Test 1: Body bbox only
--gripper-body-bbox
# Result: ✅ Red body bbox appears

# Test 2: Fingertip bbox only  
--gripper-fingertip-bbox
# Result: ✅ Blue fingertip bbox appears

# Test 3: All three together
--gripper-bbox --gripper-body-bbox --gripper-fingertip-bbox
# Result: ✅ All three bboxes appear (orange + red + blue)
```

### Bbox Color Guide
- **Orange (255, 128, 0)**: Contact/gripper bbox (`--gripper-bbox`)
- **Red (255, 0, 0)**: Full body bbox (`--gripper-body-bbox`)
- **Blue (0, 0, 255)**: Fingertip bbox (`--gripper-fingertip-bbox`)

---

## Summary

✅ **Fix 1 (ba5cb60)**: Gripper bounding box now computes correctly for Config 3 (WSG-50) using ee_link fallback  
✅ **Fix 2 (e13bc1e)**: All bbox flags now work independently and in combination  
✅ **Backward Compatible**: Config 4 (Robotiq) still uses the more accurate finger pad method when available  
⚠️ **Known Issue**: Boxes computed but not saved to NPZ (separate bug, affects both configs)
