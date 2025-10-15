# ROOT CAUSE ANALYSIS: Config 3 vs Config 4 Bounding Box Issue

## Executive Summary

**Config 3 (WSG-50 gripper) cannot compute bounding boxes** because:
1. The `ur5.urdf` file used for cfg3 **contains NO gripper finger links**
2. The bbox computation function requires finger link transforms from Forward Kinematics (FK)
3. Without finger links, FK cannot provide the transforms needed to compute bbox orientation

**Config 4 (Robotiq 2F-85 gripper) works** because `ur5_robotiq_85.urdf` includes full gripper kinematics.

## Evidence Summary

### Configuration Differences

#### Config 3 (WSG-50 - FAILING)
```json
{
  "conf_num": 3,
  "robot": "ur5",
  "robot_urdf": "./models/ur5/urdf/ur5.urdf",  // ← NO GRIPPER LINKS
  "robot_joint_sequence": [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
  ],  // ← 6 joints, NO finger_joint
  "gripper": "WSG-50"
}
```

**URDF Contents (ur5.urdf):**
```xml
<link name="base_link">
<link name="shoulder_link">
<link name="upper_arm_link">
<link name="forearm_link">
<link name="wrist_1_link">
<link name="wrist_2_link">
<link name="wrist_3_link">
<link name="ee_link">  <!-- ENDS HERE - NO GRIPPER -->
```

**CRITICAL: No finger, pad, knuckle, or wsg links exist!**

```bash
$ grep -i "finger\|pad\|knuckle\|wsg" RH20T/models/ur5/urdf/ur5.urdf
# Returns NOTHING - NO MATCHES
```

#### Config 4 (Robotiq 2F-85 - WORKING)
```json
{
  "conf_num": 4,
  "robot": "ur5",
  "robot_urdf": "./models/ur5/urdf/ur5_robotiq_85.urdf",  // ← HAS GRIPPER
  "robot_joint_sequence": [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint", "finger_joint"
  ],  // ← 7 joints, includes finger_joint
  "gripper": "Robotiq 2F-85"
}
```

**URDF Contents (ur5_robotiq_85.urdf):**
```xml
<!-- Same arm links as cfg3 -->
<link name="wrist_3_link">
<link name="ee_link">
<!-- THEN GRIPPER LINKS: -->
<link name="robotiq_arg2f_base_link">
<joint name="finger_joint" type="revolute">
<link name="left_outer_knuckle">
<link name="left_outer_finger">
<link name="left_inner_finger">
<link name="left_inner_finger_pad">  ← CRITICAL: Has pad links!
<link name="left_inner_knuckle">
<link name="right_outer_knuckle">
<link name="right_outer_finger">
<link name="right_inner_finger">
<link name="right_inner_finger_pad">  ← CRITICAL: Has pad links!
<link name="right_inner_knuckle">
```

## Technical Flow: How Bbox Computation Works

### Step 1: Robot Model Update (RH20T/utils/robot.py:117-164)

```python
def update(self, rotates: np.ndarray, first_time: bool):
    transformations = {joint: rotates[i] for i, joint in enumerate(self._robot_joint_sequence)}
    
    # For Robotiq 2F-85: automatically couples finger joints
    if 'finger_joint' in transformations:
        finger_value = transformations['finger_joint']
        transformations['left_inner_finger_joint'] = -finger_value
        transformations['right_inner_finger_joint'] = -finger_value
        # ... etc for other coupled joints
    
    # Forward Kinematics: compute transform for EVERY link in URDF
    cur_transforms = self._model_chain.forward_kinematics(transformations)
    self._latest_transforms = cur_transforms  # ← Cached for bbox computation
```

**What happens for each config:**

**Config 3 (ur5.urdf):**
```python
# FK only computes transforms for links that EXIST in URDF
self._latest_transforms = {
    'base_link': Transform(...),
    'shoulder_link': Transform(...),
    ...
    'wrist_3_link': Transform(...),
    'ee_link': Transform(...),
    # NOTHING ELSE - NO FINGER LINKS
}
```

**Config 4 (ur5_robotiq_85.urdf):**
```python
self._latest_transforms = {
    'base_link': Transform(...),
    ...
    'wrist_3_link': Transform(...),
    'ee_link': Transform(...),
    'robotiq_arg2f_base_link': Transform(...),
    'left_inner_finger_pad': Transform(...),   # ← AVAILABLE!
    'right_inner_finger_pad': Transform(...),  # ← AVAILABLE!
    # ... all other gripper links
}
```

### Step 2: Bbox Computation (create_sparse_depth_map.py:339-514)

```python
def _compute_gripper_bbox(robot_model, robot_conf, gripper_width_mm, ...):
    # Get FK results from robot model
    fk_map = getattr(robot_model, "latest_transforms", None) or {}
    
    # Search for finger links using keyword matching
    def _find_link(keyword_groups):
        for keywords in keyword_groups:
            for name in fk_map.keys():
                lname = name.lower()
                if all(k in lname for k in keywords):
                    return name
        return None
    
    # Try to find left finger pad link
    left_link = _find_link([
        ("left", "inner", "finger", "pad"),  # ← Looks for "left_inner_finger_pad"
        ("left", "finger", "pad"),
        ("left", "inner", "finger"),
        ("left", "finger"),
        ("left", "knuckle"),
    ])
    
    # Try to find right finger pad link
    right_link = _find_link([
        ("right", "inner", "finger", "pad"),  # ← Looks for "right_inner_finger_pad"
        # ... similar fallbacks
    ])
```

**What happens for each config:**

**Config 3:**
```python
fk_map.keys() = ['base_link', 'shoulder_link', ..., 'wrist_3_link', 'ee_link']

# Search for links containing "left" AND "inner" AND "finger" AND "pad"
left_link = None  # ← NO MATCH - no such link exists!

# Search for links containing "right" AND "inner" AND "finger" AND "pad"
right_link = None  # ← NO MATCH - no such link exists!

# Both are None, so:
if left_pos is not None and right_pos is not None:
    # SKIPPED - both are None
elif left_tf is not None:
    # SKIPPED - left_tf is None
elif right_tf is not None:
    # SKIPPED - right_tf is None
else:
    print("[Warning] No gripper pad transforms available; cannot compute gripper bbox.")
    return None, None, None  # ← FAILS HERE
```

**Config 4:**
```python
fk_map.keys() = [..., 'left_inner_finger_pad', 'right_inner_finger_pad', ...]

# Search for links containing "left" AND "inner" AND "finger" AND "pad"
left_link = 'left_inner_finger_pad'  # ← FOUND!

# Search for links containing "right" AND "inner" AND "finger" AND "pad"
right_link = 'right_inner_finger_pad'  # ← FOUND!

# Extract transforms
left_tf = fk_map['left_inner_finger_pad'].matrix()  # ← 4x4 transform
right_tf = fk_map['right_inner_finger_pad'].matrix()  # ← 4x4 transform

# Both available, compute bbox!
if left_pos is not None and right_pos is not None:
    print("[Info] Both gripper pad transforms available; computing gripper frame from finger pads.")
    # Compute bbox using pad positions and orientations
    # ... SUCCESS! ✓
```

### Step 3: Cascade Effect

When `_compute_gripper_bbox()` returns `(None, None, None)` for config 3:

```python
# In process_frames() at line 2334
bbox_entry_for_frame, base_full_bbox, fingertip_bbox_for_frame = _compute_gripper_bbox(...)

# All three are None, so:
if robot_gripper_boxes is not None:
    robot_gripper_boxes.append(None)  # ← Append None
if robot_gripper_body_boxes is not None:
    robot_gripper_body_boxes.append(None)  # ← Append None
if robot_gripper_fingertip_boxes is not None:
    robot_gripper_fingertip_boxes.append(None)  # ← Append None

# Later, query points extraction:
if query_points is not None:
    query_bbox = full_bbox_for_frame if full_bbox_for_frame is not None else bbox_entry_for_frame
    if query_bbox is not None:  # ← FAILS - query_bbox is None!
        # Extract points inside bbox
    else:
        query_points.append(None)  # ← No query points!
```

Result:
- ❌ No contact bbox (orange)
- ❌ No body bbox (red)
- ❌ No fingertip bbox (blue)
- ❌ No query points extracted
- ❌ Empty bbox video
- ❌ No gripper contact tracking

## Why TCP-Based Method Also Fails

From `tcp_bbox_issue_analysis.md`, TCP-based method (`--use_tcp`) also fails because:

1. Tries to load pre-computed TCP poses from `transformed/tcp_base.npy`
2. File contains `None` entries for some camera serials/timestamps
3. Error: `'NoneType' object is not subscriptable` when accessing `tcp_data[serial][index]["tcp"]`

This is a **data quality issue** independent of the URDF problem, but compounds the failure.

## Why FK-Based Method Can't Fallback for Config 3

Some might ask: "Why not use `ee_link` as fallback?"

**Answer:** The code does check for fallback links:

```python
# Line 355-361
candidate_links = GRIPPER_LINK_CANDIDATES.get(robot_type, [])
ee_link = ROBOT_EE_LINK_MAP.get(robot_type)
all_possible_links = candidate_links + ([ee_link] if ee_link else [])

GRIPPER_LINK_CANDIDATES = {
    "ur5": ["robotiq_arg2f_base_link", "ee_link", "wrist_3_link", "wsg_50_base_link"],
}
```

But even if `ee_link` exists, the **finger link search still requires** pad/finger links to compute bbox orientation:

```python
# Line 418-432: Searches for "left" + "finger" + "pad" keywords
# If no matches found → cannot determine gripper frame axes
# Without knowing which direction fingers open → cannot create bbox
```

The `ee_link` transform alone doesn't tell us:
- Which axis is the gripper opening direction (width)
- Which axis is the approach direction (depth)
- Which axis is the height direction
- Where the fingertips are located

## Definitive Root Cause

**PRIMARY CAUSE:** Config 3's URDF (`ur5.urdf`) is **incomplete** - it represents only the robot arm without any gripper model.

**WHY THIS HAPPENS:** The WSG-50 gripper was likely:
1. Not modeled in the URDF when the config was created
2. Considered as a "black box" end-effector without detailed kinematics
3. Only represented by TCP calibration data (position/orientation), not full kinematic model

**WHY CONFIG 4 WORKS:** The Robotiq 2F-85 has a well-defined, publicly available URDF with full gripper kinematics that was integrated into `ur5_robotiq_85.urdf`.

## Impact Assessment

### Current State for Config 3
- ✅ Robot arm visualization works (uses 6 arm joints)
- ✅ Point cloud generation works
- ✅ RGB/depth capture works
- ❌ **Gripper bbox computation: FAILS**
- ❌ **Query point extraction: FAILS**
- ❌ **Contact region tracking: IMPOSSIBLE**
- ❌ **Gripper pose estimation: LIMITED** (only TCP point, no orientation/width)

### Why Some Data Exists
You mentioned cfg3 has processed data (`task_0024_user_0010_scene_0005_cfg_0003_processed.npz`).

This is because:
- RGB/depth arrays are generated regardless of bbox success
- Robot visualization uses arm joints only (works)
- The `.npz` file may have `None` values for gripper-related fields
- Rerun visualizations show robot arm but no bboxes

## Proof of Diagnosis

Execute this to verify:

```bash
# 1. Confirm cfg3 URDF has no gripper links
grep -i "finger\|pad\|knuckle\|wsg\|gripper" RH20T/models/ur5/urdf/ur5.urdf
# Expected: NO OUTPUT (exit code 1)

# 2. Confirm cfg4 URDF has gripper links
grep -i "finger\|pad\|knuckle" RH20T/models/ur5/urdf/ur5_robotiq_85.urdf | wc -l
# Expected: ~40+ lines

# 3. Compare link counts
echo "Config 3 links:"
grep "<link name=" RH20T/models/ur5/urdf/ur5.urdf | wc -l
echo "Config 4 links:"
grep "<link name=" RH20T/models/ur5/urdf/ur5_robotiq_85.urdf | wc -l
# Expected: cfg4 has 10+ more links
```

## Solutions (Not Implemented Per Your Request)

### Option 1: Create WSG-50 URDF Model ⭐ RECOMMENDED
- Source/create a URDF for WSG-50 gripper with finger links
- Integrate into `ur5_wsg50.urdf`
- Update config 3 to use new URDF
- Add `finger_joint` to joint sequence
- **Pros:** Enables full bbox computation, proper gripper tracking
- **Cons:** Requires URDF development/sourcing

### Option 2: Hardcode WSG-50 Geometry
- Detect when gripper is WSG-50 and no finger links available
- Use TCP pose + hardcoded gripper dimensions to synthesize bbox
- **Pros:** Quick fix, no URDF needed
- **Cons:** Less accurate, no adaptive finger width tracking

### Option 3: Use TCP-Only Mode with Data Fixes
- Fix `tcp_base.npy` data quality issues
- Rely purely on TCP-based bbox computation
- **Pros:** Uses existing calibration data
- **Cons:** Doesn't address FK-based computation, limited by TCP data quality

### Option 4: Disable Bbox for Config 3
- Accept that cfg3 cannot track gripper contact regions
- Process data without bbox/query points
- **Pros:** Simple, no code changes
- **Cons:** Loss of gripper interaction data for cfg3

## Conclusion

The root cause is **definitively** the absence of gripper link definitions in `ur5.urdf` used by config 3. The bbox computation algorithm fundamentally requires finger link transforms from Forward Kinematics to determine gripper pose and dimensions. Without these links in the URDF, FK cannot provide the necessary data, causing bbox computation to fail.

Config 4 works because its URDF includes complete gripper kinematics for the Robotiq 2F-85. This is not a bug in the code - it's a data/configuration limitation where cfg3's robot model is incomplete.

---
**Analysis Date:** October 15, 2025
**Status:** Root cause definitively identified, no fixes implemented per user request
