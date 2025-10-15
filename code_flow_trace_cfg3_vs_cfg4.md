# Code Flow Trace: Why Config 3 Fails vs Config 4 Succeeds

This document traces the exact execution path through the code to show where and why config 3 fails.

## Execution Timeline

### Frame Processing Loop (create_sparse_depth_map.py:2162)

```python
for ti in tqdm(range(T), desc="Processing Frames"):
    # ... setup code ...
    
    if robot_model is not None:
        # Get joint angles at this timestamp
        t_low = int(per_cam_low_ts[0][ti])
        joint_angles = robot_scene.get_joint_angles_aligned(t_low)
        
        # Config 3: joint_angles = [q1, q2, q3, q4, q5, q6] (6 values)
        # Config 4: joint_angles = [q1, q2, q3, q4, q5, q6, q_gripper] (7 values)
```

### Robot Model Update (RH20T/utils/robot.py:117)

```python
def update(self, rotates: np.ndarray, first_time: bool):
    self._init_buffer()
    
    # Build transformations dictionary from joint values
    transformations = {
        joint: rotates[i] 
        for i, joint in enumerate(self._robot_joint_sequence)
    }
    
    # Config 3 transformations:
    # {
    #     'shoulder_pan_joint': rotates[0],
    #     'shoulder_lift_joint': rotates[1],
    #     'elbow_joint': rotates[2],
    #     'wrist_1_joint': rotates[3],
    #     'wrist_2_joint': rotates[4],
    #     'wrist_3_joint': rotates[5]
    # }
    # NO 'finger_joint' key!
    
    # Config 4 transformations:
    # {
    #     'shoulder_pan_joint': rotates[0],
    #     'shoulder_lift_joint': rotates[1],
    #     'elbow_joint': rotates[2],
    #     'wrist_1_joint': rotates[3],
    #     'wrist_2_joint': rotates[4],
    #     'wrist_3_joint': rotates[5],
    #     'finger_joint': rotates[6]  # ← HAS gripper joint!
    # }
    
    # Robotiq coupling: derive other gripper joints from finger_joint
    if 'finger_joint' in transformations:  # ← TRUE for cfg4, FALSE for cfg3
        finger_value = transformations['finger_joint']
        transformations['left_inner_finger_joint'] = -finger_value
        transformations['left_inner_knuckle_joint'] = -finger_value
        transformations['right_outer_knuckle_joint'] = finger_value
        transformations['right_inner_finger_joint'] = -finger_value
        transformations['right_inner_knuckle_joint'] = -finger_value
    
    # CRITICAL: Forward Kinematics computation
    # This computes transforms for ALL links defined in the URDF
    cur_transforms = self._model_chain.forward_kinematics(transformations)
    
    # Config 3 cur_transforms keys:
    # ['base_link', 'shoulder_link', 'upper_arm_link', 'forearm_link',
    #  'wrist_1_link', 'wrist_2_link', 'wrist_3_link', 'ee_link', 'world']
    # ONLY 9 links - NO finger/pad links!
    
    # Config 4 cur_transforms keys:
    # ['base_link', 'shoulder_link', 'upper_arm_link', 'forearm_link',
    #  'wrist_1_link', 'wrist_2_link', 'wrist_3_link', 'ee_link',
    #  'robotiq_arg2f_base_link', 'left_outer_knuckle', 'left_outer_finger',
    #  'left_inner_finger', 'left_inner_finger_pad', 'left_inner_knuckle',
    #  'right_outer_knuckle', 'right_outer_finger', 'right_inner_finger',
    #  'right_inner_finger_pad', 'right_inner_knuckle', 'world']
    # 20 links including finger pads!
    
    # Cache for bbox computation
    self._latest_transforms = cur_transforms
```

**Why FK returns different links:**
- Forward Kinematics computes transforms based on URDF structure
- Config 3's `ur5.urdf` defines only 9 links (arm + ee_link)
- Config 4's `ur5_robotiq_85.urdf` defines 20 links (arm + gripper)
- FK cannot compute transforms for links that don't exist in URDF!

### Gripper Bbox Computation (create_sparse_depth_map.py:2334)

```python
bbox_entry_for_frame, base_full_bbox, fingertip_bbox_for_frame = _compute_gripper_bbox(
    robot_model,
    robot_conf,
    current_gripper_width_mm,
    contact_height_m=getattr(args, "gripper_bbox_contact_height_m", None),
    contact_length_m=getattr(args, "gripper_bbox_contact_length_m", None),
    tcp_transform=None,  # Not used in FK-based computation
)
```

### Inside _compute_gripper_bbox (create_sparse_depth_map.py:339)

```python
def _compute_gripper_bbox(robot_model, robot_conf, gripper_width_mm, ...):
    # Get cached FK results
    fk_map = getattr(robot_model, "latest_transforms", None) or {}
    
    # Config 3 fk_map: 9 links (no finger pads)
    # Config 4 fk_map: 20 links (includes finger pads)
    
    # Dimension lookup
    gripper_name = getattr(robot_conf, "gripper", "")
    # Config 3: gripper_name = "WSG-50"
    # Config 4: gripper_name = "Robotiq 2F-85"
    
    # ... dimension setup code ...
```

### Link Search (create_sparse_depth_map.py:410)

```python
    def _find_link(keyword_groups):
        # Search FK cache for links matching ALL keywords in a group
        for keywords in keyword_groups:
            for name in fk_map.keys():
                lname = name.lower()
                if all(k in lname for k in keywords):
                    return name
        return None
    
    # Search for LEFT finger pad
    left_link = _find_link([
        ("left", "inner", "finger", "pad"),  # Match: "left_inner_finger_pad"
        ("left", "finger", "pad"),           # Fallback 1
        ("left", "inner", "finger"),         # Fallback 2
        ("left", "finger"),                  # Fallback 3
        ("left", "knuckle"),                 # Fallback 4
    ])
    
    # CONFIG 3 EXECUTION:
    # Iteration 1: Look for link with "left" AND "inner" AND "finger" AND "pad"
    #   Search through: ['base_link', 'shoulder_link', ..., 'ee_link', 'world']
    #   ❌ No match - no link contains all these keywords
    # Iteration 2: Look for "left" AND "finger" AND "pad"
    #   ❌ No match
    # Iteration 3: Look for "left" AND "inner" AND "finger"
    #   ❌ No match
    # Iteration 4: Look for "left" AND "finger"
    #   ❌ No match
    # Iteration 5: Look for "left" AND "knuckle"
    #   ❌ No match
    # Result: left_link = None
    
    # CONFIG 4 EXECUTION:
    # Iteration 1: Look for link with "left" AND "inner" AND "finger" AND "pad"
    #   Search through: [..., 'left_inner_finger_pad', ...]
    #   ✅ MATCH: "left_inner_finger_pad" contains all keywords!
    # Result: left_link = "left_inner_finger_pad"
    
    # Search for RIGHT finger pad (same logic)
    right_link = _find_link([
        ("right", "inner", "finger", "pad"),
        ("right", "finger", "pad"),
        ("right", "inner", "finger"),
        ("right", "finger"),
        ("right", "knuckle"),
    ])
    
    # Config 3: right_link = None
    # Config 4: right_link = "right_inner_finger_pad"
```

### Transform Extraction (create_sparse_depth_map.py:436)

```python
    # Get transforms for finger pads
    left_tf = fk_map[left_link].matrix().astype(np.float32) if left_link and left_link in fk_map else None
    right_tf = fk_map[right_link].matrix().astype(np.float32) if right_link and right_link in fk_map else None
    
    # Config 3:
    #   left_link = None → left_tf = None
    #   right_link = None → right_tf = None
    
    # Config 4:
    #   left_link = "left_inner_finger_pad"
    #   left_tf = fk_map["left_inner_finger_pad"].matrix()
    #   left_tf = 4x4 transform matrix representing finger pad pose in world frame
    #   right_tf = similar for right pad
    
    left_pos = left_tf[:3, 3] if left_tf is not None else None
    right_pos = right_tf[:3, 3] if right_tf is not None else None
    
    # Config 3: left_pos = None, right_pos = None
    # Config 4: left_pos = [x, y, z], right_pos = [x, y, z]
```

### Frame Computation Decision Tree (create_sparse_depth_map.py:444)

```python
    # Try to compute gripper frame from available transforms
    if left_pos is not None and right_pos is not None:
        # BEST CASE: Both finger pads available
        # Config 4 takes this path ✅
        print("[Info] Both gripper pad transforms available; computing gripper frame from finger pads.")
        
        # Compute width axis (jaw separation direction)
        width_vec = left_pos - right_pos
        width_axis = width_vec / np.linalg.norm(width_vec)
        
        # Compute approach axis (toward grasped object)
        approach_axis = (left_tf[:3, 2] + right_tf[:3, 2]) * 0.5
        approach_axis = approach_axis / np.linalg.norm(approach_axis)
        
        # Compute height axis (perpendicular to width and approach)
        height_axis = np.cross(approach_axis, width_axis)
        height_axis = height_axis / np.linalg.norm(height_axis)
        
        # Use pad midpoint as frame origin
        pad_midpoint = (left_pos + right_pos) * 0.5
        
        # SUCCESS: Have complete gripper frame!
        
    elif left_tf is not None:
        # FALLBACK 1: Only left pad available
        # Config 3 SKIPS this (left_tf is None)
        print("[Warning] Only left gripper pad transform available; using left pad frame directly.")
        # ... use left pad pose ...
        
    elif right_tf is not None:
        # FALLBACK 2: Only right pad available
        # Config 3 SKIPS this (right_tf is None)
        print("[Warning] Only right gripper pad transform available; negating X axis...")
        # ... use right pad pose ...
        
    else:
        # FAILURE CASE: No transforms available
        # Config 3 takes this path ❌
        print("[Warning] No gripper pad transforms available; cannot compute gripper bbox.")
        return None, None, None  # ← FAILS HERE
```

**Config 3 Execution Path:**
```
start → fk_map has 9 links (no fingers) → left_link = None → left_tf = None 
     → right_link = None → right_tf = None → all branches skip → return (None, None, None)
```

**Config 4 Execution Path:**
```
start → fk_map has 20 links (with fingers) → left_link = "left_inner_finger_pad" 
     → left_tf = Transform(...) → right_link = "right_inner_finger_pad" 
     → right_tf = Transform(...) → first branch executes → compute bbox → return (bbox, body, tip)
```

### Back to Frame Processing (create_sparse_depth_map.py:2463)

```python
    # Receive results from bbox computation
    if robot_gripper_boxes is not None:
        # Config 3: bbox_entry_for_frame = None
        # Config 4: bbox_entry_for_frame = {'center': [x,y,z], 'half_sizes': [...], ...}
        robot_gripper_boxes.append(bbox_entry_for_frame)
    
    if robot_gripper_body_boxes is not None:
        # Config 3: full_bbox_for_frame = None
        # Config 4: full_bbox_for_frame = {...}
        robot_gripper_body_boxes.append(full_bbox_for_frame)
```

### Query Point Extraction (create_sparse_depth_map.py:2477)

```python
    if query_points is not None:
        query_bbox = full_bbox_for_frame if full_bbox_for_frame is not None else bbox_entry_for_frame
        
        # Config 3: query_bbox = None (both sources are None)
        # Config 4: query_bbox = {...} (has valid bbox)
        
        if query_bbox is not None and points_world_np is not None and points_world_np.size > 0:
            # Extract points inside bbox
            # Config 4 executes this ✅
            inside_pts, inside_cols = _extract_points_inside_bbox(
                points_world_np,
                query_bbox,
                colors=colors_world_np,
            )
            query_points.append(inside_pts)
        else:
            # Config 3 executes this ❌
            query_points.append(None)
```

### Final Result

**Config 3 Output:**
```python
robot_gripper_boxes = [None, None, None, ..., None]  # All None
robot_gripper_body_boxes = [None, None, None, ..., None]  # All None
robot_gripper_fingertip_boxes = [None, None, None, ..., None]  # All None
query_points = [None, None, None, ..., None]  # All None
```

**Config 4 Output:**
```python
robot_gripper_boxes = [
    {'center': [x,y,z], 'half_sizes': [...], 'axes': [...]},
    {'center': [x,y,z], 'half_sizes': [...], 'axes': [...]},
    ...
]  # Valid bboxes for each frame

robot_gripper_body_boxes = [...]  # Valid body bboxes
robot_gripper_fingertip_boxes = [...]  # Valid fingertip bboxes
query_points = [
    np.array([[x1,y1,z1], [x2,y2,z2], ...]),  # Points inside bbox
    np.array([[x1,y1,z1], [x2,y2,z2], ...]),
    ...
]  # Valid query points for each frame
```

## Summary: The Critical Difference

| Step | Config 3 (WSG-50) | Config 4 (Robotiq 2F-85) |
|------|------------------|------------------------|
| URDF links | 9 (arm only) | 20 (arm + gripper) |
| Joint sequence | 6 joints | 7 joints (includes finger_joint) |
| FK output | 9 link transforms | 20 link transforms |
| Left pad link search | ❌ None found | ✅ "left_inner_finger_pad" |
| Right pad link search | ❌ None found | ✅ "right_inner_finger_pad" |
| Gripper frame computation | ❌ FAILS (no transforms) | ✅ SUCCESS (both pads) |
| Bbox output | None, None, None | contact_bbox, body_bbox, fingertip_bbox |
| Query points | None | Valid point arrays |
| Rerun visualization | ❌ No bboxes shown | ✅ Bboxes visualized |

## The Root Cause Chain

```
1. Config 3 uses ur5.urdf (arm only, no gripper model)
                ↓
2. URDF defines only 9 links (no finger/pad links)
                ↓
3. Forward Kinematics can only compute transforms for links in URDF
                ↓
4. FK output has 9 transforms (no finger pad transforms)
                ↓
5. _find_link() searches for "left_inner_finger_pad" pattern
                ↓
6. No match found (link doesn't exist in FK output)
                ↓
7. left_link = None, right_link = None
                ↓
8. No transform data available to compute gripper frame
                ↓
9. Return (None, None, None) - bbox computation FAILS
                ↓
10. No bboxes → no query points → no contact tracking
```

**The fix requires one of:**
- Add WSG-50 gripper to URDF (proper solution)
- Hardcode WSG-50 geometry when no finger links found (workaround)
- Use TCP-only mode with fixed data (alternative)

---
**This trace definitively proves the URDF is the root cause.**
