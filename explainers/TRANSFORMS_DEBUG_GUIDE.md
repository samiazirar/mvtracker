# DROID Transform Debugging Guide

## Overview
All transformation logic is now centralized in `conversions/droid/utils/transforms.py` using **pytransform3d** for math operations. This makes the transformation pipeline transparent and debuggable in one place.

## Architecture

### Transform Pipeline Structure

```
transforms.py
├── Basic Utilities (generic)
│   ├── pose6_to_T()              - Convert pose to 4x4 matrix
│   ├── rvec_tvec_to_matrix()     - Convert extrinsics to 4x4 matrix
│   ├── transform_points()         - Apply transform to point cloud
│   ├── decompose_transform()      - Extract translation & rotation
│   └── invert_transform()         - Invert transformation
│
└── DROID Functional Pipelines (specific)
    ├── Wrist Camera Pipeline
    │   ├── compute_wrist_cam_offset()      - Calculate T_ee_cam (once at t=0)
    │   ├── wrist_cam_to_world()            - Get T_world_cam at timestep t
    │   ├── wrist_points_to_world()         - Transform points: cam → world
    │   └── precompute_wrist_trajectory()   - Precompute all transforms
    │
    └── External Camera Pipeline
        ├── external_cam_to_world()         - Get static T_world_cam
        └── external_points_to_world()      - Transform points: cam → world
```

## Debugging Workflow

### 1. Debug Wrist Camera Transforms

To debug the wrist camera transformation at a specific timestep:

```python
from utils import (
    compute_wrist_cam_offset, 
    wrist_cam_to_world, 
    wrist_points_to_world
)

# Step 1: Compute constant offset (debug this once)
T_ee_cam = compute_wrist_cam_offset(wrist_pose_t0, ee_pose_t0)
print(f"EE→Cam offset:\n{T_ee_cam}")

# Step 2: Debug transform at specific timestep
t = 50  # Frame to debug
T_world_cam = wrist_cam_to_world(
    cartesian_positions[t], 
    T_ee_cam, 
    gripper_alignment_fix=True
)
print(f"World→Cam at t={t}:\n{T_world_cam}")

# Step 3: Debug point transformation
test_point = np.array([[0.1, 0, 0]])  # Point in camera frame
point_world = wrist_points_to_world(
    test_point, 
    cartesian_positions[t], 
    T_ee_cam,
    gripper_alignment_fix=True
)
print(f"Point in world: {point_world}")
```

### 2. Debug External Camera Transforms

```python
from utils import external_cam_to_world, external_points_to_world

# Debug static transform
extrinsic_params = [tx, ty, tz, rx, ry, rz]
T_world_cam = external_cam_to_world(extrinsic_params)
print(f"External cam transform:\n{T_world_cam}")

# Debug point transformation
test_point = np.array([[0.1, 0, 0]])
point_world = external_points_to_world(test_point, extrinsic_params)
print(f"Point in world: {point_world}")
```

### 3. Debug Transform Chain Step-by-Step

For wrist camera, the complete chain is:

```
Points_camera → T_ee_cam → Points_ee → T_base_ee → Points_world
```

Debug each step:

```python
# Step 1: Camera → End-Effector
points_cam = np.array([[0.1, 0, 0]])
points_ee = transform_points(points_cam, T_ee_cam)
print(f"Points in EE frame: {points_ee}")

# Step 2: End-Effector → World
T_base_ee = pose6_to_T(cartesian_positions[t])
if gripper_alignment_fix:
    R_fix = R.from_euler('z', 90, degrees=True).as_matrix()
    T_base_ee[:3, :3] = T_base_ee[:3, :3] @ R_fix

points_world = transform_points(points_ee, T_base_ee)
print(f"Points in world frame: {points_world}")
```

## Key Functions Explained

### `compute_wrist_cam_offset(wrist_pose_t0, ee_pose_t0)`
**When to use:** Once at initialization to compute the constant offset between end-effector and camera.

**Math:**
```
T_ee_cam = inv(T_base_ee0) @ T_base_cam0
```

**Debug tip:** Check that this offset makes physical sense (camera should be ~10-20cm in front of gripper).

---

### `wrist_cam_to_world(ee_pose, T_ee_cam, gripper_alignment_fix=True)`
**When to use:** To get the camera's world transform at a specific timestep.

**Math:**
```
T_base_ee = pose6_to_T(ee_pose)
if gripper_alignment_fix:
    T_base_ee = T_base_ee @ R_z(90°)
T_world_cam = T_base_ee @ T_ee_cam
```

**Debug tip:** Visualize the resulting camera pose with rerun arrows to verify orientation.

---

### `wrist_points_to_world(points_cam, ee_pose, T_ee_cam, gripper_alignment_fix=True)`
**When to use:** Transform a point cloud from wrist camera to world coordinates.

**Combines:** `wrist_cam_to_world()` + `transform_points()`

**Debug tip:** Test with a single known point (e.g., [0.1, 0, 0]) and verify it appears at the expected location in world frame.

---

### `precompute_wrist_trajectory(cartesian_positions, wrist_pose_t0, gripper_alignment_fix=True)`
**When to use:** At initialization to compute all transforms for the entire trajectory (optimization).

**Returns:** List of 4x4 transforms, one per timestep.

**Debug tip:** Check first and last transforms - they should match the start and end positions of the gripper.

---

### `external_points_to_world(points_cam, extrinsic_params)`
**When to use:** Transform external camera points to world coordinates.

**Math:**
```
T_world_cam = rvec_tvec_to_matrix(extrinsic_params)
points_world = T @ [points_cam; 1]
```

**Debug tip:** External cameras are static - the transform should not change across frames.

## Common Issues & Solutions

### Issue 1: Points appear in wrong location
**Solution:** Add debug prints at each transformation step:
```python
print(f"Input points (cam): {points_cam[:5]}")
print(f"Transform:\n{T_world_cam}")
print(f"Output points (world): {points_world[:5]}")
```

### Issue 2: Gripper visualization misaligned
**Solution:** The `gripper_alignment_fix` parameter controls the 90° Z rotation. Toggle it:
```python
# Try both:
wrist_points_to_world(..., gripper_alignment_fix=True)
wrist_points_to_world(..., gripper_alignment_fix=False)
```

### Issue 3: Transforms look correct but points are flipped
**Solution:** Check coordinate system conventions. ZED uses RIGHT_HAND_Y_UP, but world might be Z_UP. Verify with:
```python
from utils import decompose_transform
translation, rotation = decompose_transform(T)
print(f"X-axis: {rotation[:, 0]}")
print(f"Y-axis: {rotation[:, 1]}")  
print(f"Z-axis: {rotation[:, 2]}")
```

### Issue 4: Performance issues with large point clouds
**Solution:** The `transform_points()` function is optimized with NumPy. If still slow, consider:
- Downsampling points before transformation
- Using vectorized operations (already done)
- Precomputing transforms (use `precompute_wrist_trajectory()`)

## Testing Transforms

Quick test for any transform function:

```python
# Identity transform test
T_identity = np.eye(4)
points_in = np.array([[1, 2, 3]])
points_out = transform_points(points_in, T_identity)
assert np.allclose(points_in, points_out)  # Should be identical

# Translation test
T_translate = np.eye(4)
T_translate[:3, 3] = [1, 2, 3]
points_out = transform_points(points_in, T_translate)
expected = points_in + np.array([1, 2, 3])
assert np.allclose(points_out, expected)

# Rotation test (90° around Z)
from utils import pose6_to_T
T_rotate = pose6_to_T([0, 0, 0, 0, 0, np.pi/2])
point_x = np.array([[1, 0, 0]])
point_rotated = transform_points(point_x, T_rotate)
assert np.allclose(point_rotated, [[0, 1, 0]], atol=1e-10)
```

## Visualization Tips

Use rerun to visualize transforms:

```python
# Visualize camera frame
translation, rotation = decompose_transform(T_world_cam)
rr.log(
    "debug/camera_frame",
    rr.Arrows3D(
        origins=[translation] * 3,
        vectors=[
            rotation[:, 0] * 0.1,  # X-axis (red)
            rotation[:, 1] * 0.1,  # Y-axis (green)
            rotation[:, 2] * 0.1   # Z-axis (blue)
        ],
        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]
    )
)
```

## References

- **pytransform3d docs:** https://dfki-ric.github.io/pytransform3d/
- **Transform conventions:** All transforms follow right-hand rule, transform order is left-to-right multiplication
- **Coordinate frames:**
  - World: Z-up (RIGHT_HAND_Z_UP)
  - ZED Camera: Y-down, Z-forward
  - End-Effector: Varies by robot (see DROID calibration)
