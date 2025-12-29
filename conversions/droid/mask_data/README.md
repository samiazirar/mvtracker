# DROID Mask Data Generation

This module provides utilities for generating 2D masks from 3D robot meshes.

## Methods Available

### Method 1: Direct Mesh Projection (Stage 1)
Projects the robot mesh (Panda arm + Robotiq gripper) directly to 2D using FK.
- Requires: `joint_positions` from original trajectory.h5
- Output: Arm mask, gripper mask, combined mask

### Method 2: Sensor Point Classification (Stage 2)
Uses the 3D robot mesh to classify depth sensor points, then reprojects to RGB.
- Requires: `joint_positions` + depth images
- More accurate as it uses actual sensor data

## Quick Start

### Full Robot Body Mask (Stage 1) - Direct Mesh Projection

```bash
cd /workspace/conversions/droid/mask_data
python robot_mask_from_depth.py \
    --h5_path /data/droid/data/droid_raw/1.0.1/IRIS/success/2023-05-12/Fri_May_12_10:51:00_2023/trajectory.h5 \
    --processed_dir /workspace/droid_processed/IRIS/success/2023-05-12/Fri_May_12_10:51:00_2023 \
    --output_dir ./robot_mask_output \
    --max_frames 163
```

Output:
- `{camera}/arm_mask.mp4` - Panda arm (link0-link7) in orange
- `{camera}/gripper_mask.mp4` - Robotiq gripper in green
- `{camera}/full_robot_mask.mp4` - Combined in cyan

### Sensor Point Classification (Stage 2)

```bash
python robot_mask_from_sensor_points.py \
    --h5_path /data/droid/data/droid_raw/1.0.1/IRIS/success/2023-05-12/Fri_May_12_10:51:00_2023/trajectory.h5 \
    --processed_dir /workspace/droid_processed/IRIS/success/2023-05-12/Fri_May_12_10:51:00_2023 \
    --output_dir ./sensor_robot_mask_output \
    --distance_threshold 0.03 \
    --max_frames 163
```

Output:
- `{camera}/sensor_robot_mask.mp4` - Robot points from depth sensor in yellow

### Gripper-Only Mask (from processed data)

For episodes without joint_positions (only gripper_poses):

```bash
python demo_mask_on_rgb.py \
    --episode_dir /workspace/droid_processed/IRIS/success/2023-05-12/Fri_May_12_10:51:00_2023 \
    --output_dir ./mask_output \
    --max_frames 163
```

## Python API

### Gripper Mask Rendering (Robotiq 85 only)

```python
from mask_data import GripperMaskRenderer

# Initialize renderer (NO Panda hand by default - matches original code)
gripper = GripperMaskRenderer(include_hand=False)

# Render mask for a single frame
mask = gripper.render_mask(
    T_world_ee,      # 4x4 end-effector transform (from tracks['gripper_poses'])
    gripper_pos,     # 0.0 (open) to 1.0 (closed)
    K,               # 3x3 camera intrinsics
    T_cam_world,     # 4x4 camera extrinsics (world to camera)
    width, height,   # Image dimensions
    min_depth=0.01   # Minimum depth threshold
)
# Returns: Binary mask (height x width) as uint8
```

### Robot Arm Mask Rendering (Requires Joint Angles)

**Note:** Full arm rendering requires joint angles (joint_positions) from the
original trajectory.h5 file. Processed episodes only have gripper_poses.

```python
from mask_data import RobotArmMaskRenderer

# Initialize renderer
robot = RobotArmMaskRenderer()

# Render mask for a single frame
mask = robot.render_mask(
    joint_angles,    # 7-element array of joint angles (radians)
    K,               # 3x3 camera intrinsics
    T_cam_world,     # 4x4 camera extrinsics
    width, height,   # Image dimensions
    T_world_base=np.eye(4),  # Robot base transform
    min_depth=0.01,
    exclude_hand=True  # Exclude hand mesh (gripper attachment point)
)
```

### Combined Robot + Gripper

```python
from mask_data import CombinedRobotMaskRenderer

# Initialize combined renderer
robot = CombinedRobotMaskRenderer()

# Render all masks
gripper_mask, arm_mask, combined_mask = robot.render_masks(
    joint_angles,    # 7-element joint angles
    gripper_pos,     # Gripper position
    K, T_cam_world,
    width, height
)
```

### Forward Kinematics

```python
from mask_data import panda_forward_kinematics, robotiq_gripper_transforms

# Get all link transforms for Panda arm
link_transforms = panda_forward_kinematics(joint_angles)
# Returns: dict mapping "link0", "link1", ..., "link7", "hand" to 4x4 transforms

# Get gripper component transforms
gripper_transforms = robotiq_gripper_transforms(T_world_ee, gripper_pos)
# Returns: dict with "base", "left_inner_finger", "right_outer_knuckle", etc.
```

## Mesh Files

The module uses the following mesh files:

### Robotiq 85 Gripper
Located in `/workspace/external/robotiq_arg85_description/meshes/`:
- `robotiq_85_base_link_fine.STL` - Base
- `outer_knuckle_fine.STL` - Outer knuckles (x2)
- `outer_finger_fine.STL` - Outer fingers (x2)
- `inner_knuckle_fine.STL` - Inner knuckles (x2)
- `inner_finger_fine.STL` - Inner fingers (x2, contact surfaces)

### Panda Robot Arm
Located in `/workspace/third_party/CtRNet-X/urdfs/Panda/meshes/collision/`:
- `link0.obj` through `link7.obj` - Arm links
- `hand.obj` - Gripper attachment flange

## File Structure

```
mask_data/
├── __init__.py              # Module exports
├── mask_utils.py            # Core mask rendering utilities
├── generate_mask_video.py   # DROID episode mask video generator
├── demo_mask_rendering.py   # Synthetic demo video generator
├── README.md                # This file
└── demo_output/             # Demo output directory
    ├── demo_gripper_mask.mp4
    ├── demo_robot_mask.mp4
    └── demo_combined_mask.mp4
```

## Requirements

- `numpy`
- `opencv-python`
- `trimesh`
- `scipy`

For DROID episode processing:
- `h5py`
- `pyzed` (optional, for SVO file reading)
- `pyyaml`

## Notes

### Coordinate Systems
- Robot base is at world origin (can be overridden with `T_world_base`)
- Camera extrinsics `T_cam_world` transforms from world to camera frame
- Gripper uses DROID convention: 0.0 = open, 1.0 = closed
- The `gripper_poses` in `tracks.npz` already have the R_fix (90° Z rotation) applied

### Data Availability

**Processed DROID episodes** (`droid_processed/`) contain:
- `tracks.npz` with `gripper_poses` (end-effector transforms, 4x4 x T)
- `extrinsics.npz` with camera poses (`external_{serial}` = world_T_cam)
- RGB frames in `{serial}/` subdirectories

**What's NOT available in processed data:**
- `joint_positions` (required for full arm rendering via FK)
- Original trajectory.h5 files

For full arm rendering, you need access to the original HuggingFace DROID dataset
which contains joint_positions in the trajectory.h5 files.

### Panda Hand Transform

The Panda hand is positioned behind the gripper base. From the EE frame:
- Translation: [0, 0, -0.058] meters (slightly behind gripper mount)
- Rotation: None (aligned with EE frame)

This positions the `hand.obj` mesh correctly relative to the Robotiq gripper.

### Performance
- Mesh loading is done once at initialization
- Each frame renders independently (no GPU required)
- Typical render time: ~10-20ms per frame per camera

## Validation

The mask projection has been validated against the HuggingFace pipeline:
- Projection formula matches `project_points_to_image()` in video_utils.py
- At frame 100, mask centroid (499, 208) is within ~15px of EE position (490, 223)
- Mask correctly follows gripper trajectory across all frames
