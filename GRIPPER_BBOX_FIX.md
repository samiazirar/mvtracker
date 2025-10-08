# Gripper Bounding Box Precision Fix

## Problem
The gripper bounding boxes were showing incorrect positioning (off in orientation, height, and x-dimension) because they were being computed from a generic URDF model and joint angles using Forward Kinematics (FK), which can have inaccuracies.

## Solution
Use the RH20T dataset's official `get_tcp_aligned()` API method to obtain the precise Tool Center Point (TCP) pose directly from the robot's high-fidelity internal controller recordings.

## Changes Made

### 1. Added Helper Function (`_pose_7d_to_matrix`)
**Location:** Lines ~173-197

```python
def _pose_7d_to_matrix(pose_7d: np.ndarray) -> np.ndarray:
    """Convert a 7D pose [x, y, z, qx, qy, qz, qw] to a 4x4 transformation matrix.
    
    Args:
        pose_7d: 7-element array with position (x,y,z) and quaternion (qx,qy,qz,qw)
        
    Returns:
        4x4 transformation matrix
    """
```

This function converts the 7D pose array returned by `get_tcp_aligned()` into a 4x4 transformation matrix that can be used for bounding box computation.

### 2. Modified TCP Computation in `process_frames`
**Location:** Lines ~1683-1720

**Before:**
- Used FK (Forward Kinematics) to compute TCP from joint angles
- Less accurate due to URDF model simplifications

**After:**
- **Primary method:** Call `scene_low.get_tcp_aligned(t_low)` to get the official TCP pose
- **Fallback:** If API call fails, fall back to FK computation
- Convert 7D pose to 4x4 matrix using `_pose_7d_to_matrix()`
- Pass the precise TCP transform to `_compute_gripper_bbox()`

## Technical Details

### TCP Pose Format
The `get_tcp_aligned()` function returns a 7-element array:
- **Elements 0-2:** Position (x, y, z) in meters
- **Elements 3-6:** Orientation as quaternion (qx, qy, qz, qw)

### Why This is More Accurate

1. **Direct from Controller:** The TCP pose comes directly from the robot's internal controller, which has high-fidelity sensor data
2. **Pre-processed:** The dataset maintainers have already processed and aligned this data
3. **No Model Approximation:** Bypasses any potential inaccuracies or simplifications in the public URDF model
4. **Ground Truth:** This is as close to "ground truth" as we can get for the gripper's actual position

### Backwards Compatibility
- The change includes a fallback to the original FK-based method if the API call fails
- Existing functionality for computing gripper boxes from FK remains intact
- Debug logging on first frame shows which method is being used

## Testing
To test the fix, run the script with gripper bounding boxes enabled:

```bash
python create_sparse_depth_map.py \
  --task-folder /path/to/task_folder \
  --out-dir ./output \
  --add-robot \
  --gripper-bbox \
  --max-frames 50
```

Look for the log message on first frame:
```
[INFO] Using API TCP pose for gripper bbox: position=[...], quat=[...]
```

## Expected Results
- Gripper bounding boxes should now be precisely positioned at the bottom of the gripper (contact surface)
- Orientation should match the actual gripper orientation in the scene
- Position in all dimensions (x, y, z) should be accurate
- The bounding box should align with where the gripper makes contact with objects
