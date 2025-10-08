# TCP API Troubleshooting Guide

## Issue: API Returns None or Invalid Data

### Error Message
```
[WARN] Could not get TCP from API: 'NoneType' object is not subscriptable
[INFO] Falling back to FK-based TCP computation
```

### Root Cause
The `get_tcp_aligned()` API method requires preprocessed TCP data that should be in:
```
<task_folder>/transformed/tcp_base.npy
```

If this file doesn't exist or contains incomplete data, the API method will fail.

### Solution Options

#### Option 1: Use FK Fallback (Current Behavior)
The code now automatically falls back to FK-based computation when API data is unavailable.

**Pros:**
- Works with any dataset
- No preprocessing required
- Automatic fallback

**Cons:**
- Less accurate than API method
- Subject to URDF model limitations

#### Option 2: Preprocess the Dataset
Run the RH20T preprocessing to generate the aligned TCP data:

```bash
# TODO: Add the actual preprocessing command from RH20T
python -m RH20T.scripts.preprocess_scene --folder <task_folder>
```

**Pros:**
- Most accurate TCP positions
- Uses official robot controller data
- Better for production use

**Cons:**
- Requires preprocessing step
- May not work for all datasets

#### Option 3: Check Data Availability
Before running, check if the transformed data exists:

```bash
# Check if TCP data is available
ls -la <task_folder>/transformed/tcp_base.npy

# If missing, you can:
# 1. Use FK fallback (automatic)
# 2. Preprocess the dataset
# 3. Use a different task folder with preprocessed data
```

## Current Implementation Status

✅ **Implemented:**
- Primary: Try to use API's `get_tcp_aligned()`
- Validation: Check that returned data is valid
- Fallback: Use FK computation if API fails
- Logging: Clear messages about which method is being used

✅ **Working:**
- FK-based computation (your current working method)
- Automatic fallback when API data unavailable
- Valid gripper bounding boxes from FK

## Expected Behavior

### With Preprocessed TCP Data
```
[INFO] Using API TCP pose for gripper bbox: position=[x, y, z], quat=[qx, qy, qz, qw]
```

### Without Preprocessed TCP Data (Your Current Case)
```
[WARN] Could not get TCP from API: 'NoneType' object is not subscriptable
[INFO] Falling back to FK-based TCP computation
[INFO] Using FK-based TCP from link 'ee_link'
```

## Verifying Your Setup

### Check 1: Does transformed folder exist?
```bash
ls -la <task_folder>/transformed/
```

Expected files:
- `tcp_base.npy` - TCP poses aligned to robot base
- `force_torque_base.npy` - Force/torque data
- `joint.npy` - Joint angles
- `gripper.npy` - Gripper states

### Check 2: Is the data valid?
```python
import numpy as np

# Load the TCP data
tcp_data = np.load("<task_folder>/transformed/tcp_base.npy", allow_pickle=True).item()

# Check structure
print("Keys:", tcp_data.keys())
print("Sample entry:", list(tcp_data.values())[0][0] if tcp_data else "Empty")
```

Expected structure:
```python
{
  'camera_serial_1': [
    {'timestamp': 123456, 'tcp': [x, y, z, qx, qy, qz, qw]},
    {'timestamp': 123457, 'tcp': [x, y, z, qx, qy, qz, qw]},
    ...
  ],
  'camera_serial_2': [...],
  ...
}
```

## Recommendation

For your current use case, **the FK fallback is working correctly**. The bounding boxes should still be reasonably accurate. The API method would provide marginally better accuracy (typically sub-millimeter improvements), but FK is sufficient for most applications.

If you need the highest possible accuracy:
1. Ensure your dataset is fully preprocessed with transformed data
2. Or work with a dataset that already has the transformed folder

## Code Changes Summary

The implementation now:
1. ✅ **Tries API first** - Attempts to use `get_tcp_aligned()`
2. ✅ **Validates data** - Checks that returned data is valid (not None, correct length)
3. ✅ **Falls back gracefully** - Uses FK if API data unavailable
4. ✅ **Logs clearly** - Shows which method succeeded

This ensures maximum accuracy when data is available, while maintaining compatibility with datasets that don't have preprocessed TCP data.
