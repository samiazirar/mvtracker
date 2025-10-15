# TCP-Based Bbox Detection Issue Analysis for Config 3

## Problem Summary
The TCP-based gripper bounding box detection is failing for config 3 with the error:
```
[WARN] Could not get TCP from API: 'NoneType' object is not subscriptable
[WARN] Cannot compute TCP-based bbox, skipping frame
```

This leads to:
- No bounding boxes being computed for the gripper
- No query points being generated
- Empty bbox video export

## Root Cause Analysis

### Error Location Chain

1. **Error originates in**: `/workspace/create_sparse_depth_map.py` line 2288
   ```python
   tcp_pose_7d = robot_scene.get_tcp_aligned(t_low)
   ```

2. **The method** `get_tcp_aligned` is defined in: `/workspace/RH20T/rh20t_api/scene.py` lines 579-600
   ```python
   def get_tcp_aligned(self, timestamp:int, serial:str="base"):
       if self.is_high_freq:
           # High frequency mode handling...
       if serial == "base":
           tcp_base_aligned = self.tcp_base_aligned  # Loads the tcp_base property
           _idx_1, _idx_2 = binary_search_closest_two_idx(self._base_aligned_timestamps, timestamp)
           (serial_1, serial_idx_1) = self._base_aligned_timestamps_in_serial[_idx_1]
           (serial_2, serial_idx_2) = self._base_aligned_timestamps_in_serial[_idx_2]
           return interpolate_linear(
               timestamp,
               self._base_aligned_timestamps[_idx_1],
               self._base_aligned_timestamps[_idx_2],
               tcp_base_aligned[serial_1][serial_idx_1]["tcp"],  # ← FAILS HERE
               tcp_base_aligned[serial_2][serial_idx_2]["tcp"]
           )
   ```

3. **The data is loaded by**: `_load_tcp_base_aligned()` method (lines 168-175)
   ```python
   def _load_tcp_base_aligned(self): 
       self._tcp_base_aligned = load_dict_npy(os.path.join(
           self.folder, 
           self._used_aligned_folder,  # Default: "transformed"
           "tcp_base.npy"
       ))
       sort_by_timestamp(self._tcp_base_aligned)
       if self._base_aligned_timestamps is None:
           _t_v = []
           for _k in self._tcp_base_aligned: 
               _t_v.extend([(_item["timestamp"], _k, _i) for _i, _item in enumerate(self._tcp_base_aligned[_k])])
           _t_v.sort()
           self._base_aligned_timestamps = [_item[0] for _item in _t_v]
           self._base_aligned_timestamps_in_serial = [(_item[1], _item[2]) for _item in _t_v]
   ```

### The Actual Issue

The error `'NoneType' object is not subscriptable` occurs when:

```python
tcp_base_aligned[serial_1][serial_idx_1]["tcp"]  # Trying to access ["tcp"] on None
```

This means that `tcp_base_aligned[serial_1][serial_idx_1]` returns `None` instead of a dictionary containing a "tcp" key.

## Why Config 3 Fails

### Possible Causes

1. **Missing or Corrupted Data File**
   - The file `<task_folder>/transformed/tcp_base.npy` may not exist for config 3
   - OR the file exists but contains `None` values for certain timestamps/indices

2. **Data Structure Mismatch**
   - The expected structure is: `tcp_base_aligned[serial][index] = {"tcp": <7d_pose>, "timestamp": <int>}`
   - But for config 3, some entries are `None` instead of dictionaries

3. **Camera/Serial Issues**
   - From the logs, cam_f0172289 had "No valid color/depth frame pairs found"
   - cam_036422060909 was skipped for "No color frames found"
   - The `serial_1` or `serial_2` being looked up might correspond to a camera that has no TCP data

4. **Preprocessing/Alignment Issues**
   - Config 3 might not have been preprocessed with TCP alignment
   - The transformation/alignment step that creates `transformed/tcp_base.npy` may have failed silently for this config

### Logs Show
```
[INFO] Camera cam_f0172289: No valid color/depth frame pairs found.
[INFO] Camera cam_036422060909: Aligned 154 frames (median gap 108.0 ms).
[WARNING] Achieved FPS (4.41) is more than 5% below target (10.00).
[WARNING] Dropped 56.13% of timeline slots due to missing frames.
```

This indicates significant data quality issues for config 3 - many missing frames and low FPS.

## Why FK-Based Method Might Work Better

The script has two bbox computation methods:
1. **TCP-based** (failing): Uses `scene.get_tcp_aligned()` from the dataset API
2. **FK-based** (default): Computes TCP from forward kinematics using joint angles

The FK-based method only needs joint angles, which are retrieved via:
```python
joint_angles = robot_scene.get_joint_angles_aligned(t_low)
```

This likely works because joint angle data is more robust and complete than the pre-computed TCP poses.

## Fix Recommendations

### Short-term Workarounds
1. **Disable TCP-based mode**: Remove `--use_tcp` flag or ensure `args.use_tcp=False`
2. **Use FK-based bbox computation**: This is the default and should work

### Long-term Fixes
1. **Add data validation** before trying to access TCP data:
   ```python
   tcp_pose_7d = robot_scene.get_tcp_aligned(t_low)
   if tcp_pose_7d is None:
       # Fallback to FK-based computation
   ```

2. **Regenerate preprocessed data** for config 3:
   - Check if `<task_folder>/transformed/tcp_base.npy` exists and is valid
   - Rerun the alignment/preprocessing pipeline for this configuration

3. **Better error handling** in `get_tcp_aligned()`:
   ```python
   # In scene.py, add validation:
   tcp_data_1 = tcp_base_aligned[serial_1][serial_idx_1]
   tcp_data_2 = tcp_base_aligned[serial_2][serial_idx_2]
   
   if tcp_data_1 is None or tcp_data_2 is None:
       raise ValueError(f"TCP data missing for serial {serial_1} or {serial_2}")
   ```

4. **Check the data file directly**:
   ```python
   import numpy as np
   data = np.load("<task_folder>/transformed/tcp_base.npy", allow_pickle=True)
   # Inspect the structure and check for None values
   ```

## Verification Steps

To diagnose the exact issue:

1. Check if the file exists:
   ```bash
   ls -la <task_folder>/transformed/tcp_base.npy
   ```

2. Load and inspect the data:
   ```python
   import numpy as np
   tcp_data = np.load("<task_folder>/transformed/tcp_base.npy", allow_pickle=True).item()
   
   # Check structure
   print("Keys (serials):", tcp_data.keys())
   
   # Check for None values
   for serial, entries in tcp_data.items():
       none_count = sum(1 for e in entries if e is None)
       print(f"Serial {serial}: {none_count}/{len(entries)} None entries")
   ```

3. Compare with a working config (like config 0) to see structural differences

## Impact

Without TCP-based bbox:
- ✅ Robot visualization still works (uses FK)
- ✅ Point cloud generation works
- ❌ TCP-based bbox computation fails → Falls back to FK
- ❌ If FK also fails, no query points are generated
- ❌ No gripper contact region tracking

The fact that "No valid query points found" suggests that even the FK fallback isn't producing bboxes, which may indicate a deeper issue with the gripper width or joint angle data for config 3.
