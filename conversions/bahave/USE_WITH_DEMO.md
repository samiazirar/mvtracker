# Using BEHAVE Dataset with demo.py

This guide shows how to use your converted BEHAVE `.npz` files with the main `demo.py` tracking script.

## Quick Start

### Step 1: Convert BEHAVE Data

First, convert a BEHAVE scene to the format expected by demo.py:

```bash
python conversions/behave_to_npz.py \
    --scene Date01_Sub01_backpack_back \
    --max_frames 100 \
    --downscale_factor 2
```

This creates: `conversions/behave_converted/Date01_Sub01_backpack_back.npz`

### Step 2: Run Tracking with demo.py

```bash
python demo.py \
    --sample-path conversions/behave_converted/Date01_Sub01_backpack_back.npz \
    --tracker mvtracker \
    --depth_estimator gt \
    --rerun save \
    --rrd behave_tracking.rrd
```

## Data Format Compatibility

The converted BEHAVE `.npz` files are **directly compatible** with `demo.py`! Here's the mapping:

| BEHAVE .npz Key | demo.py Expected | Shape | Notes |
|-----------------|------------------|-------|-------|
| `rgbs` | `rgbs` | (C, T, 3, H, W) | ✅ Direct match |
| `depths` | `depths` | (C, T, H, W) | ✅ Direct match |
| `intrinsics` | `intrs` | (C, 3, 3) | Need to broadcast to (C, T, 3, 3) |
| `extrinsics` | `extrs` | (C, 3, 4) | Need to broadcast to (C, T, 3, 4) |
| `query_points` | `query_points` | (N, 3) or (T, C, N, 2) | Need to convert 2D→3D |

## Format Conversion Script

The BEHAVE data needs minor format adjustments. Here's a helper script:

```python
# conversions/adapt_behave_for_demo.py
import numpy as np
import argparse
from pathlib import Path

def adapt_behave_for_demo(input_npz: Path, output_npz: Path):
    """Adapt BEHAVE .npz format to demo.py format."""
    
    print(f"Loading: {input_npz}")
    data = np.load(input_npz)
    
    # Load data
    rgbs = data['rgbs']              # (C, T, 3, H, W)
    depths = data['depths']          # (C, T, H, W)
    intrinsics = data['intrinsics']  # (C, 3, 3)
    extrinsics = data['extrinsics']  # (C, 3, 4)
    query_points_2d = data['query_points']  # (T, C, N, 2)
    
    C, T = rgbs.shape[:2]
    
    # 1. Broadcast intrinsics to (C, T, 3, 3)
    intrs = np.tile(intrinsics[:, None, :, :], (1, T, 1, 1))
    
    # 2. Broadcast extrinsics to (C, T, 3, 4)
    extrs = np.tile(extrinsics[:, None, :, :], (1, T, 1, 1))
    
    # 3. Convert query points from 2D to 3D
    # Use first frame, first camera query points and unproject to 3D
    qpts_2d = query_points_2d[0, 0]  # (N, 2) from first frame, first camera
    depth_map = depths[0, 0]  # (H, W)
    K = intrinsics[0]  # (3, 3)
    
    # Unproject 2D points to 3D
    query_points_3d = []
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    for x, y in qpts_2d:
        x_int, y_int = int(x), int(y)
        if 0 <= y_int < depth_map.shape[0] and 0 <= x_int < depth_map.shape[1]:
            z = depth_map[y_int, x_int]
            if z > 0:
                X = (x - cx) * z / fx
                Y = (y - cy) * z / fy
                Z = z
                query_points_3d.append([X, Y, Z])
    
    query_points = np.array(query_points_3d, dtype=np.float32)
    
    print(f"\nConverted format:")
    print(f"  rgbs: {rgbs.shape}")
    print(f"  depths: {depths.shape}")
    print(f"  intrs: {intrs.shape}")
    print(f"  extrs: {extrs.shape}")
    print(f"  query_points: {query_points.shape}")
    
    # Save in demo.py format
    np.savez_compressed(
        output_npz,
        rgbs=rgbs,
        depths=depths,
        intrs=intrs,
        extrs=extrs,
        query_points=query_points,
    )
    
    print(f"\nSaved to: {output_npz}")
    print(f"File size: {output_npz.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_npz", help="Input BEHAVE .npz file")
    parser.add_argument("--output", help="Output path (default: add _demo suffix)")
    args = parser.parse_args()
    
    input_path = Path(args.input_npz)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_demo.npz"
    
    adapt_behave_for_demo(input_path, output_path)
```

## Usage Workflow

### Full Pipeline

```bash
# 1. Convert BEHAVE scene
python conversions/behave_to_npz.py \
    --scene Date01_Sub01_backpack_back \
    --max_frames 100 \
    --downscale_factor 2

# 2. Adapt for demo.py
python conversions/adapt_behave_for_demo.py \
    conversions/behave_converted/Date01_Sub01_backpack_back.npz

# 3. Run tracking
python demo.py \
    --sample-path conversions/behave_converted/Date01_Sub01_backpack_back_demo.npz \
    --tracker mvtracker \
    --depth_estimator gt \
    --rerun save \
    --rrd behave_backpack.rrd
```

### With Different Trackers

**MVTracker (multi-view):**
```bash
python demo.py \
    --sample-path conversions/behave_converted/Date01_Sub01_backpack_back_demo.npz \
    --tracker mvtracker \
    --depth_estimator gt \
    --rerun save
```

**CoTracker (monocular baseline):**
```bash
python demo.py \
    --sample-path conversions/behave_converted/Date01_Sub01_backpack_back_demo.npz \
    --tracker cotracker3_offline \
    --depth_estimator gt \
    --rerun save
```

**SpatialTracker V2:**
```bash
python demo.py \
    --sample-path conversions/behave_converted/Date01_Sub01_backpack_back_demo.npz \
    --tracker spatialtrackerv2 \
    --depth_estimator gt \
    --rerun save
```

## Advanced Options

### Use Random Query Points

Instead of mask-based query points, sample from depth:

```bash
python demo.py \
    --sample-path conversions/behave_converted/Date01_Sub01_backpack_back_demo.npz \
    --tracker mvtracker \
    --depth_estimator gt \
    --random_query_points \
    --rerun save
```

### Batch Processing (for large scenes)

```bash
python demo.py \
    --sample-path conversions/behave_converted/Date01_Sub01_backpack_back_demo.npz \
    --tracker mvtracker \
    --depth_estimator gt \
    --batch_processing \
    --batch_size_views 2 \
    --batch_size_frames 50 \
    --optimize_performance \
    --rerun save
```

### Temporal/Spatial Downsampling

```bash
python demo.py \
    --sample-path conversions/behave_converted/Date01_Sub01_backpack_back_demo.npz \
    --tracker mvtracker \
    --depth_estimator gt \
    --temporal_stride 2 \
    --spatial_downsample 2 \
    --rerun save
```

### Save Tracking Results

```bash
python demo.py \
    --sample-path conversions/behave_converted/Date01_Sub01_backpack_back_demo.npz \
    --tracker mvtracker \
    --depth_estimator gt \
    --save-npz results/behave_tracks.npz \
    --rerun save
```

## Visualization

After running `demo.py`, view results in Rerun:

```bash
# If you used --rerun save
rerun behave_backpack.rrd

# Or live streaming (use --rerun stream in demo.py)
# View at http://localhost:9876
```

## Memory Considerations

BEHAVE scenes can be large. Use these strategies:

1. **Limit frames during conversion:**
   ```bash
   python conversions/behave_to_npz.py --scene SCENE --max_frames 50
   ```

2. **Increase downscaling:**
   ```bash
   python conversions/behave_to_npz.py --scene SCENE --downscale_factor 4
   ```

3. **Use batch processing in demo.py:**
   ```bash
   python demo.py --batch_processing --batch_size_views 2 --batch_size_frames 30
   ```

4. **Further downsample in demo.py:**
   ```bash
   python demo.py --spatial_downsample 2 --temporal_stride 2
   ```

## Troubleshooting

**Problem: "RuntimeError: CUDA out of memory"**

Solution:
```bash
# Reduce batch sizes
python demo.py --batch_processing --batch_size_views 1 --batch_size_frames 20

# Or increase downscaling
python demo.py --spatial_downsample 4
```

**Problem: "Query points are all zeros"**

Solution: The masks might be empty. Use random query points:
```bash
python demo.py --random_query_points
```

**Problem: "Shape mismatch for intrinsics/extrinsics"**

Solution: Make sure to run the `adapt_behave_for_demo.py` script first to broadcast the camera parameters to the correct temporal dimensions.

**Problem: "Tracking quality is poor"**

Solutions:
- Use ground-truth depth: `--depth_estimator gt`
- Try different trackers: `--tracker spatialtrackerv2`
- Reduce temporal stride: `--temporal_stride 1`
- Use more query points in conversion: `--num_query_points 512`

## Complete Example

Here's a complete workflow from raw BEHAVE to tracking results:

```bash
# 1. Convert with good settings
python conversions/behave_to_npz.py \
    --scene Date01_Sub01_basketball \
    --max_frames 150 \
    --downscale_factor 2 \
    --num_query_points 512 \
    --mask_type person

# 2. Adapt format
python conversions/adapt_behave_for_demo.py \
    conversions/behave_converted/Date01_Sub01_basketball.npz

# 3. Track with MVTracker
python demo.py \
    --sample-path conversions/behave_converted/Date01_Sub01_basketball_demo.npz \
    --tracker mvtracker \
    --depth_estimator gt \
    --batch_processing \
    --optimize_performance \
    --save-npz results/basketball_tracks.npz \
    --rerun save \
    --rrd basketball_tracking.rrd

# 4. View results
rerun basketball_tracking.rrd
```

## Next Steps

- Use tracked points for 3D reconstruction
- Export to video with `export_tracks_to_videos.py`
- Refine tracks with `refine_tracks.py`
- Analyze interaction patterns
