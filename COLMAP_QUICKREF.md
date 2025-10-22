# COLMAP Integration - Quick Reference

## Quick Start

### 1. Basic COLMAP Refinement
Filter to best 4 cameras based on geometric consistency:
```bash
python create_sparse_depth_map.py \
  --task-folder /path/to/task \
  --out-dir ./output \
  --refine-colmap \
  --limit-num-cameras 4
```

### 2. With Camera Cropping
Crop to ROI before COLMAP processing (faster):
```bash
python create_sparse_depth_map.py \
  --task-folder /path/to/task \
  --out-dir ./output \
  --add-robot \
  --gripper-body-bbox \
  --crop-camera \
  --refine-colmap \
  --limit-num-cameras 4
```

### 3. With Densification
Create dense point cloud with COLMAP MVS:
```bash
python create_sparse_depth_map.py \
  --task-folder /path/to/task \
  --out-dir ./output \
  --refine-colmap \
  --colmap-densification
```

### 4. Full Pipeline
All features enabled:
```bash
python create_sparse_depth_map.py \
  --task-folder /path/to/task \
  --high-res-folder /path/to/high_res \
  --out-dir ./output \
  --max-frames 50 \
  --add-robot \
  --gripper-body-bbox \
  --crop-camera \
  --crop-margin 0.2 \
  --refine-colmap \
  --limit-num-cameras 6 \
  --colmap-densification
```

## Argument Reference

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--crop-camera` | flag | False | Crop cameras to ROI |
| `--crop-margin` | float | 0.1 | Margin ratio around bbox |
| `--refine-colmap` | flag | False | Run COLMAP refinement |
| `--limit-num-cameras` | int | None | Keep N best cameras |
| `--colmap-densification` | flag | False | Create dense point cloud |
| `--colmap-workspace` | path | temp | COLMAP workspace directory |

## Feature Combinations

### Recommended: Camera Selection
```bash
--refine-colmap --limit-num-cameras 6
```
Use COLMAP to pick the 6 best cameras for your scene.

### Recommended: Fast Processing
```bash
--crop-camera --crop-margin 0.15
```
Crop to ROI before processing to speed up everything.

### Advanced: Dense Reconstruction
```bash
--refine-colmap --colmap-densification --colmap-workspace ./colmap
```
Keep COLMAP workspace for inspection, create dense cloud.

### Full Quality Pipeline
```bash
--crop-camera --refine-colmap --limit-num-cameras 4 --colmap-densification
```
Crop → Filter cameras → Densify

## Performance Tips

1. **Use cropping for speed**: `--crop-camera` reduces image size
2. **Limit cameras early**: Fewer cameras = faster COLMAP
3. **Skip densification if not needed**: It's the slowest step
4. **Reuse workspace**: `--colmap-workspace` to avoid recomputing

## Troubleshooting

### COLMAP not found
```bash
# Ubuntu/Debian
sudo apt install colmap

# macOS
brew install colmap
```

### Out of memory during densification
- Use `--crop-camera` to reduce image size
- Reduce `--max-frames`
- Use `--limit-num-cameras` with fewer cameras

### No cameras selected
- Remove `--limit-num-cameras` to see all scores
- Check that cameras have sufficient overlap
- Try with more frames

## Output

### Standard Output
- `<task_name>_processed.npz`: Filtered data with best cameras
- `<task_name>_reprojected.rrd`: Rerun visualization

### COLMAP Workspace (if `--colmap-workspace` specified)
- `images/`: Exported images
- `database.db`: Feature database
- `sparse/0/`: Sparse reconstruction
- `dense/`: Dense reconstruction (if `--colmap-densification`)

## Integration with Existing Features

COLMAP works with all existing features:
- ✅ `--high-res-folder`: Reproject to high-res after filtering
- ✅ `--add-robot`: Use gripper bbox for cropping
- ✅ `--sam2-tracking`: Track on filtered cameras
- ✅ `--color-alignment-check`: Apply after COLMAP

## When to Use Each Feature

### Use `--crop-camera` when:
- You have a robot and want to focus on gripper area
- Images are large and processing is slow
- You want to reduce computational overhead

### Use `--refine-colmap` when:
- You have multiple cameras and want to select the best ones
- Camera calibration might be imperfect
- You want to verify geometric consistency

### Use `--limit-num-cameras` when:
- You have too many cameras (>8)
- Processing time is too long
- You want consistent camera count across datasets

### Use `--colmap-densification` when:
- You need a dense point cloud
- Sparse depth maps are too sparse
- You want to compare with sensor depth

## See Also

- `COLMAP_INTEGRATION.md` - Full documentation
- `IMPLEMENTATION_SUMMARY.md` - Implementation details
- COLMAP website: https://colmap.github.io/
