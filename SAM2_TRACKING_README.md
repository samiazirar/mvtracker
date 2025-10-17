# SAM2 Gripper Tracking Implementation

## Overview

This implementation adds SAM2 (Segment Anything Model 2) video tracking capability to track the robot gripper throughout video sequences. The tracking uses query points from the gripper bounding box as initial prompts and maintains continuous tracking without re-initialization.

## Features

### 1. SAM2 Video Tracking
- **Continuous Tracking**: Tracks gripper through all frames without re-initialization
- **Query Point Initialization**: Uses 3D query points (from gripper bbox) projected to 2D as initial prompts
- **Automatic Frame Selection**: Finds the first frame with valid query points if specified init frame has none
- **Flexible Initialization**: Configurable frame index for starting the tracking

### 2. 3D Visualization
- **Mask Display**: Shows 2D segmentation masks in Rerun
- **Mesh Conversion**: Converts 2D masks to 3D meshes using depth information
- **Poisson Reconstruction**: Creates smooth 3D meshes from masked point clouds
- **Color Preservation**: Maintains RGB colors from the original frames

### 3. Robust Error Handling
- Validates SAM2 checkpoint existence
- Handles missing query points gracefully
- Provides detailed error messages with stack traces
- Automatic cleanup of temporary files

## Command-Line Arguments

### SAM2 Tracking Arguments

```bash
--sam2-tracking / --no-sam2-tracking
    Enable/disable SAM2 tracking for gripper segmentation
    Default: False
    Requires: query points from gripper bbox

--sam2-checkpoint PATH
    Path to SAM2 model checkpoint file
    Default: sam2/checkpoints/sam2.1_hiera_large.pt

--sam2-config CONFIG
    SAM2 model configuration file (relative to sam2 package)
    Default: configs/sam2.1/sam2.1_hiera_l.yaml
    Options: 
      - configs/sam2.1/sam2.1_hiera_t.yaml (tiny - fastest)
      - configs/sam2.1/sam2.1_hiera_s.yaml (small)
      - configs/sam2.1/sam2.1_hiera_b+.yaml (base+)
      - configs/sam2.1/sam2.1_hiera_l.yaml (large - most accurate)

--sam2-init-frame N
    Frame index to initialize SAM2 tracking
    Default: 0
    Tip: Choose a frame where gripper is clearly visible and not touching objects

--visualize-sam2-masks / --no-visualize-sam2-masks
    Enable/disable SAM2 mask visualization in Rerun
    Default: True

--sam2-mask-as-mesh / --no-sam2-mask-as-mesh
    Convert SAM2 masks to 3D meshes for visualization
    Default: True
```

## Usage Examples

### Basic SAM2 Tracking

```bash
python create_sparse_depth_map.py \
  --task-folder /path/to/task \
  --high-res-folder /path/to/high_res_task \
  --out-dir ./output \
  --add-robot \
  --gripper-bbox \
  --sam2-tracking \
  --max-frames 50
```

### Advanced Configuration

```bash
python create_sparse_depth_map.py \
  --task-folder /path/to/task \
  --high-res-folder /path/to/high_res_task \
  --out-dir ./output \
  --add-robot \
  --gripper-bbox \
  --sam2-tracking \
  --sam2-checkpoint sam2/checkpoints/sam2.1_hiera_base_plus.pt \
  --sam2-config configs/sam2.1/sam2.1_hiera_b+.yaml \
  --sam2-init-frame 5 \
  --sam2-mask-as-mesh \
  --max-frames 100
```

### Fast Tracking (Tiny Model)

```bash
python create_sparse_depth_map.py \
  --task-folder /path/to/task \
  --out-dir ./output \
  --add-robot \
  --gripper-bbox \
  --sam2-tracking \
  --sam2-checkpoint sam2/checkpoints/sam2.1_hiera_tiny.pt \
  --sam2-config configs/sam2.1/sam2.1_hiera_t.yaml \
  --no-sam2-mask-as-mesh \
  --max-frames 200
```

## Prerequisites

### 1. SAM2 Installation

```bash
cd /workspace/sam2
pip install -e .
```

### 2. Download SAM2 Checkpoints

```bash
cd /workspace/sam2/checkpoints

# Download the model you want to use:

# Tiny (fastest, ~42MB)
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt

# Small (~182MB)
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt

# Base+ (~230MB)
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt

# Large (best quality, ~894MB) - RECOMMENDED
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

## Implementation Details

### Workflow

1. **Frame Preparation**: Saves RGB frames from selected camera to temporary directory
2. **Initialization**: Creates SAM2 inference state with video frames
3. **Point Projection**: Projects 3D query points to 2D image coordinates
4. **Prompt Creation**: Uses projected points as foreground prompts for SAM2
5. **Propagation**: Tracks object through all frames using SAM2's temporal model
6. **Cleanup**: Removes temporary frame directory

### Key Functions

#### `track_gripper_with_sam2()`
Main tracking function that:
- Initializes SAM2 video predictor
- Projects query points to 2D
- Adds points as prompts to SAM2
- Propagates masks through video
- Returns list of binary masks

#### `convert_mask_to_mesh()`
Converts 2D masks to 3D meshes:
- Applies mask to depth and RGB
- Creates RGBD point cloud
- Estimates normals
- Performs Poisson surface reconstruction
- Filters low-density vertices

### Rerun Visualization

SAM2 results are logged to Rerun with:
- **2D Masks**: `camera/{camera_id}/sam2_mask` - Segmentation overlays
- **3D Meshes**: `sam2/gripper_mesh` - Colored 3D gripper mesh

## Performance Considerations

### Model Selection
- **Tiny**: ~10-20 FPS, good for real-time preview
- **Small**: ~5-10 FPS, balanced speed/quality
- **Base+**: ~3-5 FPS, better quality
- **Large**: ~1-3 FPS, best quality (recommended for final results)

### Memory Usage
- Frames are temporarily saved to disk (not kept in memory)
- GPU memory usage depends on model size:
  - Tiny: ~2GB
  - Small: ~4GB
  - Base+: ~6GB
  - Large: ~8GB

### Tips for Better Results
1. **Choose good init frame**: Pick a frame where gripper is clearly visible
2. **More query points**: More points = better initial mask
3. **Clean backgrounds**: SAM2 works best with clear object boundaries
4. **Consistent lighting**: Avoid drastic lighting changes between frames

## Troubleshooting

### SAM2 checkpoint not found
```
[ERROR] SAM2 checkpoint not found at: sam2/checkpoints/sam2.1_hiera_large.pt
```
**Solution**: Download the checkpoint using wget (see Prerequisites)

### No query points found
```
[ERROR] No frames with query points found. Cannot initialize SAM2.
```
**Solution**: Enable gripper bbox generation with `--gripper-bbox`

### CUDA out of memory
```
RuntimeError: CUDA out of memory
```
**Solution**: Use a smaller model (tiny or small) or reduce `--max-frames`

### Import error
```
Import "sam2.build_sam" could not be resolved
```
**Solution**: Install SAM2 package: `cd sam2 && pip install -e .`

## Commits

1. **Initial Implementation** (`2b88791`):
   - Added SAM2 tracking function
   - Added command-line arguments
   - Implemented mask-to-mesh conversion
   - Integrated with Rerun visualization

2. **Bug Fixes** (`e0f7cc2`):
   - Fixed frame directory handling
   - Corrected SAM2 API usage (add_new_points)
   - Improved error handling
   - Proper cleanup of temporary files

## Future Enhancements

Potential improvements:
- [ ] Multi-object tracking (track multiple grippers or objects)
- [ ] Refinement clicks (manual correction of masks)
- [ ] Mask smoothing/filtering
- [ ] Export masks to video files
- [ ] Integration with MVTracker for point tracking
- [ ] Automatic quality assessment of masks

## References

- SAM2 Paper: https://arxiv.org/abs/2408.00714
- SAM2 GitHub: https://github.com/facebookresearch/sam2
- Rerun Visualization: https://www.rerun.io/
