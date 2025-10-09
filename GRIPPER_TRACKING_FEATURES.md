# Gripper Tracking and Alignment Features

This document describes the new features added for gripper tracking, bbox alignment, and object segmentation.

## Features Overview

### 1. Bbox Alignment with Point Cloud Center of Mass

Automatically aligns gripper bounding boxes with the center of mass of nearby point cloud points. This improves bbox accuracy by adjusting position and rotation to match the actual geometry.

**Key Characteristics:**
- Adjusts bbox center in x-y plane (preserves height/z-axis)
- Adjusts rotation based on point cloud covariance
- Preserves bbox dimensions
- Filters points by distance (configurable search radius)

**CLI Arguments:**
```bash
--align-bbox-with-points              # Enable alignment (default: True)
--no-align-bbox-with-points           # Disable alignment
--align-bbox-search-radius-scale 2.0  # Search radius scale factor
```

**Implementation:**
- Function: `_align_bbox_with_point_cloud_com()`
- Location: `create_sparse_depth_map.py`
- Applied to: contact bbox, body bbox, fingertip bbox

### 2. MVTracker Integration for Gripper Tracking

Tracks gripper bounding boxes across frames using MVTracker (multi-view point tracker).

**Key Characteristics:**
- Invokes MVTracker 3 times independently:
  1. Contact bbox (orange points in visualization)
  2. Body bbox (red points)
  3. Fingertip bbox (blue points)
- Generates query points from bbox corners
- Tracks 8 corner points per bbox
- Logs tracks to Rerun with color coding

**CLI Arguments:**
```bash
--track-gripper-with-mvtracker   # Enable tracking (default: False)
```

**Implementation:**
- Function: `_track_gripper_with_mvtracker()`
- Helper: `_generate_query_points_from_bbox()`
- Location: `create_sparse_depth_map.py`
- Output: Dictionary with tracks and visibility for each bbox type

**Visualization:**
- Orange points: Contact bbox tracks
- Red points: Body bbox tracks
- Blue points: Fingertip bbox tracks
- Logged to `tracks/gripper_tracks`, `tracks/body_tracks`, `tracks/fingertip_tracks`

### 3. SAM Integration for Object Tracking

Uses Segment Anything Model (SAM) to segment and track objects in contact with the gripper.

**Key Characteristics:**
- Projects 3D gripper bboxes to 2D per camera view
- Expands bbox by contact threshold to include nearby objects
- Segments objects frame-by-frame using SAM
- Optional dependency (gracefully handled if missing)

**CLI Arguments:**
```bash
--track-objects-with-sam          # Enable SAM tracking (default: False)
--sam-model-type vit_b            # SAM model: vit_b, vit_l, vit_h
--sam-contact-threshold 50.0      # Contact distance in pixels
```

**Implementation:**
- Function: `_segment_object_with_sam()`
- Function: `_track_gripper_contact_objects_with_sam()`
- Location: `create_sparse_depth_map.py`
- SAM checkpoint location: `~/.cache/sam/sam_<model_type>.pth`

**Visualization:**
- Binary segmentation masks
- Logged to `segmentation/contact_objects`
- Can be overlaid on RGB images in Rerun

## Installation

### Core Dependencies (Already Included)
```bash
# These are already in requirements.txt
torch
numpy
open3d
rerun-sdk
```

### Optional Dependencies

#### MVTracker (For Gripper Tracking)
```bash
pip install git+https://github.com/ethz-vlg/mvtracker.git
```

#### SAM (For Object Segmentation)
```bash
pip install segment-anything

# Download SAM checkpoint
mkdir -p ~/.cache/sam
wget -P ~/.cache/sam https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

## Usage Examples

### Basic Usage with Alignment (Default)
```bash
python create_sparse_depth_map.py \
  --task-folder /path/to/task_folder \
  --out-dir ./output \
  --add-robot \
  --gripper-bbox
  # Alignment is enabled by default
```

### Disable Alignment
```bash
python create_sparse_depth_map.py \
  --task-folder /path/to/task_folder \
  --out-dir ./output \
  --add-robot \
  --gripper-bbox \
  --no-align-bbox-with-points
```

### Full Pipeline with All Features
```bash
python create_sparse_depth_map.py \
  --task-folder /path/to/task_folder \
  --high-res-folder /path/to/high_res_folder \
  --out-dir ./output \
  --max-frames 50 \
  --add-robot \
  --gripper-bbox \
  --gripper-body-bbox \
  --gripper-fingertip-bbox \
  --align-bbox-with-points \
  --align-bbox-search-radius-scale 2.5 \
  --track-gripper-with-mvtracker \
  --track-objects-with-sam \
  --sam-model-type vit_b \
  --sam-contact-threshold 40.0
```

### Run Example Script
```bash
bash scripts/run_gripper_example.sh
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python tests/test_gripper_features.py

# Run visual tests (requires rerun)
python tests/test_gripper_features.py --visual
```

### Test Coverage

1. **Bbox Alignment Tests:**
   - Shift in x-y plane
   - Preserve height (z-axis)
   - Rotation from point cloud covariance
   - Handle empty point clouds
   - Filter distant outliers

2. **MVTracker Integration Tests:**
   - Track generation from bboxes
   - Query point generation
   - Multiple bbox types
   - Handle missing bboxes

3. **SAM Integration Tests:**
   - Segmentation mask generation
   - Multi-frame tracking
   - Contact detection

## Architecture

All features are implemented as isolated, modular functions:

1. **Alignment:** `_align_bbox_with_point_cloud_com()`
   - Called after bbox computation in `process_frames()`
   - Applied to all bbox types independently
   - Configurable via CLI arguments

2. **MVTracker:** `_track_gripper_with_mvtracker()`
   - Called after `process_frames()` completes
   - Separate function, doesn't modify main pipeline
   - Results passed to visualization

3. **SAM:** `_track_gripper_contact_objects_with_sam()`
   - Called after MVTracker step
   - Separate function, doesn't modify main pipeline
   - Results passed to visualization

This modular design ensures:
- Features don't break each other
- Easy to enable/disable individually
- Clear separation of concerns
- Straightforward testing

## Visualization in Rerun

Open the generated `.rrd` file in Rerun viewer to see:

1. **Point Clouds:** RGB point clouds from all camera views
2. **Robot Model:** URDF-based robot visualization
3. **Bounding Boxes:**
   - Orange: Contact bbox (aligned)
   - Red: Body bbox (aligned)
   - Blue: Fingertip bbox (aligned)
4. **Tracks:**
   - Orange points: Contact bbox tracks
   - Red points: Body bbox tracks
   - Blue points: Fingertip bbox tracks
5. **Segmentation:** Object masks overlaid on camera views

Navigate through time using the timeline slider to see tracked points and masks evolve.

## Performance Notes

- **Alignment:** Minimal overhead (~1-5ms per frame)
- **MVTracker:** GPU-accelerated, ~100-500ms per invocation (3 invocations total)
- **SAM:** CPU/GPU depending on model, ~1-3 seconds per frame
  - Use `vit_b` for faster inference
  - Use `vit_h` for better quality (slower)

For faster processing:
- Reduce `--max-frames`
- Use `--no-track-objects-with-sam` if object segmentation not needed
- Use smaller SAM model (`vit_b` instead of `vit_h`)

## Known Limitations

1. **Alignment:**
   - Requires sufficient point cloud density near gripper
   - May not work well for very sparse depth data
   - Currently preserves z-coordinate (height) - this is by design

2. **MVTracker:**
   - Requires GPU for reasonable performance
   - Query points from bbox corners may not represent gripper perfectly
   - Tracks bbox structure, not gripper surface directly

3. **SAM:**
   - Requires SAM checkpoint download (~350MB for vit_b)
   - Only processes first camera view currently (can be extended)
   - May segment background objects near gripper
   - Requires good RGB image quality

## Future Improvements

Potential enhancements:
- [ ] Multi-view SAM tracking (aggregate across cameras)
- [ ] Temporal consistency for SAM masks (track same object)
- [ ] Dynamic query point generation (adaptive to gripper state)
- [ ] Real-time mode with frame skipping
- [ ] Integration with object pose estimation
- [ ] Export tracks to standard formats (BOP, etc.)

## Citation

If you use these features in your research, please cite:

```bibtex
@software{gripper_tracking_2024,
  title = {Gripper Tracking and Alignment Features for RH20T},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/your-repo}
}
```

## Support

For issues or questions:
1. Check test outputs: `python tests/test_gripper_features.py`
2. Enable debug mode: `--debug-mode`
3. Verify dependencies are installed
4. Check Rerun visualization for visual debugging
