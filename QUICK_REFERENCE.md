# Quick Reference Card

## New Features Summary

### 1️⃣ Bbox Alignment (Default: ON)
Aligns bounding boxes with point cloud geometry.

**Enable/Disable:**
```bash
--align-bbox-with-points      # Default (ON)
--no-align-bbox-with-points   # Disable
```

**Tune:**
```bash
--align-bbox-search-radius-scale 2.0  # Search radius (default: 2.0)
```

**What it does:** Adjusts bbox x-y position and rotation to match nearby points, preserves height.

---

### 2️⃣ MVTracker Gripper Tracking (Default: OFF)
Tracks gripper bounding boxes across frames.

**Enable:**
```bash
--track-gripper-with-mvtracker
```

**Prerequisites:**
```bash
pip install git+https://github.com/ethz-vlg/mvtracker.git
```

**What it does:** Tracks 8 corner points of each bbox type (contact, body, fingertip) independently.

**Output:** Orange, red, and blue point tracks in Rerun.

---

### 3️⃣ SAM Object Tracking (Default: OFF)
Segments objects in contact with gripper.

**Enable:**
```bash
--track-objects-with-sam
--sam-model-type vit_b          # vit_b (fast) or vit_h (accurate)
--sam-contact-threshold 50.0    # Distance threshold in pixels
```

**Prerequisites:**
```bash
pip install segment-anything
wget -P ~/.cache/sam https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

**What it does:** Segments objects near gripper in each frame using SAM.

**Output:** Binary segmentation masks in Rerun.

---

## Common Workflows

### Minimal (Alignment Only)
```bash
python create_sparse_depth_map.py \
  --task-folder /path/to/task \
  --out-dir ./output \
  --add-robot \
  --gripper-bbox
# Alignment is ON by default
```

### With Tracking (No SAM)
```bash
python create_sparse_depth_map.py \
  --task-folder /path/to/task \
  --out-dir ./output \
  --add-robot \
  --gripper-bbox \
  --gripper-body-bbox \
  --gripper-fingertip-bbox \
  --track-gripper-with-mvtracker
```

### Full Pipeline (All Features)
```bash
bash scripts/run_gripper_example.sh
```

Or manually:
```bash
python create_sparse_depth_map.py \
  --task-folder /path/to/task \
  --out-dir ./output \
  --add-robot \
  --gripper-bbox \
  --gripper-body-bbox \
  --gripper-fingertip-bbox \
  --align-bbox-with-points \
  --track-gripper-with-mvtracker \
  --track-objects-with-sam
```

---

## Visualization Guide

Open `.rrd` file in Rerun viewer:

**Bounding Boxes:**
- 🟠 Orange: Contact bbox (aligned)
- 🔴 Red: Body bbox (aligned)
- 🔵 Blue: Fingertip bbox (aligned)

**Tracks (if MVTracker enabled):**
- 🟠 Orange points: Contact bbox tracks
- 🔴 Red points: Body bbox tracks  
- 🔵 Blue points: Fingertip bbox tracks

**Segmentation (if SAM enabled):**
- Binary masks in `segmentation/contact_objects`
- Overlay on RGB images

**Navigation:**
- Timeline slider: Scrub through frames
- Camera view: Switch between camera angles
- 3D view: Rotate and zoom point clouds

---

## Troubleshooting

### Alignment not working
✓ Check point cloud density (needs points near gripper)
✓ Try larger `--align-bbox-search-radius-scale`
✓ Verify `--gripper-bbox` is enabled

### MVTracker fails
✓ Install: `pip install git+https://github.com/ethz-vlg/mvtracker.git`
✓ Check GPU available: `nvidia-smi`
✓ Verify bboxes computed: enable `--gripper-bbox`

### SAM not segmenting
✓ Install: `pip install segment-anything`
✓ Download checkpoint (see above)
✓ Check SAM checkpoint path: `~/.cache/sam/`
✓ Try larger `--sam-contact-threshold`

### Performance slow
✓ Reduce `--max-frames`
✓ Disable SAM if not needed
✓ Use smaller SAM model: `vit_b` instead of `vit_h`
✓ Use GPU for MVTracker

---

## Testing

### Run Tests
```bash
# Unit tests
python tests/test_gripper_features.py

# Visual tests
python tests/test_gripper_features.py --visual
```

### Validate Installation
```bash
python -c "from create_sparse_depth_map import _align_bbox_with_point_cloud_com; print('OK')"
```

---

## Files Created/Modified

**New:**
- `tests/test_gripper_features.py` - Comprehensive tests
- `GRIPPER_TRACKING_FEATURES.md` - Detailed documentation
- `IMPLEMENTATION_SUMMARY.md` - Implementation summary
- `PIPELINE_DIAGRAM.txt` - Visual pipeline diagram

**Modified:**
- `create_sparse_depth_map.py` - All new features
- `scripts/run_gripper_example.sh` - Updated with new flags

---

## Git Commits

All features committed atomically:

```
4e46d11 docs: Add pipeline diagram showing data flow
e448543 docs: Add implementation summary
eee9ea3 docs: Add comprehensive documentation for new features
971c1e6 feat: Add SAM integration for object tracking
2c526cd feat: Integrate MVTracker for gripper tracking
bd0cf3f feat: Add bbox alignment with point cloud center of mass
271b036 test: Add comprehensive tests for bbox alignment, mvtracker, and SAM integration
```

Each commit can be reviewed/reverted independently.

---

## Performance Tips

**Fast (Interactive):**
```bash
--max-frames 10
--no-track-objects-with-sam
--no-track-gripper-with-mvtracker
```

**Balanced:**
```bash
--max-frames 50
--track-gripper-with-mvtracker
--no-track-objects-with-sam  # SAM is slowest
```

**Full Quality:**
```bash
--max-frames 100
--track-gripper-with-mvtracker
--track-objects-with-sam
--sam-model-type vit_h  # Best quality, slowest
```

---

## Next Steps

1. ✅ Test on real data
2. ✅ Verify Rerun visualization
3. ✅ Check alignment accuracy
4. ✅ Validate tracking quality
5. ✅ Tune performance parameters

For more details, see `GRIPPER_TRACKING_FEATURES.md`
