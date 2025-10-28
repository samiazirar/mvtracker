# Quick Reference: Object Tracking Scripts

## 📁 New Files Created

| File | Size | Purpose |
|------|------|---------|
| `filter_grasped_object_with_sam.py` | 18K | Main script - segments & tracks grasped objects |
| `identify_grasp_timestamps.py` | 9.3K | Helper - auto-detects when grasping occurs |
| `run_object_tracking_example.sh` | 2.8K | Simple example with hardcoded paths |
| `run_complete_object_tracking_pipeline.sh` | 8.4K | Full automated pipeline (all stages) |
| `OBJECT_TRACKING_README.md` | 7.3K | Detailed documentation |
| `OBJECT_TRACKING_SUMMARY.md` | 5.8K | Overview and design explanation |

## 🚀 Quick Start (3 Commands)

```bash
# 1. Auto-detect grasp timestamps
python scripts/identify_grasp_timestamps.py \
  --npz data/human_high_res_filtered/task_*_hand_tracked.npz \
  --plot grasp_analysis.png

# 2. Edit script with suggested timestamps
vim scripts/run_object_tracking_example.sh
# Update: GRASP_TIMESTAMPS="10 15 20"

# 3. Run tracking
bash scripts/run_object_tracking_example.sh
```

## 🔄 Complete Automated Pipeline

Run everything at once (with user prompts):

```bash
bash scripts/run_complete_object_tracking_pipeline.sh
```

This runs all 5 stages:
1. ✅ Create sparse depth maps
2. ✅ Detect hands with HaMeR + SAM
3. ✅ Auto-identify grasp timestamps
4. ✅ Track object with SAM2.1
5. ✅ Optional: Run MVTracker for 3D

## 📊 What You Get

### Input
- `*_hand_tracked.npz` - From hand tracking pipeline
- Grasp timestamps - When gripping occurs

### Output
- `*_with_object_masks.npz` - New NPZ with object masks
- `*_object_tracked.mp4` - Videos (cyan overlay = object)
- `grasp_analysis.png` - Visualization of finger distances

### NPZ Structure
```python
{
    # Original fields:
    "rgbs": [C, T, 3, H, W],
    "depths": [C, T, 1, H, W],
    "sam_hand_masks": [C, T, H, W],   # Magenta in videos
    
    # NEW fields:
    "sam_object_masks": [C, T, H, W], # Cyan in videos
    "sam_object_scores": [C, T, H, W],
    "grasp_timestamps": [K]
}
```

## 🎨 Color Coding

- **Magenta** = Hands (from `sam_hand_masks`)
- **Cyan** = Grasped object (from `sam_object_masks`)
- **Red dots** = SAM prompt points (on grasp frame)
- **Green** = Hand keypoints/bones (if available)

## 🛠️ Core Functions

### identify_grasp_timestamps.py
```python
# Analyzes finger distances
distances = compute_finger_distances(query_points, num_frames)

# Finds grasp periods
periods = identify_grasp_periods(distances, threshold=0.05)

# Suggests specific frames
timestamps = suggest_timestamps(periods)
```

### filter_grasped_object_with_sam.py
```python
# Gets finger positions in 2D
prompt_points = get_gripper_prompt_points(query_points, frame_idx)

# Segments with SAM2.1 video predictor
masks, scores = segment_object_with_sam2(video_frames, grasp_frame, prompts)

# Saves results
np.savez_compressed(output_path, sam_object_masks=masks, ...)
```

## 🔧 Common Parameters

### Grasp Detection
- `--threshold 0.05` - Max finger distance (meters) for grasping
- `--min-duration 3` - Min consecutive frames
- `--num-samples 3` - Samples per grasp period

### Object Tracking
- `--grasp-timestamps` - Space-separated frame indices
- `--device cuda` - Use GPU (recommended)
- `--output-dir` - Where to save results

## 📈 Example Workflow

```bash
# Stage 1: Get hand tracking
bash scripts/run_human_example.sh
# → Creates: task_*_hand_tracked.npz

# Stage 2: Find grasp times
python scripts/identify_grasp_timestamps.py \
  --npz data/human_high_res_filtered/task_*_hand_tracked.npz \
  --plot grasp_analysis.png
# → Outputs: "10 15 20 25"

# Stage 3: Track object
python scripts/filter_grasped_object_with_sam.py \
  --npz data/human_high_res_filtered/task_*_hand_tracked.npz \
  --grasp-timestamps 10 15 20 25 \
  --output-dir data/human_high_res_filtered/object_tracking
# → Creates: task_*_with_object_masks.npz + videos

# Stage 4: 3D tracking (optional)
python demo.py \
  --sample-path data/human_high_res_filtered/object_tracking/task_*_with_object_masks.npz \
  --tracker cotracker3_offline
```

## 🎯 Key Features

✅ **Simple & Clear** - Descriptive names, extensive comments
✅ **Modular** - Each function does one thing well
✅ **Robust** - Handles edge cases, validates inputs
✅ **Visual** - Progress bars, plots, overlay videos
✅ **Documented** - README + inline docs + examples
✅ **Automated** - Can run entire pipeline with one command
✅ **Flexible** - Static/dynamic cameras, multiple views

## 📚 Documentation

- **OBJECT_TRACKING_README.md** - Complete guide
  - How to determine grasp timestamps (3 methods)
  - Parameter explanations
  - Troubleshooting tips
  - Integration with MVTracker

- **OBJECT_TRACKING_SUMMARY.md** - Design overview
  - What was created and why
  - Code structure explanation
  - Comparison with hand tracking
  - Technical details

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| No grasp timestamps found | Lower `--threshold` or reduce `--min-duration` |
| Poor segmentation | Choose frames with clear grasping |
| No prompt points | Check that hands were detected at grasp frames |
| CUDA OOM | Use `--device cpu` or reduce resolution |

## 💡 Tips

1. **Check the plot** - `grasp_analysis.png` shows finger distances over time
2. **Use multiple frames** - More grasp timestamps = better segmentation
3. **Verify in videos** - Watch hand tracking videos to confirm grasp times
4. **Start automated** - Run `run_complete_object_tracking_pipeline.sh` first

## 📞 Need Help?

1. Check `OBJECT_TRACKING_README.md` for detailed docs
2. Look at examples in `run_object_tracking_example.sh`
3. Review `OBJECT_TRACKING_SUMMARY.md` for design details
4. Check script `--help` for parameter info
