# Quick Reference: Visualizing HOISTFormer Predictions

## The Problem
Your original command failed because HOISTFormer outputs predictions in a nested dictionary format, not the standard [C, T, H, W] array format expected by the visualization tool.

## The Solution
A **wrapper** in `third_party/HOISTFormer/` converts HOISTFormer predictions to standard format **without modifying HOISTFormer code**.

---

## Quick Start (3 options)

### Option 1: One-line convenience script (Easiest!)
```bash
# Just run it with hardcoded defaults (no arguments needed!)
cd third_party/HOISTFormer
./visualize_hoist_predictions.sh

# Or specify a different file
./visualize_hoist_predictions.sh hand_object_output/YOUR_FILE.npz

# Or override all parameters
./visualize_hoist_predictions.sh hand_object_output/YOUR_FILE.npz union 100
```

### Option 2: Two-step manual process
```bash
# Step 1: Convert
python third_party/HOISTFormer/convert_hoist_predictions_to_standard_format.py \
    --input third_party/HOISTFormer/hand_object_output/task_0045_user_0020_scene_0004_cfg_0006_human_processed_hand_object.npz \
    --output third_party/HOISTFormer/hand_object_output/task_0045_user_0020_scene_0004_cfg_0006_human_processed_hand_object_converted.npz

# Step 2: Visualize
python lift_and_visualize_masks.py \
    --npz third_party/HOISTFormer/hand_object_output/task_0045_user_0020_scene_0004_cfg_0006_human_processed_hand_object_converted.npz \
    --mask-key hoist_masks \
    --color 255 0 255 \
    --max-frames 50
```

### Option 3: From workspace root with short path
```bash
# From /workspace directory
python third_party/HOISTFormer/convert_hoist_predictions_to_standard_format.py \
    --input third_party/HOISTFormer/hand_object_output/task_0045_user_0020_scene_0004_cfg_0006_human_processed_hand_object.npz

python lift_and_visualize_masks.py \
    --npz third_party/HOISTFormer/hand_object_output/task_0045_user_0020_scene_0004_cfg_0006_human_processed_hand_object_converted.npz \
    --mask-key hoist_masks \
    --color 255 0 255 \
    --max-frames 50
```

---

## Merge Strategies

Control how multiple hand/object instances are combined:

```bash
# Union (default): Combine all instances
./visualize_hoist_predictions.sh YOUR_FILE.npz union

# First: Only highest-confidence instance
./visualize_hoist_predictions.sh YOUR_FILE.npz first

# Max score: Instance with highest score
./visualize_hoist_predictions.sh YOUR_FILE.npz max_score
```

---

## Files Created

| Location | File | Description |
|----------|------|-------------|
| `third_party/HOISTFormer/` | `convert_hoist_predictions_to_standard_format.py` | Conversion wrapper script |
| `third_party/HOISTFormer/` | `visualize_hoist_predictions.sh` | Convenience script (all-in-one) |
| `third_party/HOISTFormer/` | `README_CONVERSION_WRAPPER.md` | Full documentation |

---

## What the wrapper does

1. **Loads** HOISTFormer's nested dictionary format
2. **Extracts** per-camera, per-frame instance masks
3. **Merges** multiple instances using your chosen strategy
4. **Converts** to standard [C, T, H, W] boolean array
5. **Saves** with all original data intact

**No HOISTFormer code is modified!**

---

## View the result

```bash
rerun third_party/HOISTFormer/hand_object_output/YOUR_FILE_converted_masks_3d.rrd --web-viewer
```

---

## Full Documentation

See `third_party/HOISTFormer/README_CONVERSION_WRAPPER.md` for complete details on:
- Input/output formats
- Merge strategies explained
- Technical details
- Advanced options
