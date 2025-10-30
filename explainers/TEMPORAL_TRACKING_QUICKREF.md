# 🚀 Quick Reference: Temporal Tracking Mode

## When to Use

**Use `--use-temporal-global-ids` when:**
- ✅ Masks appear at different times in different cameras
- ✅ Masks disappear and reappear
- ✅ You want the most robust matching
- ✅ You have sufficient frames (10+)

**Use `--use-global-ids` when:**
- ⚠️ All masks exist at one specific frame
- ⚠️ You want faster processing
- ⚠️ You have very few frames

**Use neither (per-camera mode) when:**
- ⚠️ You don't need cross-camera matching
- ⚠️ Just visualizing each camera separately

## Quick Commands

### 🎯 Recommended (Temporal Tracking)
```bash
python lift_and_visualize_masks.py \
    --npz YOUR_FILE.npz \
    --mask-key YOUR_KEY \
    --use-temporal-global-ids \
    --temporal-distance-threshold 0.10 \
    --distance-threshold 0.15 \
    --spawn
```

### ⚡ Fast (Single Frame)
```bash
python lift_and_visualize_masks.py \
    --npz YOUR_FILE.npz \
    --mask-key YOUR_KEY \
    --use-global-ids \
    --distance-threshold 0.15 \
    --frame-for-matching 0
```

### 🎨 Simple (Per Camera)
```bash
python lift_and_visualize_masks.py \
    --npz YOUR_FILE.npz \
    --mask-key YOUR_KEY
```

## Parameter Quick Guide

| Parameter | Default | Description | When to Adjust |
|-----------|---------|-------------|----------------|
| `--temporal-distance-threshold` | 0.10 | Frame-to-frame (10cm) | Objects move fast/slow |
| `--distance-threshold` | 0.15 | Cross-camera (15cm) | Cameras far apart |

### Tuning Tips

**Objects not tracking well?**
```bash
--temporal-distance-threshold 0.20  # Allow more movement
```

**Too many tracks (breaking)?**
```bash
--temporal-distance-threshold 0.15  # Increase threshold
```

**Cameras not matching?**
```bash
--distance-threshold 0.30  # Increase threshold
```

**Wrong objects matching?**
```bash
--distance-threshold 0.10  # Decrease threshold
```

## Example Scenarios

### Hand Tracking
```bash
# Hands move ~10cm between frames
python lift_and_visualize_masks.py \
    --npz hand_masks.npz \
    --mask-key hand_masks \
    --use-temporal-global-ids \
    --temporal-distance-threshold 0.12 \
    --distance-threshold 0.15
```

### Object Tracking (Slow Motion)
```bash
# Objects on table, minimal movement
python lift_and_visualize_masks.py \
    --npz object_masks.npz \
    --mask-key object_masks \
    --use-temporal-global-ids \
    --temporal-distance-threshold 0.08 \
    --distance-threshold 0.20
```

### Object Tracking (Fast Motion)
```bash
# Objects thrown/moved quickly
python lift_and_visualize_masks.py \
    --npz fast_object_masks.npz \
    --mask-key object_masks \
    --use-temporal-global-ids \
    --temporal-distance-threshold 0.25 \
    --distance-threshold 0.30
```

## Test First!

Always test with your data:
```bash
# 1. Test the implementation
python test_temporal_global_ids.py

# 2. Try with a few frames first
python lift_and_visualize_masks.py \
    --npz YOUR_FILE.npz \
    --mask-key YOUR_KEY \
    --use-temporal-global-ids \
    --max-frames 20 \
    --spawn

# 3. Check results in Rerun, adjust thresholds

# 4. Run on full dataset
python lift_and_visualize_masks.py \
    --npz YOUR_FILE.npz \
    --mask-key YOUR_KEY \
    --use-temporal-global-ids \
    --spawn
```

## Expected Output

```
[INFO] ========== Temporal Tracking for 'hand' ==========
[INFO] Tracking camera 0...
[INFO]   Camera 0: Found 2 temporal tracks
[INFO] Tracking camera 1...
[INFO]   Camera 1: Found 2 temporal tracks
[INFO] Tracking camera 2...
[INFO]   Camera 2: Found 1 temporal tracks

[INFO] Matching tracks across cameras...
[INFO]   Camera 1, track 0 -> global ID 0 (dist: 0.089m)
[INFO]   Camera 1, track 1 -> global ID 1 (dist: 0.126m)
[INFO]   Camera 2, track 0 -> global ID 0 (dist: 0.112m)
[INFO] Assigned 2 global IDs for 'hand' using temporal tracking
```

This means:
- ✅ Camera 0 has 2 hands (global IDs 0, 1)
- ✅ Camera 1 sees both hands (matched!)
- ✅ Camera 2 sees 1 hand (global ID 0 matched)

## Rerun Visualization

Toggle layers to see each object:
```
world/masks/
  hand/
    global_id_0/  ← Toggle ON/OFF (Red - e.g., left hand)
    global_id_1/  ← Toggle ON/OFF (Green - e.g., right hand)
```

## Common Issues

| Issue | Solution |
|-------|----------|
| "No tracks found" | Check masks have integer IDs (not binary) |
| "All new global IDs" | Increase `--distance-threshold` |
| "Too many tracks" | Increase `--temporal-distance-threshold` |
| "Tracks break often" | Increase `--temporal-distance-threshold` |
| ImportError scipy | `pip install scipy` |

## More Info

- 📖 Full docs: `TEMPORAL_TRACKING_IMPLEMENTATION.md`
- 🧪 Test script: `test_temporal_global_ids.py`
- 📚 Original docs: `GLOBAL_MASK_ID_README.md`

## TL;DR

**Just use this:**
```bash
python lift_and_visualize_masks.py \
    --npz YOUR_FILE.npz \
    --mask-key YOUR_KEY \
    --use-temporal-global-ids \
    --spawn
```

Adjust thresholds if needed. Done! 🎉
