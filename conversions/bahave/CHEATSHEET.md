# BEHAVE + demo.py Quick Reference

## TL;DR

```bash
# One command to convert and track
python conversions/behave_to_demo.py --scene Date01_Sub01_backpack_back
python demo.py --sample-path conversions/behave_converted/Date01_Sub01_backpack_back_demo.npz \
    --tracker mvtracker --depth_estimator gt --rerun save
```

## Common Commands

### Convert Scene for demo.py

```bash
# Basic (100 frames, downscale 2x)
python conversions/behave_to_demo.py --scene SCENE_NAME

# More frames
python conversions/behave_to_demo.py --scene SCENE_NAME --max_frames 200

# Less downscaling (better quality, larger file)
python conversions/behave_to_demo.py --scene SCENE_NAME --downscale_factor 1

# More query points
python conversions/behave_to_demo.py --scene SCENE_NAME --num_query_points 512
```

### Track with demo.py

```bash
# MVTracker (multi-view, best quality)
python demo.py --sample-path FILE_demo.npz --tracker mvtracker --depth_estimator gt --rerun save

# CoTracker (monocular baseline)
python demo.py --sample-path FILE_demo.npz --tracker cotracker3_offline --depth_estimator gt --rerun save

# SpatialTracker V2
python demo.py --sample-path FILE_demo.npz --tracker spatialtrackerv2 --depth_estimator gt --rerun save

# Save results to file
python demo.py --sample-path FILE_demo.npz --tracker mvtracker --depth_estimator gt \
    --save-npz results/tracks.npz --rerun save
```

### Memory-Saving Options

```bash
# Use random query points (don't need mask-based points)
python demo.py --sample-path FILE_demo.npz --tracker mvtracker --depth_estimator gt \
    --random_query_points --rerun save

# Batch processing for large scenes
python demo.py --sample-path FILE_demo.npz --tracker mvtracker --depth_estimator gt \
    --batch_processing --batch_size_views 2 --batch_size_frames 30 --rerun save

# Downsample further
python demo.py --sample-path FILE_demo.npz --tracker mvtracker --depth_estimator gt \
    --spatial_downsample 2 --temporal_stride 2 --rerun save
```

## File Structure

```
conversions/
├── behave_to_demo.py          # 👈 USE THIS (one-step converter)
├── behave_to_npz.py            # Low-level converter
├── adapt_behave_for_demo.py    # Format adapter
├── inspect_behave_npz.py       # Visualization tool
├── batch_convert_behave.py     # Batch processor
├── example_usage.py            # Usage examples
├── README.md                   # Full documentation
├── USE_WITH_DEMO.md           # Integration guide
└── QUICKSTART.md              # Quick start guide
```

## Output Sizes (Approximate)

| Frames | Downscale | File Size |
|--------|-----------|-----------|
| 50     | 2x        | ~120 MB   |
| 100    | 2x        | ~230 MB   |
| 200    | 2x        | ~460 MB   |
| 100    | 1x        | ~800 MB   |

## Available Scenes (examples)

```bash
# List scenes matching pattern
ls /data/behave-dataset/behave_all/ | grep "Date01_Sub01"
```

Examples:
- `Date01_Sub01_backpack_back`
- `Date01_Sub01_basketball`
- `Date01_Sub01_chairblack_sit`
- `Date01_Sub01_keyboard_typing`
- `Date01_Sub01_suitcase`

## Troubleshooting

| Problem | Solution |
|---------|----------|
| CUDA OOM | Use `--batch_processing --batch_size_views 1` |
| File too large | Increase `--downscale_factor` or reduce `--max_frames` |
| Poor tracking | Use `--depth_estimator gt` and `--tracker mvtracker` |
| No query points | Use `--random_query_points` in demo.py |

## Cheat Sheet

```bash
# 1. List available scenes
ls /data/behave-dataset/behave_all/

# 2. Convert scene (one command)
python conversions/behave_to_demo.py --scene SCENE_NAME

# 3. Track with MVTracker
python demo.py --sample-path conversions/behave_converted/SCENE_NAME_demo.npz \
    --tracker mvtracker --depth_estimator gt --rerun save --rrd SCENE_NAME.rrd

# 4. View results
rerun SCENE_NAME.rrd
```

## Full Documentation

- **Complete guide:** `conversions/README.md`
- **demo.py integration:** `conversions/USE_WITH_DEMO.md`
- **Quick start:** `conversions/QUICKSTART.md`
