# Batch Processing for Large Masks

When masks generate too many query points (e.g., 10,000+), the tracker may exhaust GPU memory or become too slow. Batch processing splits query points into smaller chunks and processes them sequentially.

## Quick Start

```bash
# Enable batch processing with max 5000 points per batch
python process_masks_independently.py \
  --npz path/to/masks.npz \
  --mask-key sam2_masks \
  --tracker mvtracker \
  --output-dir ./results \
  --max-query-points-per-batch 5000
```

## When to Use Batch Processing

### ✅ Use batching when:
- Single mask generates >10,000 query points
- Getting CUDA out of memory errors
- Tracker becomes extremely slow (>5 min per instance)
- Tracking large objects across many frames
- Using high-resolution masks

### ❌ Don't use batching when:
- Masks are small (<5,000 query points)
- Tracking works fine without batching
- You need real-time performance (batching adds overhead)

## How It Works

### Without Batching (Default)
```
Mask → Query Points (15,000) → Tracker → RRD
                                  ↓
                         (may fail: OOM)
```

### With Batching
```
Mask → Query Points (15,000)
         ↓
       Split into batches:
         - Batch 0: 5,000 points → Tracker → RRD_batch_0
         - Batch 1: 5,000 points → Tracker → RRD_batch_1
         - Batch 2: 5,000 points → Tracker → RRD_batch_2
         ↓
       View all batches together in Rerun
```

## Batching Strategies

### Temporal (Recommended)
Groups query points by frames to preserve temporal coherence.

**Pros:**
- Maintains tracking continuity within batches
- Better for video sequences
- Points in same frame stay together

**Cons:**
- Batches may have uneven sizes if frames vary

```bash
--batch-strategy temporal
```

### Random
Randomly distributes query points across batches.

**Pros:**
- Guaranteed even batch sizes
- Simpler implementation

**Cons:**
- Breaks temporal coherence
- May reduce tracking quality

```bash
--batch-strategy random
```

## Command-Line Options

```bash
--max-query-points-per-batch N    # Max points per batch (e.g., 5000)
--batch-strategy STRATEGY         # "temporal" or "random" (default: temporal)
```

## Example Usage

### Basic Batch Processing
```bash
python process_masks_independently.py \
  --npz masks.npz \
  --mask-key sam2_masks \
  --tracker mvtracker \
  --output-dir ./results \
  --max-query-points-per-batch 5000
```

### With Custom Strategy
```bash
python process_masks_independently.py \
  --npz masks.npz \
  --mask-key sam2_masks \
  --tracker spatialtrackerv2 \
  --output-dir ./results \
  --max-query-points-per-batch 3000 \
  --batch-strategy temporal
```

### Shell Script (Recommended)
```bash
# Edit scripts/run_per_mask_tracking.sh
MAX_QUERY_POINTS_PER_BATCH="5000"  # Enable batching

# Run
bash scripts/run_per_mask_tracking.sh
```

## Output Structure

```
results/
├── instances/
│   ├── scene_instance_0.npz
│   ├── scene_instance_0_query.npz
│   ├── scene_instance_0_batches/          # Batch NPZ files
│   │   ├── scene_instance_0_query_batch_0.npz
│   │   ├── scene_instance_0_query_batch_1.npz
│   │   └── scene_instance_0_query_batch_2.npz
│   └── scene_instance_1.npz
├── scene_mvtracker_instance_0_batch_0.rrd  # Individual batch RRDs
├── scene_mvtracker_instance_0_batch_1.rrd
├── scene_mvtracker_instance_0_batch_2.rrd
├── scene_mvtracker_instance_1.rrd          # No batching needed
└── view_scene_mvtracker_combined.sh        # Views all together
```

## Viewing Batch Results

### All Batches Together
```bash
# Use generated viewing script
bash results/view_scene_mvtracker_combined.sh

# Or manually
rerun results/*.rrd
```

### Individual Batches
```bash
# View specific batch
rerun results/scene_mvtracker_instance_0_batch_0.rrd

# View all batches of one instance
rerun results/scene_mvtracker_instance_0_batch_*.rrd
```

## Performance Considerations

### Memory Usage
- **Without batching:** All query points loaded at once
- **With batching:** Only one batch in memory at a time
- **Savings:** Can reduce peak memory by 3-5x

### Processing Time
- **Overhead:** ~5-10% per batch for loading/unloading
- **Total time:** Roughly same as without batching (sequential processing)
- **Trade-off:** Slightly longer total time, but won't crash

### Recommended Batch Sizes

| Tracker | GPU | Recommended Batch Size |
|---------|-----|----------------------|
| MVTracker | 24GB | 5,000-8,000 |
| MVTracker | 16GB | 3,000-5,000 |
| MVTracker | 8GB  | 1,500-3,000 |
| SpatialTracker | 24GB | 8,000-10,000 |
| SpatialTracker | 16GB | 5,000-8,000 |
| CoTracker | 24GB | 5,000-7,000 |

*Adjust based on your specific hardware and sequence length*

## Troubleshooting

### "Still getting OOM errors"
- Reduce batch size: `--max-query-points-per-batch 2000`
- Reduce temporal stride: `--temporal-stride 2`
- Reduce spatial resolution: `--spatial-downsample 2`

### "Batches seem uneven"
- This is normal with temporal strategy (preserves frame groups)
- Use `--batch-strategy random` for perfectly even batches
- Check batch info in output logs

### "Tracking quality decreased"
- Temporal strategy should preserve quality
- If using random strategy, switch to temporal
- Consider using fewer, larger batches if possible

### "Too many RRD files"
- This is expected - one RRD per batch
- Use viewing script to see all together
- Rerun can handle 100+ files easily

## Advanced: Manual Batch Processing

If you need more control, you can batch manually:

```python
from utils.batch_processing_utils import (
    split_query_points_into_batches,
    create_batch_npzs,
)
import numpy as np
from pathlib import Path

# Load query points
data = np.load("scene_query.npz", allow_pickle=True)
query_points = data["query_points"]

# Split into batches
batches = split_query_points_into_batches(
    query_points=query_points,
    max_points_per_batch=5000,
    strategy="temporal",
)

# Create batch NPZ files
batch_npzs = create_batch_npzs(
    input_npz=Path("scene_query.npz"),
    query_points_batches=batches,
    output_dir=Path("./batches/"),
    base_name="scene",
)

# Process each batch with demo.py
for i, batch_npz in enumerate(batch_npzs):
    # Run tracking on batch_npz
    ...
```

## Integration with Existing Scripts

### Add to run_human_example.sh

```bash
# After SAM2 tracking, run per-mask with batching
python process_masks_independently.py \
  --npz "$SAMPLE_PATH_HAND_TRACKED" \
  --mask-key sam2_masks \
  --tracker mvtracker \
  --output-dir "./tracking_per_mask/${TASK_FOLDER}" \
  --max-query-points-per-batch 5000 \
  --batch-strategy temporal
```

### Dynamic Batching Based on Size

```bash
# Check query point count and batch if needed
QUERY_COUNT=$(python -c "
import numpy as np
data = np.load('$SAMPLE_PATH_HAND_TRACKED', allow_pickle=True)
if 'query_points' in data:
    print(len(data['query_points']))
else:
    print(0)
")

BATCH_ARG=""
if [ "$QUERY_COUNT" -gt 8000 ]; then
    BATCH_ARG="--max-query-points-per-batch 5000"
    echo "Large mask detected ($QUERY_COUNT points), enabling batching"
fi

python process_masks_independently.py \
  --npz "$SAMPLE_PATH_HAND_TRACKED" \
  --tracker mvtracker \
  --output-dir "./results" \
  $BATCH_ARG
```

## Summary

**Batch processing splits large masks into manageable chunks:**

✅ **Enable when:** Getting OOM errors or slow tracking  
✅ **Recommended size:** 3,000-8,000 points per batch  
✅ **Best strategy:** Temporal (preserves frame coherence)  
✅ **Viewing:** All batches appear together in Rerun  

**Quick command:**
```bash
python process_masks_independently.py \
  --npz masks.npz \
  --max-query-points-per-batch 5000 \
  --output-dir ./results
```
