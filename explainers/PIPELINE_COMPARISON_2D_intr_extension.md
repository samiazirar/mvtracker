# Pipeline Comparison Quick Reference

## Side-by-Side Comparison

| Feature | Metadata-Only | Full Pipeline |
|---------|---------------|---------------|
| **Script** | `run_pipeline_cluster_huggingface_metadata_only_no_depth.sh` | `run_pipeline_cluster_huggingface_compressed_lossy.sh` |
| **GPU Required** | ❌ No | ✅ Yes (CUDA) |
| **Processing** | CPU-only | Multi-GPU |
| **Workers** | 16 (default) | 3 per GPU |
| **Time/Episode** | ~30 seconds | ~2-3 minutes |
| **Output Size** | ~50 MB | ~2 GB |

## Output Files Comparison

### Both Pipelines Include:

✅ **tracks.npz**
- 3D tracks in world coordinates [T, N, 3]
- 2D projections per camera [T, N, 2]
- Normalized flow at 1mm steps [M, N, 3]
- Contact frames and centroids
- Gripper poses and state

✅ **extrinsics.npz**
- External camera poses (static)
- Wrist camera trajectory (per-frame)

✅ **quality.json**
- Episode metadata
- Calibration info
- Camera lists

✅ **recordings/{camera}/intrinsics.json**
- Focal lengths (fx, fy)
- Principal point (cx, cy)
- Distortion coefficients
- Image dimensions

### Full Pipeline ALSO Includes:

✅ **recordings/{camera}/depth.mkv**
- FFV1 lossless video
- 16-bit depth in millimeters
- ~70% compression vs raw frames

✅ **recordings/{camera}/depth_meta.json**
- Decoding instructions
- Scale factors

## HuggingFace Uploads

### Metadata-Only Pipeline
- **Repo:** `sazirarrwth99/droid_metadata_only`
- **Contents:** tracks.npz, extrinsics.npz, quality.json, intrinsics.json
- **Size:** ~50 MB per episode

### Full Pipeline
- **Repo 1:** `sazirarrwth99/lossy_comr_traject`
  - **Contents:** Everything (depth videos + metadata)
  - **Size:** ~2 GB per episode
  
- **Repo 2:** `sazirarrwth99/droid_metadata_only`
  - **Contents:** tracks.npz, extrinsics.npz, quality.json, intrinsics.json
  - **Size:** ~50 MB per episode
  - (Same metadata uploaded to both repos)

## Use Cases

### Choose Metadata-Only When:
- ✅ Training vision-based policies (RGB only)
- ✅ Trajectory prediction / imitation learning
- ✅ Flow estimation / contact prediction
- ✅ No GPU available
- ✅ Fast batch processing needed
- ✅ Storage constraints

### Choose Full Pipeline When:
- ✅ 3D reconstruction needed
- ✅ Depth-aware models
- ✅ Complete sensor suite required
- ✅ Building comprehensive dataset
- ✅ GPU resources available

## What Both Pipelines Generate

### 3D Track Data (Identical)
```python
tracks_3d: [T, N, 3]               # Track points in world frame
contact_centroids: [T, 3]          # Center of contact per frame
contact_frames: [T, 4, 4]          # Contact pose transforms
left_contact_frames: [T, 4, 4]     # Left finger frames
right_contact_frames: [T, 4, 4]    # Right finger frames
```

### Normalized Flow (Identical)
```python
normalized_tracks_3d: [M, N, 3]         # Resampled at 1mm steps
normalized_centroids: [M, 3]            # Centroids at 1mm steps
normalized_frames: [M, 4, 4]            # Frames at 1mm steps
normalized_left_frames: [M, 4, 4]       # Left finger at 1mm steps
normalized_right_frames: [M, 4, 4]      # Right finger at 1mm steps
cumulative_distance_mm: [T]             # Distance traveled
frame_to_normalized_idx: [T]            # Frame to normalized mapping
normalized_step_size_mm: 1.0            # Step size
num_normalized_steps: M                 # Total steps
```

### 2D Projections (Identical)
```python
tracks_2d_{camera_serial}: [T, N, 2]    # Per-camera projections
intrinsics_{camera_serial}: [3, 3]      # Camera matrix
image_size_{camera_serial}: [2]         # Width, height
cameras_with_2d_tracks: List[str]       # Available cameras
```

### Robot State (Identical)
```python
gripper_poses: [T, 4, 4]           # End-effector poses
gripper_positions: [T]             # Aperture values
cartesian_positions: [T, 6]        # Robot state
```

## Performance Numbers

### Metadata-Only Pipeline
- **Throughput:** ~120 episodes/hour (16 workers)
- **CPU Usage:** High (100% across all cores)
- **GPU Usage:** None
- **Memory:** ~8GB per worker
- **Network:** Low (small files)

### Full Pipeline
- **Throughput:** ~20 episodes/hour (3 workers/GPU, 4 GPUs)
- **CPU Usage:** Medium (50-70%)
- **GPU Usage:** High (80-90% during depth extraction)
- **Memory:** ~16GB per worker
- **Network:** High (large depth videos)

## Storage Requirements

### Per Episode
- Metadata-only: ~50 MB
- Full dataset: ~2 GB (depth videos dominate)

### For 10,000 Episodes
- Metadata-only: ~500 GB
- Full dataset: ~20 TB

### Batch Upload Settings
Both pipelines use batch uploads every 10 minutes (configurable):
```bash
BATCH_UPLOAD_INTERVAL=600  # seconds (default: 10 minutes)
```

## Command Examples

### Metadata-Only
```bash
# Process 100 episodes, 16 workers
./run_pipeline_cluster_huggingface_metadata_only_no_depth.sh 100

# Process 100 episodes, 32 workers (more parallelism)
./run_pipeline_cluster_huggingface_metadata_only_no_depth.sh 100 32

# Process all episodes
./run_pipeline_cluster_huggingface_metadata_only_no_depth.sh -1

# Skip HuggingFace existence check (faster)
SKIP_HF_CHECK=1 ./run_pipeline_cluster_huggingface_metadata_only_no_depth.sh 100
```

### Full Pipeline
```bash
# Auto-detect GPUs, 3 workers/GPU
./run_pipeline_cluster_huggingface_compressed_lossy.sh 100

# 6 workers/GPU, auto-detect GPUs
./run_pipeline_cluster_huggingface_compressed_lossy.sh 100 6

# 6 workers/GPU, use 4 GPUs
./run_pipeline_cluster_huggingface_compressed_lossy.sh 100 6 4

# Process all episodes
./run_pipeline_cluster_huggingface_compressed_lossy.sh -1

# Use specific GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_pipeline_cluster_huggingface_compressed_lossy.sh 100
```

## Quality Assurance

Both pipelines:
- ✅ Validate episode structure before processing
- ✅ Check HuggingFace for existing episodes (skip if present)
- ✅ Log errors per episode with detailed stack traces
- ✅ Generate timing statistics (CSV format)
- ✅ Atomic batch uploads (resume on failure)
- ✅ Preserve failed episode lists for retry

## Error Handling

Common failure modes:
1. **SVO file corruption** → Skip episode, log error
2. **Missing calibration** → Skip episode, log error
3. **Network timeout** → Retry batch upload next interval
4. **Out of memory** → Reduce workers per GPU
5. **Disk full** → Pipeline pauses, cleans up staging

Both pipelines clean up scratch space on:
- ✅ Success (immediate cleanup)
- ✅ Failure (cleanup after logging)
- ✅ Interrupt (signal handler cleanup)
