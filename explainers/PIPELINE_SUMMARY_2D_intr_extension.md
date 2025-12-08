# DROID Training Data Pipeline Summary

## Overview

Two pipeline scripts for processing DROID episodes for training data:

1. **Metadata-Only Pipeline** - Lightweight, CPU-only, no GPU needed
2. **Full Pipeline with Compressed Depth** - GPU-required, saves compressed depth videos

Both pipelines now generate **complete normalized training data** including:
- ✅ 3D tracks (world coordinates)
- ✅ 2D track projections per camera
- ✅ Normalized flow (resampled at 1mm distance steps)
- ✅ Camera intrinsics
- ✅ Camera extrinsics
- ✅ Quality metadata

---

## Pipeline 1: Metadata-Only (No Depth)

**Script:** `run_pipeline_cluster_huggingface_metadata_only_no_depth.sh`

### What It Does
- ✅ Extracts camera intrinsics from ZED SVO files (no GPU needed)
- ✅ Generates 3D gripper contact tracks
- ✅ Projects tracks to 2D for each camera
- ✅ Computes normalized flow (1mm resampling)
- ✅ Saves camera extrinsics
- ✅ Uploads to HuggingFace: `sazirarrwth99/droid_metadata_only`

### Output Files
```
{lab}/success/{date}/{timestamp}/
├── tracks.npz                    # 3D + 2D tracks, normalized flow
├── extrinsics.npz                # Camera poses
├── quality.json                  # Metadata
└── recordings/
    └── {camera_serial}/
        └── intrinsics.json       # Camera parameters
```

### Requirements
- **CPU only** (no GPU needed)
- ZED SDK (for reading SVO files)
- Fast local storage: `/data/droid_scratch`

### Usage
```bash
# Process 10 episodes with 16 workers (CPU-only)
./run_pipeline_cluster_huggingface_metadata_only_no_depth.sh

# Process 100 episodes with 32 workers
./run_pipeline_cluster_huggingface_metadata_only_no_depth.sh 100 32

# Process all episodes
./run_pipeline_cluster_huggingface_metadata_only_no_depth.sh -1

# Skip HuggingFace checking (faster)
SKIP_HF_CHECK=1 ./run_pipeline_cluster_huggingface_metadata_only_no_depth.sh 100
```

### Key Features
- **No GPU required** - runs on CPU-only nodes
- **High parallelism** - default 16 workers (vs 3 workers/GPU for depth extraction)
- **Lightweight** - only extracts intrinsics, not full depth maps
- **Complete metadata** - includes all tracks, projections, and normalization

---

## Pipeline 2: Full Dataset with Compressed Depth

**Script:** `run_pipeline_cluster_huggingface_compressed_lossy.sh`

### What It Does
- ✅ Extracts depth maps from ZED (GPU required)
- ✅ Saves depth as FFV1 lossless video (compressed, ~70% size reduction)
- ✅ Extracts camera intrinsics
- ✅ Generates 3D gripper contact tracks
- ✅ Projects tracks to 2D for each camera
- ✅ Computes normalized flow (1mm resampling)
- ✅ Saves camera extrinsics
- ✅ Uploads to TWO repos:
  - Full data: `sazirarrwth99/lossy_comr_traject` (depth videos + metadata)
  - Metadata only: `sazirarrwth99/droid_metadata_only` (tracks, extrinsics, quality)

### Output Files
```
{lab}/success/{date}/{timestamp}/
├── tracks.npz                    # 3D + 2D tracks, normalized flow
├── extrinsics.npz                # Camera poses
├── quality.json                  # Metadata
└── recordings/
    └── {camera_serial}/
        ├── depth.mkv             # FFV1 lossless depth video (16-bit, mm)
        ├── depth_meta.json       # Depth decoding metadata
        └── intrinsics.json       # Camera parameters
```

### Requirements
- **GPU with CUDA** (for ZED depth extraction)
- ZED SDK with GPU support
- Fast local storage: `/data/droid_scratch`
- FFV1 codec for video compression

### Usage
```bash
# Auto-detect GPUs, 3 workers/GPU, process 10 episodes
./run_pipeline_cluster_huggingface_compressed_lossy.sh

# Process 100 episodes, 6 workers/GPU, auto-detect GPUs
./run_pipeline_cluster_huggingface_compressed_lossy.sh 100 6

# Process 100 episodes, 6 workers/GPU, use 4 GPUs explicitly
./run_pipeline_cluster_huggingface_compressed_lossy.sh 100 6 4

# Process all episodes
./run_pipeline_cluster_huggingface_compressed_lossy.sh -1

# Skip HuggingFace checking (faster)
SKIP_HF_CHECK=1 ./run_pipeline_cluster_huggingface_compressed_lossy.sh 100
```

### Key Features
- **Compressed depth** - FFV1 lossless video saves ~70% space vs raw frames
- **Dual upload** - uploads to both full and metadata-only repos
- **Multi-GPU** - automatically distributes work across available GPUs
- **Complete dataset** - depth maps + all metadata for full reconstruction

---

## Data Format Details

### tracks.npz Contents

Both pipelines generate identical `tracks.npz` files with:

**3D Tracks (Original)**
- `tracks_3d`: [T, N, 3] - 3D track points in world frame (meters)
- `contact_centroids`: [T, 3] - Center of mass of contact points per frame
- `contact_frames`: [T, 4, 4] - Contact frame transforms (centroid + EE orientation)
- `left_contact_frames`: [T, 4, 4] - Left finger contact frames
- `right_contact_frames`: [T, 4, 4] - Right finger contact frames

**Normalized Flow (1mm Resampling)**
- `normalized_tracks_3d`: [M, N, 3] - Tracks resampled at 1mm distance steps
- `normalized_centroids`: [M, 3] - Centroids at 1mm steps
- `normalized_frames`: [M, 4, 4] - Contact frames at 1mm steps
- `normalized_left_frames`: [M, 4, 4] - Left finger frames at 1mm steps
- `normalized_right_frames`: [M, 4, 4] - Right finger frames at 1mm steps
- `cumulative_distance_mm`: [T] - Distance traveled at each frame (mm)
- `frame_to_normalized_idx`: [T] - Mapping from frame to normalized index
- `normalized_step_size_mm`: float - Step size (default: 1.0)
- `num_normalized_steps`: int (M) - Number of normalized steps

**2D Projections (Per Camera)**
- `tracks_2d_{camera_serial}`: [T, N, 2] - 2D projections (NaN for invalid)
- `intrinsics_{camera_serial}`: [3, 3] - Camera intrinsic matrix
- `image_size_{camera_serial}`: [2] - Image dimensions [width, height]
- `cameras_with_2d_tracks`: list - Camera serials with 2D tracks

**Robot State**
- `gripper_poses`: [T, 4, 4] - End-effector poses
- `gripper_positions`: [T] - Gripper aperture values
- `cartesian_positions`: [T, 6] - Robot cartesian state

**Metadata**
- `num_frames`: int - Number of frames
- `num_points_per_finger`: int - Track points per finger
- `fps`: float - Frame rate

### Compression Details

**Depth Video (FFV1)**
- Codec: FFV1 (lossless)
- Format: 16-bit grayscale (z16)
- Unit: millimeters
- Decoding: `depth_meters = pixel_value / 1000.0`
- Size: ~70% reduction vs PNG frames

**Intrinsics JSON**
- fx, fy: focal lengths
- cx, cy: principal point
- width, height: image dimensions
- k1, k2, k3: radial distortion
- p1, p2: tangential distortion

---

## Storage Configuration

Both pipelines use fast local storage for processing:

```bash
FAST_LOCAL_DIR="/data/droid_scratch"         # Node-local NVMe
STAGING_DIR="/data/droid_staging"            # Pre-upload staging
BATCH_UPLOAD_DIR="/data/droid_batch_upload"  # Full dataset queue
BATCH_METADATA_DIR="/data/droid_batch_metadata"  # Metadata-only queue
```

**Batch Upload:** Episodes are staged locally and uploaded every 10 minutes (configurable) to avoid HuggingFace rate limits.

---

## Choosing a Pipeline

### Use Metadata-Only When:
- ✅ Training vision-based models (don't need depth)
- ✅ Working on trajectory planning / imitation learning
- ✅ No GPU available
- ✅ Want fast processing (16+ CPU workers)
- ✅ Only need tracks, intrinsics, extrinsics

### Use Full Pipeline When:
- ✅ Need depth maps for 3D reconstruction
- ✅ Training depth-aware models
- ✅ Building complete dataset with sensor data
- ✅ GPU resources available

---

## HuggingFace Repositories

### sazirarrwth99/droid_metadata_only
- Tracks (3D + 2D + normalized)
- Camera intrinsics
- Camera extrinsics
- Quality metadata
- **Size:** ~50MB per episode
- **Source:** Both pipelines

### sazirarrwth99/lossy_comr_traject
- Everything from metadata-only repo
- PLUS compressed depth videos (FFV1)
- **Size:** ~2GB per episode
- **Source:** Full pipeline only

---

## Helper Scripts

### extract_intrinsics_only.py
Standalone script to extract ONLY camera intrinsics from ZED SVO files.

**Features:**
- No GPU required
- No depth extraction
- Fast (< 5 seconds per camera)
- Called by metadata-only pipeline

**Usage:**
```bash
python extract_intrinsics_only.py --episode_id "AUTOLab+hash+2023-08-18-12h-01m-10s"
```

### generate_tracks_and_metadata.py
Generates tracks, projections, and metadata from trajectory + intrinsics.

**Features:**
- Computes 3D contact tracks
- Projects to 2D using intrinsics
- Normalizes flow at 1mm steps
- Saves extrinsics and quality

**Usage:**
```bash
python generate_tracks_and_metadata.py --episode_id "AUTOLab+hash+2023-08-18-12h-01m-10s"
```

---

## Performance Comparison

| Metric | Metadata-Only | Full Pipeline |
|--------|---------------|---------------|
| GPU Required | ❌ No | ✅ Yes |
| Workers | 16 (CPU) | 3 per GPU |
| Time/Episode | ~30s | ~2-3 min |
| Output Size | ~50MB | ~2GB |
| Includes Depth | ❌ No | ✅ Yes (FFV1) |
| 2D Tracks | ✅ Yes | ✅ Yes |
| Normalized Flow | ✅ Yes | ✅ Yes |

---

## Notes

1. **Both pipelines generate identical metadata** - the only difference is depth video inclusion
2. **FFV1 is lossless** - despite "lossy" in filename, compression is lossless
3. **Normalized flow** - all tracks resampled at constant 1mm distance for uniform flow learning
4. **2D projections** - computed using camera intrinsics and extrinsics for all cameras
5. **Batch uploads** - automatic rate-limited uploads to HuggingFace every 10 minutes
