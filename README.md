# MVTracker: Multi-View Point Tracking for Robot Manipulation

**MVTracker** is a comprehensive toolbox for multi-view 3D point tracking, contact flow visualization, and robot manipulation analysis. It provides pipelines for processing DROID robotics dataset and generating rich visualizations.

## 🚀 Features

- **Multi-View 3D Point Tracking**: Track points across multiple camera views in 3D space
- **Contact Flow Visualization**: Visualize gripper contact points as they interact with objects
- **DROID Dataset Processing**: Complete pipeline for processing DROID robot manipulation data
- **Training Data Generation**: Create training shards for robot learning tasks
- **Rerun Visualization**: Interactive 3D visualization using Rerun SDK
- **Hand-Object Interaction**: Integration with HOISTFormer and HaMeR for human hand tracking

## 📋 Requirements

- NVIDIA GPU with CUDA 12.8+ support
- Docker with NVIDIA Container Toolkit
- 64GB+ system RAM recommended
- 100GB+ disk space for data

## 🐳 Quick Start (DevContainer)

### Option 1: VS Code DevContainer (Recommended)

1. Open this folder in VS Code
2. Install the "Dev Containers" extension
3. Press `F1` → "Dev Containers: Reopen in Container"
4. Select the "MVTracker Latest" configuration
5. Wait for the container to build and run `post-create-latest.sh`

### Option 2: Manual Docker Run

```bash
# Build the image
cd .devcontainer/latest
docker build -t mvtracker-latest -f Dockerfile ../..

# Run with same arguments as devcontainer
docker run -it --gpus all \
    --shm-size=64g \
    --ipc=host \
    -v /home/nfs/datasets/internal:/data:rw \
    -v ~/.ssh:/ssh-host:ro \
    -v ~/.codex:/root/.codex \
    -v $(pwd):/workspace \
    -w /workspace \
    -p 7860:7860 \
    mvtracker-latest bash

# Inside container, run post-create setup
bash .devcontainer/latest/post-create-latest.sh
```

### Option 3: Standalone Docker (without ZED SDK)

For visualization without raw DROID data (uses pre-processed shards):

```bash
docker run --gpus all --rm -it \
    -v $PWD:/workspace \
    -v /path/to/droid/training_shards:/data/droid/training_shards:ro \
    nvcr.io/nvidia/cuda:12.8.1-devel-ubuntu22.04 bash

# Inside container
cd /workspace
pip install -r requirements.txt
pip install rerun-sdk opencv-python
python visualize_training_shards.py --shard_path /data/droid/training_shards/shard_0000.tar
```

## 📊 Visualization

### Contact Flow Visualization

Visualize how gripper contact points move through 3D space during manipulation:

```bash
# Single shard visualization
python visualize_training_shards.py \
    --shard_path /data/droid/training_shards/shard_0000.tar \
    --num_samples 10 \
    --output_dir visualizations

# Generate visualizations for all shards
python generate_all_visualizations.py \
    --num_samples_per_shard 10 \
    --max_shards 7

# View the results
rerun visualizations/shard_0000_contact_flow.rrd
```

### Demo Visualization

```bash
python demo.py \
    --batch_processing \
    --optimize_performance \
    --temporal_stride 1 \
    --spatial_downsample 1 \
    --depth_estimator gt \
    --depth_cache_dir ./depth_cache \
    --rerun save \
    --random_query_points
```

### Point Cloud Visualization (requires ZED SDK)

```bash
# Edit config to point to your data
vim conversions/droid/config.yaml

# Generate fused point cloud
python conversions/droid/generate_pointcloud_from_droid.py

# View output
rerun point_clouds/droid_full_fusion.rrd
```

## 🗂️ Training Data Processing

### Pipeline 1: Metadata-Only (CPU-only)

```bash
cd conversions/droid/training_data
./run_pipeline_cluster_huggingface_metadata_only_no_depth.sh 100
```

**Output per episode:**
- `tracks.npz`: 3D + 2D normalized tracks
- `extrinsics.npz`: Camera extrinsics
- `quality.json`: Episode quality metrics
- `intrinsics.json`: Camera intrinsics

### Pipeline 2: Full with Depth (GPU required)

```bash
cd conversions/droid/training_data
./run_pipeline_cluster_huggingface_compressed_lossy.sh 100
```

**Output per episode:**
- `depth.mkv`: FFV1 lossless compressed depth
- All metadata files from Pipeline 1

## 📁 Training Shard Format

Each training shard (`shard_XXXX.tar`) contains processed episodes:

```
shard_0000.tar
├── s0000_000000.npz   # Trajectory data
├── s0000_000000.json  # Metadata
├── s0000_000000.mp4   # Video
├── s0000_000001.npz
...
```

**NPZ contents:**
- `normalized_frames`: Gripper poses (T, 4, 4)
- `contact_points_local`: Contact points in gripper frame (N, 3)
- `normalized_centroids`: Centroid positions (T, 3)
- `gripper_positions`: Gripper open/close state (T,)
- `raw_frames`: Original poses
- `camera_X_intrinsics/extrinsics`: Per-camera calibration

## 🔧 Third-Party Dependencies

MVTracker integrates several libraries (installed via `post-create-latest.sh`):

| Library | Purpose | Location |
|---------|---------|----------|
| **SAM2** | Segment Anything Model 2 | `third_party/sam2` |
| **PyTorch3D** | 3D transforms and rendering | `third_party/pytorch3d` |
| **SpaTrackerV2** | Spatial point tracking | `third_party/spatialtrackerv2` |
| **HaMeR** | Hand mesh recovery | `third_party/hamer` |
| **HOISTFormer** | Hand-object interaction | `third_party/HOISTFormer` |
| **3DFlowAction** | 3D flow for action prediction | `third_party/3DFlowAction` |

## 📖 Project Structure

```
mvtracker/
├── .devcontainer/          # DevContainer configurations
│   ├── latest/             # Latest build (pulls deps fresh)
│   └── stable/             # Stable cached build
├── configs/                # Hydra configuration files
│   ├── model/              # Model configurations
│   └── experiment/         # Experiment configurations
├── conversions/            # Data conversion pipelines
│   └── droid/              # DROID dataset processing
│       ├── configs/        # Per-lab configurations
│       ├── training_data/  # Training data pipelines
│       └── utils/          # Utility functions
├── explainers/             # Documentation and guides
├── mvtracker/              # Main package
│   ├── cli/                # Command-line interfaces
│   ├── datasets/           # Dataset loaders
│   ├── evaluation/         # Evaluation metrics
│   ├── models/             # Tracking models
│   └── utils/              # Utilities and visualization
├── third_party/            # External dependencies
├── demo.py                 # Main demo script
├── visualize_training_shards.py    # Shard visualization
└── generate_all_visualizations.py  # Batch visualization
```

## 🐛 Troubleshooting

### NVIDIA Driver Mismatch
```
Failed to initialize NVML: Driver/library version mismatch
NVML library version: 570.195
```
**Solution**: Restart the host machine without stopping the container, or restart Docker daemon.

### ZED SDK Not Found
```
ModuleNotFoundError: No module named 'pyzed.sl'
```
**Solution**: ZED SDK requires manual download from [Stereolabs](https://www.stereolabs.com/developers). For visualization-only workflows, use pre-processed training shards.

### EGL Library Missing (HaMeR)
```
ImportError: Unable to load EGL library
```
**Solution**: Install headless OpenGL:
```bash
apt-get install -y libosmesa6 libosmesa6-dev mesa-utils xvfb
export PYOPENGL_PLATFORM=osmesa
```

## 📚 Additional Documentation

- [Training Data Pipeline Summary](conversions/droid/training_data/PIPELINE_SUMMARY.md)
- [Mask Lifting Guide](explainers/MASK_LIFTING_README.md)
- [GPU Requirements](explainers/GPU_REQUIREMENT.md)
- [HOISTFormer Integration](third_party/HOISTFormer/USAGE_EXAMPLES.md)

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Open a Pull Request

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

## 🎬 Generated Visualizations

The following visualizations have been generated and are available:

### Rerun RRD Files (Interactive 3D)

| File | Size | Description |
|------|------|-------------|
| `shard_0000_contact_flow.rrd` | 1.6G | AUTOLab episodes with contact flow |
| `shard_0001_contact_flow.rrd` | 1.2G | Multi-lab contact flow visualization |
| `shard_0002_contact_flow.rrd` | 1.2G | Contact flow from various labs |
| `shard_0003_contact_flow.rrd` | 1.4G | Additional episode visualizations |
| `shard_0004_contact_flow.rrd` | 1.2G | Contact point tracking |
| `shard_0005_contact_flow.rrd` | 1.2G | Gripper trajectory visualization |
| `shard_0006_contact_flow.rrd` | 1.5G | Full contact flow analysis |
| `combined_episodes.rrd` | 1.5G | Multi-lab combined view |

**Total: ~10.8GB of 3D visualization data**

### Video Outputs (MP4)

Videos with 2D projected contact point tracks overlaid:
- `s0000_000000_tracks.mp4` - `s0000_000002_tracks.mp4`
- `s0001_000000_tracks.mp4` - `s0001_000002_tracks.mp4`

### How to View

```bash
# View RRD files with Rerun
rerun visualizations/shard_0000_contact_flow.rrd

# Play video files
ffplay output_videos/s0000_000000_tracks.mp4
```

## 🔬 Visualization Scripts

| Script | Purpose |
|--------|---------|
| `visualize_training_shards.py` | Basic contact flow from training shards |
| `generate_all_visualizations.py` | Batch process all shards |
| `visualize_combined_episodes.py` | Multi-episode 3D view with orientation |
| `visualize_videos_with_tracks.py` | Create MP4 with 2D track overlays |
