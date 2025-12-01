# DROID Training Data Generation

This directory contains scripts for generating training data from DROID episodes.

## Output Structure

The scripts maintain the DROID folder structure:

```
{output_root}/
└── {lab}/
    └── {success|failure}/
        └── {date}/
            └── {timestamp}/
                ├── tracks.npz           # Gripper contact surface tracks
                ├── extrinsics.npz       # Camera extrinsics (external + wrist)
                ├── quality.json         # Calibration snippet + metadata
                └── recordings/
                    ├── extraction_metadata.json
                    ├── {camera_serial_1}/
                    │   ├── intrinsics.json
                    │   ├── rgb/
                    │   │   ├── 000000.png
                    │   │   ├── 000001.png
                    │   │   └── ...
                    │   └── depth/
                    │       ├── 000000.png  # 16-bit PNG, depth in mm
                    │       └── ...
                    └── {camera_serial_2}/
                        └── ...
```

## Episode ID Format

Episodes are identified by a unique ID:
```
{lab}+{hash}+{date}-{hour}h-{min}m-{sec}s
```

Example: `AUTOLab+84bd5053+2023-08-18-12h-01m-10s`

## Scripts

### 1. `generate_tracks_and_metadata.py` (CPU)

Generates gripper tracks and camera extrinsics without GPU requirements.

**Outputs:**
- `tracks.npz` - Gripper contact surface points in world coordinates
- `extrinsics.npz` - Camera extrinsics (fixed for external, per-frame for wrist)
- `quality.json` - Calibration data and metadata

**Usage:**
```bash
python generate_tracks_and_metadata.py --episode_id "AUTOLab+84bd5053+2023-08-18-12h-01m-10s"
```

### 2. `extract_rgb_depth.py` (GPU)

Extracts RGB and depth frames from SVO recordings. Requires GPU for ZED depth computation.

**Outputs:**
- `recordings/{camera_serial}/rgb/*.png` - Lossless RGB frames
- `recordings/{camera_serial}/depth/*.png` - 16-bit depth in millimeters
- `recordings/{camera_serial}/intrinsics.json` - Camera intrinsics

**Usage:**
```bash
python extract_rgb_depth.py --episode_id "AUTOLab+84bd5053+2023-08-18-12h-01m-10s"

# Process specific cameras only
python extract_rgb_depth.py --episode_id "..." --cameras 12345678 87654321
```

### 3. `process_episode.py` (Convenience Wrapper)

Runs both scripts in sequence.

**Usage:**
```bash
# Full processing
python process_episode.py --episode_id "AUTOLab+84bd5053+2023-08-18-12h-01m-10s"

# Skip frame extraction (tracks only)
python process_episode.py --episode_id "..." --skip-frames

# Skip track generation (frames only)
python process_episode.py --episode_id "..." --skip-tracks
```

## Configuration

Edit `config.yaml` to set paths and parameters:

```yaml
# Input paths
droid_root: "/data/droid/data/droid_raw/1.0.1"
cam2base_extrinsics_path: "/data/droid/calib_and_annot/droid/cam2base_extrinsic_superset.json"

# Output root
output_root: "/data/droid/training_data"

# Processing parameters
max_frames: null  # null = all frames
num_track_points: 480
fps: 30.0

# PNG settings
png_compression: 3  # 0-9, higher = smaller but slower
depth_scale: 1000.0  # meters -> millimeters
```

## Data Formats

### tracks.npz

```python
import numpy as np
data = np.load('tracks.npz')

# Main arrays
tracks_3d = data['tracks_3d']           # [T, N, 3] - world coords
gripper_poses = data['gripper_poses']   # [T, 4, 4] - EE transforms
gripper_positions = data['gripper_positions']  # [T] - gripper open/close
cartesian_positions = data['cartesian_positions']  # [T, 6] - raw EE poses

# Metadata
num_frames = data['num_frames']
num_points_per_finger = data['num_points_per_finger']
fps = data['fps']
```

### extrinsics.npz

```python
data = np.load('extrinsics.npz', allow_pickle=True)

# External cameras (fixed, 4x4 each)
external_12345678 = data['external_12345678']  # [4, 4]

# Wrist camera (per-frame)
wrist_extrinsics = data['wrist_extrinsics']  # [T, 4, 4]
wrist_serial = str(data['wrist_serial'])
```

### depth PNGs

Depth is stored as 16-bit PNG in millimeters:

```python
import cv2
depth_mm = cv2.imread('000000.png', cv2.IMREAD_UNCHANGED)  # uint16
depth_m = depth_mm.astype(np.float32) / 1000.0  # Convert to meters
depth_m[depth_mm == 0] = np.nan  # Invalid depth
```

## Cluster Deployment

For large-scale processing on a cluster:

1. Generate a list of episode IDs from `cam2base_extrinsic_superset.json`
2. Submit jobs with SLURM/PBS arrays
3. Use `--skip-tracks` on GPU nodes and `--skip-frames` on CPU nodes

Example SLURM job:
```bash
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --array=0-999

EPISODE_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" episode_ids.txt)
python extract_rgb_depth.py --episode_id "$EPISODE_ID"
```
