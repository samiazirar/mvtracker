#!/bin/bash
# Test script to verify pipeline outputs are complete
# Usage: ./verify_pipeline_outputs.sh <episode_output_dir>

set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <episode_output_dir>"
    echo "Example: $0 /data/droid_staging/AUTOLab/success/2023-08-18/..."
    exit 1
fi

EPISODE_DIR=$1

echo "=== Verifying Pipeline Output ==="
echo "Directory: ${EPISODE_DIR}"
echo ""

# Check required files exist
echo "[1/5] Checking required files..."
REQUIRED_FILES=(
    "tracks.npz"
    "extrinsics.npz"
    "quality.json"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "${EPISODE_DIR}/${file}" ]; then
        echo "  ✅ ${file}"
    else
        echo "  ❌ ${file} MISSING"
        exit 1
    fi
done

# Check for recordings directory and intrinsics
echo ""
echo "[2/5] Checking camera intrinsics..."
RECORDINGS_DIR="${EPISODE_DIR}/recordings"
if [ ! -d "${RECORDINGS_DIR}" ]; then
    echo "  ❌ recordings/ directory MISSING"
    exit 1
fi

INTRINSICS_COUNT=0
for cam_dir in "${RECORDINGS_DIR}"/*; do
    if [ -d "${cam_dir}" ]; then
        cam_name=$(basename "${cam_dir}")
        if [ -f "${cam_dir}/intrinsics.json" ]; then
            echo "  ✅ ${cam_name}/intrinsics.json"
            INTRINSICS_COUNT=$((INTRINSICS_COUNT + 1))
        else
            echo "  ❌ ${cam_name}/intrinsics.json MISSING"
        fi
    fi
done

if [ ${INTRINSICS_COUNT} -eq 0 ]; then
    echo "  ❌ No intrinsics.json files found"
    exit 1
fi

# Verify tracks.npz contents using Python
echo ""
echo "[3/5] Verifying tracks.npz contents..."
python3 << 'PYTHON_CHECK'
import sys
import numpy as np

try:
    # Load tracks.npz
    data = np.load(sys.argv[1] + '/tracks.npz', allow_pickle=True)
    keys = set(data.keys())
    
    # Check required 3D data
    required_3d = [
        'tracks_3d', 'contact_centroids', 'contact_frames',
        'gripper_poses', 'gripper_positions', 'cartesian_positions'
    ]
    
    missing_3d = [k for k in required_3d if k not in keys]
    if missing_3d:
        print(f"  ❌ Missing 3D data: {missing_3d}")
        sys.exit(1)
    else:
        print(f"  ✅ 3D tracks data complete")
    
    # Check normalized flow data
    required_normalized = [
        'normalized_tracks_3d', 'normalized_centroids', 'normalized_frames',
        'normalized_left_frames', 'normalized_right_frames',
        'cumulative_distance_mm', 'frame_to_normalized_idx',
        'normalized_step_size_mm', 'num_normalized_steps'
    ]
    
    missing_norm = [k for k in required_normalized if k not in keys]
    if missing_norm:
        print(f"  ❌ Missing normalized data: {missing_norm}")
        sys.exit(1)
    else:
        M = int(data['num_normalized_steps'])
        dist = float(data['cumulative_distance_mm'][-1])
        print(f"  ✅ Normalized flow data complete (M={M} steps, {dist:.1f}mm total)")
    
    # Check for 2D projections
    tracks_2d_keys = [k for k in keys if k.startswith('tracks_2d_')]
    if not tracks_2d_keys:
        print(f"  ⚠️  No 2D track projections found")
        print(f"     This is expected if intrinsics extraction failed")
    else:
        # Get camera list
        cameras = data.get('cameras_with_2d_tracks', [])
        print(f"  ✅ 2D projections for {len(tracks_2d_keys)} camera(s): {list(cameras)}")
        
        # Verify intrinsics are included
        intrinsics_keys = [k for k in keys if k.startswith('intrinsics_')]
        if len(intrinsics_keys) != len(tracks_2d_keys):
            print(f"  ⚠️  Intrinsics mismatch: {len(intrinsics_keys)} intrinsics vs {len(tracks_2d_keys)} 2D tracks")
        else:
            print(f"  ✅ Camera intrinsics included in tracks.npz")
    
    # Print summary
    T = data['tracks_3d'].shape[0]
    N = data['tracks_3d'].shape[1]
    print(f"\n  Summary:")
    print(f"    Frames (T): {T}")
    print(f"    Track points (N): {N}")
    print(f"    Normalized steps (M): {M}")
    print(f"    Cameras with 2D: {len(tracks_2d_keys)}")
    
except Exception as e:
    print(f"  ❌ Error loading tracks.npz: {e}")
    sys.exit(1)

PYTHON_CHECK "${EPISODE_DIR}"

# Verify extrinsics.npz
echo ""
echo "[4/5] Verifying extrinsics.npz..."
python3 << 'PYTHON_CHECK'
import sys
import numpy as np

try:
    data = np.load(sys.argv[1] + '/extrinsics.npz', allow_pickle=True)
    keys = set(data.keys())
    
    # Check for external cameras
    external_keys = [k for k in keys if k.startswith('external_')]
    if not external_keys:
        print(f"  ❌ No external camera extrinsics found")
        sys.exit(1)
    else:
        print(f"  ✅ External cameras: {len(external_keys)}")
    
    # Check for wrist camera
    if 'wrist_extrinsics' in keys and 'wrist_serial' in keys:
        T = data['wrist_extrinsics'].shape[0]
        serial = str(data['wrist_serial'])
        print(f"  ✅ Wrist camera: {serial} ({T} frames)")
    else:
        print(f"  ⚠️  No wrist camera data")
    
except Exception as e:
    print(f"  ❌ Error loading extrinsics.npz: {e}")
    sys.exit(1)

PYTHON_CHECK "${EPISODE_DIR}"

# Verify quality.json
echo ""
echo "[5/5] Verifying quality.json..."
python3 << 'PYTHON_CHECK'
import sys
import json

try:
    with open(sys.argv[1] + '/quality.json', 'r') as f:
        data = json.load(f)
    
    required_keys = ['episode_id', 'num_frames', 'num_track_points', 'external_cameras']
    missing = [k for k in required_keys if k not in data]
    
    if missing:
        print(f"  ❌ Missing keys: {missing}")
        sys.exit(1)
    else:
        print(f"  ✅ Episode: {data['episode_id']}")
        print(f"  ✅ Frames: {data['num_frames']}")
        print(f"  ✅ Track points: {data['num_track_points']}")
        print(f"  ✅ External cameras: {len(data['external_cameras'])}")
        
        if 'cameras_with_2d_tracks' in data and data['cameras_with_2d_tracks']:
            print(f"  ✅ 2D tracks for: {data['cameras_with_2d_tracks']}")
    
except Exception as e:
    print(f"  ❌ Error loading quality.json: {e}")
    sys.exit(1)

PYTHON_CHECK "${EPISODE_DIR}"

echo ""
echo "=== ✅ ALL CHECKS PASSED ==="
echo ""
echo "Output summary:"
echo "  - 3D tracks: ✅"
echo "  - Normalized flow: ✅"
echo "  - 2D projections: $(python3 -c "import numpy as np; d=np.load('${EPISODE_DIR}/tracks.npz', allow_pickle=True); print('✅ ' + str(len([k for k in d.keys() if k.startswith('tracks_2d_')])) + ' cameras' if any(k.startswith('tracks_2d_') for k in d.keys()) else '⚠️  None (check intrinsics)')")"
echo "  - Camera intrinsics: ✅ ${INTRINSICS_COUNT} cameras"
echo "  - Camera extrinsics: ✅"
echo "  - Metadata: ✅"
echo ""
