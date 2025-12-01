"""Extract RGB frames as PNG and depth frames as per-frame NPY from DROID SVO recordings.

This script processes DROID episodes and extracts frames to:
    {output_root}/{lab}/success/{date}/{timestamp}/
        └── recordings/
            ├── {camera_serial}/
            │   ├── rgb/
            │   │   ├── 000000.png
            │   │   ├── 000001.png
            │   │   └── ...
            │   └── depth/
            │       ├── 000000.npy  (float32 array, depth in meters)
            │       ├── 000001.npy
            │       └── ...
            └── ...

Usage:
    python extract_rgb_depth.py --episode_id "AUTOLab+84bd5053+2023-08-18-12h-01m-10s"
    python extract_rgb_depth.py --episode_id "AUTOLab+84bd5053+2023-08-18-12h-01m-10s" --config custom_config.yaml

Requirements:
    - GPU with CUDA support (for ZED depth computation)
    - pyzed SDK installed
"""

import argparse
import json
import os
import sys
import glob
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
import yaml

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ZED SDK import
try:
    import pyzed.sl as sl
except ImportError:
    print("[ERROR] pyzed SDK not installed. Please install the ZED SDK.")
    sys.exit(1)


def parse_episode_id(episode_id: str) -> dict:
    """Parse episode ID into components.
    
    Episode ID format: "{lab}+{hash}+{date}-{hour}h-{min}m-{sec}s"
    Example: "AUTOLab+84bd5053+2023-08-18-12h-01m-10s"
    
    Returns:
        dict with keys: lab, hash, date, time_str, timestamp_folder
    """
    parts = episode_id.split('+')
    if len(parts) != 3:
        raise ValueError(f"Invalid episode ID format: {episode_id}")
    
    lab = parts[0]
    episode_hash = parts[1]
    datetime_part = parts[2]
    
    # Parse datetime: "2023-08-18-12h-01m-10s"
    match = re.match(r'(\d{4}-\d{2}-\d{2})-(\d+)h-(\d+)m-(\d+)s', datetime_part)
    if not match:
        raise ValueError(f"Invalid datetime format in episode ID: {datetime_part}")
    
    date = match.group(1)
    hour = match.group(2)
    minute = match.group(3)
    second = match.group(4)
    time_str = f"{hour}:{minute}:{second}"
    
    # Reconstruct DROID timestamp folder format: "Fri_Aug_18_11:58:51_2023"
    from datetime import datetime
    dt = datetime.strptime(f"{date} {hour}:{minute}:{second}", "%Y-%m-%d %H:%M:%S")
    timestamp_folder = dt.strftime("%a_%b_%d_%H:%M:%S_%Y")
    
    return {
        'lab': lab,
        'hash': episode_hash,
        'date': date,
        'time_str': time_str,
        'timestamp_folder': timestamp_folder,
        'full_id': episode_id,
    }


def find_episode_paths(droid_root: str, episode_info: dict) -> dict:
    """Find all paths for an episode.
    
    Returns:
        dict with keys: h5_path, recordings_dir, metadata_path, relative_path
    """
    lab = episode_info['lab']
    date = episode_info['date']
    timestamp_folder = episode_info['timestamp_folder']
    
    # Search in both success and failure folders
    for outcome in ['success', 'failure']:
        base_path = os.path.join(droid_root, lab, outcome, date, timestamp_folder)
        h5_path = os.path.join(base_path, 'trajectory.h5')
        
        if os.path.exists(h5_path):
            recordings_dir = os.path.join(base_path, 'recordings', 'SVO')
            
            # Find metadata file
            metadata_files = glob.glob(os.path.join(base_path, 'metadata_*.json'))
            metadata_path = metadata_files[0] if metadata_files else None
            
            # Relative path for output structure
            relative_path = os.path.join(lab, outcome, date, timestamp_folder)
            
            return {
                'h5_path': h5_path,
                'recordings_dir': recordings_dir,
                'metadata_path': metadata_path,
                'relative_path': relative_path,
                'outcome': outcome,
            }
    
    raise FileNotFoundError(f"Episode not found: {episode_info['full_id']}")


def find_all_svo_files(recordings_dir: str) -> dict:
    """Find all SVO files in recordings directory.
    
    Returns:
        dict mapping camera serial to SVO path
    """
    svo_files = {}
    
    # Search for SVO and SVO2 files
    for pattern in ['*.svo', '*.svo2']:
        for svo_path in glob.glob(os.path.join(recordings_dir, pattern)):
            filename = os.path.basename(svo_path)
            # Extract serial number from filename
            # Format typically: "SN12345678.svo" or similar
            serial = filename.replace('.svo2', '').replace('.svo', '')
            # Clean up serial - remove common prefixes
            serial = serial.replace('SN', '').replace('sn', '')
            svo_files[serial] = svo_path
    
    return svo_files


def save_frame_png(
    rgb: np.ndarray,
    depth: np.ndarray,
    frame_idx: int,
    rgb_dir: str,
    depth_dir: str,
) -> Tuple[str, str]:
    """Save RGB as PNG and depth as raw NPY files."""
    filename = f"{frame_idx:06d}"
    
    # Save RGB (lossless/uncompressed PNG)
    rgb_path = os.path.join(rgb_dir, f"{filename}.png")
    # OpenCV expects BGR, convert from RGB
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(rgb_path, rgb_bgr)
    
    # Save depth as float32 NPY (meters)
    depth_path = os.path.join(depth_dir, f"{filename}.npy")
    depth_to_save = depth.copy()
    depth_to_save[~np.isfinite(depth_to_save)] = 0.0  # Keep invalid values consistent
    np.save(depth_path, depth_to_save)
    
    return rgb_path, depth_path


def process_camera(
    svo_path: str,
    camera_serial: str,
    output_base_dir: str,
    max_frames: Optional[int] = None,
) -> dict:
    """Process a single camera SVO file and extract frames.
    
    Args:
        svo_path: Path to SVO file
        camera_serial: Camera serial number for output folder
        output_base_dir: Base output directory
        max_frames: Maximum frames to process (None = all)
    
    Returns:
        dict with processing statistics
    """
    print(f"  Processing camera: {camera_serial}")
    
    # Create output directories
    cam_dir = os.path.join(output_base_dir, 'recordings', camera_serial)
    rgb_dir = os.path.join(cam_dir, 'rgb')
    depth_dir = os.path.join(cam_dir, 'depth')
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    
    # Initialize ZED camera
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo_path)
    init_params.svo_real_time_mode = False
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL  # Best quality depth
    
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"    [ERROR] Failed to open SVO: {err}")
        return {'serial': camera_serial, 'status': 'failed', 'error': str(err)}
    
    # Get camera info
    cam_info = zed.get_camera_information()
    width = cam_info.camera_configuration.resolution.width
    height = cam_info.camera_configuration.resolution.height
    total_frames = zed.get_svo_number_of_frames()
    
    if max_frames is not None:
        process_frames = min(max_frames, total_frames)
    else:
        process_frames = total_frames
    
    print(f"    Resolution: {width}x{height}")
    print(f"    Total frames: {total_frames}, processing: {process_frames}")
    
    # Prepare matrices
    runtime_params = sl.RuntimeParameters()
    mat_rgb = sl.Mat()
    mat_depth = sl.Mat()
    
    # Process frames
    frames_saved = 0
    for frame_idx in range(process_frames):
        if frame_idx % 100 == 0:
            print(f"    Frame {frame_idx}/{process_frames}")
        
        err = zed.grab(runtime_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"    [WARN] Failed to grab frame {frame_idx}: {err}")
            continue
        
        # Retrieve RGB image
        zed.retrieve_image(mat_rgb, sl.VIEW.LEFT)
        rgb_data = mat_rgb.get_data()
        # ZED returns BGRA, convert to RGB
        rgb = cv2.cvtColor(rgb_data, cv2.COLOR_BGRA2RGB)
        
        # Retrieve depth map
        zed.retrieve_measure(mat_depth, sl.MEASURE.DEPTH)
        depth = mat_depth.get_data().copy()  # float32, meters
        
        # Save frames
        save_frame_png(
            rgb, depth, frame_idx,
            rgb_dir, depth_dir,
        )
        frames_saved += 1
    
    # Get intrinsics and save
    calib = cam_info.camera_configuration.calibration_parameters.left_cam
    intrinsics = {
        'fx': float(calib.fx),
        'fy': float(calib.fy),
        'cx': float(calib.cx),
        'cy': float(calib.cy),
        'width': width,
        'height': height,
        'k1': float(calib.disto[0]) if len(calib.disto) > 0 else 0.0,
        'k2': float(calib.disto[1]) if len(calib.disto) > 1 else 0.0,
        'p1': float(calib.disto[2]) if len(calib.disto) > 2 else 0.0,
        'p2': float(calib.disto[3]) if len(calib.disto) > 3 else 0.0,
        'k3': float(calib.disto[4]) if len(calib.disto) > 4 else 0.0,
    }
    
    intrinsics_path = os.path.join(cam_dir, 'intrinsics.json')
    with open(intrinsics_path, 'w') as f:
        json.dump(intrinsics, f, indent=2)
    
    zed.close()
    
    return {
        'serial': camera_serial,
        'status': 'success',
        'frames_saved': frames_saved,
        'total_frames': total_frames,
        'width': width,
        'height': height,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract RGB and depth frames from DROID SVO recordings."
    )
    parser.add_argument(
        "--episode_id",
        required=True,
        help='Episode ID, e.g., "AUTOLab+84bd5053+2023-08-18-12h-01m-10s"',
    )
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
        help="Path to config YAML file.",
    )
    parser.add_argument(
        "--cameras",
        nargs="*",
        help="Specific camera serials to process (default: all)",
    )
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"=== Extracting RGB and Depth Frames ===")
    print(f"Episode: {args.episode_id}")
    
    # Parse episode ID
    episode_info = parse_episode_id(args.episode_id)
    print(f"  Lab: {episode_info['lab']}")
    print(f"  Date: {episode_info['date']}")
    
    # Find episode paths
    episode_paths = find_episode_paths(config['droid_root'], episode_info)
    print(f"  Found: {episode_paths['relative_path']}")
    print(f"  Recordings: {episode_paths['recordings_dir']}")
    
    # Find all SVO files
    svo_files = find_all_svo_files(episode_paths['recordings_dir'])
    print(f"  Found {len(svo_files)} cameras: {list(svo_files.keys())}")
    
    # Filter cameras if specified
    if args.cameras:
        svo_files = {k: v for k, v in svo_files.items() if k in args.cameras}
        print(f"  Filtering to: {list(svo_files.keys())}")
    
    if not svo_files:
        print("[ERROR] No SVO files found!")
        sys.exit(1)
    
    # Create output directory
    output_dir = os.path.join(config['output_root'], episode_paths['relative_path'])
    os.makedirs(output_dir, exist_ok=True)
    print(f"  Output: {output_dir}")
    
    # Process each camera
    results = []
    for serial, svo_path in svo_files.items():
        result = process_camera(
            svo_path=svo_path,
            camera_serial=serial,
            output_base_dir=output_dir,
            max_frames=config.get('max_frames'),
        )
        results.append(result)
    
    # Save extraction metadata
    metadata = {
        'episode_id': args.episode_id,
        'source_path': episode_paths['relative_path'],
        'cameras': results,
    }
    
    metadata_path = os.path.join(output_dir, 'recordings', 'extraction_metadata.json')
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Summary
    print("\n=== Extraction Summary ===")
    for result in results:
        if result['status'] == 'success':
            print(f"  {result['serial']}: {result['frames_saved']} frames saved")
        else:
            print(f"  {result['serial']}: FAILED - {result.get('error', 'unknown')}")
    
    print(f"\nOutput: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
