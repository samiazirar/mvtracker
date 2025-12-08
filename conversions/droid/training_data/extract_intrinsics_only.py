"""Extract camera intrinsics from DROID SVO recordings WITHOUT extracting depth/RGB.

This lightweight script only extracts camera intrinsics from ZED SVO files and saves
them as intrinsics.json files. It does NOT require GPU or extract any depth/RGB data.

Output:
    {output_root}/{lab}/success/{date}/{timestamp}/
        └── recordings/
            ├── {camera_serial}/
            │   └── intrinsics.json
            └── ...

Usage:
    python extract_intrinsics_only.py --episode_id "AUTOLab+84bd5053+2023-08-18-12h-01m-10s"
"""

import argparse
import json
import os
import sys
import glob
import re
from pathlib import Path
from typing import Optional

import yaml

# ZED SDK import
try:
    import pyzed.sl as sl
except ImportError:
    print("[ERROR] pyzed SDK not installed. Please install the ZED SDK.")
    sys.exit(1)


def parse_episode_id(episode_id: str) -> dict:
    """Parse episode ID into components."""
    parts = episode_id.split('+')
    if len(parts) != 3:
        raise ValueError(f"Invalid episode ID format: {episode_id}")
    
    lab = parts[0]
    episode_hash = parts[1]
    datetime_part = parts[2]
    
    match = re.match(r'(\d{4}-\d{2}-\d{2})-(\d+)h-(\d+)m-(\d+)s', datetime_part)
    if not match:
        raise ValueError(f"Invalid datetime format in episode ID: {datetime_part}")
    
    date = match.group(1)
    hour = match.group(2)
    minute = match.group(3)
    second = match.group(4)
    time_str = f"{hour}:{minute}:{second}"
    
    from datetime import datetime
    dt = datetime.strptime(f"{date} {hour}:{minute}:{second}", "%Y-%m-%d %H:%M:%S")
    timestamp_folder = dt.strftime("%a_%b_%e_%H:%M:%S_%Y").replace(" ", "_")
    
    return {
        'lab': lab,
        'hash': episode_hash,
        'date': date,
        'time_str': time_str,
        'timestamp_folder': timestamp_folder,
        'full_id': episode_id,
    }


def find_episode_paths(droid_root: str, episode_info: dict, extra_roots: list = None) -> dict:
    """Find all paths for an episode."""
    lab = episode_info['lab']
    date = episode_info['date']
    timestamp_folder = episode_info['timestamp_folder']
    
    roots_to_search = [droid_root]
    if extra_roots:
        roots_to_search.extend(extra_roots)
    
    for root in roots_to_search:
        for outcome in ['success', 'failure']:
            base_path = os.path.join(root, lab, outcome, date, timestamp_folder)
            recordings_dir = os.path.join(base_path, 'recordings', 'SVO')
            
            if os.path.exists(recordings_dir):
                relative_path = os.path.join(lab, outcome, date, timestamp_folder)
                return {
                    'recordings_dir': recordings_dir,
                    'relative_path': relative_path,
                    'outcome': outcome,
                }
    
    raise FileNotFoundError(f"Episode not found: {episode_info['full_id']}")


def extract_intrinsics_from_svo(svo_path: str, output_dir: str, camera_serial: str) -> dict:
    """Extract ONLY camera intrinsics from an SVO file (no depth/RGB extraction).
    
    Args:
        svo_path: Path to SVO file
        output_dir: Directory to save intrinsics.json
        camera_serial: Camera serial number
        
    Returns:
        dict with status and intrinsics info
    """
    # Initialize ZED camera with SVO in CPU mode
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo_path)
    init_params.svo_real_time_mode = False
    init_params.coordinate_units = sl.UNIT.METER
    # CRITICAL: Enable CPU mode to avoid GPU requirement for just reading intrinsics
    init_params.sdk_gpu_id = -1  # Use CPU-only mode
    init_params.depth_mode = sl.DEPTH_MODE.NONE  # Disable depth computation
    
    # Open camera (CPU mode - no GPU needed for intrinsics)
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        return {
            'serial': camera_serial,
            'status': 'error',
            'error': f"Failed to open SVO: {status}",
        }
    
    # Get camera information
    cam_info = zed.get_camera_information()
    calib = cam_info.camera_configuration.calibration_parameters.left_cam
    
    # Extract resolution
    width = cam_info.camera_configuration.resolution.width
    height = cam_info.camera_configuration.resolution.height
    
    # Build intrinsics dict
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
    
    # Create output directory and save
    os.makedirs(output_dir, exist_ok=True)
    intrinsics_path = os.path.join(output_dir, 'intrinsics.json')
    with open(intrinsics_path, 'w') as f:
        json.dump(intrinsics, f, indent=2)
    
    zed.close()
    
    return {
        'serial': camera_serial,
        'status': 'success',
        'width': width,
        'height': height,
        'intrinsics_path': intrinsics_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract camera intrinsics only from DROID SVO recordings."
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
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"=== Extracting Camera Intrinsics ===")
    print(f"Episode: {args.episode_id}")
    
    # Parse episode ID
    episode_info = parse_episode_id(args.episode_id)
    
    # Find episode paths
    extra_roots = [config.get('download_dir', './droid_downloads')]
    episode_paths = find_episode_paths(config['droid_root'], episode_info, extra_roots)
    
    recordings_dir = episode_paths['recordings_dir']
    print(f"  Recordings: {recordings_dir}")
    
    # Create output directory
    output_base = os.path.join(config['output_root'], episode_paths['relative_path'], 'recordings')
    os.makedirs(output_base, exist_ok=True)
    
    # Find all SVO files
    svo_files = glob.glob(os.path.join(recordings_dir, '*.svo*'))
    if not svo_files:
        print(f"[ERROR] No SVO files found in {recordings_dir}")
        sys.exit(1)
    
    print(f"\nFound {len(svo_files)} camera(s)")
    
    # Process each camera
    results = []
    for svo_path in svo_files:
        # Extract camera serial from filename
        filename = os.path.basename(svo_path)
        camera_serial = os.path.splitext(filename)[0]
        
        print(f"\n[{camera_serial}] Extracting intrinsics...")
        
        cam_output_dir = os.path.join(output_base, camera_serial)
        result = extract_intrinsics_from_svo(svo_path, cam_output_dir, camera_serial)
        results.append(result)
        
        if result['status'] == 'success':
            print(f"  ✓ Saved: {result['intrinsics_path']}")
            print(f"    Resolution: {result['width']}x{result['height']}")
        else:
            print(f"  ✗ Error: {result.get('error', 'Unknown error')}")
    
    # Summary
    print(f"\n=== Summary ===")
    success_count = sum(1 for r in results if r['status'] == 'success')
    print(f"Success: {success_count}/{len(results)}")
    
    if success_count < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
