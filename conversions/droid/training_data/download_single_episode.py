"""Download a single DROID episode from GCS.

Usage:
    python download_single_episode.py --episode_id "AUTOLab+84bd5053+2023-08-18-12h-01m-10s"
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime


def format_timestamp_folder(dt: datetime) -> str:
    """Format timestamp folder to match DROID GCS layout (space-padded day)."""
    return dt.strftime("%a_%b_%e_%H:%M:%S_%Y").replace(" ", "_")


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
        raise ValueError(f"Invalid datetime format: {datetime_part}")
    
    date = match.group(1)
    hour = match.group(2)
    minute = match.group(3)
    second = match.group(4)
    dt = datetime.strptime(f"{date} {hour}:{minute}:{second}", "%Y-%m-%d %H:%M:%S")
    timestamp_folder = format_timestamp_folder(dt)

    return {
        'lab': lab,
        'hash': episode_hash,
        'date': date,
        'timestamp_folder': timestamp_folder,
    }


def find_relative_path(episode_info: dict, cam2base_path: str = None) -> str:
    """Find relative path, trying success then failure."""
    lab = episode_info['lab']
    date = episode_info['date']
    timestamp = episode_info['timestamp_folder']
    
    # Default to success, the download will work regardless
    return f"{lab}/success/{date}/{timestamp}"


def run_gsutil(command: str, check: bool = True) -> bool:
    """Run gsutil command."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=check,
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"[WARN] gsutil error: {e.stderr}", file=sys.stderr)
        return False


def download_episode(
    episode_id: str,
    gcs_bucket: str,
    output_dir: str,
    cam2base_path: str = None
) -> str:
    """Download episode files from GCS.
    
    Returns local path to downloaded episode.
    """
    episode_info = parse_episode_id(episode_id)
    
    # Try both success and failure paths
    for outcome in ['success', 'failure']:
        rel_path = f"{episode_info['lab']}/{outcome}/{episode_info['date']}/{episode_info['timestamp_folder']}"
        gcs_path = f"{gcs_bucket}/{rel_path}"
        local_path = os.path.join(output_dir, rel_path)
        
        # Check if trajectory.h5 exists at this path
        check_cmd = f'gsutil -q stat "{gcs_path}/trajectory.h5"'
        if run_gsutil(check_cmd, check=False):
            print(f"Found episode at: {rel_path}")
            break
    else:
        raise FileNotFoundError(f"Episode not found in GCS: {episode_id}")
    
    # Create local directory
    os.makedirs(local_path, exist_ok=True)
    svo_path = os.path.join(local_path, "recordings", "SVO")
    os.makedirs(svo_path, exist_ok=True)
    
    # Download trajectory.h5
    print("Downloading trajectory.h5...")
    cmd = f'gsutil cp "{gcs_path}/trajectory.h5" "{local_path}/"'
    run_gsutil(cmd)
    
    # Download metadata JSON
    print("Downloading metadata...")
    cmd = f'gsutil cp "{gcs_path}/metadata*.json" "{local_path}/"'
    run_gsutil(cmd, check=False)  # May not exist
    
    # Download SVO files
    print("Downloading SVO recordings...")
    cmd = f'gsutil -m cp -r "{gcs_path}/recordings/SVO/*" "{svo_path}/"'
    run_gsutil(cmd)
    
    print(f"Downloaded to: {local_path}")
    return local_path


def main():
    parser = argparse.ArgumentParser(description="Download a single DROID episode.")
    parser.add_argument(
        "--episode_id",
        required=True,
        help='Episode ID, e.g., "AUTOLab+84bd5053+2023-08-18-12h-01m-10s"',
    )
    parser.add_argument(
        "--gcs_bucket",
        default="gs://gresearch/robotics/droid_raw/1.0.1",
        help="GCS bucket path",
    )
    parser.add_argument(
        "--output_dir",
        default="./droid_downloads",
        help="Local output directory",
    )
    parser.add_argument(
        "--cam2base",
        default=None,
        help="Path to cam2base JSON (optional, for path lookup)",
    )
    args = parser.parse_args()
    
    download_episode(
        args.episode_id,
        args.gcs_bucket,
        args.output_dir,
        args.cam2base
    )


if __name__ == "__main__":
    main()
