"""Download a single DROID episode from local storage or GCS.

Usage:
    python download_single_episode.py --episode_id "AUTOLab+84bd5053+2023-08-18-12h-01m-10s"
    
Local mode (default):
    python download_single_episode.py --episode_id "..." --local_source /data/droid/data/droid_raw/1.0.1
    
GCS mode:
    python download_single_episode.py --episode_id "..." --use_gcs --gcs_bucket gs://gresearch/robotics/droid_raw/1.0.1
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime


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
    # Use ctime-style format: space-padded day becomes underscore (e.g., "Jul  7" -> "Jul__7")
    timestamp_folder = dt.strftime("%a_%b_%e_%H:%M:%S_%Y").replace(" ", "_")
    
    return {
        'lab': lab,
        'hash': episode_hash,
        'date': date,
        'timestamp_folder': timestamp_folder,
        'full_id': episode_id,
    }


def find_episode_local(local_source: str, episode_info: dict) -> tuple:
    """Find episode in local storage.
    
    Returns:
        tuple: (source_path, outcome) if found, raises FileNotFoundError otherwise
    """
    lab = episode_info['lab']
    date = episode_info['date']
    timestamp_folder = episode_info['timestamp_folder']
    
    # Try both success and failure paths
    for outcome in ['success', 'failure']:
        source_path = os.path.join(local_source, lab, outcome, date, timestamp_folder)
        h5_path = os.path.join(source_path, 'trajectory.h5')
        
        if os.path.exists(h5_path):
            return source_path, outcome
    
    # Episode not found - search for similar paths to help debug
    search_results = search_for_episode(local_source, episode_info)
    
    raise FileNotFoundError(
        f"Episode not found locally: {episode_info['full_id']}\n"
        f"  Expected path: {local_source}/{lab}/{{success|failure}}/{date}/{timestamp_folder}\n"
        f"  Search results: {search_results}"
    )


def search_for_episode(local_source: str, episode_info: dict) -> str:
    """Search for episode with fuzzy matching to help debug."""
    lab = episode_info['lab']
    date = episode_info['date']
    timestamp_folder = episode_info['timestamp_folder']
    
    results = []
    
    # Check if lab exists
    lab_path = os.path.join(local_source, lab)
    if not os.path.exists(lab_path):
        available_labs = [d for d in os.listdir(local_source) 
                        if os.path.isdir(os.path.join(local_source, d)) and not d.startswith('.')]
        return f"Lab '{lab}' not found. Available: {available_labs}"
    
    # Check if date exists
    for outcome in ['success', 'failure']:
        date_path = os.path.join(lab_path, outcome, date)
        if os.path.exists(date_path):
            # List timestamps in that date folder
            timestamps = os.listdir(date_path)
            # Find similar timestamps
            for ts in timestamps:
                if timestamp_folder[:10] in ts:  # Match day of week + month
                    results.append(f"{outcome}/{date}/{ts}")
    
    if results:
        return f"Similar paths found: {results[:5]}"
    
    # Check what dates exist for this lab
    for outcome in ['success', 'failure']:
        outcome_path = os.path.join(lab_path, outcome)
        if os.path.exists(outcome_path):
            dates = sorted(os.listdir(outcome_path))[:10]
            results.append(f"{outcome}: dates={dates}")
    
    return f"Date '{date}' search: {results}" if results else "No matching dates found"


def copy_episode_local(source_path: str, output_path: str, use_symlinks: bool = True) -> str:
    """Copy or symlink episode from local source to output directory.
    
    Args:
        source_path: Source episode directory
        output_path: Destination directory
        use_symlinks: If True, create symlinks; if False, copy files
        
    Returns:
        Path to the output directory
    """
    os.makedirs(output_path, exist_ok=True)
    svo_output = os.path.join(output_path, "recordings", "SVO")
    os.makedirs(svo_output, exist_ok=True)
    
    # Copy/link trajectory.h5
    src_h5 = os.path.join(source_path, "trajectory.h5")
    dst_h5 = os.path.join(output_path, "trajectory.h5")
    if os.path.exists(src_h5):
        if use_symlinks:
            if os.path.exists(dst_h5):
                os.remove(dst_h5)
            os.symlink(src_h5, dst_h5)
        else:
            shutil.copy2(src_h5, dst_h5)
        print(f"  trajectory.h5: {'linked' if use_symlinks else 'copied'}")
    else:
        raise FileNotFoundError(f"trajectory.h5 not found at {src_h5}")
    
    # Copy/link metadata files
    for metadata_file in glob.glob(os.path.join(source_path, "metadata*.json")):
        dst_meta = os.path.join(output_path, os.path.basename(metadata_file))
        if use_symlinks:
            if os.path.exists(dst_meta):
                os.remove(dst_meta)
            os.symlink(metadata_file, dst_meta)
        else:
            shutil.copy2(metadata_file, dst_meta)
        print(f"  {os.path.basename(metadata_file)}: {'linked' if use_symlinks else 'copied'}")
    
    # Copy/link SVO files
    svo_source = os.path.join(source_path, "recordings", "SVO")
    if os.path.exists(svo_source):
        svo_count = 0
        for svo_file in glob.glob(os.path.join(svo_source, "*.svo*")):
            dst_svo = os.path.join(svo_output, os.path.basename(svo_file))
            if use_symlinks:
                if os.path.exists(dst_svo):
                    os.remove(dst_svo)
                os.symlink(svo_file, dst_svo)
            else:
                shutil.copy2(svo_file, dst_svo)
            svo_count += 1
        print(f"  SVO files: {svo_count} {'linked' if use_symlinks else 'copied'}")
    else:
        print(f"  WARNING: No SVO directory at {svo_source}")
    
    return output_path


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


def download_episode_gcs(
    episode_id: str,
    gcs_bucket: str,
    output_dir: str,
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


def download_episode(
    episode_id: str,
    output_dir: str,
    local_source: str = None,
    gcs_bucket: str = None,
    use_gcs: bool = False,
    use_symlinks: bool = True,
) -> str:
    """Download/copy episode to output directory.
    
    Args:
        episode_id: Episode identifier
        output_dir: Where to put the episode
        local_source: Local DROID data root (for local mode)
        gcs_bucket: GCS bucket path (for GCS mode)
        use_gcs: If True, download from GCS; if False, use local source
        use_symlinks: If True, symlink local files; if False, copy them
        
    Returns:
        Path to the episode in output_dir
    """
    episode_info = parse_episode_id(episode_id)
    
    if use_gcs:
        if not gcs_bucket:
            raise ValueError("gcs_bucket required for GCS mode")
        return download_episode_gcs(episode_id, gcs_bucket, output_dir)
    
    # Local mode
    if not local_source:
        raise ValueError("local_source required for local mode")
    
    # Find episode in local storage
    source_path, outcome = find_episode_local(local_source, episode_info)
    
    # Build output path
    rel_path = f"{episode_info['lab']}/{outcome}/{episode_info['date']}/{episode_info['timestamp_folder']}"
    output_path = os.path.join(output_dir, rel_path)
    
    print(f"Found episode locally: {rel_path}")
    
    # Copy/symlink to output
    return copy_episode_local(source_path, output_path, use_symlinks)


def main():
    parser = argparse.ArgumentParser(description="Download/copy a single DROID episode.")
    parser.add_argument(
        "--episode_id",
        required=True,
        help='Episode ID, e.g., "AUTOLab+84bd5053+2023-08-18-12h-01m-10s"',
    )
    parser.add_argument(
        "--output_dir",
        default="./droid_downloads",
        help="Local output directory",
    )
    parser.add_argument(
        "--local_source",
        default="/data/droid/data/droid_raw/1.0.1",
        help="Local DROID data root (default: /data/droid/data/droid_raw/1.0.1)",
    )
    parser.add_argument(
        "--use_gcs",
        action="store_true",
        help="Download from GCS instead of local source",
    )
    parser.add_argument(
        "--gcs_bucket",
        default="gs://gresearch/robotics/droid_raw/1.0.1",
        help="GCS bucket path (only used with --use_gcs)",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of symlinking (local mode only)",
    )
    parser.add_argument(
        "--cam2base",
        default=None,
        help="Path to cam2base JSON (optional, unused but kept for compatibility)",
    )
    args = parser.parse_args()
    
    download_episode(
        episode_id=args.episode_id,
        output_dir=args.output_dir,
        local_source=args.local_source,
        gcs_bucket=args.gcs_bucket,
        use_gcs=args.use_gcs,
        use_symlinks=not args.copy,
    )


if __name__ == "__main__":
    main()
