"""Get episodes sorted by quality per lab from cam2base_extrinsic_superset.json.

This script reads the calibration JSON and outputs episodes sorted by quality,
interleaved across labs (best from each lab, then second best from each, etc.)

Balanced mode (--balanced):
    Queries GCS to determine success/failure status for each episode,
    then interleaves success and failure episodes per lab:
    lab1_success, lab1_failure, lab2_success, lab2_failure, ...
    
Usage:
    python get_episodes_by_quality.py --limit 100
    python get_episodes_by_quality.py --limit -1  # All episodes
    python get_episodes_by_quality.py --output episodes.txt
    python get_episodes_by_quality.py --balanced --limit 100  # Balance success/failure per lab
"""

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime


def load_cam2base(cam2base_path: str) -> dict:
    """Load the cam2base extrinsics superset JSON."""
    with open(cam2base_path, 'r') as f:
        return json.load(f)


def parse_episode_key(key: str) -> dict:
    """Parse episode key into components.
    
    Key format: "{lab}+{hash}+{date}-{hour}h-{min}m-{sec}s"
    """
    parts = key.split('+')
    if len(parts) != 3:
        return None
    
    return {
        'lab': parts[0],
        'hash': parts[1],
        'datetime': parts[2],
        'full_id': key,
    }


def get_episode_quality(episode_data: dict) -> float:
    """Calculate quality score for an episode.
    
    Higher score = better quality.
    Factors:
    - Number of external cameras (more = better)
    - Presence of all expected calibration data
    """
    score = 0.0
    
    # Count external cameras (numeric keys are camera serials)
    num_cameras = sum(1 for k in episode_data.keys() if k.isdigit())
    score += num_cameras * 10.0
    
    # Check for quality indicators in the data
    for cam_id, transform in episode_data.items():
        if cam_id.isdigit() and isinstance(transform, list) and len(transform) == 6:
            # Valid 6-DOF transform
            score += 1.0
    
    return score


def episode_id_to_gcs_paths(episode_id: str, gcs_bucket: str) -> tuple:
    """Convert episode ID to potential GCS paths (success and failure).
    
    Returns (success_path, failure_path)
    """
    parts = episode_id.split('+')
    if len(parts) != 3:
        return None, None
    
    lab = parts[0]
    datetime_part = parts[2]
    
    match = re.match(r'(\d{4}-\d{2}-\d{2})-(\d+)h-(\d+)m-(\d+)s', datetime_part)
    if not match:
        return None, None
    
    date = match.group(1)
    hour = match.group(2)
    minute = match.group(3)
    second = match.group(4)
    
    dt = datetime.strptime(f"{date} {hour}:{minute}:{second}", "%Y-%m-%d %H:%M:%S")
    timestamp_folder = dt.strftime("%a_%b_%e_%H:%M:%S_%Y").replace(" ", "_")
    
    success_path = f"{gcs_bucket}/{lab}/success/{date}/{timestamp_folder}/trajectory.h5"
    failure_path = f"{gcs_bucket}/{lab}/failure/{date}/{timestamp_folder}/trajectory.h5"
    
    return success_path, failure_path


def check_episode_outcome(episode_id: str, gcs_bucket: str) -> str:
    """Check if episode is success or failure by querying GCS.
    
    Returns 'success', 'failure', or None if not found.
    """
    success_path, failure_path = episode_id_to_gcs_paths(episode_id, gcs_bucket)
    if not success_path:
        return None
    
    # Check success first (more common)
    try:
        result = subprocess.run(
            f'gsutil -q stat "{success_path}"',
            shell=True,
            capture_output=True,
            timeout=30
        )
        if result.returncode == 0:
            return 'success'
    except subprocess.TimeoutExpired:
        pass
    
    # Check failure
    try:
        result = subprocess.run(
            f'gsutil -q stat "{failure_path}"',
            shell=True,
            capture_output=True,
            timeout=30
        )
        if result.returncode == 0:
            return 'failure'
    except subprocess.TimeoutExpired:
        pass
    
    return None


def get_episode_outcomes_parallel(episode_ids: list, gcs_bucket: str, max_workers: int = 32) -> dict:
    """Get outcomes for multiple episodes in parallel.
    
    Returns dict mapping episode_id -> outcome ('success' or 'failure')
    """
    outcomes = {}
    total = len(episode_ids)
    
    print(f"[INFO] Checking success/failure status for {total} episodes...", file=sys.stderr)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_episode = {
            executor.submit(check_episode_outcome, ep_id, gcs_bucket): ep_id
            for ep_id in episode_ids
        }
        
        completed = 0
        for future in as_completed(future_to_episode):
            episode_id = future_to_episode[future]
            completed += 1
            
            if completed % 100 == 0:
                print(f"[INFO] Checked {completed}/{total} episodes...", file=sys.stderr)
            
            try:
                outcome = future.result()
                if outcome:
                    outcomes[episode_id] = outcome
            except Exception as e:
                print(f"[WARN] Error checking {episode_id}: {e}", file=sys.stderr)
    
    success_count = sum(1 for o in outcomes.values() if o == 'success')
    failure_count = sum(1 for o in outcomes.values() if o == 'failure')
    print(f"[INFO] Found {success_count} success, {failure_count} failure episodes", file=sys.stderr)
    
    return outcomes


def get_episodes_sorted_by_quality(cam2base_path: str, limit: int = -1) -> list:
    """Get episodes sorted by quality, interleaved by lab.
    
    Returns list of episode IDs in processing order.
    """
    data = load_cam2base(cam2base_path)
    
    # Group episodes by lab with quality scores
    labs = defaultdict(list)
    
    for episode_id, episode_data in data.items():
        parsed = parse_episode_key(episode_id)
        if parsed is None:
            continue
        
        quality = get_episode_quality(episode_data)
        labs[parsed['lab']].append({
            'id': episode_id,
            'quality': quality,
            'data': episode_data,
        })
    
    # Sort each lab's episodes by quality (descending)
    for lab in labs:
        labs[lab].sort(key=lambda x: x['quality'], reverse=True)
    
    # Interleave: best from each lab, then second best, etc.
    result = []
    lab_names = sorted(labs.keys())
    max_episodes = max(len(labs[lab]) for lab in lab_names) if lab_names else 0
    
    for rank in range(max_episodes):
        for lab in lab_names:
            if rank < len(labs[lab]):
                result.append(labs[lab][rank]['id'])
    
    # Apply limit
    if limit > 0:
        result = result[:limit]
    
    return result


def get_episodes_balanced(cam2base_path: str, gcs_bucket: str, limit: int = -1) -> list:
    """Get episodes with balanced success/failure per lab.
    
    For each lab, interleaves success and failure episodes:
    lab1_success1, lab1_failure1, lab2_success1, lab2_failure1, ...
    lab1_success2, lab1_failure2, lab2_success2, lab2_failure2, ...
    
    Returns list of episode IDs in processing order.
    """
    data = load_cam2base(cam2base_path)
    
    # Get all episode IDs
    all_episode_ids = list(data.keys())
    print(f"[INFO] Total episodes in cam2base: {len(all_episode_ids)}", file=sys.stderr)
    
    # Query GCS for outcomes (parallel)
    outcomes = get_episode_outcomes_parallel(all_episode_ids, gcs_bucket)
    
    # Group by (lab, outcome) with quality scores
    # Structure: labs[lab]['success'] = [...], labs[lab]['failure'] = [...]
    labs = defaultdict(lambda: {'success': [], 'failure': []})
    
    for episode_id, episode_data in data.items():
        parsed = parse_episode_key(episode_id)
        if parsed is None:
            continue
        
        outcome = outcomes.get(episode_id)
        if outcome is None:
            continue  # Skip episodes we couldn't find
        
        quality = get_episode_quality(episode_data)
        labs[parsed['lab']][outcome].append({
            'id': episode_id,
            'quality': quality,
            'outcome': outcome,
        })
    
    # Sort each lab's episodes by quality (descending) within each outcome
    for lab in labs:
        labs[lab]['success'].sort(key=lambda x: x['quality'], reverse=True)
        labs[lab]['failure'].sort(key=lambda x: x['quality'], reverse=True)
    
    # Print stats per lab
    print(f"\n[INFO] Episodes per lab:", file=sys.stderr)
    for lab in sorted(labs.keys()):
        s_count = len(labs[lab]['success'])
        f_count = len(labs[lab]['failure'])
        print(f"  {lab}: {s_count} success, {f_count} failure", file=sys.stderr)
    print("", file=sys.stderr)
    
    # Interleave: for each rank, for each lab, add success then failure
    # This gives us balanced coverage across labs AND outcomes
    result = []
    lab_names = sorted(labs.keys())
    
    # Find max episodes per outcome per lab
    max_per_outcome = 0
    for lab in lab_names:
        max_per_outcome = max(max_per_outcome, len(labs[lab]['success']), len(labs[lab]['failure']))
    
    for rank in range(max_per_outcome):
        for lab in lab_names:
            # Add success episode at this rank if available
            if rank < len(labs[lab]['success']):
                result.append(labs[lab]['success'][rank]['id'])
            # Add failure episode at this rank if available
            if rank < len(labs[lab]['failure']):
                result.append(labs[lab]['failure'][rank]['id'])
    
    # Apply limit
    if limit > 0:
        result = result[:limit]
    
    # Count final distribution
    final_success = sum(1 for ep in result if outcomes.get(ep) == 'success')
    final_failure = sum(1 for ep in result if outcomes.get(ep) == 'failure')
    print(f"[INFO] Final selection: {final_success} success, {final_failure} failure (total: {len(result)})", file=sys.stderr)
    
    return result


def get_relative_path_for_episode(episode_id: str, cam2base_data: dict) -> str:
    """Construct relative path for episode based on ID.
    
    Returns path like: AUTOLab/success/2023-08-18/Fri_Aug_18_12:01:10_2023
    """
    import re
    from datetime import datetime
    
    parts = episode_id.split('+')
    if len(parts) != 3:
        return None
    
    lab = parts[0]
    datetime_part = parts[2]
    
    # Parse datetime: "2023-08-18-12h-01m-10s"
    match = re.match(r'(\d{4}-\d{2}-\d{2})-(\d+)h-(\d+)m-(\d+)s', datetime_part)
    if not match:
        return None
    
    date = match.group(1)
    hour = match.group(2)
    minute = match.group(3)
    second = match.group(4)
    
    # Reconstruct timestamp folder using ctime-style format
    # DROID uses space-padded days (e.g., "Jul  7" -> "Jul__7" when replacing spaces with underscores)
    dt = datetime.strptime(f"{date} {hour}:{minute}:{second}", "%Y-%m-%d %H:%M:%S")
    timestamp_folder = dt.strftime("%a_%b_%e_%H:%M:%S_%Y").replace(" ", "_")
    
    # We don't know success/failure from just the ID, use success as default
    # The actual scripts will search for both
    return f"{lab}/success/{date}/{timestamp_folder}"


def main():
    parser = argparse.ArgumentParser(
        description="Get episodes sorted by quality from cam2base JSON."
    )
    parser.add_argument(
        "--cam2base",
        default="/data/droid/calib_and_annot/droid/cam2base_extrinsic_superset.json",
        help="Path to cam2base_extrinsic_superset.json",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Maximum number of episodes (-1 for all)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--format",
        choices=["ids", "paths", "json"],
        default="ids",
        help="Output format",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Balance success/failure episodes per lab (requires GCS access)",
    )
    parser.add_argument(
        "--gcs_bucket",
        default="gs://gresearch/robotics/droid_raw/1.0.1",
        help="GCS bucket path (used with --balanced)",
    )
    args = parser.parse_args()
    
    if args.balanced:
        episodes = get_episodes_balanced(args.cam2base, args.gcs_bucket, args.limit)
    else:
        episodes = get_episodes_sorted_by_quality(args.cam2base, args.limit)
    
    if args.format == "ids":
        output = "\n".join(episodes)
    elif args.format == "paths":
        data = load_cam2base(args.cam2base)
        paths = []
        for ep_id in episodes:
            path = get_relative_path_for_episode(ep_id, data)
            if path:
                paths.append(path)
        output = "\n".join(paths)
    elif args.format == "json":
        data = load_cam2base(args.cam2base)
        result = []
        for ep_id in episodes:
            path = get_relative_path_for_episode(ep_id, data)
            result.append({
                "episode_id": ep_id,
                "relative_path": path,
            })
        output = json.dumps(result, indent=2)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output + "\n")
        print(f"Wrote {len(episodes)} episodes to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
