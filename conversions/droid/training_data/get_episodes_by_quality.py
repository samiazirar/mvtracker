"""Get episodes sorted by quality per lab from cam2base_extrinsic_superset.json.

This script reads the calibration JSON and outputs episodes sorted by quality,
interleaved across labs (best from each lab, then second best from each, etc.)

Usage:
    python get_episodes_by_quality.py --limit 100
    python get_episodes_by_quality.py --limit -1  # All episodes
    python get_episodes_by_quality.py --output episodes.txt
"""

import argparse
import json
import os
import sys
from collections import defaultdict


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
    args = parser.parse_args()
    
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
