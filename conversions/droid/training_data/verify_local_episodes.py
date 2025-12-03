#!/usr/bin/env python3
"""Verify which episodes from cam2base extrinsics are available locally.

Usage:
    python verify_local_episodes.py --cam2base /data/cam2base_extrinsic_superset.json
    python verify_local_episodes.py --cam2base /data/cam2base_extrinsic_superset.json --local_source /data/droid/data/droid_raw/1.0.1
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime


def parse_episode_id(episode_id: str) -> dict:
    """Parse episode ID into components."""
    parts = episode_id.split('+')
    if len(parts) != 3:
        return None
    
    lab = parts[0]
    episode_hash = parts[1]
    datetime_part = parts[2]
    
    match = re.match(r'(\d{4}-\d{2}-\d{2})-(\d+)h-(\d+)m-(\d+)s', datetime_part)
    if not match:
        return None
    
    date = match.group(1)
    hour = match.group(2)
    minute = match.group(3)
    second = match.group(4)
    
    dt = datetime.strptime(f"{date} {hour}:{minute}:{second}", "%Y-%m-%d %H:%M:%S")
    timestamp_folder = dt.strftime("%a_%b_%e_%H:%M:%S_%Y").replace(" ", "_")
    
    return {
        'lab': lab,
        'hash': episode_hash,
        'date': date,
        'timestamp_folder': timestamp_folder,
        'full_id': episode_id,
    }


def find_episode_local(local_source: str, episode_info: dict) -> tuple:
    """Check if episode exists locally.
    
    Returns:
        tuple: (found: bool, path_or_reason: str)
    """
    lab = episode_info['lab']
    date = episode_info['date']
    timestamp_folder = episode_info['timestamp_folder']
    
    # Try both success and failure paths
    for outcome in ['success', 'failure']:
        source_path = os.path.join(local_source, lab, outcome, date, timestamp_folder)
        h5_path = os.path.join(source_path, 'trajectory.h5')
        
        if os.path.exists(h5_path):
            return True, source_path
    
    # Not found - figure out why
    lab_path = os.path.join(local_source, lab)
    if not os.path.exists(lab_path):
        return False, f"Lab '{lab}' not found"
    
    for outcome in ['success', 'failure']:
        date_path = os.path.join(lab_path, outcome, date)
        if os.path.exists(date_path):
            # Date exists, but timestamp doesn't match
            timestamps = os.listdir(date_path)
            # Look for similar timestamps
            similar = [t for t in timestamps if timestamp_folder[:15] in t]
            if similar:
                return False, f"Similar timestamps: {similar[:3]}"
            return False, f"Date exists but timestamp '{timestamp_folder}' not found"
    
    return False, f"Date '{date}' not found for lab '{lab}'"


def main():
    parser = argparse.ArgumentParser(description="Verify local episode availability")
    parser.add_argument(
        "--cam2base",
        required=True,
        help="Path to cam2base_extrinsic_superset.json",
    )
    parser.add_argument(
        "--local_source",
        default="/data/droid/data/droid_raw/1.0.1",
        help="Local DROID data root",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file for missing episodes (optional)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Limit number of episodes to check (-1 for all)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show details for each episode",
    )
    args = parser.parse_args()
    
    # Load cam2base
    print(f"Loading {args.cam2base}...")
    with open(args.cam2base, 'r') as f:
        cam2base = json.load(f)
    
    episodes = list(cam2base.keys())
    if args.limit > 0:
        episodes = episodes[:args.limit]
    
    print(f"Checking {len(episodes)} episodes against {args.local_source}...")
    print()
    
    # Track results
    found = []
    missing = []
    missing_reasons = defaultdict(list)
    lab_stats = defaultdict(lambda: {'found': 0, 'missing': 0})
    
    for i, episode_id in enumerate(episodes):
        info = parse_episode_id(episode_id)
        if not info:
            missing.append((episode_id, "Invalid episode ID format"))
            continue
        
        exists, path_or_reason = find_episode_local(args.local_source, info)
        
        if exists:
            found.append(episode_id)
            lab_stats[info['lab']]['found'] += 1
            if args.verbose:
                print(f"[OK] {episode_id}")
        else:
            missing.append((episode_id, path_or_reason))
            missing_reasons[path_or_reason].append(episode_id)
            lab_stats[info['lab']]['missing'] += 1
            if args.verbose:
                print(f"[MISSING] {episode_id}: {path_or_reason}")
        
        # Progress
        if (i + 1) % 1000 == 0:
            print(f"  Checked {i + 1}/{len(episodes)}...")
    
    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total episodes: {len(episodes)}")
    print(f"Found locally:  {len(found)} ({100*len(found)/len(episodes):.1f}%)")
    print(f"Missing:        {len(missing)} ({100*len(missing)/len(episodes):.1f}%)")
    print()
    
    # Per-lab stats
    print("Per-lab breakdown:")
    print("-" * 40)
    for lab in sorted(lab_stats.keys()):
        stats = lab_stats[lab]
        total = stats['found'] + stats['missing']
        pct = 100 * stats['found'] / total if total > 0 else 0
        print(f"  {lab:15s}: {stats['found']:5d}/{total:5d} ({pct:5.1f}%) found")
    print()
    
    # Missing reasons summary
    if missing:
        print("Missing episode reasons:")
        print("-" * 40)
        for reason, eps in sorted(missing_reasons.items(), key=lambda x: -len(x[1])):
            print(f"  {len(eps):5d} episodes: {reason}")
            if args.verbose and len(eps) <= 5:
                for ep in eps:
                    print(f"         - {ep}")
        print()
    
    # Write missing episodes to file
    if args.output and missing:
        with open(args.output, 'w') as f:
            for ep, reason in missing:
                f.write(f"{ep}\t{reason}\n")
        print(f"Missing episodes written to: {args.output}")
    
    # Exit with error if any missing
    if missing:
        print(f"\n[ERROR] {len(missing)} episodes not found locally!")
        sys.exit(1)
    else:
        print("\n[SUCCESS] All episodes found locally!")
        sys.exit(0)


if __name__ == "__main__":
    main()


