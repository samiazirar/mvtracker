#!/usr/bin/env python3
"""Check episode availability in local data and GCS (no downloading).

This script diagnoses why episodes fail to process by checking:
1. If episode exists in local pre-downloaded data (/data/droid/data/droid_raw/1.0.1/)
2. If episode exists in GCS bucket (optional, requires gsutil)

Usage:
    # Check a single episode
    python check_episode_availability.py --episode_id "ILIAD+7ae1bcff+2023-06-03-19h-32m-29s"
    
    # Check episodes from a file
    python check_episode_availability.py --episodes_file failed_episodes.txt
    
    # Check all episodes from cam2base and output availability report
    python check_episode_availability.py --cam2base /data/cam2base_extrinsic_superset.json --limit 100
    
    # Skip GCS check (faster, local-only)
    python check_episode_availability.py --episodes_file episodes.txt --skip_gcs
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


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


def check_local_path(local_root: str, episode_info: dict) -> Optional[str]:
    """Check if episode exists in local data root.
    
    Returns the full path if found, None otherwise.
    Handles case-insensitive lab name matching (e.g., 'tri' -> 'TRI').
    """
    lab = episode_info['lab']
    date = episode_info['date']
    timestamp = episode_info['timestamp_folder']
    
    # Build list of lab name variants to try (original, upper, title case)
    lab_variants = [lab, lab.upper(), lab.lower(), lab.title()]
    # Also check actual directories for case-insensitive match
    if os.path.isdir(local_root):
        for existing_lab in os.listdir(local_root):
            if existing_lab.lower() == lab.lower():
                lab_variants.append(existing_lab)
    
    # Remove duplicates while preserving order
    seen = set()
    lab_variants = [x for x in lab_variants if not (x in seen or seen.add(x))]
    
    for lab_name in lab_variants:
        for outcome in ['success', 'failure']:
            path = os.path.join(local_root, lab_name, outcome, date, timestamp)
            h5_path = os.path.join(path, 'trajectory.h5')
            if os.path.exists(h5_path):
                return path
    
    return None


def check_gcs_path(gcs_bucket: str, episode_info: dict) -> Optional[str]:
    """Check if episode exists in GCS bucket.
    
    Returns the GCS path if found, None otherwise.
    """
    lab = episode_info['lab']
    date = episode_info['date']
    timestamp = episode_info['timestamp_folder']
    
    for outcome in ['success', 'failure']:
        rel_path = f"{lab}/{outcome}/{date}/{timestamp}"
        gcs_path = f"{gcs_bucket}/{rel_path}"
        
        # Check if trajectory.h5 exists at this path
        check_cmd = f'gsutil -q stat "{gcs_path}/trajectory.h5"'
        try:
            result = subprocess.run(
                check_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                return gcs_path
        except subprocess.TimeoutExpired:
            print(f"[WARN] GCS check timed out for {rel_path}", file=sys.stderr)
        except Exception as e:
            print(f"[WARN] GCS check error: {e}", file=sys.stderr)
    
    return None


def check_episode(
    episode_id: str,
    local_roots: list,
    gcs_bucket: str = None,
    skip_gcs: bool = False
) -> dict:
    """Check availability of a single episode.
    
    Returns dict with:
        - episode_id: str
        - parsed: bool (whether parsing succeeded)
        - local_path: str or None
        - gcs_path: str or None (if checked)
        - status: 'local', 'gcs_only', 'missing', 'parse_error'
    """
    result = {
        'episode_id': episode_id,
        'parsed': False,
        'local_path': None,
        'gcs_path': None,
        'status': 'unknown',
    }
    
    # Parse episode ID
    try:
        episode_info = parse_episode_id(episode_id)
        result['parsed'] = True
        result['lab'] = episode_info['lab']
        result['date'] = episode_info['date']
        result['expected_folder'] = episode_info['timestamp_folder']
    except ValueError as e:
        result['status'] = 'parse_error'
        result['error'] = str(e)
        return result
    
    # Check local paths
    for local_root in local_roots:
        if os.path.isdir(local_root):
            local_path = check_local_path(local_root, episode_info)
            if local_path:
                result['local_path'] = local_path
                result['status'] = 'local'
                break
    
    # Check GCS (if not found locally and not skipped)
    if not result['local_path'] and gcs_bucket and not skip_gcs:
        gcs_path = check_gcs_path(gcs_bucket, episode_info)
        if gcs_path:
            result['gcs_path'] = gcs_path
            result['status'] = 'gcs_only'
    
    # Final status
    if result['local_path']:
        result['status'] = 'local'
    elif result.get('gcs_path'):
        result['status'] = 'gcs_only'
    elif result['parsed']:
        result['status'] = 'missing'
    
    return result


def load_episodes_from_cam2base(cam2base_path: str, limit: int = None) -> list:
    """Load episode IDs from cam2base JSON."""
    with open(cam2base_path, 'r') as f:
        data = json.load(f)
    
    episodes = list(data.keys())
    if limit and limit > 0:
        episodes = episodes[:limit]
    
    return episodes


def main():
    parser = argparse.ArgumentParser(
        description="Check episode availability in local data and GCS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Check single episode
    python check_episode_availability.py --episode_id "ILIAD+7ae1bcff+2023-06-03-19h-32m-29s"
    
    # Check from failed episodes file
    python check_episode_availability.py --episodes_file /data/logs/pipeline_xxx/failed_episodes.txt
    
    # Check all from cam2base (first 100)
    python check_episode_availability.py --cam2base /data/cam2base_extrinsic_superset.json --limit 100
        """
    )
    
    # Input sources (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--episode_id",
        help="Single episode ID to check"
    )
    input_group.add_argument(
        "--episodes_file",
        help="File with episode IDs (one per line, or CSV with episode_id in first column)"
    )
    input_group.add_argument(
        "--cam2base",
        help="Path to cam2base JSON to get all episode IDs"
    )
    
    # Options
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of episodes to check (for --cam2base)"
    )
    parser.add_argument(
        "--local_roots",
        nargs='+',
        default=["/data/droid/data/droid_raw/1.0.1"],
        help="Local data roots to search (default: /data/droid/data/droid_raw/1.0.1)"
    )
    parser.add_argument(
        "--gcs_bucket",
        default="gs://gresearch/robotics/droid_raw/1.0.1",
        help="GCS bucket path"
    )
    parser.add_argument(
        "--skip_gcs",
        action="store_true",
        help="Skip GCS checks (faster, local-only)"
    )
    parser.add_argument(
        "--output",
        help="Output file for results (CSV format)"
    )
    parser.add_argument(
        "--output_missing",
        help="Output file for missing episodes only (one per line)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Collect episodes to check
    episodes = []
    
    if args.episode_id:
        episodes = [args.episode_id]
    elif args.episodes_file:
        with open(args.episodes_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                # Handle CSV format (episode_id,step)
                if ',' in line:
                    episode_id = line.split(',')[0].strip()
                else:
                    episode_id = line
                if episode_id:
                    episodes.append(episode_id)
    elif args.cam2base:
        episodes = load_episodes_from_cam2base(args.cam2base, args.limit)
    
    print(f"Checking {len(episodes)} episodes...")
    print(f"Local roots: {args.local_roots}")
    if not args.skip_gcs:
        print(f"GCS bucket: {args.gcs_bucket}")
    else:
        print("GCS check: SKIPPED")
    print()
    
    # Check each episode
    results = []
    stats = {'local': 0, 'gcs_only': 0, 'missing': 0, 'parse_error': 0}
    
    for i, episode_id in enumerate(episodes):
        result = check_episode(
            episode_id,
            args.local_roots,
            args.gcs_bucket if not args.skip_gcs else None,
            args.skip_gcs
        )
        results.append(result)
        stats[result['status']] = stats.get(result['status'], 0) + 1
        
        # Progress
        if args.verbose or (i + 1) % 10 == 0:
            status_char = {
                'local': '✓',
                'gcs_only': 'G',
                'missing': '✗',
                'parse_error': '?'
            }.get(result['status'], '?')
            
            if args.verbose:
                print(f"[{i+1}/{len(episodes)}] {status_char} {episode_id}")
                if result['local_path']:
                    print(f"           -> {result['local_path']}")
                elif result['gcs_path']:
                    print(f"           -> {result['gcs_path']} (GCS only)")
                elif result['status'] == 'missing':
                    print(f"           -> NOT FOUND (expected: {result.get('expected_folder', '?')})")
            else:
                print(f"\r[{i+1}/{len(episodes)}] local:{stats['local']} gcs:{stats['gcs_only']} missing:{stats['missing']} errors:{stats['parse_error']}", end='')
    
    print()
    print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total episodes checked: {len(episodes)}")
    print(f"  Found locally:        {stats['local']} ({100*stats['local']/len(episodes):.1f}%)")
    print(f"  GCS only:             {stats['gcs_only']} ({100*stats['gcs_only']/len(episodes):.1f}%)")
    print(f"  Missing (not found):  {stats['missing']} ({100*stats['missing']/len(episodes):.1f}%)")
    print(f"  Parse errors:         {stats['parse_error']} ({100*stats['parse_error']/len(episodes):.1f}%)")
    print()
    
    # Show missing episodes
    missing = [r for r in results if r['status'] == 'missing']
    if missing:
        print("=" * 60)
        print(f"MISSING EPISODES ({len(missing)})")
        print("=" * 60)
        for r in missing[:20]:  # Show first 20
            print(f"  {r['episode_id']}")
            print(f"    Lab: {r.get('lab', '?')}, Date: {r.get('date', '?')}")
            print(f"    Expected folder: {r.get('expected_folder', '?')}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
        print()
    
    # Show parse errors
    errors = [r for r in results if r['status'] == 'parse_error']
    if errors:
        print("=" * 60)
        print(f"PARSE ERRORS ({len(errors)})")
        print("=" * 60)
        for r in errors[:10]:
            print(f"  {r['episode_id']}: {r.get('error', '?')}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
        print()
    
    # Write output files
    if args.output:
        with open(args.output, 'w') as f:
            f.write("episode_id,status,lab,date,local_path,gcs_path\n")
            for r in results:
                f.write(f"{r['episode_id']},{r['status']},{r.get('lab','')},{r.get('date','')},{r.get('local_path','')},{r.get('gcs_path','')}\n")
        print(f"Results written to: {args.output}")
    
    if args.output_missing:
        with open(args.output_missing, 'w') as f:
            for r in results:
                if r['status'] == 'missing':
                    f.write(f"{r['episode_id']}\n")
        print(f"Missing episodes written to: {args.output_missing}")
    
    # Exit code based on missing count
    if stats['missing'] > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
