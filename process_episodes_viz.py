#!/usr/bin/env python3
"""
Process multiple DROID episodes: generate tracks and create visualizations.

This script:
1. Finds episodes in DROID dataset
2. Generates tracks.npz with normalized and unnormalized tracks
3. Creates Rerun visualizations for each episode

Usage:
    python process_episodes_viz.py --num_episodes 5
    python process_episodes_viz.py --episode_ids "AUTOLab+84bd5053+2023-07-12-12h-26m-46s,IRIS+f94b8622+2023-12-13-13h-05m-52s"
"""

import argparse
import os
import sys
import subprocess
import glob
from pathlib import Path


def find_episodes(droid_root: str, max_count: int = 10):
    """Find episode IDs from DROID dataset."""
    episodes = []
    
    for traj_h5 in glob.glob(f"{droid_root}/**/trajectory.h5", recursive=True):
        if len(episodes) >= max_count:
            break
            
        episode_dir = os.path.dirname(traj_h5)
        metadata_files = glob.glob(f"{episode_dir}/metadata_*.json")
        
        if metadata_files:
            meta_name = os.path.basename(metadata_files[0])
            episode_id = meta_name.replace("metadata_", "").replace(".json", "")
            episodes.append(episode_id)
    
    return episodes


def generate_tracks(episode_id: str):
    """Generate tracks.npz for an episode."""
    cmd = [
        "python", "/workspace/conversions/droid/training_data/generate_tracks_and_metadata.py",
        "--episode_id", episode_id
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"[ERROR] Failed to generate tracks for {episode_id}")
        print(result.stderr)
        return None
    
    # Find the generated tracks file
    output_lines = result.stdout.split("\n")
    for line in output_lines:
        if "Saved:" in line and "tracks.npz" in line:
            path = line.split("Saved:")[1].strip()
            return path
    
    return None


def create_visualization(tracks_path: str, output_path: str):
    """Create visualization from tracks.npz."""
    cmd = [
        "python", "/workspace/visualize_tracks_normalized.py",
        "--tracks_path", tracks_path,
        "--output_path", output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"[ERROR] Failed to create visualization for {tracks_path}")
        print(result.stderr)
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Process DROID episodes")
    parser.add_argument("--droid_root", type=str, 
                        default="/data/droid/data/droid_raw/1.0.1",
                        help="Path to DROID dataset root")
    parser.add_argument("--num_episodes", type=int, default=5,
                        help="Number of episodes to process")
    parser.add_argument("--episode_ids", type=str, default=None,
                        help="Comma-separated list of specific episode IDs")
    parser.add_argument("--output_dir", type=str, default="visualizations/episodes",
                        help="Output directory for visualizations")
    args = parser.parse_args()
    
    # Get episode list
    if args.episode_ids:
        episodes = args.episode_ids.split(",")
    else:
        print(f"Finding {args.num_episodes} episodes in {args.droid_root}...")
        episodes = find_episodes(args.droid_root, args.num_episodes)
    
    print(f"Found {len(episodes)} episodes to process")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = []
    
    for i, episode_id in enumerate(episodes):
        print(f"\n[{i+1}/{len(episodes)}] Processing: {episode_id}")
        
        # Generate tracks
        print("  Generating tracks...")
        tracks_path = generate_tracks(episode_id)
        
        if tracks_path is None:
            results.append((episode_id, False, "Track generation failed"))
            continue
        
        # Make path absolute
        if not tracks_path.startswith("/"):
            tracks_path = f"/workspace/{tracks_path}"
        
        # Create visualization
        print("  Creating visualization...")
        safe_name = episode_id.replace("+", "_").replace("-", "_")
        output_path = f"{args.output_dir}/{safe_name}_tracks.rrd"
        
        success = create_visualization(tracks_path, output_path)
        
        if success:
            results.append((episode_id, True, output_path))
            print(f"  ✓ Saved: {output_path}")
        else:
            results.append((episode_id, False, "Visualization failed"))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    successful = sum(1 for _, success, _ in results if success)
    print(f"Successfully processed: {successful}/{len(results)} episodes\n")
    
    for episode_id, success, info in results:
        status = "✓" if success else "✗"
        print(f"  {status} {episode_id}")
        if success:
            print(f"      -> {info}")
        else:
            print(f"      Error: {info}")


if __name__ == "__main__":
    main()
