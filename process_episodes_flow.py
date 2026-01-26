#!/usr/bin/env python3
"""Batch process DROID episodes: generate tracks and render flow videos.

This script:
1. Finds episodes in the DROID dataset
2. Generates tracks.npz with contact flow data (if not already present)
3. Renders flow overlay videos for each camera

Usage:
    python process_episodes_flow.py --max_episodes 10
    python process_episodes_flow.py --labs AUTOLab IRIS --max_per_lab 5
"""

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path


def find_droid_episodes(droid_path: str, labs: list = None, max_per_lab: int = None):
    """Find DROID episodes with trajectory.h5 files."""
    episodes = []
    
    # Find all trajectory.h5 files
    pattern = os.path.join(droid_path, "*", "*", "*", "*", "trajectory.h5")
    h5_files = glob.glob(pattern)
    
    lab_counts = {}
    
    for h5_path in sorted(h5_files):
        parts = h5_path.split(os.sep)
        # Extract: lab/outcome/date/timestamp
        idx = parts.index("1.0.1") if "1.0.1" in parts else -5
        if idx < 0:
            continue
            
        lab = parts[idx + 1]
        outcome = parts[idx + 2]
        date = parts[idx + 3]
        timestamp = parts[idx + 4]
        
        # Filter by labs if specified
        if labs and lab not in labs:
            continue
        
        # Check max per lab
        if max_per_lab:
            if lab_counts.get(lab, 0) >= max_per_lab:
                continue
            lab_counts[lab] = lab_counts.get(lab, 0) + 1
        
        # Parse timestamp to episode_id format
        # Wed_Jul_12_11:15:03_2023 -> 2023-07-12-11h-15m-03s
        try:
            from datetime import datetime
            # Handle format like "Wed_Jul_12_11:15:03_2023"
            dt = datetime.strptime(timestamp, "%a_%b_%d_%H:%M:%S_%Y")
            episode_id = f"{lab}+84bd5053+{dt.strftime('%Y-%m-%d-%Hh-%Mm-%Ss')}"
        except:
            continue
        
        episode_dir = os.path.dirname(h5_path)
        video_dir = os.path.join(episode_dir, "recordings", "MP4")
        
        if os.path.exists(video_dir):
            mp4_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4') and '-stereo' not in f]
            if mp4_files:
                episodes.append({
                    'episode_id': episode_id,
                    'episode_dir': episode_dir,
                    'video_dir': video_dir,
                    'lab': lab,
                    'outcome': outcome,
                    'date': date,
                    'timestamp': timestamp,
                    'rel_path': f"{lab}/{outcome}/{date}/{timestamp}",
                })
    
    return episodes


def generate_tracks(episode_id: str, config_path: str = None):
    """Generate tracks.npz for an episode."""
    cmd = [
        "python", "/workspace/conversions/droid/training_data/generate_tracks_and_metadata.py",
        "--episode_id", episode_id,
    ]
    if config_path:
        cmd.extend(["--config", config_path])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stdout + result.stderr


def render_flow_videos(tracks_path: str, video_dir: str, output_dir: str, 
                       max_frames: int = None, trail_length: int = 10):
    """Render flow overlay videos for an episode."""
    # Import the render module
    sys.path.insert(0, '/workspace')
    from render_flow_videos import render_episode_videos
    
    outputs = render_episode_videos(
        tracks_path,
        video_dir, 
        output_dir,
        max_frames=max_frames,
        trail_length=trail_length,
    )
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Batch process DROID episodes for flow visualization")
    parser.add_argument("--droid_path", default="/data/droid/data/droid_raw/1.0.1",
                        help="Path to DROID dataset")
    parser.add_argument("--tracks_dir", default="/workspace/test_data",
                        help="Output directory for tracks.npz files")
    parser.add_argument("--video_dir", default="/workspace/output_videos",
                        help="Output directory for flow videos")
    parser.add_argument("--config", default="/workspace/conversions/droid/training_data/config.yaml",
                        help="Config file for track generation")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Maximum total episodes to process")
    parser.add_argument("--labs", nargs="+", default=None,
                        help="Filter by lab names (e.g., AUTOLab IRIS)")
    parser.add_argument("--max_per_lab", type=int, default=None,
                        help="Maximum episodes per lab")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Maximum frames per video")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip episodes with existing tracks.npz")
    parser.add_argument("--regenerate", action="store_true",
                        help="Regenerate tracks even if they exist")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DROID Flow Video Pipeline")
    print("=" * 60)
    
    # Find episodes
    episodes = find_droid_episodes(
        args.droid_path,
        labs=args.labs,
        max_per_lab=args.max_per_lab,
    )
    
    if args.max_episodes:
        episodes = episodes[:args.max_episodes]
    
    print(f"\nFound {len(episodes)} episodes to process\n")
    
    if not episodes:
        print("[ERROR] No episodes found!")
        return 1
    
    os.makedirs(args.tracks_dir, exist_ok=True)
    os.makedirs(args.video_dir, exist_ok=True)
    
    total_videos = 0
    
    for i, ep in enumerate(episodes, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(episodes)}] {ep['episode_id']}")
        print(f"  Path: {ep['rel_path']}")
        print(f"{'='*60}")
        
        # Check for existing tracks
        tracks_output_dir = os.path.join(args.tracks_dir, ep['rel_path'])
        tracks_path = os.path.join(tracks_output_dir, "tracks.npz")
        
        need_tracks = True
        if os.path.exists(tracks_path):
            if args.skip_existing:
                print(f"  [SKIP] tracks.npz exists")
                need_tracks = False
            elif not args.regenerate:
                # Check if tracks have correct units
                import numpy as np
                t = np.load(tracks_path)
                x_range = t['tracks_3d'][0,:,0].max() - t['tracks_3d'][0,:,0].min()
                if x_range < 1.0:
                    print(f"  [OK] tracks.npz exists with correct units")
                    need_tracks = False
                else:
                    print(f"  [REGEN] tracks.npz has wrong units, regenerating...")
        
        # Generate tracks
        if need_tracks:
            print(f"  Generating tracks...")
            success, output = generate_tracks(ep['episode_id'], args.config)
            if not success:
                print(f"  [ERROR] Track generation failed")
                print(output[-500:] if len(output) > 500 else output)
                continue
            print(f"  [OK] Tracks generated")
        
        # Render videos
        if not os.path.exists(tracks_path):
            print(f"  [ERROR] No tracks.npz found after generation")
            continue
        
        video_output_dir = os.path.join(args.video_dir, ep['timestamp'])
        print(f"  Rendering flow videos...")
        
        try:
            outputs = render_flow_videos(
                tracks_path,
                ep['video_dir'],
                video_output_dir,
                max_frames=args.max_frames,
            )
            total_videos += len(outputs)
            print(f"  [OK] Rendered {len(outputs)} videos")
        except Exception as e:
            print(f"  [ERROR] Video rendering failed: {e}")
    
    print(f"\n{'='*60}")
    print(f"[DONE] Processed {len(episodes)} episodes, rendered {total_videos} videos")
    print(f"  Tracks: {args.tracks_dir}")
    print(f"  Videos: {args.video_dir}")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
