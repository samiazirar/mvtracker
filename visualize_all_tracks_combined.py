#!/usr/bin/env python3
"""
Create a combined visualization of all episode track data.

Combines multiple tracks.npz files into a single Rerun visualization
showing both normalized and unnormalized tracks for comparison.

Usage:
    python visualize_all_tracks_combined.py --tracks_dir /workspace/test_data
"""

import argparse
import numpy as np
import rerun as rr
from pathlib import Path
import glob


def get_episode_color(episode_idx: int, total: int):
    """Generate distinct color for each episode."""
    hue = episode_idx / max(total, 1)
    # HSV to RGB (simplified)
    r, g, b = 0, 0, 0
    i = int(hue * 6)
    f = hue * 6 - i
    
    if i == 0: r, g, b = 1, f, 0
    elif i == 1: r, g, b = 1-f, 1, 0
    elif i == 2: r, g, b = 0, 1, f
    elif i == 3: r, g, b = 0, 1-f, 1
    elif i == 4: r, g, b = f, 0, 1
    else: r, g, b = 1, 0, 1-f
    
    return [int(r*255), int(g*255), int(b*255), 255]


def main():
    parser = argparse.ArgumentParser(description="Combined track visualization")
    parser.add_argument("--tracks_dir", type=str, default="/workspace/test_data",
                        help="Directory containing tracks.npz files")
    parser.add_argument("--output_path", type=str, 
                        default="/workspace/visualizations/all_tracks_combined.rrd",
                        help="Output RRD file")
    parser.add_argument("--max_episodes", type=int, default=10,
                        help="Maximum number of episodes to include")
    args = parser.parse_args()
    
    # Find all tracks.npz files
    tracks_files = list(Path(args.tracks_dir).rglob("tracks.npz"))
    tracks_files = sorted(tracks_files)[:args.max_episodes]
    
    print(f"Found {len(tracks_files)} track files")
    
    # Initialize Rerun
    rr.init("all_tracks_combined", spawn=False)
    rr.save(args.output_path)
    print(f"Saving to: {args.output_path}")
    
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    
    # Load all episode data
    episodes_data = []
    for i, tracks_file in enumerate(tracks_files):
        try:
            data = np.load(tracks_file, allow_pickle=True)
            episodes_data.append({
                'path': str(tracks_file),
                'centroids': data['contact_centroids'],  # [T, 3]
                'norm_centroids': data['normalized_centroids'],  # [M, 3]
                'tracks': data['tracks_3d'],  # [T, N, 3]
                'norm_tracks': data['normalized_tracks_3d'],  # [M, N, 3]
                'frame_to_norm': data['frame_to_normalized_idx'],  # [T]
                'num_frames': int(data['num_frames']),
                'num_norm_steps': int(data['num_normalized_steps']),
                'fps': float(data['fps']),
                'color': get_episode_color(i, len(tracks_files))
            })
            print(f"  [{i+1}] {tracks_file.parent.name}: {data['num_frames']} frames, "
                  f"{data['num_normalized_steps']} norm steps")
        except Exception as e:
            print(f"  [ERROR] Failed to load {tracks_file}: {e}")
    
    # Log static full trajectories for all episodes
    print("\nLogging static trajectories...")
    for i, ep in enumerate(episodes_data):
        # Unnormalized trajectory
        rr.log(f"episodes/ep_{i}/unnormalized/full_trajectory", rr.LineStrips3D(
            [ep['centroids']],
            colors=[ep['color'][:3] + [128]],
            radii=[0.002]
        ), static=True)
        
        # Normalized trajectory
        rr.log(f"episodes/ep_{i}/normalized/full_trajectory", rr.LineStrips3D(
            [ep['norm_centroids']],
            colors=[ep['color'][:3] + [200]],
            radii=[0.002]
        ), static=True)
    
    # Find max frames across all episodes
    max_frames = max(ep['num_frames'] for ep in episodes_data)
    common_fps = episodes_data[0]['fps'] if episodes_data else 30.0
    
    # Animate through common timeline
    print(f"Animating {max_frames} frames...")
    for t in range(max_frames):
        rr.set_time("frame", duration=t / common_fps)
        rr.set_time("frame_idx", sequence=t)
        
        for i, ep in enumerate(episodes_data):
            if t < ep['num_frames']:
                # Current unnormalized centroid
                rr.log(f"episodes/ep_{i}/unnormalized/centroid", rr.Points3D(
                    [ep['centroids'][t]],
                    colors=[ep['color']],
                    radii=0.006
                ))
                
                # Corresponding normalized centroid
                norm_idx = min(int(ep['frame_to_norm'][t]), ep['num_norm_steps'] - 1)
                rr.log(f"episodes/ep_{i}/normalized/centroid", rr.Points3D(
                    [ep['norm_centroids'][norm_idx]],
                    colors=[ep['color']],
                    radii=0.006
                ))
                
                # Sample of track points (every 100th point to reduce data)
                if ep['tracks'].shape[1] > 0:
                    sample_points = ep['tracks'][t, ::100]  # Every 100th point
                    rr.log(f"episodes/ep_{i}/unnormalized/sample_tracks", rr.Points3D(
                        sample_points,
                        colors=[ep['color']],
                        radii=0.002
                    ))
    
    print(f"\n✓ Combined visualization saved to: {args.output_path}")
    print(f"  Episodes: {len(episodes_data)}")
    print(f"  Frames: {max_frames}")


if __name__ == "__main__":
    main()
