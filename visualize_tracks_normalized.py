#!/usr/bin/env python3
"""
Visualize normalized and unnormalized tracks from DROID track data.

This script creates a Rerun visualization showing:
1. Unnormalized tracks (raw frame-based contact points)
2. Normalized tracks (distance-based resampled at 1mm steps)

Usage:
    python visualize_tracks_normalized.py --tracks_path /path/to/tracks.npz
"""

import argparse
import numpy as np
import rerun as rr
from pathlib import Path


def visualize_tracks(tracks_path: str, output_path: str = None, spawn: bool = False):
    """Visualize normalized and unnormalized tracks from tracks.npz file."""
    
    # Load track data
    data = np.load(tracks_path, allow_pickle=True)
    
    # Extract unnormalized (raw) data
    tracks_3d = data['tracks_3d']  # [T, N, 3]
    contact_centroids = data['contact_centroids']  # [T, 3]
    contact_frames = data['contact_frames']  # [T, 4, 4]
    left_frames = data['left_contact_frames']  # [T, 4, 4]
    right_frames = data['right_contact_frames']  # [T, 4, 4]
    gripper_poses = data['gripper_poses']  # [T, 4, 4]
    gripper_positions = data['gripper_positions']  # [T]
    
    # Extract normalized data
    norm_centroids = data['normalized_centroids']  # [M, 3]
    norm_frames = data['normalized_frames']  # [M, 4, 4]
    norm_tracks = data['normalized_tracks_3d']  # [M, N, 3]
    norm_left = data['normalized_left_frames']  # [M, 4, 4]
    norm_right = data['normalized_right_frames']  # [M, 4, 4]
    
    # Mapping
    frame_to_norm_idx = data['frame_to_normalized_idx']  # [T]
    cumulative_dist = data['cumulative_distance_mm']  # [T]
    
    # Metadata
    num_frames = int(data['num_frames'])
    num_norm_steps = int(data['num_normalized_steps'])
    fps = float(data['fps'])
    step_size_mm = float(data['normalized_step_size_mm'])
    
    print(f"=== Track Visualization ===")
    print(f"Unnormalized: {num_frames} frames, {tracks_3d.shape[1]} track points")
    print(f"Normalized: {num_norm_steps} steps at {step_size_mm}mm intervals")
    print(f"Total distance: {cumulative_dist[-1]:.1f}mm")
    
    # Initialize Rerun
    rr.init("track_visualization", spawn=spawn)
    if output_path:
        rr.save(output_path)
        print(f"Saving to: {output_path}")
    
    # Set up view
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    
    # Color maps
    def get_track_colors(num_tracks):
        """Generate colors for track points."""
        colors = np.zeros((num_tracks, 4), dtype=np.uint8)
        for i in range(num_tracks):
            t = i / max(num_tracks - 1, 1)
            # Rainbow gradient
            if t < 0.5:
                colors[i] = [int(255 * (1 - 2*t)), int(255 * 2*t), 0, 255]
            else:
                colors[i] = [0, int(255 * (2 - 2*t)), int(255 * (2*t - 1)), 255]
        return colors
    
    num_track_points = tracks_3d.shape[1]
    track_colors = get_track_colors(num_track_points)
    
    # Log static normalized trajectory overview
    rr.log("normalized/full_trajectory", rr.LineStrips3D(
        [norm_centroids],
        colors=[[255, 165, 0, 128]],  # Orange
        radii=[0.001]
    ), static=True)
    
    rr.log("unnormalized/full_trajectory", rr.LineStrips3D(
        [contact_centroids],
        colors=[[0, 191, 255, 128]],  # Deep sky blue
        radii=[0.001]
    ), static=True)
    
    # Animate through frames (unnormalized timeline)
    for t in range(num_frames):
        rr.set_time("frame", duration=t / fps)
        rr.set_time("frame_idx", sequence=t)
        
        # Current distance and normalized index
        dist = cumulative_dist[t]
        norm_idx = frame_to_norm_idx[t]
        
        rr.set_time("distance_mm", sequence=int(dist))
        
        # Log metrics as scalars  
        rr.log("metrics/cumulative_distance_mm", rr.Scalars([dist]))
        rr.log("metrics/gripper_position", rr.Scalars([gripper_positions[t]]))
        rr.log("metrics/normalized_step", rr.Scalars([float(norm_idx)]))
        
        # === UNNORMALIZED VISUALIZATION ===
        # Current contact points
        current_points = tracks_3d[t]  # [N, 3]
        rr.log("unnormalized/contact_points", rr.Points3D(
            current_points,
            colors=track_colors,
            radii=0.002
        ))
        
        # Centroid
        rr.log("unnormalized/centroid", rr.Points3D(
            [contact_centroids[t]],
            colors=[[0, 191, 255, 255]],  # Deep sky blue
            radii=0.008
        ))
        
        # Contact frame axes (orientation)
        frame = contact_frames[t]
        origin = frame[:3, 3]
        x_axis = frame[:3, 0] * 0.03
        y_axis = frame[:3, 1] * 0.03
        z_axis = frame[:3, 2] * 0.03
        
        rr.log("unnormalized/frame_axes", rr.Arrows3D(
            origins=[origin, origin, origin],
            vectors=[x_axis, y_axis, z_axis],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            radii=0.002
        ))
        
        # Left/right finger positions
        left_pos = left_frames[t][:3, 3]
        right_pos = right_frames[t][:3, 3]
        rr.log("unnormalized/finger_positions", rr.Points3D(
            [left_pos, right_pos],
            colors=[[255, 100, 100], [100, 100, 255]],  # Red-ish, Blue-ish
            radii=0.005
        ))
        
        # Gripper pose
        ee_pos = gripper_poses[t][:3, 3]
        rr.log("unnormalized/gripper_ee", rr.Points3D(
            [ee_pos],
            colors=[[200, 200, 200]],
            radii=0.01
        ))
        
        # Trail (last 20 frames)
        trail_start = max(0, t - 20)
        if trail_start < t:
            trail_points = contact_centroids[trail_start:t+1]
            rr.log("unnormalized/centroid_trail", rr.LineStrips3D(
                [trail_points],
                colors=[[0, 191, 255, 150]],
                radii=0.002
            ))
        
        # === NORMALIZED VISUALIZATION ===
        # Show current normalized position
        if norm_idx < len(norm_centroids):
            # Current normalized centroid
            rr.log("normalized/centroid", rr.Points3D(
                [norm_centroids[norm_idx]],
                colors=[[255, 165, 0, 255]],  # Orange
                radii=0.008
            ))
            
            # Current normalized contact points  
            norm_points = norm_tracks[norm_idx]  # [N, 3]
            rr.log("normalized/contact_points", rr.Points3D(
                norm_points,
                colors=track_colors,
                radii=0.002
            ))
            
            # Normalized frame axes
            n_frame = norm_frames[norm_idx]
            n_origin = n_frame[:3, 3]
            n_x = n_frame[:3, 0] * 0.03
            n_y = n_frame[:3, 1] * 0.03
            n_z = n_frame[:3, 2] * 0.03
            
            rr.log("normalized/frame_axes", rr.Arrows3D(
                origins=[n_origin, n_origin, n_origin],
                vectors=[n_x, n_y, n_z],
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                radii=0.002
            ))
            
            # Trail (last N normalized steps)
            trail_n_start = max(0, norm_idx - 50)
            if trail_n_start < norm_idx:
                trail_n = norm_centroids[trail_n_start:norm_idx+1]
                rr.log("normalized/centroid_trail", rr.LineStrips3D(
                    [trail_n],
                    colors=[[255, 165, 0, 150]],
                    radii=0.002
                ))
    
    print(f"Visualization complete: {num_frames} frames logged")


def main():
    parser = argparse.ArgumentParser(description="Visualize normalized and unnormalized tracks")
    parser.add_argument("--tracks_path", type=str, required=True,
                        help="Path to tracks.npz file")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output .rrd file path (optional)")
    parser.add_argument("--spawn", action="store_true",
                        help="Spawn Rerun viewer")
    args = parser.parse_args()
    
    if args.output_path is None:
        tracks_file = Path(args.tracks_path)
        args.output_path = f"visualizations/{tracks_file.stem}_viz.rrd"
    
    visualize_tracks(args.tracks_path, args.output_path, args.spawn)


if __name__ == "__main__":
    main()
