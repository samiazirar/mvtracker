#!/usr/bin/env python3
"""
Visualize DROID training shards with Rerun.
This script works with pre-processed training shards and doesn't require ZED SDK.

Usage:
    python visualize_training_shards.py --shard_path /data/droid/training_shards/shard_0000.tar --num_samples 5 --output_dir visualizations
"""

import argparse
import json
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rerun as rr
import torch


def load_sample_from_tar(
    tar_path: str, sample_id: str, temp_dir: str
) -> Tuple[Dict, np.ndarray, Optional[np.ndarray]]:
    """Load a sample (npz, json, mp4) from a tar archive."""
    with tarfile.open(tar_path, "r") as tar:
        # Extract files
        npz_name = f"{sample_id}.npz"
        json_name = f"{sample_id}.json"
        mp4_name = f"{sample_id}.mp4"

        for name in [npz_name, json_name, mp4_name]:
            try:
                tar.extract(name, temp_dir)
            except KeyError:
                print(f"Warning: {name} not found in archive")

        # Load NPZ
        npz_path = os.path.join(temp_dir, npz_name)
        data = dict(np.load(npz_path, allow_pickle=True))

        # Load JSON
        json_path = os.path.join(temp_dir, json_name)
        with open(json_path, "r") as f:
            metadata = json.load(f)

        # Load video frames
        mp4_path = os.path.join(temp_dir, mp4_name)
        frames = load_video_frames(mp4_path) if os.path.exists(mp4_path) else None

    return metadata, data, frames


def load_video_frames(video_path: str) -> np.ndarray:
    """Load all frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    return np.array(frames)


def list_samples_in_tar(tar_path: str) -> List[str]:
    """List all sample IDs in a tar archive."""
    samples = set()
    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getnames():
            # Extract sample ID (e.g., s0000_000000)
            base = os.path.splitext(member)[0]
            samples.add(base)
    return sorted(list(samples))


def visualize_contact_flow(
    sample_id: str,
    metadata: Dict,
    data: Dict,
    frames: Optional[np.ndarray],
    fps: float = 30.0,
):
    """
    Visualize contact points flow over time using Rerun.
    
    Contact Flow shows how the gripper contact points move through 3D space
    as the robot manipulates objects.
    """
    episode_id = metadata.get("episode_id", sample_id)
    
    # Extract data
    normalized_frames = data.get("normalized_frames")  # (T, 4, 4) - gripper poses
    contact_points_local = data.get("contact_points_local")  # (N, 3) - contact points in gripper frame
    normalized_centroids = data.get("normalized_centroids")  # (T, 3) - centroid positions
    gripper_positions = data.get("gripper_positions")  # (T,) - gripper open/close
    raw_frames = data.get("raw_frames")  # Original poses
    
    if normalized_frames is None:
        print(f"[WARN] No normalized_frames for {sample_id}")
        return
        
    num_timesteps = normalized_frames.shape[0]
    
    # Transform contact points to world frame for each timestep
    if contact_points_local is not None:
        contact_points_world = []
        for t in range(min(num_timesteps, len(raw_frames) if raw_frames is not None else num_timesteps)):
            T_world_gripper = normalized_frames[t]  # 4x4 transform
            
            # Transform contact points
            pts_homo = np.hstack([contact_points_local, np.ones((len(contact_points_local), 1))])
            pts_world = (T_world_gripper @ pts_homo.T).T[:, :3]
            contact_points_world.append(pts_world)
        contact_points_world = np.array(contact_points_world)  # (T, N, 3)
    else:
        contact_points_world = None
    
    # Log to Rerun
    rr.set_time("time", duration=0.0)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    
    # Create color palette for contact points
    num_contact_pts = contact_points_local.shape[0] if contact_points_local is not None else 0
    colors = np.zeros((num_contact_pts, 3), dtype=np.uint8)
    
    # Color left finger points blue, right finger green
    half = num_contact_pts // 2
    colors[:half] = [51, 127, 255]  # Blue
    colors[half:] = [51, 255, 127]  # Green
    
    # Log static contact point cloud at first frame
    if contact_points_world is not None and len(contact_points_world) > 0:
        rr.log(
            f"episode/{episode_id}/contact_points/initial",
            rr.Points3D(contact_points_world[0], colors=colors, radii=0.003),
            static=True,
        )
    
    # Animate through timesteps
    actual_frames = min(num_timesteps, len(raw_frames)) if raw_frames is not None else num_timesteps
    
    for t in range(actual_frames):
        rr.set_time("time", duration=t / fps)
        
        # Log gripper pose as transform
        T = normalized_frames[t]
        rr.log(
            f"episode/{episode_id}/gripper",
            rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]),
        )
        
        # Log contact points at current frame
        if contact_points_world is not None and t < len(contact_points_world):
            rr.log(
                f"episode/{episode_id}/contact_points/current",
                rr.Points3D(contact_points_world[t], colors=colors, radii=0.004),
            )
            
            # Log contact flow lines (trails)
            trail_length = min(10, t)
            if trail_length > 1:
                for pt_idx in range(num_contact_pts):
                    trail_pts = contact_points_world[t - trail_length + 1:t + 1, pt_idx, :]
                    rr.log(
                        f"episode/{episode_id}/contact_flow/pt_{pt_idx}",
                        rr.LineStrips3D([trail_pts], colors=[colors[pt_idx]], radii=0.001),
                    )
        
        # Log centroid
        if normalized_centroids is not None and t < len(normalized_centroids):
            rr.log(
                f"episode/{episode_id}/centroid",
                rr.Points3D([normalized_centroids[t]], colors=[[255, 255, 0]], radii=0.008),
            )
        
        # Log video frame if available
        if frames is not None and t < len(frames):
            rr.log(
                f"episode/{episode_id}/video",
                rr.Image(frames[t]),
            )
        
        # Log gripper state
        if gripper_positions is not None and t < len(gripper_positions):
            grip = gripper_positions[t]
            rr.log(
                f"episode/{episode_id}/gripper_state",
                rr.Scalars([float(grip)]),
            )
    
    print(f"[INFO] Visualized {actual_frames} frames of contact flow for {episode_id}")


def visualize_3d_trajectory(
    sample_id: str,
    metadata: Dict,
    data: Dict,
):
    """Visualize the full 3D trajectory of the gripper."""
    episode_id = metadata.get("episode_id", sample_id)
    normalized_frames = data.get("normalized_frames")
    
    if normalized_frames is None:
        return
    
    # Extract positions from transforms
    positions = normalized_frames[:, :3, 3]  # (T, 3)
    
    # Log complete trajectory as line strip
    rr.set_time("time", duration=0.0)
    rr.log(
        f"episode/{episode_id}/trajectory",
        rr.LineStrips3D([positions], colors=[[255, 128, 0]], radii=0.002),
        static=True,
    )
    
    # Log start and end points
    rr.log(
        f"episode/{episode_id}/trajectory/start",
        rr.Points3D([positions[0]], colors=[[0, 255, 0]], radii=0.01),
        static=True,
    )
    rr.log(
        f"episode/{episode_id}/trajectory/end",
        rr.Points3D([positions[-1]], colors=[[255, 0, 0]], radii=0.01),
        static=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Visualize DROID training shards")
    parser.add_argument(
        "--shard_path",
        type=str,
        default="/data/droid/training_shards/shard_0000.tar",
        help="Path to shard tar file",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to visualize",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="visualizations",
        help="Output directory for RRD files",
    )
    parser.add_argument(
        "--spawn",
        action="store_true",
        help="Spawn Rerun viewer (requires display)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frames per second for animation",
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # List samples
    samples = list_samples_in_tar(args.shard_path)
    print(f"[INFO] Found {len(samples)} samples in {args.shard_path}")
    
    samples_to_process = samples[: args.num_samples]
    
    # Initialize Rerun
    shard_name = Path(args.shard_path).stem
    rrd_path = os.path.join(args.output_dir, f"{shard_name}_contact_flow.rrd")
    
    rr.init("droid_contact_flow", spawn=args.spawn)
    if not args.spawn:
        rr.save(rrd_path)
    
    # Process samples
    with tempfile.TemporaryDirectory() as temp_dir:
        for sample_id in samples_to_process:
            print(f"[INFO] Processing {sample_id}...")
            try:
                metadata, data, frames = load_sample_from_tar(
                    args.shard_path, sample_id, temp_dir
                )
                
                # Visualize contact flow
                visualize_contact_flow(sample_id, metadata, data, frames, fps=args.fps)
                
                # Visualize 3D trajectory
                visualize_3d_trajectory(sample_id, metadata, data)
                
            except Exception as e:
                print(f"[ERROR] Failed to process {sample_id}: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"[INFO] Visualization saved to {rrd_path}")
    print(f"[INFO] View with: rerun {rrd_path}")


if __name__ == "__main__":
    main()
