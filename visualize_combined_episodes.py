#!/usr/bin/env python3
"""
Create combined visualization showing multiple episodes side by side.
Also visualizes gripper trajectory and contact point flow with enhanced 3D views.

Usage:
    python visualize_combined_episodes.py --output visualizations/combined_episodes.rrd
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


def load_sample_from_tar(
    tar_path: str, sample_id: str, temp_dir: str
) -> Tuple[Dict, np.ndarray, Optional[np.ndarray]]:
    """Load a sample (npz, json, mp4) from a tar archive."""
    with tarfile.open(tar_path, "r") as tar:
        npz_name = f"{sample_id}.npz"
        json_name = f"{sample_id}.json"
        mp4_name = f"{sample_id}.mp4"

        for name in [npz_name, json_name, mp4_name]:
            try:
                tar.extract(name, temp_dir)
            except KeyError:
                pass

        npz_path = os.path.join(temp_dir, npz_name)
        data = dict(np.load(npz_path, allow_pickle=True))

        json_path = os.path.join(temp_dir, json_name)
        with open(json_path, "r") as f:
            metadata = json.load(f)

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
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    return np.array(frames) if frames else None


def list_samples_in_tar(tar_path: str) -> List[str]:
    """List all sample IDs in a tar archive."""
    samples = set()
    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getnames():
            base = os.path.splitext(member)[0]
            samples.add(base)
    return sorted(list(samples))


def get_lab_from_episode_id(episode_id: str) -> str:
    """Extract lab name from episode ID."""
    if "+" in episode_id:
        return episode_id.split("+")[0]
    return "Unknown"


def visualize_episode_with_3d_view(
    sample_id: str,
    metadata: Dict,
    data: Dict,
    frames: Optional[np.ndarray],
    episode_num: int,
    fps: float = 30.0,
):
    """
    Visualize episode with enhanced 3D visualization including:
    - Gripper trajectory as a colored path
    - Contact points with flow trails
    - Video frames
    - 3D coordinate frame
    """
    episode_id = metadata.get("episode_id", sample_id)
    lab = get_lab_from_episode_id(episode_id)
    
    # Extract data
    normalized_frames = data.get("normalized_frames")
    contact_points_local = data.get("contact_points_local")
    normalized_centroids = data.get("normalized_centroids")
    gripper_positions = data.get("gripper_positions")
    raw_frames = data.get("raw_frames")
    
    if normalized_frames is None:
        print(f"[WARN] No normalized_frames for {sample_id}")
        return
        
    num_timesteps = normalized_frames.shape[0]
    
    # Entity path for this episode
    entity_base = f"episodes/{lab}/ep_{episode_num}"
    
    # Transform contact points to world frame
    if contact_points_local is not None:
        contact_points_world = []
        for t in range(min(num_timesteps, len(raw_frames) if raw_frames is not None else num_timesteps)):
            T_world_gripper = normalized_frames[t]
            pts_homo = np.hstack([contact_points_local, np.ones((len(contact_points_local), 1))])
            pts_world = (T_world_gripper @ pts_homo.T).T[:, :3]
            contact_points_world.append(pts_world)
        contact_points_world = np.array(contact_points_world)
    else:
        contact_points_world = None
    
    # Get gripper positions for full trajectory
    positions = normalized_frames[:, :3, 3]  # (T, 3)
    
    # Create color gradient for trajectory (blue to red)
    t_colors = np.zeros((len(positions), 3), dtype=np.uint8)
    for i in range(len(positions)):
        ratio = i / max(len(positions) - 1, 1)
        t_colors[i] = [int(255 * ratio), 0, int(255 * (1 - ratio))]
    
    # Log static trajectory
    rr.set_time("time", duration=0.0)
    rr.log(
        f"{entity_base}/trajectory",
        rr.LineStrips3D([positions], colors=[[255, 128, 0]], radii=0.002),
        static=True,
    )
    
    # Log start/end markers
    rr.log(
        f"{entity_base}/start",
        rr.Points3D([positions[0]], colors=[[0, 255, 0]], radii=0.015),
        static=True,
    )
    rr.log(
        f"{entity_base}/end",
        rr.Points3D([positions[-1]], colors=[[255, 0, 0]], radii=0.015),
        static=True,
    )
    
    # Create color palette for contact points
    num_contact_pts = contact_points_local.shape[0] if contact_points_local is not None else 0
    colors = np.zeros((num_contact_pts, 3), dtype=np.uint8)
    half = num_contact_pts // 2
    colors[:half] = [100, 149, 237]  # Cornflower blue for left
    colors[half:] = [144, 238, 144]  # Light green for right
    
    # Animate through timesteps
    actual_frames = min(num_timesteps, len(raw_frames)) if raw_frames is not None else num_timesteps
    
    for t in range(actual_frames):
        rr.set_time("time", duration=t / fps)
        
        # Log current gripper position
        T = normalized_frames[t]
        rr.log(
            f"{entity_base}/gripper/current",
            rr.Points3D([T[:3, 3]], colors=[[255, 255, 0]], radii=0.01),
        )
        
        # Log gripper orientation as arrows
        arrow_length = 0.03
        origin = T[:3, 3]
        x_axis = origin + T[:3, 0] * arrow_length
        y_axis = origin + T[:3, 1] * arrow_length
        z_axis = origin + T[:3, 2] * arrow_length
        
        rr.log(
            f"{entity_base}/gripper/axes",
            rr.Arrows3D(
                origins=[origin, origin, origin],
                vectors=[T[:3, 0] * arrow_length, T[:3, 1] * arrow_length, T[:3, 2] * arrow_length],
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                radii=0.002,
            ),
        )
        
        # Log contact points
        if contact_points_world is not None and t < len(contact_points_world):
            rr.log(
                f"{entity_base}/contact/points",
                rr.Points3D(contact_points_world[t], colors=colors, radii=0.004),
            )
            
            # Flow trails
            trail_length = min(15, t)
            if trail_length > 2:
                for pt_idx in range(0, num_contact_pts, max(1, num_contact_pts // 20)):  # Sample points
                    trail_pts = contact_points_world[t - trail_length + 1:t + 1, pt_idx, :]
                    rr.log(
                        f"{entity_base}/contact/flow_{pt_idx}",
                        rr.LineStrips3D([trail_pts], colors=[colors[pt_idx]], radii=0.001),
                    )
        
        # Log video frame
        if frames is not None and t < len(frames):
            rr.log(
                f"{entity_base}/video",
                rr.Image(frames[t]),
            )
        
        # Log gripper state
        if gripper_positions is not None and t < len(gripper_positions):
            rr.log(
                f"{entity_base}/gripper_state",
                rr.Scalars([float(gripper_positions[t])]),
            )
    
    print(f"[INFO] Visualized {actual_frames} frames for {lab} episode {episode_num}")
    return lab


def main():
    parser = argparse.ArgumentParser(description="Combined multi-lab episode visualization")
    parser.add_argument(
        "--shards_dir",
        type=str,
        default="/data/droid/training_shards",
        help="Directory containing shard tar files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="visualizations/combined_episodes.rrd",
        help="Output RRD file",
    )
    parser.add_argument(
        "--samples_per_shard",
        type=int,
        default=3,
        help="Number of samples per shard to include",
    )
    parser.add_argument(
        "--max_shards",
        type=int,
        default=4,
        help="Maximum number of shards",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frames per second",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Find shards
    shards_dir = Path(args.shards_dir)
    shards = sorted(shards_dir.glob("shard_*.tar"))[:args.max_shards]
    
    print(f"[INFO] Processing {len(shards)} shards")
    
    # Initialize Rerun
    rr.init("droid_combined_visualization", spawn=False)
    rr.save(args.output)
    
    # World coordinates
    rr.set_time("time", duration=0.0)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    
    episode_num = 0
    labs_seen = set()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for shard_path in shards:
            samples = list_samples_in_tar(str(shard_path))[:args.samples_per_shard]
            
            for sample_id in samples:
                try:
                    metadata, data, frames = load_sample_from_tar(
                        str(shard_path), sample_id, temp_dir
                    )
                    
                    lab = visualize_episode_with_3d_view(
                        sample_id, metadata, data, frames, episode_num, args.fps
                    )
                    
                    if lab:
                        labs_seen.add(lab)
                    episode_num += 1
                    
                except Exception as e:
                    print(f"[ERROR] Failed to process {sample_id}: {e}")
    
    print(f"\n[INFO] Visualization saved to {args.output}")
    print(f"[INFO] Labs included: {sorted(labs_seen)}")
    print(f"[INFO] Total episodes: {episode_num}")


if __name__ == "__main__":
    main()
