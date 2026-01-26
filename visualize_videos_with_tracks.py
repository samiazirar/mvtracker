#!/usr/bin/env python3
"""
Create video output with 2D projected contact points overlaid.
This outputs MP4 videos that can be viewed without Rerun.

Usage:
    python visualize_videos_with_tracks.py --shard_path /data/droid/training_shards/shard_0000.tar --num_samples 5 --output_dir output_videos
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


def load_sample_from_tar(
    tar_path: str, sample_id: str, temp_dir: str
) -> Tuple[Dict, np.ndarray, Optional[np.ndarray]]:
    """Load a sample from tar archive."""
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
        frames.append(frame)  # Keep BGR for OpenCV
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


def project_3d_to_2d(
    points_3d: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    img_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Project 3D points to 2D image coordinates.
    
    Args:
        points_3d: (N, 3) 3D points in world frame
        intrinsics: (3, 3) camera intrinsic matrix
        extrinsics: (4, 4) or (3, 4) camera extrinsic matrix (world-to-camera)
        img_shape: (H, W) image dimensions
        
    Returns:
        (N, 2) 2D image coordinates, or (-1, -1) for invalid points
    """
    N = points_3d.shape[0]
    H, W = img_shape
    
    # Handle both 3x4 and 4x4 extrinsics
    if extrinsics.shape == (3, 4):
        R = extrinsics[:3, :3]
        t = extrinsics[:3, 3]
    else:
        R = extrinsics[:3, :3]
        t = extrinsics[:3, 3]
    
    points_2d = np.zeros((N, 2), dtype=np.float32)
    
    for i, pt in enumerate(points_3d):
        # Transform to camera space
        cam_pt = R @ pt + t
        
        # Skip points behind camera
        if cam_pt[2] <= 0.01:
            points_2d[i] = [-1, -1]
            continue
        
        # Project
        img_pt = intrinsics @ cam_pt
        img_pt = img_pt[:2] / img_pt[2]
        
        # Check bounds
        if 0 <= img_pt[0] < W and 0 <= img_pt[1] < H:
            points_2d[i] = img_pt
        else:
            points_2d[i] = [-1, -1]
    
    return points_2d


def draw_contact_points(
    frame: np.ndarray,
    points_2d: np.ndarray,
    colors: List[Tuple[int, int, int]],
    radius: int = 4,
    thickness: int = -1,
) -> np.ndarray:
    """Draw contact points on frame."""
    frame_out = frame.copy()
    
    for i, pt in enumerate(points_2d):
        if pt[0] >= 0 and pt[1] >= 0:
            cv2.circle(
                frame_out,
                (int(pt[0]), int(pt[1])),
                radius,
                colors[i % len(colors)],
                thickness,
            )
    
    return frame_out


def draw_track_trails(
    frame: np.ndarray,
    trail_history: List[np.ndarray],
    colors: List[Tuple[int, int, int]],
    thickness: int = 2,
) -> np.ndarray:
    """Draw track trails on frame."""
    frame_out = frame.copy()
    
    num_points = trail_history[0].shape[0] if trail_history else 0
    
    for pt_idx in range(0, num_points, max(1, num_points // 30)):  # Sample points
        for t in range(len(trail_history) - 1):
            pt1 = trail_history[t][pt_idx]
            pt2 = trail_history[t + 1][pt_idx]
            
            if pt1[0] >= 0 and pt1[1] >= 0 and pt2[0] >= 0 and pt2[1] >= 0:
                alpha = (t + 1) / len(trail_history)  # Fade older points
                color = tuple(int(c * alpha) for c in colors[pt_idx % len(colors)])
                cv2.line(
                    frame_out,
                    (int(pt1[0]), int(pt1[1])),
                    (int(pt2[0]), int(pt2[1])),
                    color,
                    thickness,
                )
    
    return frame_out


def create_video_with_tracks(
    sample_id: str,
    metadata: Dict,
    data: Dict,
    frames: np.ndarray,
    output_path: str,
    fps: float = 30.0,
    trail_length: int = 10,
):
    """Create video with contact point tracks overlaid."""
    episode_id = metadata.get("episode_id", sample_id)
    
    # Extract data
    normalized_frames = data.get("normalized_frames")
    contact_points_local = data.get("contact_points_local")
    raw_frames = data.get("raw_frames")
    
    # Get camera intrinsics (use first camera)
    intrinsics = data.get("camera_0_intrinsics")
    extrinsics = data.get("camera_0_extrinsics")
    
    if normalized_frames is None or contact_points_local is None:
        print(f"[WARN] Missing data for {sample_id}")
        return False
    
    if intrinsics is None or extrinsics is None:
        print(f"[WARN] Missing camera params for {sample_id}")
        return False
    
    H, W = frames.shape[1:3]
    
    # Calculate contact points in world frame for each timestep
    actual_frames = min(len(frames), len(raw_frames) if raw_frames is not None else len(normalized_frames))
    
    # Create colors for contact points
    num_pts = contact_points_local.shape[0]
    half = num_pts // 2
    colors = []
    for i in range(num_pts):
        if i < half:
            colors.append((255, 127, 51))  # Orange-ish (BGR)
        else:
            colors.append((51, 255, 127))  # Green-ish (BGR)
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    
    trail_history = []
    
    for t in range(actual_frames):
        frame = frames[t].copy()
        
        # Transform contact points to world frame
        T_world_gripper = normalized_frames[t]
        pts_homo = np.hstack([contact_points_local, np.ones((num_pts, 1))])
        pts_world = (T_world_gripper @ pts_homo.T).T[:, :3]
        
        # Project to 2D
        pts_2d = project_3d_to_2d(pts_world, intrinsics, extrinsics, (H, W))
        
        # Update trail history
        trail_history.append(pts_2d.copy())
        if len(trail_history) > trail_length:
            trail_history.pop(0)
        
        # Draw trails
        if len(trail_history) > 1:
            frame = draw_track_trails(frame, trail_history, colors)
        
        # Draw current points
        frame = draw_contact_points(frame, pts_2d, colors, radius=5)
        
        # Add text overlay
        cv2.putText(
            frame,
            f"Episode: {episode_id[:30]}...",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Frame: {t}/{actual_frames}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        
        out.write(frame)
    
    out.release()
    print(f"[INFO] Created {output_path} ({actual_frames} frames)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Create videos with track overlays")
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
        help="Number of samples to process",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_videos",
        help="Output directory",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frames per second",
    )
    parser.add_argument(
        "--trail_length",
        type=int,
        default=15,
        help="Length of track trails",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    samples = list_samples_in_tar(args.shard_path)[:args.num_samples]
    print(f"[INFO] Processing {len(samples)} samples from {args.shard_path}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for sample_id in samples:
            try:
                metadata, data, frames = load_sample_from_tar(
                    args.shard_path, sample_id, temp_dir
                )
                
                if frames is None:
                    print(f"[WARN] No video frames for {sample_id}")
                    continue
                
                output_path = os.path.join(args.output_dir, f"{sample_id}_tracks.mp4")
                create_video_with_tracks(
                    sample_id, metadata, data, frames, output_path,
                    fps=args.fps, trail_length=args.trail_length
                )
                
            except Exception as e:
                print(f"[ERROR] Failed to process {sample_id}: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n[INFO] Videos saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
