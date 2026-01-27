#!/usr/bin/env python3
"""Render flow track overlays on training shard videos.

This script reads training shards (tar files containing npz + mp4) and renders
contact flow overlays by projecting 3D contact points onto the video frames.

The training shards contain:
- npz: normalized_frames, normalized_centroids, contact_points_local, camera intrinsics/extrinsics
- mp4: video frames for each camera

Usage:
    python render_flow_from_shards.py --shards_dir /data/droid/training_shards_v2 --max_shards 2
"""

import argparse
import glob
import os
import sys
import tarfile
import tempfile
from pathlib import Path

import cv2
import numpy as np


def project_3d_to_2d(points_3d: np.ndarray, K: np.ndarray, 
                     world_T_cam: np.ndarray, width: int, height: int,
                     min_depth: float = 0.01):
    """Project 3D points to 2D image coordinates."""
    if points_3d is None or len(points_3d) == 0:
        return None
    
    cam_T_world = np.linalg.inv(world_T_cam)
    R = cam_T_world[:3, :3]
    t = cam_T_world[:3, 3]
    
    points_cam = (R @ points_3d.T).T + t
    uv = np.full((len(points_3d), 2), np.nan)
    
    valid = points_cam[:, 2] > min_depth
    if np.any(valid):
        z = points_cam[valid, 2]
        x = points_cam[valid, 0] / z
        y = points_cam[valid, 1] / z
        
        u = K[0, 0] * x + K[0, 2]
        v = K[1, 1] * y + K[1, 2]
        
        in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        valid_indices = np.where(valid)[0]
        for i, idx in enumerate(valid_indices):
            if in_bounds[i]:
                uv[idx, 0] = u[i]
                uv[idx, 1] = v[i]
    
    return uv


def compute_contact_points_3d(contact_points_local: np.ndarray,
                              normalized_frames: np.ndarray,
                              gripper_positions: np.ndarray,
                              step_idx: int):
    """Compute 3D contact points for a normalized step."""
    if contact_points_local is None or len(contact_points_local) == 0:
        return None
    
    frame = normalized_frames[step_idx]
    gripper_pos = gripper_positions[step_idx]
    
    finger_offset = 0.040 + gripper_pos * 0.02
    
    n_pts = len(contact_points_local)
    pts_local_hom = np.hstack([contact_points_local, np.ones((n_pts, 1))])
    
    T_left = frame.copy()
    T_left[:3, 3] += frame[:3, 1] * finger_offset
    
    T_right = frame.copy()
    T_right[:3, 3] -= frame[:3, 1] * finger_offset
    
    pts_left = (T_left @ pts_local_hom.T).T[:, :3]
    pts_right = (T_right @ pts_local_hom.T).T[:, :3]
    
    return np.vstack([pts_left, pts_right])


def draw_points(frame: np.ndarray, uv: np.ndarray, colors: np.ndarray = None,
                radius: int = 4):
    """Draw 2D points on frame."""
    if uv is None:
        return frame
    
    n_pts = len(uv)
    half = n_pts // 2
    
    if colors is None:
        colors = np.zeros((n_pts, 3), dtype=np.uint8)
        colors[:half, :] = [255, 127, 51]
        colors[half:, :] = [127, 255, 51]
    
    for i, pt in enumerate(uv):
        if np.isnan(pt[0]) or np.isnan(pt[1]):
            continue
        x, y = int(pt[0]), int(pt[1])
        color = tuple(int(c) for c in colors[i])
        cv2.circle(frame, (x, y), radius, color, -1)
    
    return frame


def draw_centroid(frame: np.ndarray, centroid_3d: np.ndarray,
                  K: np.ndarray, world_T_cam: np.ndarray,
                  width: int, height: int, color=(0, 255, 255)):
    """Draw centroid marker."""
    uv = project_3d_to_2d(centroid_3d.reshape(1, 3), K, world_T_cam, width, height)
    if uv is not None and not np.isnan(uv[0, 0]):
        x, y = int(uv[0, 0]), int(uv[0, 1])
        cv2.circle(frame, (x, y), 8, color, -1)
        cv2.circle(frame, (x, y), 8, (0, 0, 0), 2)
    return frame


def render_shard_episode(npz_path: str, mp4_path: str, output_path: str,
                         camera_idx: int = 0, trail_length: int = 10):
    """Render flow overlay for a single shard episode."""
    data = np.load(npz_path, allow_pickle=True)
    
    normalized_frames = data['normalized_frames']
    normalized_centroids = data['normalized_centroids']
    gripper_positions = data['gripper_positions']
    contact_points_local = data['contact_points_local']
    frame_to_step = data['frame_to_step']
    num_frames = int(data['num_frames'])
    num_steps = int(data['num_steps'])
    
    K = data[f'camera_{camera_idx}_intrinsics']
    extr = data[f'camera_{camera_idx}_extrinsics']
    width = int(data[f'camera_{camera_idx}_width'])
    height = int(data[f'camera_{camera_idx}_height'])
    
    print(f"  Episode: {num_frames} frames, {num_steps} normalized steps")
    print(f"  Camera {camera_idx}: {width}x{height}")
    print(f"  Contact points: {len(contact_points_local)} per finger")
    
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open video: {mp4_path}")
        return False
    
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    actual_frames = min(video_frames, num_frames)
    
    n_pts = len(contact_points_local) * 2
    colors = np.zeros((n_pts, 3), dtype=np.uint8)
    colors[:n_pts//2, :] = [255, 127, 51]
    colors[n_pts//2:, :] = [127, 255, 51]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    uv_history = []
    
    for frame_idx in range(actual_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        step_idx = int(frame_to_step[frame_idx]) if frame_idx < len(frame_to_step) else 0
        step_idx = min(step_idx, num_steps - 1)
        
        pts_3d = compute_contact_points_3d(
            contact_points_local, normalized_frames, gripper_positions, step_idx
        )
        
        if pts_3d is not None:
            uv = project_3d_to_2d(pts_3d, K, extr, width, height)
            
            uv_history.append(uv)
            if len(uv_history) > trail_length:
                uv_history.pop(0)
            
            for t, hist_uv in enumerate(uv_history[:-1]):
                alpha = (t + 1) / len(uv_history)
                next_uv = uv_history[t + 1]
                for i in range(len(hist_uv)):
                    if np.isnan(hist_uv[i, 0]) or np.isnan(next_uv[i, 0]):
                        continue
                    pt1 = (int(hist_uv[i, 0]), int(hist_uv[i, 1]))
                    pt2 = (int(next_uv[i, 0]), int(next_uv[i, 1]))
                    color = tuple(int(c * alpha) for c in colors[i])
                    cv2.line(frame, pt1, pt2, color, 1)
            
            frame = draw_points(frame, uv, colors, radius=4)
        
        centroid = normalized_centroids[step_idx]
        frame = draw_centroid(frame, centroid, K, extr, width, height)
        
        text = f"Frame {frame_idx}/{actual_frames-1} | Step {step_idx}/{num_steps-1}"
        cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        writer.write(frame)
        
        if frame_idx % 30 == 0:
            print(f"    Frame {frame_idx}/{actual_frames}")
    
    cap.release()
    writer.release()
    print(f"  Saved: {output_path}")
    return True


def process_shard_tar(tar_path: str, output_dir: str, max_episodes: int = None):
    """Process a shard tar file and render flow videos."""
    shard_name = Path(tar_path).stem
    print(f"\nProcessing shard: {shard_name}")
    
    outputs = []
    
    with tempfile.TemporaryDirectory() as tmpdir:
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(tmpdir)
        
        npz_files = sorted(glob.glob(os.path.join(tmpdir, "*.npz")))
        
        if max_episodes:
            npz_files = npz_files[:max_episodes]
        
        print(f"  Found {len(npz_files)} episodes")
        
        for npz_path in npz_files:
            episode_name = Path(npz_path).stem
            mp4_path = npz_path.replace('.npz', '.mp4')
            
            if not os.path.exists(mp4_path):
                print(f"  [SKIP] No mp4 for {episode_name}")
                continue
            
            print(f"\n  Episode: {episode_name}")
            
            output_path = os.path.join(output_dir, f"{episode_name}_flow.mp4")
            
            success = render_shard_episode(
                npz_path, mp4_path, output_path, camera_idx=0
            )
            
            if success:
                outputs.append(output_path)
    
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Render flow videos from training shards")
    parser.add_argument("--shards_dir", default="/data/droid/training_shards_v2",
                        help="Directory containing shard tar files")
    parser.add_argument("--output_dir", default="/workspace/shard_flow_videos",
                        help="Output directory for videos")
    parser.add_argument("--max_shards", type=int, default=None,
                        help="Maximum shards to process")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Maximum episodes per shard")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Shard Flow Video Renderer")
    print("=" * 60)
    
    shard_files = sorted(glob.glob(os.path.join(args.shards_dir, "shard_*.tar")))
    
    if not shard_files:
        print(f"[ERROR] No shard tar files found in {args.shards_dir}")
        return 1
    
    if args.max_shards:
        shard_files = shard_files[:args.max_shards]
    
    print(f"\nProcessing {len(shard_files)} shards")
    
    os.makedirs(args.output_dir, exist_ok=True)
    all_outputs = []
    
    for i, tar_path in enumerate(shard_files, 1):
        print(f"\n[{i}/{len(shard_files)}] {os.path.basename(tar_path)}")
        
        try:
            outputs = process_shard_tar(
                tar_path,
                args.output_dir,
                max_episodes=args.max_episodes,
            )
            all_outputs.extend(outputs)
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"[DONE] Rendered {len(all_outputs)} videos")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
