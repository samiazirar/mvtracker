#!/usr/bin/env python3
"""
Generate sample mask videos using actual DROID RGB frames.

This script:
1. Loads a processed DROID episode (tracks.npz + RGB frames)
2. Renders 2D masks for gripper on the actual RGB frames
3. Creates overlay videos showing masks on real camera images

Usage:
    python demo_mask_on_rgb.py --episode_dir /path/to/episode --output_dir ./output
    python demo_mask_on_rgb.py  # Uses default sample episode
"""

import argparse
import numpy as np
import cv2
import os
import sys
import json
import glob
from typing import Dict, Optional, Tuple, List

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mask_utils import (
    GripperMaskRenderer,
    RobotArmMaskRenderer,
    pose6_to_T,
)
from scipy.spatial.transform import Rotation as R


# =============================================================================
# DEFAULT SAMPLE EPISODE
# =============================================================================

DEFAULT_EPISODE = "/workspace/droid_processed/IRIS/success/2023-05-12/Fri_May_12_10:51:00_2023"


# =============================================================================
# UTILITIES
# =============================================================================

def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float = 0.4
) -> np.ndarray:
    """Overlay a binary mask on an image with specified color and alpha."""
    result = image.copy()
    mask_bool = mask > 0
    
    if not np.any(mask_bool):
        return result
    
    # Create colored overlay
    overlay = np.zeros_like(image)
    overlay[mask_bool] = color
    
    # Blend
    result[mask_bool] = cv2.addWeighted(
        image[mask_bool].astype(np.float32), 1 - alpha,
        overlay[mask_bool].astype(np.float32), alpha,
        0
    ).astype(np.uint8)
    
    # Draw contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, color, 2)
    
    return result


def load_episode_data(episode_dir: str) -> Dict:
    """Load all data from a processed DROID episode."""
    
    # Load tracks.npz
    tracks_path = os.path.join(episode_dir, "tracks.npz")
    if not os.path.exists(tracks_path):
        raise FileNotFoundError(f"tracks.npz not found in {episode_dir}")
    
    tracks_data = np.load(tracks_path, allow_pickle=True)
    
    # Load extrinsics.npz
    extrinsics_path = os.path.join(episode_dir, "extrinsics.npz")
    extrinsics_data = None
    if os.path.exists(extrinsics_path):
        extrinsics_data = np.load(extrinsics_path, allow_pickle=True)
    
    # Find cameras (subdirectories in recordings/)
    recordings_dir = os.path.join(episode_dir, "recordings")
    cameras = {}
    
    if os.path.exists(recordings_dir):
        for item in os.listdir(recordings_dir):
            cam_dir = os.path.join(recordings_dir, item)
            if os.path.isdir(cam_dir):
                rgb_dir = os.path.join(cam_dir, "rgb")
                intrinsics_path = os.path.join(cam_dir, "intrinsics.json")
                
                if os.path.exists(rgb_dir) and os.path.exists(intrinsics_path):
                    with open(intrinsics_path, 'r') as f:
                        intrinsics = json.load(f)
                    
                    # Get RGB frame paths
                    rgb_frames = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
                    
                    cameras[item] = {
                        'rgb_dir': rgb_dir,
                        'rgb_frames': rgb_frames,
                        'intrinsics': intrinsics,
                    }
    
    return {
        'tracks': tracks_data,
        'extrinsics': extrinsics_data,
        'cameras': cameras,
        'episode_dir': episode_dir,
    }


def get_camera_matrix(intrinsics: Dict) -> np.ndarray:
    """Build 3x3 camera intrinsic matrix from intrinsics dict."""
    fx = intrinsics.get('fx', intrinsics.get('focal_length_x', 500))
    fy = intrinsics.get('fy', intrinsics.get('focal_length_y', 500))
    cx = intrinsics.get('cx', intrinsics.get('principal_point_x', 320))
    cy = intrinsics.get('cy', intrinsics.get('principal_point_y', 180))
    
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)


def get_camera_extrinsics(extrinsics_data, camera_serial: str, frame_idx: int = 0) -> Optional[np.ndarray]:
    """Get camera extrinsics (world_T_cam) from extrinsics.npz."""
    if extrinsics_data is None:
        return None
    
    # Check for external camera (static pose)
    external_key = f"external_{camera_serial}"
    if external_key in extrinsics_data:
        return extrinsics_data[external_key]
    
    # Check for wrist camera (dynamic pose per frame)
    wrist_serial_key = 'wrist_serial'
    if wrist_serial_key in extrinsics_data:
        wrist_serial = str(extrinsics_data[wrist_serial_key].item())
        if camera_serial == wrist_serial:
            wrist_extr = extrinsics_data.get('wrist_extrinsics')
            if wrist_extr is not None and len(wrist_extr) > frame_idx:
                return wrist_extr[frame_idx]
    
    # Try different possible keys (legacy formats)
    for key in [camera_serial, f"{camera_serial}_world_T_cam", "world_T_cam"]:
        if key in extrinsics_data:
            data = extrinsics_data[key]
            if isinstance(data, np.ndarray) and data.shape == (4, 4):
                return data
    
    return None


# =============================================================================
# MAIN
# =============================================================================

def generate_mask_videos(
    episode_dir: str,
    output_dir: str,
    max_frames: int = 200,
    gripper_color: Tuple[int, int, int] = (0, 255, 0),  # Green
    robot_color: Tuple[int, int, int] = (255, 100, 0),  # Orange-ish blue
    alpha: float = 0.4,
):
    """Generate mask overlay videos for a DROID episode."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("DROID Mask Overlay Video Generator")
    print("=" * 60)
    print(f"Episode: {episode_dir}")
    print(f"Output: {output_dir}")
    print()
    
    # Load episode data
    print("[1/4] Loading episode data...")
    data = load_episode_data(episode_dir)
    
    tracks = data['tracks']
    extrinsics = data['extrinsics']
    cameras = data['cameras']
    
    # Get robot data
    gripper_poses = tracks['gripper_poses']  # (N, 4, 4)
    gripper_positions = tracks['gripper_positions']  # (N,)
    cartesian_positions = tracks['cartesian_positions']  # (N, 6)
    num_frames = int(tracks['num_frames'])
    
    print(f"  Frames in episode: {num_frames}")
    print(f"  Cameras found: {list(cameras.keys())}")
    print(f"  Gripper poses shape: {gripper_poses.shape}")
    
    # Limit frames
    num_frames = min(num_frames, max_frames)
    print(f"  Processing frames: {num_frames}")
    
    # Initialize mask renderer
    print("\n[2/4] Initializing gripper renderer...")
    gripper_renderer = GripperMaskRenderer()
    print("  OK")
    
    # Process each camera
    print("\n[3/4] Generating mask videos...")
    
    for cam_serial, cam_data in cameras.items():
        print(f"\n  Camera: {cam_serial}")
        
        rgb_frames = cam_data['rgb_frames']
        intrinsics = cam_data['intrinsics']
        
        if len(rgb_frames) == 0:
            print("    [SKIP] No RGB frames")
            continue
        
        # Get camera matrix
        K = get_camera_matrix(intrinsics)
        print(f"    Intrinsics: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")
        
        # Check if this is the wrist camera (dynamic extrinsics per frame)
        is_wrist_camera = False
        wrist_extrinsics = None
        if extrinsics is not None and 'wrist_serial' in extrinsics:
            wrist_serial = str(extrinsics['wrist_serial'].item())
            if cam_serial == wrist_serial:
                is_wrist_camera = True
                wrist_extrinsics = extrinsics.get('wrist_extrinsics')
                print(f"    Type: Wrist camera (dynamic pose)")
        
        # Get static camera extrinsics (for external cameras)
        static_world_T_cam = None
        if not is_wrist_camera:
            static_world_T_cam = get_camera_extrinsics(extrinsics, cam_serial, 0)
            if static_world_T_cam is not None:
                print(f"    Type: External camera (static pose)")
        
        if static_world_T_cam is None and not is_wrist_camera:
            # Estimate camera pose from first frame (looking at robot base)
            print("    [WARN] No extrinsics, using estimated camera pose")
            # Place camera 1m in front of robot, looking at origin
            static_world_T_cam = np.eye(4)
            static_world_T_cam[:3, 3] = [0.8, 0.5, 0.6]  # Camera position
            # Look at origin
            look_at = np.array([0.3, 0, 0.3])
            z_axis = look_at - static_world_T_cam[:3, 3]
            z_axis = z_axis / np.linalg.norm(z_axis)
            up = np.array([0, 0, 1])
            x_axis = np.cross(z_axis, up)
            x_axis = x_axis / np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)
            static_world_T_cam[:3, :3] = np.stack([x_axis, y_axis, z_axis], axis=1)
        
        # Get first frame to determine size
        first_frame = cv2.imread(rgb_frames[0])
        if first_frame is None:
            print(f"    [SKIP] Could not read first frame")
            continue
        
        height, width = first_frame.shape[:2]
        print(f"    Resolution: {width}x{height}")
        
        # Adjust intrinsics if needed (in case K was for different resolution)
        # Assume K is for this resolution
        
        # Initialize video writer
        video_path = os.path.join(output_dir, f"{cam_serial}_gripper_mask.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = float(tracks.get('fps', 30.0))
        writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        # Process frames
        frames_processed = 0
        
        for frame_idx in range(min(num_frames, len(rgb_frames))):
            # Load RGB frame
            rgb_path = rgb_frames[frame_idx]
            frame = cv2.imread(rgb_path)
            
            if frame is None:
                continue
            
            # Get gripper pose for this frame (already a 4x4 matrix)
            T_world_ee = gripper_poses[frame_idx]
            grip_pos = gripper_positions[frame_idx]
            
            # Get camera extrinsics for this frame
            if is_wrist_camera and wrist_extrinsics is not None:
                # Wrist camera has per-frame extrinsics
                world_T_cam = wrist_extrinsics[frame_idx] if frame_idx < len(wrist_extrinsics) else wrist_extrinsics[-1]
            else:
                # External camera has static extrinsics
                world_T_cam = static_world_T_cam
            
            T_cam_world = np.linalg.inv(world_T_cam)
            
            # Render gripper mask
            gripper_mask = gripper_renderer.render_mask(
                T_world_ee, grip_pos, K, T_cam_world,
                width, height, min_depth=0.01
            )
            
            # Overlay on frame
            result = overlay_mask(frame, gripper_mask, gripper_color, alpha)
            
            # Add frame info text
            cv2.putText(result, f"Frame: {frame_idx+1}/{num_frames}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(result, f"Gripper: {grip_pos:.2f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(result, f"Cam: {cam_serial}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            writer.write(result)
            frames_processed += 1
            
            if (frame_idx + 1) % 50 == 0:
                print(f"    Frame {frame_idx + 1}/{num_frames}")
        
        writer.release()
        print(f"    Saved: {video_path} ({frames_processed} frames)")
    
    print("\n[4/4] Complete!")
    print(f"Videos saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate mask videos on actual DROID RGB")
    parser.add_argument('--episode_dir', type=str, default=DEFAULT_EPISODE,
                        help='Path to processed DROID episode directory')
    parser.add_argument('--output_dir', type=str, default='./mask_rgb_output',
                        help='Output directory for videos')
    parser.add_argument('--max_frames', type=int, default=200,
                        help='Maximum frames to process')
    parser.add_argument('--alpha', type=float, default=0.4,
                        help='Mask overlay alpha (0-1)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.episode_dir):
        print(f"[ERROR] Episode directory not found: {args.episode_dir}")
        print("\nAvailable episodes in droid_processed:")
        import subprocess
        result = subprocess.run(
            ["find", "/workspace/droid_processed", "-name", "tracks.npz", "-type", "f"],
            capture_output=True, text=True
        )
        for line in result.stdout.strip().split('\n')[:5]:
            if line:
                print(f"  {os.path.dirname(line)}")
        sys.exit(1)
    
    generate_mask_videos(
        episode_dir=args.episode_dir,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        alpha=args.alpha,
    )


if __name__ == '__main__':
    main()
