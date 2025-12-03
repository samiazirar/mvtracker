"""Render validation RRD from extracted RGB, depth, and tracks.

This script generates a Rerun RRD visualization from pre-extracted data:
- RGB frames (PNG files)
- Depth frames (NPY files)  
- Tracks (tracks.npz)
- Extrinsics (extrinsics.npz)
- Camera intrinsics (intrinsics.json per camera)

This is used for validation after the pipeline runs, without needing SVO files.

Usage:
    python render_episode_validation.py --episode_id "AUTOLab+84bd5053+2023-08-18-12h-01m-10s"
    python render_episode_validation.py --episode_id "AUTOLab+84bd5053+2023-08-18-12h-01m-10s" --config custom_config.yaml

Output:
    {output_root}/{lab}/success/{date}/{timestamp}/point_clouds/
        ├── validation.rrd
        └── videos/
            ├── {camera_serial}_validation.mp4
            └── ...
"""

import argparse
import json
import os
import sys
import glob
import re
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import cv2
import numpy as np
import yaml
import rerun as rr
from scipy.spatial.transform import Rotation as R

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.transforms import pose6_to_T, transform_points, invert_transform
from utils.tracking import MinimalGripperVisualizer, ContactSurfaceTracker
from utils.video_utils import (
    VideoRecorder,
    project_points_to_image,
    draw_points_on_image,
    draw_points_on_image_fast,
    draw_track_trails_on_image,
)


def parse_episode_id(episode_id: str) -> dict:
    """Parse episode ID into components.
    
    Episode ID format: "{lab}+{hash}+{date}-{hour}h-{min}m-{sec}s"
    Example: "AUTOLab+84bd5053+2023-08-18-12h-01m-10s"
    
    Returns:
        dict with keys: lab, hash, date, time_str, timestamp_folder
    """
    parts = episode_id.split('+')
    if len(parts) != 3:
        raise ValueError(f"Invalid episode ID format: {episode_id}")
    
    lab = parts[0]
    episode_hash = parts[1]
    datetime_part = parts[2]
    
    # Parse datetime: "2023-08-18-12h-01m-10s"
    match = re.match(r'(\d{4}-\d{2}-\d{2})-(\d+)h-(\d+)m-(\d+)s', datetime_part)
    if not match:
        raise ValueError(f"Invalid datetime format in episode ID: {datetime_part}")
    
    date = match.group(1)
    hour = match.group(2)
    minute = match.group(3)
    second = match.group(4)
    time_str = f"{hour}:{minute}:{second}"
    
    # Reconstruct DROID timestamp folder format: "Fri_Aug_18_11:58:51_2023"
    from datetime import datetime
    dt = datetime.strptime(f"{date} {hour}:{minute}:{second}", "%Y-%m-%d %H:%M:%S")
    timestamp_folder = dt.strftime("%a_%b_%e_%H:%M:%S_%Y").replace(" ", "_")
    
    return {
        'lab': lab,
        'hash': episode_hash,
        'date': date,
        'time_str': time_str,
        'timestamp_folder': timestamp_folder,
        'full_id': episode_id,
    }


def find_processed_episode(output_root: str, episode_info: dict) -> dict:
    """Find processed episode output paths.
    
    Returns:
        dict with paths to all processed data
    """
    lab = episode_info['lab']
    date = episode_info['date']
    timestamp_folder = episode_info['timestamp_folder']
    
    # Search both success and failure folders
    for outcome in ['success', 'failure']:
        base_path = os.path.join(output_root, lab, outcome, date, timestamp_folder)
        tracks_path = os.path.join(base_path, 'tracks.npz')
        
        if os.path.exists(tracks_path):
            return {
                'base_path': base_path,
                'tracks_path': tracks_path,
                'extrinsics_path': os.path.join(base_path, 'extrinsics.npz'),
                'quality_path': os.path.join(base_path, 'quality.json'),
                'recordings_dir': os.path.join(base_path, 'recordings'),
                'relative_path': os.path.join(lab, outcome, date, timestamp_folder),
                'outcome': outcome,
            }
    
    raise FileNotFoundError(f"Processed episode not found: {episode_info['full_id']}")


def load_camera_data(recordings_dir: str) -> Dict[str, dict]:
    """Load all camera data (intrinsics and frame paths).
    
    Returns:
        dict mapping camera serial -> camera info dict
    """
    cameras = {}
    
    # Find all camera directories
    for cam_dir in glob.glob(os.path.join(recordings_dir, '*')):
        if not os.path.isdir(cam_dir):
            continue
        
        serial = os.path.basename(cam_dir)
        rgb_dir = os.path.join(cam_dir, 'rgb')
        depth_dir = os.path.join(cam_dir, 'depth')
        intrinsics_path = os.path.join(cam_dir, 'intrinsics.json')
        
        # Skip if no frames
        if not os.path.exists(rgb_dir) or not os.path.exists(intrinsics_path):
            continue
        
        # Load intrinsics
        with open(intrinsics_path, 'r') as f:
            intrinsics = json.load(f)
        
        # Build intrinsic matrix
        K = np.array([
            [intrinsics['fx'], 0, intrinsics['cx']],
            [0, intrinsics['fy'], intrinsics['cy']],
            [0, 0, 1]
        ])
        
        # Count frames
        rgb_frames = sorted(glob.glob(os.path.join(rgb_dir, '*.png')))
        depth_frames = sorted(glob.glob(os.path.join(depth_dir, '*.npy')))
        
        cameras[serial] = {
            'serial': serial,
            'rgb_dir': rgb_dir,
            'depth_dir': depth_dir,
            'K': K,
            'width': intrinsics['width'],
            'height': intrinsics['height'],
            'rgb_frames': rgb_frames,
            'depth_frames': depth_frames,
            'num_frames': len(rgb_frames),
        }
        
        print(f"  Camera {serial}: {len(rgb_frames)} frames, {intrinsics['width']}x{intrinsics['height']}")
    
    return cameras


def depth_to_pointcloud(
    depth: np.ndarray,
    rgb: np.ndarray,
    K: np.ndarray,
    max_depth: float = 2.0,
    min_depth: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert depth image to colored point cloud in camera frame.
    
    Args:
        depth: HxW depth image in meters (float32)
        rgb: HxWx3 RGB image (uint8)
        K: 3x3 intrinsic matrix
        max_depth: Maximum depth threshold
        min_depth: Minimum depth threshold
        
    Returns:
        Tuple of (xyz, rgb) arrays
    """
    h, w = depth.shape
    
    # Create pixel grid
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u = u.flatten()
    v = v.flatten()
    z = depth.flatten()
    
    # Filter by depth
    valid = (z > min_depth) & (z < max_depth) & np.isfinite(z)
    u = u[valid]
    v = v[valid]
    z = z[valid]
    
    if len(z) == 0:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)
    
    # Backproject to 3D
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    xyz = np.stack([x, y, z], axis=-1)
    
    # Get colors
    colors = rgb.reshape(-1, 3)[valid]
    
    return xyz, colors


def render_episode(
    episode_paths: dict,
    config: dict,
    episode_info: dict,
    headless: bool = True,
) -> str:
    """Render validation RRD for an episode.
    
    Returns:
        Path to saved RRD file
    """
    # Create output directory for point clouds
    point_cloud_dir = os.path.join(episode_paths['base_path'], 'point_clouds')
    os.makedirs(point_cloud_dir, exist_ok=True)
    
    rrd_path = os.path.join(point_cloud_dir, 'validation.rrd')
    video_dir = os.path.join(point_cloud_dir, 'videos')
    os.makedirs(video_dir, exist_ok=True)
    
    # Initialize Rerun
    rr.init("droid_validation", spawn=not headless)
    rr.save(rrd_path)
    
    # Set world coordinate system (Z-up for robotics)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    
    # Load tracks
    print("[INFO] Loading tracks...")
    tracks_data = np.load(episode_paths['tracks_path'])
    tracks_3d = tracks_data['tracks_3d']
    gripper_poses = tracks_data['gripper_poses']
    gripper_positions = tracks_data['gripper_positions']
    num_frames = int(tracks_data['num_frames'])
    num_contact_pts = int(tracks_data['num_points_per_finger'])
    total_track_pts = num_contact_pts * 2
    fps = float(tracks_data.get('fps', 30.0))
    
    print(f"  Tracks: {total_track_pts} points, {num_frames} frames")
    
    # Load extrinsics
    print("[INFO] Loading extrinsics...")
    extrinsics_data = np.load(episode_paths['extrinsics_path'], allow_pickle=True)
    
    # Parse external camera extrinsics
    external_extrinsics = {}
    wrist_extrinsics = None
    wrist_serial = None
    
    for key in extrinsics_data.files:
        if key.startswith('external_'):
            cam_id = key.replace('external_', '')
            external_extrinsics[cam_id] = extrinsics_data[key]
        elif key == 'wrist_extrinsics':
            wrist_extrinsics = extrinsics_data[key]
        elif key == 'wrist_serial':
            wrist_serial = str(extrinsics_data[key])
    
    print(f"  External cams: {list(external_extrinsics.keys())}")
    print(f"  Wrist cam: {wrist_serial}")
    
    # Load camera data
    print("[INFO] Loading camera data...")
    cameras = load_camera_data(episode_paths['recordings_dir'])
    
    # Initialize gripper visualizer
    gripper_viz = MinimalGripperVisualizer(
        num_track_points=num_contact_pts,
        mesh_path=config.get('finger_mesh_path', "/workspace/third_party/robotiq_arg85_description/meshes/inner_finger_fine.STL")
    )
    gripper_viz.init_rerun()
    
    # Prepare track colors
    track_colors_rgb = np.zeros((total_track_pts, 3), dtype=np.uint8)
    if total_track_pts > 0:
        track_colors_rgb[:num_contact_pts, :] = [51, 127, 255]  # Blue for left
        track_colors_rgb[num_contact_pts:, :] = [51, 255, 127]  # Green for right
    
    track_colors_float = np.zeros((total_track_pts, 4), dtype=np.float32)
    if total_track_pts > 0:
        track_colors_float[:num_contact_pts, :] = [0.2, 0.5, 1.0, 1.0]
        track_colors_float[num_contact_pts:, :] = [0.2, 1.0, 0.5, 1.0]
    
    # Setup video recorders
    recorders = {}
    for serial, cam in cameras.items():
        recorders[serial] = VideoRecorder(
            video_dir, serial, "validation",
            cam['width'], cam['height'], fps=fps
        )
    
    # Config params
    max_depth_ext = config.get('ext_max_depth', 1.5)
    max_depth_wrist = config.get('wrist_max_depth', 0.75)
    min_depth = config.get('min_depth', 0.1)
    min_depth_wrist = config.get('min_depth_wrist', 0.01)
    radii_size = config.get('radii_size', 0.002)
    track_trail_length = config.get('track_trail_length', 10)
    track_trail_length_video = config.get('track_trail_length_video', 10)
    
    # Determine actual frame count (minimum across all data)
    actual_frames = num_frames
    for serial, cam in cameras.items():
        actual_frames = min(actual_frames, cam['num_frames'])
    
    max_frames_config = config.get('max_frames')
    if max_frames_config is not None:
        actual_frames = min(actual_frames, max_frames_config)
    
    print(f"[INFO] Rendering {actual_frames} frames...")
    print(f"[INFO] Track points per frame: {total_track_pts}")
    
    # Debug: Check track validity on first frame
    if total_track_pts > 0 and actual_frames > 0:
        sample_tracks = tracks_3d[0]
        valid_tracks = np.isfinite(sample_tracks).all(axis=1)
        print(f"[DEBUG] Frame 0 tracks: {valid_tracks.sum()}/{total_track_pts} valid, range: {sample_tracks.min():.3f} to {sample_tracks.max():.3f}")
    
    # Main render loop
    for frame_idx in range(actual_frames):
        if frame_idx % 50 == 0:
            print(f"  Frame {frame_idx}/{actual_frames}")
        
        rr.set_time(timeline="frame_index", sequence=frame_idx)
        
        # Update gripper
        T_base_ee = gripper_poses[frame_idx]
        gripper_pos = gripper_positions[frame_idx]
        gripper_viz.update(T_base_ee, gripper_pos)
        
        # Get track points for this frame
        track_points_world = tracks_3d[frame_idx] if total_track_pts > 0 else None
        
        # Track trail window
        tracks_window = None
        if total_track_pts > 0 and track_trail_length_video > 1:
            start_idx = max(0, frame_idx - track_trail_length_video + 1)
            tracks_window = tracks_3d[start_idx:frame_idx + 1]
        
        # Log track points in 3D
        if track_points_world is not None:
            rr.log(
                "world/gripper_tracks/points",
                rr.Points3D(
                    positions=track_points_world,
                    colors=(track_colors_float[:, :3] * 255).astype(np.uint8),
                    radii=0.003
                )
            )
        
        # Process each camera
        for serial, cam in cameras.items():
            if frame_idx >= cam['num_frames']:
                continue
            
            # Load RGB and depth
            rgb_path = cam['rgb_frames'][frame_idx]
            depth_path = cam['depth_frames'][frame_idx] if frame_idx < len(cam['depth_frames']) else None
            
            rgb = cv2.imread(rgb_path)
            if rgb is None:
                continue
            rgb_for_cloud = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            
            depth = None
            if depth_path and os.path.exists(depth_path):
                depth = np.load(depth_path)
            
            # Determine camera type and extrinsics
            is_wrist = (serial == wrist_serial)
            
            if is_wrist:
                if wrist_extrinsics is None or frame_idx >= len(wrist_extrinsics):
                    continue
                T_world_cam = wrist_extrinsics[frame_idx]
                max_d = max_depth_wrist
                min_d = min_depth_wrist
                cam_path = "world/wrist_cam"
            else:
                if serial not in external_extrinsics:
                    continue
                T_world_cam = external_extrinsics[serial]
                max_d = max_depth_ext
                min_d = min_depth
                cam_path = f"world/external_cams/{serial}"
            
            # Log camera transform
            rr.log(
                cam_path,
                rr.Transform3D(
                    translation=T_world_cam[:3, 3],
                    mat3x3=T_world_cam[:3, :3],
                    axis_length=0.1
                )
            )
            
            # Log pinhole
            rr.log(
                f"{cam_path}/pinhole",
                rr.Pinhole(
                    image_from_camera=cam['K'],
                    width=cam['width'],
                    height=cam['height']
                )
            )
            
            # Log RGB image
            rr.log(
                f"{cam_path}/pinhole/image",
                rr.Image(rgb_for_cloud)
            )
            
            # Generate and log point cloud from depth
            xyz_world = None
            cloud_colors = None
            if depth is not None:
                xyz_cam, cloud_colors = depth_to_pointcloud(
                    depth, rgb_for_cloud, cam['K'],
                    max_depth=max_d, min_depth=min_d
                )
                
                if len(xyz_cam) > 0:
                    # Transform to world frame
                    xyz_world = transform_points(xyz_cam, T_world_cam)
                    
                    # Log points
                    points_path = "world/points/wrist_cam" if is_wrist else f"world/points/external_cams/{serial}"
                    rr.log(
                        points_path,
                        rr.Points3D(xyz_world, colors=cloud_colors, radii=radii_size)
                    )
            
            # Create video overlay
            frame_overlay = rgb.copy()
            
            # Draw point cloud on video (optional, can be skipped for cleaner track visualization)
            if xyz_world is not None and len(xyz_world) > 0:
                # Project point cloud onto image
                uv_cloud, cols_cloud = project_points_to_image(
                    xyz_world, cam['K'], T_world_cam, cam['width'], cam['height'],
                    colors=cloud_colors, min_depth=min_d
                )
                frame_overlay = draw_points_on_image_fast(frame_overlay, uv_cloud, colors=cols_cloud)
            
            # Project tracks onto image - use very small min_depth to ensure tracks are visible
            # Tracks are at the gripper which may be very close to cameras
            track_min_depth = 0.001  # 1mm - tracks should always be visible
            if track_points_world is not None and len(track_points_world) > 0:
                uv_tracks, cols_tracks = project_points_to_image(
                    track_points_world, cam['K'], T_world_cam, cam['width'], cam['height'],
                    colors=track_colors_rgb, min_depth=track_min_depth
                )
                # Debug: log projection results on first frame
                if frame_idx == 0:
                    print(f"[DEBUG] Camera {serial}: projected {len(uv_tracks)}/{len(track_points_world)} track points")
                
                # Draw tracks with larger radius for visibility
                frame_overlay = draw_points_on_image(
                    frame_overlay, uv_tracks, colors=cols_tracks,
                    radius=5, default_color=(0, 0, 255)
                )
            
            # Draw track trails
            if tracks_window is not None and len(tracks_window) > 0:
                frame_overlay = draw_track_trails_on_image(
                    frame_overlay, tracks_window, cam['K'], T_world_cam,
                    cam['width'], cam['height'], track_colors_rgb, min_depth=track_min_depth
                )
            
            # Write video frame
            if serial in recorders:
                recorders[serial].write_frame(frame_overlay)
    
    # Log track trails in 3D (after main loop for better visualization)
    if total_track_pts > 0:
        for t in range(1, actual_frames):
            rr.set_time(timeline="frame_index", sequence=t)
            trail_len = min(t, track_trail_length)
            for n in range(total_track_pts):
                trail_points = tracks_3d[max(0, t - trail_len):t + 1, n, :]
                if len(trail_points) > 1:
                    segments = np.stack([trail_points[:-1], trail_points[1:]], axis=1)
                    color = track_colors_float[n, :3]
                    rr.log(
                        f"world/gripper_tracks/trails/track_{n:03d}",
                        rr.LineStrips3D(
                            strips=segments,
                            colors=[color] * len(segments),
                            radii=0.001
                        )
                    )
    
    # Cleanup
    for rec in recorders.values():
        rec.close()
    
    print(f"[SUCCESS] Saved RRD to: {rrd_path}")
    print(f"[SUCCESS] Saved videos to: {video_dir}")
    
    return rrd_path


def main():
    parser = argparse.ArgumentParser(
        description="Render validation RRD from extracted RGB, depth, and tracks."
    )
    parser.add_argument(
        "--episode_id",
        required=True,
        help='Episode ID, e.g., "AUTOLab+84bd5053+2023-08-18-12h-01m-10s"',
    )
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
        help="Path to config YAML file.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run without spawning Rerun viewer (default: True)",
    )
    parser.add_argument(
        "--spawn-viewer",
        action="store_true",
        help="Spawn Rerun viewer while rendering",
    )
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"=== Rendering Validation RRD ===")
    print(f"Episode: {args.episode_id}")
    
    # Parse episode ID
    episode_info = parse_episode_id(args.episode_id)
    print(f"  Lab: {episode_info['lab']}")
    print(f"  Date: {episode_info['date']}")
    
    # Find processed episode
    episode_paths = find_processed_episode(config['output_root'], episode_info)
    print(f"  Found: {episode_paths['relative_path']}")
    
    # Render
    headless = not args.spawn_viewer
    rrd_path = render_episode(episode_paths, config, episode_info, headless=headless)
    
    print("\n=== Done ===")
    print(f"Output: {rrd_path}")


if __name__ == "__main__":
    main()
