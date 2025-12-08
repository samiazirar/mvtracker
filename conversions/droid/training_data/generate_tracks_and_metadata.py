"""Generate tracks.npz, extrinsics.npz, and quality.json for DROID training data.

This script processes DROID episodes and generates:
1. tracks.npz - Gripper contact surface tracks in world coordinates + 2D projections per camera
2. extrinsics.npz - Camera extrinsics (fixed external cams + per-frame wrist cam)
3. quality.json - Copy of cam2base calibration snippet + metadata

tracks.npz contents:
    - tracks_3d: [T, N, 3] float32 - 3D track points in world frame
    - tracks_2d_{camera_serial}: [T, N, 2] float32 - 2D projections per camera (NaN for invalid)
    - intrinsics_{camera_serial}: [3, 3] float64 - Camera intrinsic matrix
    - image_size_{camera_serial}: [2] int - Image dimensions [width, height]
    - cameras_with_2d_tracks: list of camera serials that have 2D tracks
    - contact_points_local: Local coordinates of contact points on gripper
    - gripper_poses: [T, 4, 4] float32 - End-effector poses
    - gripper_positions: [T] float32 - Gripper aperture values
    - cartesian_positions: [T, 6] float32 - Robot cartesian state
    - contact_centroids: [T, 3] float32 - Center of mass of all contact points per frame
    - contact_frames: [T, 4, 4] float32 - Combined contact frame (centroid + EE orientation)
    - left_contact_frames: [T, 4, 4] float32 - Left finger contact frames
    - right_contact_frames: [T, 4, 4] float32 - Right finger contact frames
    - normalized_centroids: [M, 3] float32 - Centroids resampled at 1mm distance steps
    - normalized_frames: [M, 4, 4] float32 - Contact frames resampled at 1mm distance steps
    - normalized_tracks_3d: [M, N, 3] float32 - Contact points resampled at 1mm distance steps
    - normalized_left_frames: [M, 4, 4] float32 - Left finger frames resampled at 1mm steps
    - normalized_right_frames: [M, 4, 4] float32 - Right finger frames resampled at 1mm steps
    - cumulative_distance_mm: [T] float32 - Cumulative distance traveled at each frame (mm)
    - frame_to_normalized_idx: [T] int32 - Mapping from original frame to normalized index
    - normalized_step_size_mm: float - Step size used for normalization (default: 1.0)
    - num_normalized_steps: int (M) - Number of normalized flow steps
    - num_frames: int - Number of frames
    - num_points_per_finger: int - Number of track points per finger
    - fps: float - Frame rate

Usage:
    python generate_tracks_and_metadata.py --episode_id "AUTOLab+84bd5053+2023-08-18-12h-01m-10s"
    python generate_tracks_and_metadata.py --episode_id "AUTOLab+84bd5053+2023-08-18-12h-01m-10s" --config custom_config.yaml

The output maintains DROID folder structure:
    {output_root}/{lab}/success/{date}/{timestamp}/
        ├── tracks.npz
        ├── extrinsics.npz
        └── quality.json

Note: For 2D tracks to be generated, extract_rgb_depth.py must be run first to save
camera intrinsics.json files in the recordings directory.
"""

import argparse
import json
import os
import sys
import glob
import re
from pathlib import Path

import h5py
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

# Minimal imports from utils without pulling the full package (which brings heavy deps)
UTILS_DIR = Path(__file__).resolve().parents[1] / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from transforms import (  # type: ignore
    pose6_to_T,
    rvec_tvec_to_matrix,
    compute_wrist_cam_offset,
    precompute_wrist_trajectory,
    external_cam_to_world,
    project_tracks_to_2d,
)
from tracking import ContactSurfaceTracker, compute_finger_transforms, compute_normalized_flow  # type: ignore


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


def find_episode_paths(droid_root: str, episode_info: dict, extra_roots: list = None) -> dict:
    """Find all paths for an episode.
    
    Searches in droid_root and any extra_roots (e.g., download directory).
    
    Returns:
        dict with keys: h5_path, recordings_dir, metadata_path, relative_path
    """
    lab = episode_info['lab']
    date = episode_info['date']
    timestamp_folder = episode_info['timestamp_folder']
    
    # Build list of roots to search
    roots_to_search = [droid_root]
    if extra_roots:
        roots_to_search.extend(extra_roots)
    
    # Search in all roots, both success and failure folders
    for root in roots_to_search:
        for outcome in ['success', 'failure']:
            base_path = os.path.join(root, lab, outcome, date, timestamp_folder)
            h5_path = os.path.join(base_path, 'trajectory.h5')
            
            if os.path.exists(h5_path):
                recordings_dir = os.path.join(base_path, 'recordings', 'SVO')
                
                # Find metadata file
                metadata_files = glob.glob(os.path.join(base_path, 'metadata_*.json'))
                metadata_path = metadata_files[0] if metadata_files else None
                
                # Relative path for output structure
                relative_path = os.path.join(lab, outcome, date, timestamp_folder)
                
                return {
                    'h5_path': h5_path,
                    'recordings_dir': recordings_dir,
                    'metadata_path': metadata_path,
                    'relative_path': relative_path,
                    'outcome': outcome,
                }
    
    raise FileNotFoundError(f"Episode not found: {episode_info['full_id']}")


def load_cam2base_for_episode(cam2base_path: str, episode_info: dict) -> dict:
    """Load camera-to-base calibration for specific episode from superset."""
    with open(cam2base_path, 'r') as f:
        all_calibs = json.load(f)
    
    # Build the target key suffix from episode ID
    date = episode_info['date']
    time_parts = episode_info['time_str'].split(':')
    target_suffix = f"{date}-{time_parts[0]}h-{time_parts[1]}m-{time_parts[2]}s"
    
    # Search for matching key
    for key in all_calibs.keys():
        if key.endswith(target_suffix):
            return {key: all_calibs[key]}
    
    # Fallback: try partial match
    for key in all_calibs.keys():
        if episode_info['date'] in key:
            # Check if time roughly matches
            if time_parts[2] in key:  # Match seconds at least
                return {key: all_calibs[key]}
    
    raise KeyError(f"No calibration found for episode: {episode_info['full_id']}")


def load_camera_intrinsics(recordings_dir: str) -> dict:
    """Load camera intrinsics from recordings output directory.
    
    Looks for intrinsics.json files in each camera subdirectory.
    
    Args:
        recordings_dir: Path to recordings directory containing camera folders
        
    Returns:
        dict mapping camera_serial to {'K': 3x3 matrix, 'width': int, 'height': int}
    """
    intrinsics = {}
    
    if not os.path.exists(recordings_dir):
        return intrinsics
    
    # Look for camera subdirectories
    for entry in os.listdir(recordings_dir):
        cam_dir = os.path.join(recordings_dir, entry)
        if not os.path.isdir(cam_dir):
            continue
        
        intrinsics_path = os.path.join(cam_dir, 'intrinsics.json')
        if os.path.exists(intrinsics_path):
            with open(intrinsics_path, 'r') as f:
                data = json.load(f)
            
            # Build 3x3 intrinsic matrix
            K = np.array([
                [data['fx'], 0, data['cx']],
                [0, data['fy'], data['cy']],
                [0, 0, 1]
            ], dtype=np.float64)
            
            intrinsics[entry] = {
                'K': K,
                'width': data['width'],
                'height': data['height'],
            }
    
    return intrinsics


def compute_2d_tracks(
    tracks_3d: np.ndarray,
    camera_intrinsics: dict,
    extrinsics_data: dict,
    min_depth: float = 0.01,
) -> dict:
    """Compute 2D track projections for all cameras.
    
    Args:
        tracks_3d: [T, N, 3] array of 3D track points
        camera_intrinsics: dict from load_camera_intrinsics()
        extrinsics_data: dict from compute_extrinsics()
        min_depth: Minimum depth for valid projection
        
    Returns:
        dict mapping camera_serial to [T, N, 2] array of 2D projections
    """
    tracks_2d = {}
    
    # Process external cameras (static extrinsics)
    for cam_id, T_world_cam in extrinsics_data['external_extrinsics'].items():
        if cam_id not in camera_intrinsics:
            print(f"  [WARN] No intrinsics for external camera {cam_id}, skipping 2D projection")
            continue
        
        cam_info = camera_intrinsics[cam_id]
        tracks_2d[cam_id] = project_tracks_to_2d(
            tracks_3d,
            cam_info['K'],
            T_world_cam,  # Static extrinsics
            cam_info['width'],
            cam_info['height'],
            min_depth=min_depth,
            clip_to_bounds=False,  # Keep all projections, even outside image
        )
    
    # Process wrist camera (per-frame extrinsics)
    wrist_serial = extrinsics_data.get('wrist_serial')
    wrist_extrinsics = extrinsics_data.get('wrist_extrinsics')
    
    if wrist_serial and wrist_extrinsics is not None:
        if wrist_serial not in camera_intrinsics:
            print(f"  [WARN] No intrinsics for wrist camera {wrist_serial}, skipping 2D projection")
        else:
            cam_info = camera_intrinsics[wrist_serial]
            tracks_2d[wrist_serial] = project_tracks_to_2d(
                tracks_3d,
                cam_info['K'],
                wrist_extrinsics,  # Per-frame extrinsics [T, 4, 4]
                cam_info['width'],
                cam_info['height'],
                min_depth=min_depth,
                clip_to_bounds=False,
            )
    
    return tracks_2d


def generate_tracks(h5_path: str, num_track_points: int, max_frames: int = None, mesh_path: str = "/workspace/third_party/robotiq_arg85_description/meshes/inner_finger_fine.STL") -> dict:
    """Generate gripper contact surface tracks.

    Returns:
        dict containing tracks_3d, contact frames, gripper_poses, and metadata
    """
    # Load trajectory
    with h5py.File(h5_path, 'r') as h5_file:
        cartesian_positions = h5_file['observation/robot_state/cartesian_position'][:]
        gripper_positions = h5_file['observation/robot_state/gripper_position'][:]

    num_frames = len(cartesian_positions)
    if max_frames is not None:
        actual_frames = min(max_frames, num_frames)
    else:
        actual_frames = num_frames

    # Initialize contact tracker
    contact_tracker = ContactSurfaceTracker(num_track_points=num_track_points,mesh_path=mesh_path)
    num_contact_pts = len(contact_tracker.contact_points_local) if contact_tracker.contact_points_local is not None else 0
    total_track_pts = num_contact_pts * 2  # Both fingers

    print(f"[INFO] Tracking {total_track_pts} contact points across both fingers")

    # Storage
    tracks_3d = np.zeros((actual_frames, total_track_pts, 3), dtype=np.float32)
    gripper_poses = []
    contact_centroids = np.zeros((actual_frames, 3), dtype=np.float32)
    contact_frames = np.zeros((actual_frames, 4, 4), dtype=np.float32)
    left_contact_frames = np.zeros((actual_frames, 4, 4), dtype=np.float32)
    right_contact_frames = np.zeros((actual_frames, 4, 4), dtype=np.float32)

    # Rotation fix for gripper
    R_fix = R.from_euler('z', 90, degrees=True).as_matrix()

    for i in range(actual_frames):
        if i % 100 == 0:
            print(f"  Processing frame {i}/{actual_frames}")

        # Compute end-effector pose with rotation fix
        T_base_ee = pose6_to_T(cartesian_positions[i])
        T_base_ee[:3, :3] = T_base_ee[:3, :3] @ R_fix
        gripper_poses.append(T_base_ee.copy())

        # Get contact points AND contact frames in world space
        if num_contact_pts > 0:
            result = contact_tracker.get_contact_points_and_frames(
                T_base_ee, gripper_positions[i]
            )
            pts_left, pts_right, centroid, frame, left_frame, right_frame = result

            if pts_left is not None:
                tracks_3d[i, :num_contact_pts, :] = pts_left
                tracks_3d[i, num_contact_pts:, :] = pts_right
                contact_centroids[i] = centroid
                contact_frames[i] = frame
                left_contact_frames[i] = left_frame
                right_contact_frames[i] = right_frame

    # Compute normalized flow (resampled at 1mm steps)
    # This normalizes centroids, frames, contact points, AND per-finger frames
    step_size_mm = 1.0
    print(f"[INFO] Computing normalized flow at {step_size_mm}mm steps...")
    norm_result = compute_normalized_flow(
        contact_centroids, contact_frames, step_size_mm=step_size_mm,
        tracks_3d=tracks_3d,
        left_contact_frames=left_contact_frames,
        right_contact_frames=right_contact_frames,
    )
    print(f"[INFO] Normalized flow: {norm_result['num_normalized_steps']} steps "
          f"(total distance: {norm_result['cumulative_distance_mm'][-1]:.1f}mm)")

    return {
        # Original (unnormalized) data
        'tracks_3d': tracks_3d,
        'contact_points_local': contact_tracker.contact_points_local,
        'gripper_poses': np.stack(gripper_poses, axis=0),
        'gripper_positions': gripper_positions[:actual_frames],
        'cartesian_positions': cartesian_positions[:actual_frames],
        'contact_centroids': contact_centroids,
        'contact_frames': contact_frames,
        'left_contact_frames': left_contact_frames,
        'right_contact_frames': right_contact_frames,
        # Normalized flow data (all resampled at 1mm steps)
        'normalized_centroids': norm_result['normalized_centroids'],
        'normalized_frames': norm_result['normalized_frames'],
        'normalized_tracks_3d': norm_result['normalized_tracks_3d'],
        'normalized_left_frames': norm_result['normalized_left_frames'],
        'normalized_right_frames': norm_result['normalized_right_frames'],
        'cumulative_distance_mm': norm_result['cumulative_distance_mm'],
        'frame_to_normalized_idx': norm_result['frame_to_normalized_idx'],
        'normalized_step_size_mm': step_size_mm,
        'num_normalized_steps': norm_result['num_normalized_steps'],
        'num_frames': actual_frames,
        'num_points_per_finger': num_contact_pts,
    }


def compute_extrinsics(
    h5_path: str,
    metadata_path: str,
    cam2base_snippet: dict,
    max_frames: int = None
) -> dict:
    """Compute camera extrinsics for all cameras.
    
    Returns:
        dict with external_extrinsics (fixed) and wrist_extrinsics (per-frame)
    """
    # Load trajectory
    with h5py.File(h5_path, 'r') as h5_file:
        cartesian_positions = h5_file['observation/robot_state/cartesian_position'][:]
    
    num_frames = len(cartesian_positions)
    if max_frames is not None:
        actual_frames = min(max_frames, num_frames)
    else:
        actual_frames = num_frames
    
    # Extract external camera extrinsics from cam2base snippet
    external_extrinsics = {}
    episode_key = list(cam2base_snippet.keys())[0]
    episode_data = cam2base_snippet[episode_key]
    
    for cam_id, transform_list in episode_data.items():
        if cam_id.isdigit():
            # Convert to 4x4 matrix
            T_world_cam = rvec_tvec_to_matrix(transform_list)
            external_extrinsics[cam_id] = T_world_cam
    
    # Compute wrist camera extrinsics (per-frame)
    wrist_extrinsics = None
    wrist_serial = None
    
    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            meta = json.load(f)
        
        wrist_serial = str(meta.get("wrist_cam_serial", ""))
        wrist_pose_t0 = meta.get("wrist_cam_extrinsics")
        
        if wrist_pose_t0:
            # Compute wrist camera offset
            T_ee_cam = compute_wrist_cam_offset(wrist_pose_t0, cartesian_positions[0])
            
            # Rotation fix
            R_fix = R.from_euler('z', 90, degrees=True).as_matrix()
            
            # Precompute wrist transforms with rotation fix
            wrist_transforms = []
            for i in range(actual_frames):
                T_base_ee_t = pose6_to_T(cartesian_positions[i])
                T_base_ee_t[:3, :3] = T_base_ee_t[:3, :3] @ R_fix
                T_world_cam = T_base_ee_t @ T_ee_cam
                wrist_transforms.append(T_world_cam)
            
            wrist_extrinsics = np.stack(wrist_transforms, axis=0)  # [T, 4, 4]
    
    return {
        'external_extrinsics': external_extrinsics,  # dict of cam_id -> [4, 4]
        'wrist_extrinsics': wrist_extrinsics,  # [T, 4, 4] or None
        'wrist_serial': wrist_serial,
        'num_frames': actual_frames,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate tracks and metadata for DROID training data."
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
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"=== Generating Tracks and Metadata ===")
    print(f"Episode: {args.episode_id}")
    
    # Parse episode ID
    episode_info = parse_episode_id(args.episode_id)
    print(f"  Lab: {episode_info['lab']}")
    print(f"  Date: {episode_info['date']}")
    
    # Find episode paths (search both droid_root and download directory)
    extra_roots = [config.get('download_dir', './droid_downloads')]
    episode_paths = find_episode_paths(config['droid_root'], episode_info, extra_roots)
    print(f"  Found: {episode_paths['relative_path']}")
    
    # Load cam2base calibration
    cam2base_snippet = load_cam2base_for_episode(
        config['cam2base_extrinsics_path'], episode_info
    )
    print(f"  Calibration key: {list(cam2base_snippet.keys())[0]}")
    
    # Create output directory
    output_dir = os.path.join(config['output_root'], episode_paths['relative_path'])
    os.makedirs(output_dir, exist_ok=True)
    print(f"  Output: {output_dir}")
    
    # --- Generate Tracks ---
    print("\n[1/4] Generating gripper tracks...")
    tracks_data = generate_tracks(
        episode_paths['h5_path'],
        num_track_points=config.get('num_track_points', 480),
        max_frames=config.get('max_frames'),
        mesh_path=config.get('finger_mesh_path', "/workspace/third_party/robotiq_arg85_description/meshes/inner_finger_fine.STL"),
    )
    
    # --- Compute Extrinsics ---
    print("\n[2/4] Computing camera extrinsics...")
    extrinsics_data = compute_extrinsics(
        episode_paths['h5_path'],
        episode_paths['metadata_path'],
        cam2base_snippet,
        max_frames=config.get('max_frames'),
    )
    
    # Save extrinsics
    extrinsics_path = os.path.join(output_dir, 'extrinsics.npz')
    save_dict = {
        'num_frames': extrinsics_data['num_frames'],
    }
    
    # Add external camera extrinsics
    for cam_id, T in extrinsics_data['external_extrinsics'].items():
        save_dict[f'external_{cam_id}'] = T
    
    # Add wrist camera extrinsics
    if extrinsics_data['wrist_extrinsics'] is not None:
        save_dict['wrist_extrinsics'] = extrinsics_data['wrist_extrinsics']
        save_dict['wrist_serial'] = extrinsics_data['wrist_serial']
    
    np.savez(extrinsics_path, **save_dict)
    print(f"  Saved: {extrinsics_path}")
    print(f"  External cams: {list(extrinsics_data['external_extrinsics'].keys())}")
    print(f"  Wrist cam: {extrinsics_data['wrist_serial']}")
    
    # --- Load Camera Intrinsics and Compute 2D Tracks ---
    print("\n[3/4] Computing 2D track projections for each camera...")
    
    # Load intrinsics from the output recordings directory (saved by extract_rgb_depth.py)
    output_recordings_dir = os.path.join(output_dir, 'recordings')
    camera_intrinsics = load_camera_intrinsics(output_recordings_dir)
    
    tracks_2d_data = {}
    cameras_with_2d_tracks = []
    
    if camera_intrinsics:
        print(f"  Found intrinsics for cameras: {list(camera_intrinsics.keys())}")
        
        # Compute 2D projections for all cameras
        tracks_2d_data = compute_2d_tracks(
            tracks_data['tracks_3d'],
            camera_intrinsics,
            extrinsics_data,
            min_depth=config.get('min_depth', 0.01),
        )
        
        cameras_with_2d_tracks = list(tracks_2d_data.keys())
        for cam_id, tracks_2d in tracks_2d_data.items():
            # Count valid projections (non-NaN)
            valid_count = np.sum(~np.isnan(tracks_2d[:, :, 0]))
            total_count = tracks_2d.shape[0] * tracks_2d.shape[1]
            print(f"    {cam_id}: {tracks_2d.shape} ({valid_count}/{total_count} valid projections)")
    else:
        print("  [WARN] No camera intrinsics found - skipping 2D track projections")
        print("         (Run extract_rgb_depth.py first to generate intrinsics.json files)")
    
    # --- Save tracks.npz with both 3D and 2D tracks ---
    tracks_path = os.path.join(output_dir, 'tracks.npz')
    tracks_save_dict = {
        'tracks_3d': tracks_data['tracks_3d'],
        'contact_points_local': tracks_data['contact_points_local'],
        'gripper_poses': tracks_data['gripper_poses'],
        'gripper_positions': tracks_data['gripper_positions'],
        'cartesian_positions': tracks_data['cartesian_positions'],
        # Contact frame data (simplified pose of contact points)
        'contact_centroids': tracks_data['contact_centroids'],
        'contact_frames': tracks_data['contact_frames'],
        'left_contact_frames': tracks_data['left_contact_frames'],
        'right_contact_frames': tracks_data['right_contact_frames'],
        # Normalized flow data (resampled at fixed distance steps)
        'normalized_centroids': tracks_data['normalized_centroids'],
        'normalized_frames': tracks_data['normalized_frames'],
        'normalized_tracks_3d': tracks_data['normalized_tracks_3d'],
        'normalized_left_frames': tracks_data['normalized_left_frames'],
        'normalized_right_frames': tracks_data['normalized_right_frames'],
        'cumulative_distance_mm': tracks_data['cumulative_distance_mm'],
        'frame_to_normalized_idx': tracks_data['frame_to_normalized_idx'],
        'normalized_step_size_mm': tracks_data['normalized_step_size_mm'],
        'num_normalized_steps': tracks_data['num_normalized_steps'],
        'num_frames': tracks_data['num_frames'],
        'num_points_per_finger': tracks_data['num_points_per_finger'],
        'fps': config.get('fps', 30.0),
        # List of cameras with 2D tracks (for easy discovery)
        'cameras_with_2d_tracks': np.array(cameras_with_2d_tracks, dtype=object),
    }
    
    # Add 2D tracks for each camera
    for cam_id, tracks_2d in tracks_2d_data.items():
        tracks_save_dict[f'tracks_2d_{cam_id}'] = tracks_2d
        # Also save camera intrinsics for convenience
        if cam_id in camera_intrinsics:
            tracks_save_dict[f'intrinsics_{cam_id}'] = camera_intrinsics[cam_id]['K']
            tracks_save_dict[f'image_size_{cam_id}'] = np.array([
                camera_intrinsics[cam_id]['width'],
                camera_intrinsics[cam_id]['height']
            ])
    
    np.savez(tracks_path, **tracks_save_dict)
    print(f"  Saved: {tracks_path}")
    if cameras_with_2d_tracks:
        print(f"  2D tracks included for: {cameras_with_2d_tracks}")
    
    # --- Save Quality JSON ---
    print("\n[4/4] Saving quality.json...")
    quality_data = {
        'episode_id': args.episode_id,
        'source_path': episode_paths['relative_path'],
        'outcome': episode_paths['outcome'],
        'num_frames': tracks_data['num_frames'],
        'num_track_points': tracks_data['num_points_per_finger'] * 2,
        'external_cameras': list(extrinsics_data['external_extrinsics'].keys()),
        'wrist_camera': extrinsics_data['wrist_serial'],
        'cameras_with_2d_tracks': cameras_with_2d_tracks,
        'cam2base_calibration': cam2base_snippet,
    }
    
    quality_path = os.path.join(output_dir, 'quality.json')
    with open(quality_path, 'w') as f:
        json.dump(quality_data, f, indent=2)
    print(f"  Saved: {quality_path}")
    
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
