"""
Instrumented copy of generate_tracks_and_metadata.py with detailed timing.

Usage:
    python debug_tracks_timing.py --episode_id "LAB+hash+YYYY-MM-DD-HHh-MMm-SSs" --config conversions/droid/training_data/config.yaml
Outputs:
    - Same artifacts as the main script (tracks.npz, extrinsics.npz, quality.json)
    - Detailed timing breakdown printed to stdout with sub-step durations
"""

import argparse
import json
import os
import sys
import glob
import re
import time
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
)
from tracking import ContactSurfaceTracker, compute_finger_transforms  # type: ignore


def now():
    return time.perf_counter()


def parse_episode_id(episode_id: str) -> dict:
    parts = episode_id.split('+')
    if len(parts) != 3:
        raise ValueError(f"Invalid episode ID format: {episode_id}")
    
    lab = parts[0]
    episode_hash = parts[1]
    datetime_part = parts[2]
    
    match = re.match(r'(\d{4}-\d{2}-\d{2})-(\d+)h-(\d+)m-(\d+)s', datetime_part)
    if not match:
        raise ValueError(f"Invalid datetime format: {datetime_part}")
    
    date = match.group(1)
    hour = match.group(2)
    minute = match.group(3)
    second = match.group(4)
    
    from datetime import datetime
    dt = datetime.strptime(f"{date} {hour}:{minute}:{second}", "%Y-%m-%d %H:%M:%S")
    timestamp_folder = dt.strftime("%a_%b_%e_%H:%M:%S_%Y").replace(" ", "_")
    
    return {
        'lab': lab,
        'hash': episode_hash,
        'date': date,
        'timestamp_folder': timestamp_folder,
        'time_str': f"{hour}:{minute}:{second}",
    }


def find_episode_paths(droid_root: str, episode_info: dict, extra_roots: list = None) -> dict:
    lab = episode_info['lab']
    date = episode_info['date']
    timestamp_folder = episode_info['timestamp_folder']
    
    roots_to_search = [droid_root]
    if extra_roots:
        roots_to_search.extend(extra_roots)
    
    for root in roots_to_search:
        for outcome in ['success', 'failure']:
            base_path = os.path.join(root, lab, outcome, date, timestamp_folder)
            h5_path = os.path.join(base_path, 'trajectory.h5')
            if os.path.exists(h5_path):
                recordings_dir = os.path.join(base_path, 'recordings', 'SVO')
                metadata_files = glob.glob(os.path.join(base_path, 'metadata_*.json'))
                metadata_path = metadata_files[0] if metadata_files else None
                relative_path = os.path.join(lab, outcome, date, timestamp_folder)
                return {
                    'h5_path': h5_path,
                    'recordings_dir': recordings_dir,
                    'metadata_path': metadata_path,
                    'relative_path': relative_path,
                    'outcome': outcome,
                }
    
    raise FileNotFoundError(f"Episode not found: {episode_info}")


def load_cam2base_for_episode(cam2base_path: str, episode_info: dict) -> dict:
    with open(cam2base_path, 'r') as f:
        all_calibs = json.load(f)
    
    date = episode_info['date']
    time_parts = episode_info['time_str'].split(':')
    target_suffix = f"{date}-{time_parts[0]}h-{time_parts[1]}m-{time_parts[2]}s"
    
    for key in all_calibs.keys():
        if key.endswith(target_suffix):
            return {key: all_calibs[key]}
    
    for key in all_calibs.keys():
        if episode_info['date'] in key and time_parts[2] in key:
            return {key: all_calibs[key]}
    
    raise KeyError(f"No calibration found for episode: {episode_info}")


def generate_tracks(h5_path: str, num_track_points: int, max_frames: int = None, mesh_path: str = "/workspace/third_party/robotiq_arg85_description/meshes/inner_finger_fine.STL") -> tuple[dict, dict]:
    timing = {}
    
    t0 = now()
    with h5py.File(h5_path, 'r') as h5_file:
        cartesian_positions = h5_file['observation/robot_state/cartesian_position'][:]
        gripper_positions = h5_file['observation/robot_state/gripper_position'][:]
    timing['load_h5'] = now() - t0
    
    num_frames = len(cartesian_positions)
    actual_frames = min(max_frames, num_frames) if max_frames is not None else num_frames
    
    t_init = now()
    contact_tracker = ContactSurfaceTracker(num_track_points=num_track_points, mesh_path=mesh_path)
    num_contact_pts = len(contact_tracker.contact_points_local) if contact_tracker.contact_points_local is not None else 0
    total_track_pts = num_contact_pts * 2
    timing['init_tracker'] = now() - t_init
    
    tracks_3d = np.zeros((actual_frames, total_track_pts, 3), dtype=np.float32)
    gripper_poses = []
    R_fix = R.from_euler('z', 90, degrees=True).as_matrix()
    
    t_loop = now()
    for i in range(actual_frames):
        if i % 100 == 0:
            elapsed = now() - t_loop
            print(f"[track] frame {i}/{actual_frames} elapsed_loop={elapsed:.3f}s")
        T_base_ee = pose6_to_T(cartesian_positions[i])
        T_base_ee[:3, :3] = T_base_ee[:3, :3] @ R_fix
        gripper_poses.append(T_base_ee.copy())
        
        if num_contact_pts > 0:
            pts_left, pts_right = contact_tracker.get_contact_points_world(T_base_ee, gripper_positions[i])
            if pts_left is not None:
                tracks_3d[i, :num_contact_pts, :] = pts_left
                tracks_3d[i, num_contact_pts:, :] = pts_right
    timing['loop'] = now() - t_loop
    
    timing['frames'] = actual_frames
    timing['num_contact_points'] = total_track_pts
    
    return {
        'tracks_3d': tracks_3d,
        'contact_points_local': contact_tracker.contact_points_local,
        'gripper_poses': np.stack(gripper_poses, axis=0),
        'gripper_positions': gripper_positions[:actual_frames],
        'cartesian_positions': cartesian_positions[:actual_frames],
        'num_frames': actual_frames,
        'num_points_per_finger': num_contact_pts,
    }, timing


def compute_extrinsics(h5_path: str, metadata_path: str, cam2base_snippet: dict, max_frames: int = None) -> tuple[dict, dict]:
    timing = {}
    t0 = now()
    with h5py.File(h5_path, 'r') as h5_file:
        cartesian_positions = h5_file['observation/robot_state/cartesian_position'][:]
    timing['load_h5'] = now() - t0
    
    num_frames = len(cartesian_positions)
    actual_frames = min(max_frames, num_frames) if max_frames is not None else num_frames
    
    t_ext = now()
    external_extrinsics = {}
    episode_key = list(cam2base_snippet.keys())[0]
    episode_data = cam2base_snippet[episode_key]
    for cam_id, transform_list in episode_data.items():
        if cam_id.isdigit():
            T_world_cam = rvec_tvec_to_matrix(transform_list)
            external_extrinsics[cam_id] = T_world_cam
    timing['external_extrinsics'] = now() - t_ext
    
    wrist_extrinsics = None
    wrist_serial = None
    t_wrist_start = now()
    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            meta = json.load(f)
        wrist_serial = str(meta.get("wrist_cam_serial", ""))
        wrist_pose_t0 = meta.get("wrist_cam_extrinsics")
        if wrist_pose_t0:
            T_ee_cam = compute_wrist_cam_offset(wrist_pose_t0, cartesian_positions[0])
            R_fix = R.from_euler('z', 90, degrees=True).as_matrix()
            wrist_transforms = []
            for i in range(actual_frames):
                T_base_ee_t = pose6_to_T(cartesian_positions[i])
                T_base_ee_t[:3, :3] = T_base_ee_t[:3, :3] @ R_fix
                T_world_cam = T_base_ee_t @ T_ee_cam
                wrist_transforms.append(T_world_cam)
            wrist_extrinsics = np.stack(wrist_transforms, axis=0)
    timing['wrist_extrinsics'] = now() - t_wrist_start
    timing['frames'] = actual_frames
    return {
        'external_extrinsics': external_extrinsics,
        'wrist_extrinsics': wrist_extrinsics,
        'wrist_serial': wrist_serial,
        'num_frames': actual_frames,
    }, timing


def main():
    parser = argparse.ArgumentParser(description="Debug timing for tracks/extrinsics generation.")
    parser.add_argument("--episode_id", required=True)
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    args = parser.parse_args()
    
    total_start = now()
    stage_times = {}
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    episode_info = parse_episode_id(args.episode_id)
    
    t_find = now()
    extra_roots = [config.get('download_dir', './droid_downloads')]
    episode_paths = find_episode_paths(config['droid_root'], episode_info, extra_roots)
    stage_times['find_paths'] = now() - t_find
    
    t_cam = now()
    cam2base_snippet = load_cam2base_for_episode(config['cam2base_extrinsics_path'], episode_info)
    stage_times['load_cam2base'] = now() - t_cam
    
    output_dir = os.path.join(config['output_root'], episode_paths['relative_path'])
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"=== Timing Debug: {args.episode_id} ===")
    print(f"output_dir={output_dir}")
    
    # Tracks
    t_tracks = now()
    tracks_data, tracks_timing = generate_tracks(
        episode_paths['h5_path'],
        num_track_points=config.get('num_track_points', 480),
        max_frames=config.get('max_frames'),
        mesh_path=config.get('finger_mesh_path', "/workspace/third_party/robotiq_arg85_description/meshes/inner_finger_fine.STL"),
    )
    stage_times['tracks_total'] = now() - t_tracks
    
    tracks_path = os.path.join(output_dir, 'tracks.npz')
    np.savez(
        tracks_path,
        tracks_3d=tracks_data['tracks_3d'],
        contact_points_local=tracks_data['contact_points_local'],
        gripper_poses=tracks_data['gripper_poses'],
        gripper_positions=tracks_data['gripper_positions'],
        cartesian_positions=tracks_data['cartesian_positions'],
        num_frames=tracks_data['num_frames'],
        num_points_per_finger=tracks_data['num_points_per_finger'],
        fps=config.get('fps', 30.0),
    )
    
    # Extrinsics
    t_ext = now()
    extrinsics_data, extrinsics_timing = compute_extrinsics(
        episode_paths['h5_path'],
        episode_paths['metadata_path'],
        cam2base_snippet,
        max_frames=config.get('max_frames'),
    )
    stage_times['extrinsics_total'] = now() - t_ext
    
    extrinsics_path = os.path.join(output_dir, 'extrinsics.npz')
    save_dict = {'num_frames': extrinsics_data['num_frames']}
    for cam_id, T in extrinsics_data['external_extrinsics'].items():
        save_dict[f'external_{cam_id}'] = T
    if extrinsics_data['wrist_extrinsics'] is not None:
        save_dict['wrist_extrinsics'] = extrinsics_data['wrist_extrinsics']
        save_dict['wrist_serial'] = extrinsics_data['wrist_serial']
    np.savez(extrinsics_path, **save_dict)
    
    # Quality
    t_quality = now()
    quality_data = {
        'episode_id': args.episode_id,
        'source_path': episode_paths['relative_path'],
        'outcome': episode_paths['outcome'],
        'num_frames': tracks_data['num_frames'],
        'num_track_points': tracks_data['num_points_per_finger'] * 2,
        'external_cameras': list(extrinsics_data['external_extrinsics'].keys()),
        'wrist_camera': extrinsics_data['wrist_serial'],
        'cam2base_calibration': cam2base_snippet,
    }
    quality_path = os.path.join(output_dir, 'quality.json')
    with open(quality_path, 'w') as f:
        json.dump(quality_data, f, indent=2)
    stage_times['quality'] = now() - t_quality
    
    stage_times['total'] = now() - total_start
    
    print("\n--- Timing (seconds) ---")
    for k, v in stage_times.items():
        print(f"{k:20s} {v:.4f}")
    print("\n--- Tracks timing ---")
    for k, v in tracks_timing.items():
        print(f"{k:20s} {v}")
    print("\n--- Extrinsics timing ---")
    for k, v in extrinsics_timing.items():
        print(f"{k:20s} {v}")
    print("\nSaved:")
    print(f"  {tracks_path}")
    print(f"  {extrinsics_path}")
    print(f"  {quality_path}")


if __name__ == "__main__":
    main()
