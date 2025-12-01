"""Export gripper tracks and extrinsics for training, mirroring DROID folder layout."""

import argparse
import glob
import json
import os
from typing import Dict, Tuple

import h5py
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

from utils import (
    ContactSurfaceTracker,
    compute_wrist_cam_offset,
    external_cam_to_world,
    find_episode_data_by_date,
    pose6_to_T,
    precompute_wrist_trajectory,
)


def _derive_episode_relative_path(h5_path: str) -> str:
    """Take last 5 components to keep org/success/date/timestamp layout."""
    parts = os.path.normpath(os.path.dirname(h5_path)).split(os.sep)
    return os.path.join(*parts[-5:]) if len(parts) >= 5 else os.path.basename(os.path.dirname(h5_path))


def _load_wrist_transforms(cartesian_positions: np.ndarray, config: Dict) -> Tuple[str, list]:
    wrist_serial = None
    wrist_cam_transforms = []

    metadata_path = config.get("metadata_path")
    if metadata_path is None:
        episode_dir = os.path.dirname(config["h5_path"])
        metadata_files = glob.glob(os.path.join(episode_dir, "metadata_*.json"))
        if metadata_files:
            metadata_path = metadata_files[0]

    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            meta = json.load(f)
        wrist_serial = str(meta.get("wrist_cam_serial", "") or "")
        wrist_pose_t0 = meta.get("wrist_cam_extrinsics")
        if wrist_pose_t0:
            T_ee_cam = compute_wrist_cam_offset(wrist_pose_t0, cartesian_positions[0])
            wrist_cam_transforms = precompute_wrist_trajectory(cartesian_positions, T_ee_cam)

    return wrist_serial, wrist_cam_transforms


def _save_json(path: str, data: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Export tracks/extrinsics for training.")
    parser.add_argument("--config", default="conversions/droid/config.yaml", help="Path to YAML config file.")
    parser.add_argument("--output_root", default=None, help="Override root for training outputs.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        CONFIG = yaml.safe_load(f)

    h5_path = CONFIG["h5_path"]
    output_root = args.output_root or CONFIG.get("training_output_root", "training_output")
    fps = CONFIG.get("fps", 30.0)

    # Derive episode-relative path from H5 location
    rel_path = _derive_episode_relative_path(h5_path)
    episode_out = os.path.join(output_root, rel_path)
    os.makedirs(episode_out, exist_ok=True)

    # Load robot data
    h5_file = h5py.File(h5_path, "r")
    cartesian_positions = h5_file["observation/robot_state/cartesian_position"][:]
    gripper_positions = h5_file["observation/robot_state/gripper_position"][:]
    h5_file.close()
    num_frames = len(cartesian_positions)
    max_frames = CONFIG.get("max_frames", num_frames)
    actual_frames = min(max_frames, num_frames)

    # Wrist transforms
    wrist_serial, wrist_cam_transforms = _load_wrist_transforms(cartesian_positions, CONFIG)

    # External extrinsics
    ext_data = find_episode_data_by_date(h5_path, CONFIG["extrinsics_json_path"]) or {}
    ext_world = {}
    for cam_id, transform_list in ext_data.items():
        if cam_id.isdigit():
            ext_world[cam_id] = external_cam_to_world(transform_list)

    # Track computation (no rendering)
    track_trail_length = CONFIG.get("track_trail_length", 10)
    contact_tracker = ContactSurfaceTracker(num_track_points=CONFIG.get("num_track_points", 24))
    num_contact_pts = len(contact_tracker.contact_points_local) if contact_tracker.contact_points_local is not None else 0
    total_track_pts = num_contact_pts * 2

    tracks_3d = np.zeros((actual_frames, total_track_pts, 3), dtype=np.float32)
    gripper_poses = []

    R_fix = R.from_euler("z", 90, degrees=True).as_matrix()
    for i in range(actual_frames):
        T_base_ee = pose6_to_T(cartesian_positions[i])
        T_base_ee[:3, :3] = T_base_ee[:3, :3] @ R_fix
        gripper_poses.append(T_base_ee.copy())
        if total_track_pts == 0:
            continue
        pts_left, pts_right = contact_tracker.get_contact_points_world(T_base_ee, gripper_positions[i])
        if pts_left is not None:
            tracks_3d[i, :num_contact_pts, :] = pts_left
            tracks_3d[i, num_contact_pts:, :] = pts_right

    gripper_poses_array = np.stack(gripper_poses, axis=0) if gripper_poses else np.empty((0, 4, 4))

    # Save tracks NPZ
    tracks_path = os.path.join(episode_out, "tracks.npz")
    np.savez(
        tracks_path,
        tracks_3d=tracks_3d,
        contact_points_local=contact_tracker.contact_points_local,
        gripper_poses=gripper_poses_array,
        gripper_positions=gripper_positions[:actual_frames],
        cartesian_positions=cartesian_positions[:actual_frames],
        num_frames=actual_frames,
        num_points_per_finger=num_contact_pts,
        fps=fps,
    )

    # Save extrinsics (external static + wrist trajectory if available)
    extrinsics_path = os.path.join(episode_out, "extrinsics.npz")
    np.savez(
        extrinsics_path,
        external_world_T_cam=np.array(list(ext_world.values())) if ext_world else np.empty((0, 4, 4)),
        external_ids=np.array(list(ext_world.keys())),
        wrist_serial=wrist_serial if wrist_serial else "",
        wrist_world_T_cam=np.array(wrist_cam_transforms) if wrist_cam_transforms else np.empty((0, 4, 4)),
    )

    # Save cam2base snippet as quality.json
    quality_path = os.path.join(episode_out, "quality.json")
    _save_json(quality_path, ext_data)

    print(f"[SUCCESS] Saved training tracks to: {tracks_path}")
    print(f"[INFO] Extrinsics saved to: {extrinsics_path}")
    print(f"[INFO] Cam2base snippet saved to: {quality_path}")


if __name__ == "__main__":
    main()
