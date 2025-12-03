#!/usr/bin/env python3
"""Download a processed DROID episode from Hugging Face and render tracks + videos.

This mirrors the visualization in conversions/droid/create_video_with_tracks.py,
but runs on the preprocessed RGB/depth/tracks/extrinsics artifacts that the
training-data pipeline uploads to Hugging Face (see run_pipeline_cluster_huggingface.sh).

Usage example:
    HF_TOKEN=hf_xxx HF_REPO_ID=sazirarrwth99/trajectory_data     python conversions/droid/training_data/render_visualize_hf.py --episode_id "AUTOLab+84bd5053+2023-08-18-12h-01m-10s"    --config conversions/droid/training_data/config.yaml
"""

import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime
from typing import Dict, Tuple

import cv2
import numpy as np
import rerun as rr
import yaml
from huggingface_hub import snapshot_download

# Make repo root utils importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (  # type: ignore
    GripperVisualizer,
    VideoRecorder,
    draw_points_on_image,
    draw_points_on_image_fast,
    draw_track_trails_on_image,
    pose6_to_T,
    project_points_to_image,
    transform_points,
)


DEFAULT_CONFIG = {
    "fps": 30.0,
    "max_frames": None,
    "track_trail_length": 10,
    "track_trail_length_video": 10,
    "ext_max_depth": 1.5,
    "wrist_max_depth": 0.75,
    "min_depth": 0.1,
    "min_depth_wrist": 0.01,
    "radii_size": 0.001,
    "video_output_path": "point_clouds/videos",
    "rrd_output_path": "point_clouds/droid_full_fusion_gripper.rrd",
}


def parse_episode_id(episode_id: str) -> dict:
    """Parse episode ID into components and timestamp folder."""
    parts = episode_id.split("+")
    if len(parts) != 3:
        raise ValueError(f"Invalid episode ID format: {episode_id}")

    lab, episode_hash, datetime_part = parts
    match = re.match(r"(\d{4}-\d{2}-\d{2})-(\d+)h-(\d+)m-(\d+)s", datetime_part)
    if not match:
        raise ValueError(f"Invalid datetime format in episode ID: {datetime_part}")

    date = match.group(1)
    hour, minute, second = match.group(2), match.group(3), match.group(4)
    dt = datetime.strptime(f"{date} {hour}:{minute}:{second}", "%Y-%m-%d %H:%M:%S")
    timestamp_folder = dt.strftime("%a_%b_%e_%H:%M:%S_%Y")

    return {
        "lab": lab,
        "hash": episode_hash,
        "date": date,
        "time_str": f"{hour}:{minute}:{second}",
        "timestamp_folder": timestamp_folder,
        "full_id": episode_id,
    }


def depth_to_pointcloud(
    depth: np.ndarray,
    rgb: np.ndarray,
    K: np.ndarray,
    max_depth: float,
    min_depth: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert depth map to a colored point cloud in camera coordinates."""
    if depth is None or rgb is None:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)

    depth = depth.astype(np.float32)
    valid = np.isfinite(depth) & (depth > min_depth) & (depth < max_depth)
    if not np.any(valid):
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)

    v_idx, u_idx = np.where(valid)
    z = depth[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x = (u_idx - cx) * z / fx
    y = (v_idx - cy) * z / fy
    xyz = np.stack([x, y, z], axis=-1)
    colors = rgb[valid]
    return xyz, colors


def download_episode_from_hf(
    repo_id: str,
    episode_info: dict,
    cache_dir: str,
    token: str = None,
    revision: str = None,
    explicit_path: str = None,
) -> Tuple[str, str]:
    """
    Download a single episode folder from Hugging Face.

    If explicit_path is provided, only that path is fetched. Otherwise, the path
    is derived from the episode_id info (searching both success/failure).
    """
    if explicit_path:
        allow_patterns = [f"{explicit_path}/*"]
        snapshot_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            revision=revision,
            allow_patterns=allow_patterns,
            local_dir=cache_dir,
            local_dir_use_symlinks=False,
        )
        candidate = os.path.join(snapshot_path, explicit_path)
        if os.path.exists(candidate):
            # Outcome is embedded in the path (e.g., success/failure)
            outcome = explicit_path.split("/")[1] if len(explicit_path.split("/")) > 1 else "unknown"
            return candidate, outcome
        raise FileNotFoundError(f"Path {explicit_path} not found in {repo_id}")

    timestamp_folder = episode_info["timestamp_folder"]
    base = f"{episode_info['lab']}"
    date = episode_info["date"]

    allow_patterns = [
        f"{base}/success/{date}/{timestamp_folder}/*",
        f"{base}/failure/{date}/{timestamp_folder}/*",
    ]

    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        revision=revision,
        allow_patterns=allow_patterns,
        local_dir=cache_dir,
        local_dir_use_symlinks=False,
    )

    for outcome in ["success", "failure"]:
        candidate = os.path.join(snapshot_path, base, outcome, date, timestamp_folder)
        if os.path.exists(candidate):
            return candidate, outcome

    raise FileNotFoundError(
        f"Episode {episode_info['full_id']} not found in {repo_id} (patterns: {allow_patterns})"
    )


def build_intrinsics(cam_dir: str) -> Tuple[np.ndarray, dict]:
    """Load camera intrinsics from JSON."""
    intr_path = os.path.join(cam_dir, "intrinsics.json")
    with open(intr_path, "r") as f:
        intrinsics = json.load(f)
    K = np.array(
        [
            [intrinsics["fx"], 0, intrinsics["cx"]],
            [0, intrinsics["fy"], intrinsics["cy"]],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    return K, intrinsics


def prepare_cameras(episode_dir: str, extrinsics: dict) -> Dict[str, dict]:
    """Assemble camera metadata from extrinsics + recordings."""
    recordings_dir = os.path.join(episode_dir, "recordings")
    cameras: Dict[str, dict] = {}

    # External cameras
    for key in extrinsics.files:
        if not key.startswith("external_"):
            continue
        serial = key.split("_", 1)[1]
        cam_dir = os.path.join(recordings_dir, serial)
        if not os.path.isdir(cam_dir):
            continue
        rgb_frames = sorted(glob.glob(os.path.join(cam_dir, "rgb", "*.png")))
        depth_frames = sorted(glob.glob(os.path.join(cam_dir, "depth", "*.npy")))
        if len(rgb_frames) == 0 or len(depth_frames) == 0:
            continue

        K, intr = build_intrinsics(cam_dir)
        cameras[serial] = {
            "type": "external",
            "world_T_cam": extrinsics[key],
            "K": K,
            "width": intr["width"],
            "height": intr["height"],
            "rgb_frames": rgb_frames,
            "depth_frames": depth_frames,
        }

    # Wrist camera
    if "wrist_extrinsics" in extrinsics and extrinsics["wrist_extrinsics"] is not None:
        wrist_serial = str(extrinsics.get("wrist_serial", ""))
        if wrist_serial and wrist_serial != "None":
            cam_dir = os.path.join(recordings_dir, wrist_serial)
            if os.path.isdir(cam_dir):
                rgb_frames = sorted(glob.glob(os.path.join(cam_dir, "rgb", "*.png")))
                depth_frames = sorted(glob.glob(os.path.join(cam_dir, "depth", "*.npy")))
                if len(rgb_frames) > 0 and len(depth_frames) > 0:
                    K, intr = build_intrinsics(cam_dir)
                    cameras[wrist_serial] = {
                        "type": "wrist",
                        "transforms": extrinsics["wrist_extrinsics"],
                        "K": K,
                        "width": intr["width"],
                        "height": intr["height"],
                        "rgb_frames": rgb_frames,
                        "depth_frames": depth_frames,
                    }

    return cameras


def main():
    parser = argparse.ArgumentParser(
        description="Render RRD + videos with tracks from a Hugging Face episode."
    )
    parser.add_argument(
        "--episode_id",
        required=False,
        help='Episode ID, e.g., "AUTOLab+84bd5053+2023-08-18-12h-01m-10s"',
    )
    parser.add_argument(
        "--episode_path",
        help='Direct path inside the dataset repo, e.g., "GuptaLab/success/2023-04-30/Sun_Apr_30_15:02:19_2023"',
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("HF_REPO_ID", "sazirarrwth99/trajectory_data"),
        help="Hugging Face dataset repo id (default: env HF_REPO_ID or sazirarrwth99/trajectory_data)",
    )
    parser.add_argument(
        "--cache_dir",
        default=os.path.join(os.getcwd(), "hf_droid_cache"),
        help="Local cache directory for downloaded episodes.",
    )
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml"),
        help="Path to YAML config for visualization parameters.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional HF revision (branch/commit) to download.",
    )
    parser.add_argument(
        "--spawn-viewer",
        action="store_true",
        help="Spawn Rerun viewer (headless by default).",
    )
    args = parser.parse_args()

    # Load config and merge defaults
    CONFIG = DEFAULT_CONFIG.copy()
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            loaded = yaml.safe_load(f) or {}
            CONFIG.update({k: v for k, v in loaded.items() if v is not None})

    if not args.episode_id and not args.episode_path:
        raise ValueError("Provide either --episode_id or --episode_path")

    if args.episode_id:
        episode_info = parse_episode_id(args.episode_id)
        episode_rel_path = os.path.join(
            episode_info["lab"],
            "success",  # placeholder, actual outcome is auto-detected below
            episode_info["date"],
            episode_info["timestamp_folder"],
        )
        print(f"[INFO] Episode: {episode_info['full_id']}")
    else:
        # Use a direct repo path; infer minimal info for logging
        episode_rel_path = args.episode_path.strip().strip("/")
        parts = episode_rel_path.split("/")
        if len(parts) < 4:
            raise ValueError("episode_path must look like Lab/success/YYYY-MM-DD/TimestampFolder")
        episode_info = {
            "lab": parts[0],
            "full_id": episode_rel_path,
            "date": parts[2],
            "timestamp_folder": parts[3],
        }
        print(f"[INFO] Episode path: {episode_rel_path}")

    os.makedirs(args.cache_dir, exist_ok=True)
    episode_dir, outcome = download_episode_from_hf(
        args.repo,
        episode_info,
        cache_dir=args.cache_dir,
        token=os.environ.get("HF_TOKEN"),
        revision=args.revision,
        explicit_path=episode_rel_path if args.episode_path else None,
    )
    print(f"[INFO] Downloaded to: {episode_dir} (outcome: {outcome})")

    tracks_path = os.path.join(episode_dir, "tracks.npz")
    extrinsics_path = os.path.join(episode_dir, "extrinsics.npz")
    if not os.path.exists(tracks_path):
        raise FileNotFoundError(f"tracks.npz not found at {tracks_path}")
    if not os.path.exists(extrinsics_path):
        raise FileNotFoundError(f"extrinsics.npz not found at {extrinsics_path}")

    tracks = np.load(tracks_path)
    extrinsics = np.load(extrinsics_path, allow_pickle=True)

    tracks_3d = tracks["tracks_3d"]
    gripper_poses = tracks.get("gripper_poses")
    gripper_positions = tracks.get("gripper_positions")
    fps = float(tracks.get("fps", CONFIG["fps"]))

    num_frames_tracks = tracks_3d.shape[0]
    max_frames_cfg = CONFIG.get("max_frames")
    actual_frames = num_frames_tracks if max_frames_cfg in (None, "null") else min(num_frames_tracks, int(max_frames_cfg))

    num_contact_pts = int(tracks.get("num_points_per_finger", 0))
    total_track_pts = tracks_3d.shape[1]
    if num_contact_pts == 0 and total_track_pts > 0:
        num_contact_pts = total_track_pts // 2
    track_colors_rgb = np.zeros((total_track_pts, 3), dtype=np.uint8)
    if total_track_pts > 0:
        track_colors_rgb[:num_contact_pts, :] = [51, 127, 255]
        track_colors_rgb[num_contact_pts:, :] = [51, 255, 127]

    track_colors_float = np.zeros((total_track_pts, 4), dtype=np.float32)
    if total_track_pts > 0:
        track_colors_float[:num_contact_pts, :] = [0.2, 0.5, 1.0, 1.0]
        track_colors_float[num_contact_pts:, :] = [0.2, 1.0, 0.5, 1.0]

    cameras = prepare_cameras(episode_dir, extrinsics)
    if len(cameras) == 0:
        raise RuntimeError(f"No cameras with RGB+depth found in {episode_dir}")

    # Limit frames by camera availability
    for cam in cameras.values():
        frames_cam = min(len(cam["rgb_frames"]), len(cam["depth_frames"]))
        actual_frames = min(actual_frames, frames_cam)
        if cam["type"] == "wrist" and "transforms" in cam:
            actual_frames = min(actual_frames, cam["transforms"].shape[0])

    track_trail_length_video = CONFIG.get("track_trail_length_video", 10) or 10
    track_trail_length = CONFIG.get("track_trail_length", 10) or 10

    # Rerun setup
    rr.init("droid_full_fusion", spawn=args.spawn_viewer)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    rrd_path = os.path.join(episode_dir, CONFIG.get("rrd_output_path", "point_clouds/droid_full_fusion_gripper.rrd"))
    rrd_path = rrd_path.replace(".rrd", "")
    rrd_path = f"{rrd_path}_full_fusion.rrd"
    os.makedirs(os.path.dirname(rrd_path), exist_ok=True)
    rr.save(rrd_path)

    config_tag = os.path.splitext(os.path.basename(args.config))[0]
    video_dir = os.path.join(
        episode_dir,
        CONFIG.get("video_output_path", "point_clouds/videos"),
        config_tag,
        "tracks_reprojection",
    )
    os.makedirs(video_dir, exist_ok=True)

    recorders = {
        serial: VideoRecorder(video_dir, serial, "tracks", cam["width"], cam["height"], fps=fps, ext="avi", fourcc="MJPG")
        for serial, cam in cameras.items()
    }

    gripper_viz = GripperVisualizer()
    gripper_viz.init_rerun()

    radii_size = CONFIG.get("radii_size", 0.001)
    print(f"[INFO] Rendering {actual_frames} frames...")

    for i in range(actual_frames):
        if i % 10 == 0:
            print(f"  Frame {i}/{actual_frames}")

        rr.set_time(timeline="frame_index", sequence=i)

        # Update gripper pose (already rotation-fixed in tracks.npz)
        if gripper_poses is not None and len(gripper_poses) > i:
            gripper_viz.update(gripper_poses[i], gripper_positions[i] if gripper_positions is not None else None)
        else:
            # Fallback to cartesian positions if needed
            cart = tracks.get("cartesian_positions")
            if cart is not None and len(cart) > i:
                T_base_ee = pose6_to_T(cart[i])
                gripper_viz.update(T_base_ee, gripper_positions[i] if gripper_positions is not None else None)

        track_points_world = tracks_3d[i] if total_track_pts > 0 else None
        tracks_window = None
        if total_track_pts > 0 and track_trail_length_video > 1:
            start_idx = max(0, i - track_trail_length_video + 1)
            tracks_window = tracks_3d[start_idx : i + 1]

        for serial, cam in cameras.items():
            rgb_path = cam["rgb_frames"][i]
            depth_path = cam["depth_frames"][i]
            rgb = cv2.cvtColor(cv2.imread(rgb_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            depth = np.load(depth_path)

            is_wrist = cam["type"] == "wrist"
            if is_wrist:
                if i >= len(cam.get("transforms", [])):
                    continue
                T_world_cam = cam["transforms"][i]
                max_d = CONFIG.get("wrist_max_depth", 0.75)
                min_d = CONFIG.get("min_depth_wrist", 0.01)
                cam_path = "world/wrist_cam"
            else:
                T_world_cam = cam["world_T_cam"]
                max_d = CONFIG.get("ext_max_depth", 1.5)
                min_d = CONFIG.get("min_depth", 0.1)
                cam_path = f"world/external_cams/{serial}"

            # Log transform + pinhole
            rr.log(
                cam_path,
                rr.Transform3D(
                    translation=T_world_cam[:3, 3],
                    mat3x3=T_world_cam[:3, :3],
                    axis_length=0.1,
                ),
            )
            rr.log(
                f"{cam_path}/pinhole",
                rr.Pinhole(
                    image_from_camera=cam["K"],
                    width=cam["width"],
                    height=cam["height"],
                ),
            )

            xyz_cam, rgb_cloud = depth_to_pointcloud(depth, rgb, cam["K"], max_d, min_d)
            frame_overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            if xyz_cam.shape[0] > 0:
                xyz_world = transform_points(xyz_cam, T_world_cam)
                points_path = "world/points/wrist_cam" if is_wrist else f"world/points/external_cams/{serial}"
                rr.log(points_path, rr.Points3D(xyz_world, colors=rgb_cloud, radii=radii_size))

                uv_cloud, cols_cloud = project_points_to_image(
                    xyz_world, cam["K"], T_world_cam, cam["width"], cam["height"], colors=rgb_cloud
                )
                frame_overlay = draw_points_on_image_fast(frame_overlay, uv_cloud, colors=cols_cloud)

            if track_points_world is not None:
                uv_tracks, cols_tracks = project_points_to_image(
                    track_points_world,
                    cam["K"],
                    T_world_cam,
                    cam["width"],
                    cam["height"],
                    colors=track_colors_rgb,
                )
                frame_overlay = draw_points_on_image(
                    frame_overlay, uv_tracks, colors=cols_tracks, radius=3, default_color=(0, 0, 255)
                )

            if tracks_window is not None:
                frame_overlay = draw_track_trails_on_image(
                    frame_overlay,
                    tracks_window,
                    cam["K"],
                    T_world_cam,
                    cam["width"],
                    cam["height"],
                    track_colors_rgb,
                    min_depth=min_d,
                )

            recorders[serial].write_frame(frame_overlay)

    # Log 3D track trails like create_video_with_tracks.py
    if total_track_pts > 0:
        for t in range(actual_frames):
            rr.set_time(timeline="frame_index", sequence=t)
            rr.log(
                "world/gripper_tracks/points",
                rr.Points3D(
                    positions=tracks_3d[t],
                    colors=(track_colors_float[:, :3] * 255).astype(np.uint8),
                    radii=0.003,
                ),
            )
            if t > 0:
                trail_len = min(t, track_trail_length)
                for n in range(total_track_pts):
                    trail_points = tracks_3d[max(0, t - trail_len) : t + 1, n, :]
                    if len(trail_points) > 1:
                        segments = np.stack([trail_points[:-1], trail_points[1:]], axis=1)
                        color = track_colors_float[n, :3]
                        rr.log(
                            f"world/gripper_tracks/trails/track_{n:03d}",
                            rr.LineStrips3D(
                                strips=segments,
                                colors=[color] * len(segments),
                                radii=0.001,
                            ),
                        )

    for rec in recorders.values():
        rec.close()

    print("\n[SUCCESS] Done.")
    print(f"  Episode dir: {episode_dir}")
    print(f"  RRD saved to: {rrd_path}")
    print(f"  Videos saved to: {video_dir}")


if __name__ == "__main__":
    main()
