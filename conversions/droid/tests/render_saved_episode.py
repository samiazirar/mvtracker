"""Render saved training-data assets (RGB, depth, tracks, extrinsics) into Rerun.

This script is a lightweight validator for the training_data pipeline outputs. It
loads `tracks.npz`, `extrinsics.npz`, per-camera intrinsics, RGB, and depth PNGs
from a processed episode directory and produces:
  - An .rrd containing world-frame point clouds (from depth) and track points.
  - Per-camera MP4s with the world tracks reprojected onto each RGB view.

Example:
    python conversions/droid/tests/render_saved_episode.py \\
        --episode_dir test_data/AUTOLab/success/2023-08-18/Fri_Aug_18_12:01:10_2023 \\
        --max_frames 50
"""

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation as R

# Ensure utils package is importable when running as a standalone script
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.transforms import transform_points  # noqa: E402
from utils.video_utils import (  # noqa: E402
    VideoRecorder,
    draw_points_on_image_fast,
    draw_track_trails_on_image,
    project_points_to_image,
)


@dataclass
class CameraView:
    serial: str
    K: np.ndarray
    width: int
    height: int
    rgb_files: List[str]
    depth_files: List[str]
    world_T_cam: Optional[np.ndarray] = None
    wrist_extrinsics: Optional[np.ndarray] = None
    recorder: Optional[VideoRecorder] = None

    def frame_count(self) -> int:
        counts = [len(self.rgb_files), len(self.depth_files)]
        if self.wrist_extrinsics is not None:
            counts.append(len(self.wrist_extrinsics))
        return min(counts)


def load_intrinsics(intrinsics_path: str) -> Tuple[np.ndarray, int, int]:
    with open(intrinsics_path, "r") as f:
        intr = json.load(f)
    K = np.array(
        [
            [intr["fx"], 0, intr["cx"]],
            [0, intr["fy"], intr["cy"]],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    return K, int(intr["width"]), int(intr["height"])


def load_extrinsics(extrinsics_path: str):
    data = np.load(extrinsics_path, allow_pickle=True)
    external: Dict[str, np.ndarray] = {}
    for key in data.files:
        if key.startswith("external_"):
            serial = key.split("external_")[1]
            external[serial] = data[key]
    wrist_serial = str(data["wrist_serial"]) if "wrist_serial" in data else None
    wrist_extrinsics = data["wrist_extrinsics"] if "wrist_extrinsics" in data else None
    num_frames = int(data["num_frames"]) if "num_frames" in data else 0
    return external, wrist_serial, wrist_extrinsics, num_frames


def depth_to_points(depth: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Backproject depth map (meters) to camera-frame 3D points + RGB colors."""
    h, w = depth.shape[:2]
    u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))
    depth_flat = depth.reshape(-1)
    valid = np.isfinite(depth_flat) & (depth_flat > 0)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), valid

    u = u_coords.reshape(-1)[valid]
    v = v_coords.reshape(-1)[valid]
    d = depth_flat[valid]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x = (u - cx) * d / fx
    y = (v - cy) * d / fy
    z = d
    pts_cam = np.stack([x, y, z], axis=-1).astype(np.float32)
    return pts_cam, valid


def gather_cameras(episode_dir: str, external_extr: Dict[str, np.ndarray], wrist_serial: Optional[str], wrist_extr: Optional[np.ndarray]) -> Dict[str, CameraView]:
    recordings_dir = os.path.join(episode_dir, "recordings")
    cameras: Dict[str, CameraView] = {}

    for cam_serial in sorted(os.listdir(recordings_dir)):
        cam_dir = os.path.join(recordings_dir, cam_serial)
        rgb_dir = os.path.join(cam_dir, "rgb")
        depth_dir = os.path.join(cam_dir, "depth")
        intr_path = os.path.join(cam_dir, "intrinsics.json")

        if not (os.path.isdir(rgb_dir) and os.path.isdir(depth_dir) and os.path.exists(intr_path)):
            continue

        rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
        depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.npy")))
        if not depth_files:
            # Fallback to legacy PNG depth if present
            depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
        if not rgb_files or not depth_files:
            continue

        K, width, height = load_intrinsics(intr_path)
        cam = CameraView(
            serial=cam_serial,
            K=K,
            width=width,
            height=height,
            rgb_files=rgb_files,
            depth_files=depth_files,
            world_T_cam=external_extr.get(cam_serial),
            wrist_extrinsics=wrist_extr if wrist_serial and cam_serial == wrist_serial else None,
        )
        cameras[cam_serial] = cam

    return cameras


def main():
    parser = argparse.ArgumentParser(description="Render saved DROID episode assets into an RRD + videos.")
    parser.add_argument("--episode_dir", required=True, help="Path to processed episode directory (with tracks.npz, extrinsics.npz, recordings/).")
    parser.add_argument("--output_rrd", help="Output RRD path (default: <episode_dir>/recordings/render_from_saved.rrd)")
    parser.add_argument("--video_dir", help="Directory for per-camera videos (default: <episode_dir>/recordings/reprojected_videos)")
    parser.add_argument("--max_frames", type=int, default=None, help="Optional max frames to render.")
    parser.add_argument("--track_trail", type=int, default=10, help="Number of past frames to draw as trails on videos.")
    args = parser.parse_args()

    tracks_path = os.path.join(args.episode_dir, "tracks.npz")
    extrinsics_path = os.path.join(args.episode_dir, "extrinsics.npz")
    if not os.path.exists(tracks_path) or not os.path.exists(extrinsics_path):
        raise FileNotFoundError("tracks.npz and extrinsics.npz are required in the episode directory.")

    # Load track data
    tracks = np.load(tracks_path)
    tracks_3d = tracks["tracks_3d"]
    fps = float(tracks.get("fps", 30.0))
    num_frames_tracks = tracks_3d.shape[0]

    # Build track colors (left/right fingers)
    total_track_pts = tracks_3d.shape[1]
    half = total_track_pts // 2
    track_colors = np.zeros((total_track_pts, 3), dtype=np.uint8)
    track_colors[:half, :] = [51, 127, 255]
    track_colors[half:, :] = [51, 255, 127]

    # Load extrinsics
    external_extr, wrist_serial, wrist_extrinsics, _ = load_extrinsics(extrinsics_path)

    # Build camera views from saved RGB/depth/intrinsics
    cameras = gather_cameras(args.episode_dir, external_extr, wrist_serial, wrist_extrinsics)
    if not cameras:
        raise RuntimeError("No cameras with RGB+depth+intrinsics found in recordings/.")

    # Determine frame budget
    per_cam_counts = [cam.frame_count() for cam in cameras.values()]
    max_frames_available = min([num_frames_tracks] + per_cam_counts)
    actual_frames = min(args.max_frames, max_frames_available) if args.max_frames is not None else max_frames_available
    print(f"[INFO] Rendering {actual_frames} frames ({len(cameras)} cameras)")

    # Set default outputs
    default_rrd = os.path.join(args.episode_dir, "recordings", "render_from_saved.rrd")
    rrd_path = args.output_rrd or default_rrd
    video_dir = args.video_dir or os.path.join(args.episode_dir, "recordings", "reprojected_videos")
    os.makedirs(video_dir, exist_ok=True)

    # Init Rerun
    rr.init("droid_full_fusion", spawn=False)
    rrd_path = rrd_path.replace(".rrd", "")
    rrd_path = f"{rrd_path}_render_from_saved.rrd"
    rr.save(rrd_path)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # Create recorders
    for cam in cameras.values():
        cam.recorder = VideoRecorder(video_dir, cam.serial, "tracks_reprojection", cam.width, cam.height, fps=fps)

    for frame_idx in range(actual_frames):
        rr.set_time(timeline="frame_index", sequence=frame_idx)

        # Log tracks in world frame
        rr.log(
            "world/tracks",
            rr.Points3D(positions=tracks_3d[frame_idx], colors=track_colors),
        )

        # History for trails
        start_idx = max(0, frame_idx - args.track_trail + 1)
        track_history = tracks_3d[start_idx : frame_idx + 1]

        for cam in cameras.values():
            if frame_idx >= cam.frame_count():
                continue

            # Pose selection (external cams are static)
            world_T_cam = cam.world_T_cam
            is_wrist = False
            if world_T_cam is None and cam.wrist_extrinsics is not None:
                world_T_cam = cam.wrist_extrinsics[frame_idx]
                is_wrist = True

            if world_T_cam is None:
                continue

            if is_wrist:
                R_fix = R.from_euler("z", 90, degrees=True).as_matrix()
                world_T_cam[:3, :3] = world_T_cam[:3, :3] @ R_fix

            rgb_path = cam.rgb_files[frame_idx]
            depth_path = cam.depth_files[frame_idx]
            rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
            if depth_path.endswith(".npy"):
                depth = np.load(depth_path)
            else:
                depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if rgb_bgr is None or depth is None:
                continue

            # Build point cloud
            pts_cam, valid_mask = depth_to_points(depth, cam.K)
            rgb_flat = rgb_bgr.reshape(-1, 3)
            colors_cam = rgb_flat[valid_mask] if valid_mask.size > 0 else np.empty((0, 3), dtype=np.uint8)
            pts_world = transform_points(pts_cam, world_T_cam)

            rr.log(
                f"world/point_clouds/{cam.serial}",
                rr.Points3D(positions=pts_world, colors=colors_cam),
            )

            # Log raw image (converted to RGB for Rerun)
            rgb_for_rr = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            rr.log(f"world/cameras/{cam.serial}/rgb", rr.Image(rgb_for_rr))

            # Reproject tracks for video
            uv_tracks, colors_proj = project_points_to_image(
                tracks_3d[frame_idx], cam.K, world_T_cam, cam.width, cam.height, colors=track_colors
            )
            frame_with_tracks = draw_points_on_image_fast(rgb_bgr, uv_tracks, colors_proj)
            frame_with_tracks = draw_track_trails_on_image(
                frame_with_tracks,
                track_history,
                cam.K,
                world_T_cam,
                cam.width,
                cam.height,
                colors=track_colors,
            )
            cam.recorder.write_frame(frame_with_tracks)

    # Finalize videos
    for cam in cameras.values():
        if cam.recorder:
            cam.recorder.close()

    print(f"[INFO] RRD saved to {rrd_path}")
    print(f"[INFO] Videos saved to {video_dir}")


if __name__ == "__main__":
    main()
