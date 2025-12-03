#!/usr/bin/env python3
"""Download a compressed DROID episode and render tracks onto mp4 videos.

This utility pulls the FFV1 depth videos, tracks, and extrinsics from a
Hugging Face dataset (output of `run_pipeline_cluster_huggingface_compressed_lossy.sh`)
and fetches the corresponding RGB MP4s from the original GCS bucket. The script
then reprojects the precomputed gripper tracks onto each RGB video, saving
per-camera MP4s with overlaid trails similar to
`conversions/droid/create_video_with_tracks.py`.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from huggingface_hub import snapshot_download

# Make repo root utils importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (  # type: ignore
    VideoRecorder,
    draw_track_trails_on_image,
    project_points_to_image,
)
from training_data.extract_rgb_depth import decode_ffv1_depth_frame


DEFAULT_CONFIG = {
    "fps": 30.0,
    "max_frames": None,
    "track_trail_length": 10,
    "track_trail_length_video": 10,
    "min_depth": 0.01,
    "video_output_path": "point_clouds/videos",
}


# ---------------------------------------------------------------------------
# Episode helpers
# ---------------------------------------------------------------------------

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
    timestamp_folder = dt.strftime("%a_%b_%e_%H:%M:%S_%Y").replace(" ", "_")

    return {
        "lab": lab,
        "hash": episode_hash,
        "date": date,
        "time_str": f"{hour}:{minute}:{second}",
        "timestamp_folder": timestamp_folder,
        "full_id": episode_id,
    }


def run_gsutil(command: str, check: bool = True) -> bool:
    """Run a gsutil command, returning success."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=check,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as exc:  # pragma: no cover - runtime diagnostics
        print(f"[WARN] gsutil error: {exc.stderr}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_episode_from_hf(
    repo_id: str,
    episode_info: dict,
    cache_dir: str,
    token: Optional[str] = None,
    revision: Optional[str] = None,
) -> Tuple[str, str]:
    """Download a single episode folder from Hugging Face.

    Returns the local episode path and the outcome (success/failure).
    """

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


def download_mp4_from_gcs(
    episode_info: dict,
    local_episode_dir: str,
    gcs_bucket: str,
) -> str:
    """Fetch MP4 recordings from GCS into the episode directory."""

    lab = episode_info["lab"]
    date = episode_info["date"]
    timestamp_folder = episode_info["timestamp_folder"]

    for outcome in ["success", "failure"]:
        rel_path = f"{lab}/{outcome}/{date}/{timestamp_folder}"
        gcs_path = f"{gcs_bucket}/{rel_path}"
        stat_cmd = f'gsutil -q stat "{gcs_path}/recordings/mp4"'
        if run_gsutil(stat_cmd, check=False):
            dest_dir = os.path.join(local_episode_dir, "recordings")
            os.makedirs(dest_dir, exist_ok=True)
            cp_cmd = f'gsutil -m cp -r "{gcs_path}/recordings/mp4" "{dest_dir}/"'
            if not run_gsutil(cp_cmd, check=False):
                raise RuntimeError(f"Failed to download MP4 recordings from {gcs_path}")
            return os.path.join(dest_dir, "mp4")

    raise FileNotFoundError(
        f"MP4 recordings not found for episode {episode_info['full_id']} in {gcs_bucket}"
    )


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def load_intrinsics(cam_dir: str) -> np.ndarray:
    intr_path = os.path.join(cam_dir, "intrinsics.json")
    with open(intr_path, "r") as f:
        intrinsics = json.load(f)
    return np.array(
        [
            [intrinsics["fx"], 0, intrinsics["cx"]],
            [0, intrinsics["fy"], intrinsics["cy"]],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )


def build_cameras(
    episode_dir: str,
    extrinsics: np.lib.npyio.NpzFile,
    mp4_dir: str,
) -> Dict[str, dict]:
    """Assemble camera metadata using extrinsics and available recordings."""

    recordings_dir = os.path.join(episode_dir, "recordings")
    cameras: Dict[str, dict] = {}

    mp4_files = {os.path.splitext(os.path.basename(p))[0]: p for p in glob_mp4_files(mp4_dir)}

    # External cameras have fixed extrinsics
    for key in extrinsics.files:
        if not key.startswith("external_"):
            continue
        serial = key.split("_", 1)[1]
        cam_dir = os.path.join(recordings_dir, serial)
        depth_video = os.path.join(cam_dir, "depth.mkv")
        if not (os.path.isfile(depth_video) and os.path.isfile(os.path.join(cam_dir, "intrinsics.json"))):
            continue
        if serial not in mp4_files:
            continue
        cameras[serial] = {
            "type": "external",
            "world_T_cam": extrinsics[key],
            "intrinsics": load_intrinsics(cam_dir),
            "rgb_path": mp4_files[serial],
            "depth_path": depth_video,
        }

    # Wrist camera (per-frame extrinsics)
    if "wrist_extrinsics" in extrinsics and "wrist_serial" in extrinsics:
        wrist_serial = str(extrinsics["wrist_serial"])
        cam_dir = os.path.join(recordings_dir, wrist_serial)
        depth_video = os.path.join(cam_dir, "depth.mkv")
        if wrist_serial in mp4_files and os.path.isfile(depth_video):
            cameras[wrist_serial] = {
                "type": "wrist",
                "extrinsics": extrinsics["wrist_extrinsics"],
                "intrinsics": load_intrinsics(cam_dir),
                "rgb_path": mp4_files[wrist_serial],
                "depth_path": depth_video,
            }

    return cameras


def glob_mp4_files(mp4_dir: str) -> List[str]:
    if not os.path.isdir(mp4_dir):
        return []
    return [
        os.path.join(mp4_dir, fname)
        for fname in os.listdir(mp4_dir)
        if fname.lower().endswith(".mp4")
    ]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def decode_depth_frames(cap: cv2.VideoCapture) -> Tuple[bool, Optional[np.ndarray]]:
    ok, frame = cap.read()
    if not ok:
        return False, None
    return True, decode_ffv1_depth_frame(frame)


def compute_track_colors(total_pts: int, per_finger: int) -> np.ndarray:
    colors = np.zeros((total_pts, 3), dtype=np.uint8)
    if total_pts == 0:
        return colors
    per_finger = per_finger or total_pts // 2
    colors[:per_finger, :] = [51, 127, 255]
    colors[per_finger:, :] = [51, 255, 127]
    return colors


def render_episode(
    episode_dir: str,
    cameras: Dict[str, dict],
    tracks: np.lib.npyio.NpzFile,
    config: dict,
    output_root: str,
):
    tracks_3d = tracks["tracks_3d"]
    total_frames, total_pts, _ = tracks_3d.shape
    per_finger = int(tracks.get("num_points_per_finger", total_pts // 2))
    track_colors = compute_track_colors(total_pts, per_finger)
    fps = config.get("fps", DEFAULT_CONFIG["fps"])
    max_frames = config.get("max_frames") or total_frames
    track_trail = config.get("track_trail_length_video", DEFAULT_CONFIG["track_trail_length_video"])

    episode_tag = os.path.basename(episode_dir.rstrip("/"))
    video_dir = os.path.join(output_root, episode_tag, "tracks_reprojection")
    os.makedirs(video_dir, exist_ok=True)

    recorders: Dict[str, VideoRecorder] = {}
    captures: Dict[str, Tuple[cv2.VideoCapture, cv2.VideoCapture]] = {}

    for serial, cam in cameras.items():
        rgb_cap = cv2.VideoCapture(cam["rgb_path"])
        depth_cap = cv2.VideoCapture(cam["depth_path"])
        width = int(rgb_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(rgb_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        recorders[serial] = VideoRecorder(video_dir, serial, "tracks", width, height, fps=fps)
        captures[serial] = (rgb_cap, depth_cap)

    trails: Dict[str, List[np.ndarray]] = {k: [] for k in cameras}
    min_frames = min(
        max_frames,
        min(int(cap[0].get(cv2.CAP_PROP_FRAME_COUNT)) for cap in captures.values()),
        total_frames,
    )

    for frame_idx in range(min_frames):
        points_world = tracks_3d[frame_idx]
        for serial, cam in cameras.items():
            rgb_cap, depth_cap = captures[serial]
            ok_rgb, rgb_frame = rgb_cap.read()
            ok_depth, depth_frame = decode_depth_frames(depth_cap)
            if not (ok_rgb and ok_depth and depth_frame is not None):
                continue

            # Determine pose for this frame
            if cam["type"] == "external":
                world_T_cam = cam["world_T_cam"]
            else:
                extr = cam["extrinsics"]
                if frame_idx >= extr.shape[0]:
                    continue
                world_T_cam = extr[frame_idx]

            uv, _ = project_points_to_image(
                points_world,
                cam["intrinsics"],
                world_T_cam,
                rgb_frame.shape[1],
                rgb_frame.shape[0],
                colors=track_colors,
                min_depth=config.get("min_depth", DEFAULT_CONFIG["min_depth"]),
            )

            trails[serial].append(uv)
            if len(trails[serial]) > track_trail:
                trails[serial].pop(0)

            draw_track_trails_on_image(rgb_frame, trails[serial], trail_length=track_trail)
            recorders[serial].write(rgb_frame)

    for recorder in recorders.values():
        recorder.close()
    for rgb_cap, depth_cap in captures.values():
        rgb_cap.release()
        depth_cap.release()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and render compressed DROID episode from Hugging Face + GCS.")
    parser.add_argument("--episode_id", required=True, help="Episode ID, e.g., 'AUTOLab+84bd5053+2023-08-18-12h-01m-10s'")
    parser.add_argument("--hf_repo", default=os.environ.get("HF_REPO_ID"), help="Hugging Face dataset repo ID")
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"), help="Hugging Face token (env HF_TOKEN if unset)")
    parser.add_argument("--hf_revision", default=None, help="Optional Hugging Face revision")
    parser.add_argument("--cache_dir", default="./droid_hf_downloads", help="Local cache directory for HF snapshots")
    parser.add_argument("--gcs_bucket", default="gs://gresearch/robotics/droid_raw/1.0.1", help="GCS bucket for raw recordings")
    parser.add_argument("--output_videos", default=DEFAULT_CONFIG["video_output_path"], help="Output directory for rendered videos")
    parser.add_argument("--max_frames", type=int, default=None, help="Optional max frames to render")
    parser.add_argument("--fps", type=float, default=DEFAULT_CONFIG["fps"], help="Frame rate for output videos")
    parser.add_argument("--track_trail_length_video", type=int, default=DEFAULT_CONFIG["track_trail_length_video"], help="Trail length for track visualization")
    parser.add_argument("--min_depth", type=float, default=DEFAULT_CONFIG["min_depth"], help="Minimum depth for valid projections")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.hf_repo:
        raise SystemExit("--hf_repo or HF_REPO_ID env variable is required")

    episode_info = parse_episode_id(args.episode_id)
    episode_dir, outcome = download_episode_from_hf(
        repo_id=args.hf_repo,
        episode_info=episode_info,
        cache_dir=args.cache_dir,
        token=args.hf_token,
        revision=args.hf_revision,
    )
    print(f"[INFO] Downloaded episode to {episode_dir} ({outcome})")

    tracks_path = os.path.join(episode_dir, "tracks.npz")
    extrinsics_path = os.path.join(episode_dir, "extrinsics.npz")
    if not os.path.isfile(tracks_path) or not os.path.isfile(extrinsics_path):
        raise FileNotFoundError("tracks.npz or extrinsics.npz missing from Hugging Face download")

    mp4_dir = download_mp4_from_gcs(episode_info, episode_dir, args.gcs_bucket)
    print(f"[INFO] Downloaded MP4 recordings to {mp4_dir}")

    extrinsics = np.load(extrinsics_path, allow_pickle=True)
    cameras = build_cameras(episode_dir, extrinsics, mp4_dir)
    if not cameras:
        raise SystemExit("No cameras with both MP4 RGB and depth.mkv found. Nothing to render.")

    config = {
        "fps": args.fps,
        "max_frames": args.max_frames,
        "track_trail_length_video": args.track_trail_length_video,
        "min_depth": args.min_depth,
    }

    with np.load(tracks_path) as tracks:
        render_episode(episode_dir, cameras, tracks, config, args.output_videos)
        print(f"[INFO] Rendered videos saved under {args.output_videos}")


if __name__ == "__main__":
    main()
