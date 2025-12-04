#!/usr/bin/env python3
"""Render track overlays on MP4 videos using metadata from HuggingFace.

This script:
1. Downloads tracks.npz and extrinsics.npz from HuggingFace
2. Downloads MP4 videos from either local source or another HuggingFace repo
3. Renders the 3D tracks projected onto each video frame

REQUIREMENTS:
- tracks.npz MUST contain camera intrinsics (intrinsics_{serial} and image_size_{serial})
- This means the full pipeline with extract_rgb_depth.py must have been run first
- The metadata-only pipeline does NOT generate intrinsics (no depth extraction)

Unlike render_visualize_hf.py which requires PNG+depth data, this works with
just the metadata (tracks, extrinsics with intrinsics) and source MP4 videos.

Usage:
    # Using local MP4 videos
    python render_tracks_from_mp4.py \
        --episode_id "AUTOLab+84bd5053+2023-08-18-12h-01m-10s" \
        --metadata_repo "sazirarrwth99/droid_metadata_only" \
        --video_source "/data/droid/data/droid_raw/1.0.1"
    
    # Using MP4 videos from HuggingFace
    python render_tracks_from_mp4.py \
        --episode_id "AUTOLab+84bd5053+2023-08-18-12h-01m-10s" \
        --metadata_repo "sazirarrwth99/droid_metadata_only" \
        --video_repo "droid-dataset/droid"

Environment Variables:
    HF_TOKEN            HuggingFace token for private repos
"""

import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import yaml

# Make repo root utils importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (  # type: ignore
    VideoRecorder,
    draw_points_on_image,
    draw_track_trails_on_image,
    project_points_to_image,
)


DEFAULT_CONFIG = {
    "fps": 30.0,
    "max_frames": None,
    "track_trail_length_video": 10,
    "min_depth": 0.01,
    "output_dir": "./rendered_tracks",
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
    timestamp_folder = dt.strftime("%a_%b_%e_%H:%M:%S_%Y").replace(" ", "_")

    return {
        "lab": lab,
        "hash": episode_hash,
        "date": date,
        "time_str": f"{hour}:{minute}:{second}",
        "timestamp_folder": timestamp_folder,
        "full_id": episode_id,
    }


def download_metadata_from_hf(
    repo_id: str,
    episode_info: dict,
    cache_dir: str,
    token: str = None,
) -> str:
    """Download tracks.npz and extrinsics.npz from HuggingFace metadata repo."""
    from huggingface_hub import snapshot_download
    
    timestamp_folder = episode_info["timestamp_folder"]
    base = episode_info["lab"]
    date = episode_info["date"]

    # Try both success and failure paths
    allow_patterns = [
        f"{base}/success/{date}/{timestamp_folder}/tracks.npz",
        f"{base}/success/{date}/{timestamp_folder}/extrinsics.npz",
        f"{base}/success/{date}/{timestamp_folder}/quality.json",
        f"{base}/failure/{date}/{timestamp_folder}/tracks.npz",
        f"{base}/failure/{date}/{timestamp_folder}/extrinsics.npz",
        f"{base}/failure/{date}/{timestamp_folder}/quality.json",
    ]

    print(f"[INFO] Downloading metadata from {repo_id}...")
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        allow_patterns=allow_patterns,
        local_dir=cache_dir,
        local_dir_use_symlinks=False,
    )

    # Find the actual episode directory
    for outcome in ["success", "failure"]:
        candidate = os.path.join(snapshot_path, base, outcome, date, timestamp_folder)
        if os.path.exists(os.path.join(candidate, "tracks.npz")):
            print(f"[INFO] Metadata found at: {candidate}")
            return candidate

    raise FileNotFoundError(
        f"Episode metadata not found in {repo_id} for {episode_info['full_id']}"
    )


def find_local_mp4_videos(
    video_source: str,
    episode_info: dict,
) -> Dict[str, str]:
    """Find MP4 video files from local DROID source."""
    timestamp_folder = episode_info["timestamp_folder"]
    base = episode_info["lab"]
    date = episode_info["date"]

    videos = {}
    
    for outcome in ["success", "failure"]:
        recordings_dir = os.path.join(
            video_source, base, outcome, date, timestamp_folder, "recordings"
        )
        
        # Check for MP4 subfolder or SVO folder with MP4s
        for mp4_dir in [
            os.path.join(recordings_dir, "MP4"),
            os.path.join(recordings_dir, "mp4"),
            recordings_dir,
        ]:
            if not os.path.isdir(mp4_dir):
                continue
            
            # Find all MP4 files
            for mp4_file in glob.glob(os.path.join(mp4_dir, "*.mp4")):
                # Extract camera serial from filename
                filename = os.path.basename(mp4_file)
                # Assume format like "12345678.mp4" or "SN12345678.mp4"
                serial = filename.replace(".mp4", "").replace("SN", "").replace("sn", "")
                videos[serial] = mp4_file
        
        if videos:
            print(f"[INFO] Found {len(videos)} MP4 videos in {recordings_dir}")
            return videos
    
    return videos


def download_videos_from_hf(
    repo_id: str,
    episode_info: dict,
    cache_dir: str,
    token: str = None,
) -> Dict[str, str]:
    """Download MP4 videos from HuggingFace video repo."""
    from huggingface_hub import snapshot_download
    
    timestamp_folder = episode_info["timestamp_folder"]
    base = episode_info["lab"]
    date = episode_info["date"]

    # Download MP4 files
    allow_patterns = [
        f"{base}/success/{date}/{timestamp_folder}/recordings/**/*.mp4",
        f"{base}/failure/{date}/{timestamp_folder}/recordings/**/*.mp4",
    ]

    print(f"[INFO] Downloading videos from {repo_id}...")
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        allow_patterns=allow_patterns,
        local_dir=cache_dir,
        local_dir_use_symlinks=False,
    )

    videos = {}
    for outcome in ["success", "failure"]:
        recordings_dir = os.path.join(snapshot_path, base, outcome, date, timestamp_folder, "recordings")
        for mp4_dir in [
            os.path.join(recordings_dir, "MP4"),
            os.path.join(recordings_dir, "mp4"),
            recordings_dir,
        ]:
            if not os.path.isdir(mp4_dir):
                continue
            for mp4_file in glob.glob(os.path.join(mp4_dir, "*.mp4")):
                filename = os.path.basename(mp4_file)
                serial = filename.replace(".mp4", "").replace("SN", "").replace("sn", "")
                videos[serial] = mp4_file
        if videos:
            break

    return videos


def build_camera_info(
    extrinsics: np.lib.npyio.NpzFile,
    tracks: np.lib.npyio.NpzFile,
    video_paths: Dict[str, str],
) -> Dict[str, dict]:
    """Build camera metadata from extrinsics, tracks (for intrinsics), and video paths."""
    cameras = {}
    
    # Get list of cameras with 2D tracks (they have intrinsics)
    cameras_with_intrinsics = []
    if "cameras_with_2d_tracks" in tracks:
        cameras_with_intrinsics = list(tracks["cameras_with_2d_tracks"])
    
    # External cameras
    for key in extrinsics.files:
        if not key.startswith("external_"):
            continue
        serial = key.split("_", 1)[1]
        
        # Skip if no video for this camera
        if serial not in video_paths:
            print(f"  [SKIP] Camera {serial}: no MP4 video found")
            continue
        
        # Get intrinsics from tracks.npz (required)
        if serial not in cameras_with_intrinsics:
            raise ValueError(
                f"Camera {serial}: no intrinsics found in tracks.npz. "
                f"Run the full pipeline with extract_rgb_depth.py first to generate intrinsics."
            )
        
        if f"intrinsics_{serial}" not in tracks:
            raise ValueError(
                f"Camera {serial}: intrinsics_{serial} not found in tracks.npz"
            )
        if f"image_size_{serial}" not in tracks:
            raise ValueError(
                f"Camera {serial}: image_size_{serial} not found in tracks.npz"
            )
        
        K = tracks[f"intrinsics_{serial}"]
        size = tracks[f"image_size_{serial}"]
        width, height = int(size[0]), int(size[1])
        
        cameras[serial] = {
            "type": "external",
            "world_T_cam": extrinsics[key],
            "K": K,
            "width": width,
            "height": height,
            "video_path": video_paths[serial],
        }
        print(f"  [OK] Camera {serial}: external, {width}x{height}")

    # Wrist camera
    wrist_serial = None
    if "wrist_serial" in extrinsics:
        wrist_serial = str(extrinsics["wrist_serial"])
        if wrist_serial in ("", "None", "none"):
            wrist_serial = None
    
    if wrist_serial and "wrist_extrinsics" in extrinsics:
        if wrist_serial in video_paths:
            # Get intrinsics from tracks.npz (required)
            if wrist_serial not in cameras_with_intrinsics:
                raise ValueError(
                    f"Wrist camera {wrist_serial}: no intrinsics found in tracks.npz. "
                    f"Run the full pipeline with extract_rgb_depth.py first to generate intrinsics."
                )
            
            if f"intrinsics_{wrist_serial}" not in tracks:
                raise ValueError(
                    f"Wrist camera {wrist_serial}: intrinsics_{wrist_serial} not found in tracks.npz"
                )
            if f"image_size_{wrist_serial}" not in tracks:
                raise ValueError(
                    f"Wrist camera {wrist_serial}: image_size_{wrist_serial} not found in tracks.npz"
                )
            
            K = tracks[f"intrinsics_{wrist_serial}"]
            size = tracks[f"image_size_{wrist_serial}"]
            width, height = int(size[0]), int(size[1])
            
            cameras[wrist_serial] = {
                "type": "wrist",
                "transforms": extrinsics["wrist_extrinsics"],
                "K": K,
                "width": width,
                "height": height,
                "video_path": video_paths[wrist_serial],
            }
            print(f"  [OK] Camera {wrist_serial}: wrist, {width}x{height}")
        else:
            print(f"  [SKIP] Wrist camera {wrist_serial}: no MP4 video found")

    return cameras


def render_tracks_on_videos(
    cameras: Dict[str, dict],
    tracks_3d: np.ndarray,
    output_dir: str,
    fps: float = 30.0,
    max_frames: Optional[int] = None,
    track_trail_length: int = 10,
    min_depth: float = 0.01,
):
    """Render track overlays on all camera videos."""
    
    num_frames_tracks = tracks_3d.shape[0]
    total_track_pts = tracks_3d.shape[1]
    num_contact_pts = total_track_pts // 2
    
    # Setup track colors (blue for left finger, green for right)
    track_colors_rgb = np.zeros((total_track_pts, 3), dtype=np.uint8)
    if total_track_pts > 0:
        track_colors_rgb[:num_contact_pts, :] = [51, 127, 255]  # Blue-ish
        track_colors_rgb[num_contact_pts:, :] = [51, 255, 127]  # Green-ish
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video captures and create recorders
    captures = {}
    recorders = {}
    actual_frames = num_frames_tracks
    
    for serial, cam in cameras.items():
        cap = cv2.VideoCapture(cam["video_path"])
        if not cap.isOpened():
            print(f"[WARN] Could not open video: {cam['video_path']}")
            continue
        
        video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        actual_frames = min(actual_frames, video_frames)
        
        # Limit by wrist transforms if applicable
        if cam["type"] == "wrist" and "transforms" in cam:
            actual_frames = min(actual_frames, cam["transforms"].shape[0])
        
        captures[serial] = cap
        recorders[serial] = VideoRecorder(
            output_dir, serial, "tracks_overlay",
            cam["width"], cam["height"], fps=fps
        )
    
    if max_frames is not None:
        actual_frames = min(actual_frames, max_frames)
    
    print(f"\n[INFO] Rendering {actual_frames} frames with track overlays...")
    
    for frame_idx in range(actual_frames):
        if frame_idx % 50 == 0:
            print(f"  Frame {frame_idx}/{actual_frames}")
        
        track_points_world = tracks_3d[frame_idx] if total_track_pts > 0 else None
        
        # Build track window for trail visualization
        tracks_window = None
        if total_track_pts > 0 and track_trail_length > 1:
            start_idx = max(0, frame_idx - track_trail_length + 1)
            tracks_window = tracks_3d[start_idx:frame_idx + 1]
        
        for serial, cam in cameras.items():
            if serial not in captures:
                continue
            
            cap = captures[serial]
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Get camera pose for this frame
            if cam["type"] == "wrist":
                if frame_idx >= len(cam["transforms"]):
                    continue
                world_T_cam = cam["transforms"][frame_idx]
            else:
                world_T_cam = cam["world_T_cam"]
            
            # Project and draw track points
            if track_points_world is not None:
                uv_tracks, cols_tracks = project_points_to_image(
                    track_points_world,
                    cam["K"],
                    world_T_cam,
                    cam["width"],
                    cam["height"],
                    colors=track_colors_rgb,
                    min_depth=min_depth,
                )
                frame = draw_points_on_image(
                    frame, uv_tracks, colors=cols_tracks,
                    radius=4, default_color=(0, 0, 255)
                )
            
            # Draw track trails
            if tracks_window is not None:
                frame = draw_track_trails_on_image(
                    frame,
                    tracks_window,
                    cam["K"],
                    world_T_cam,
                    cam["width"],
                    cam["height"],
                    track_colors_rgb,
                    min_depth=min_depth,
                )
            
            recorders[serial].write_frame(frame)
    
    # Cleanup
    for cap in captures.values():
        cap.release()
    for rec in recorders.values():
        rec.close()
    
    print(f"\n[SUCCESS] Rendered videos saved to: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Render track overlays on MP4 videos using metadata from HuggingFace."
    )
    parser.add_argument(
        "--episode_id",
        required=True,
        help='Episode ID, e.g., "AUTOLab+84bd5053+2023-08-18-12h-01m-10s"',
    )
    parser.add_argument(
        "--metadata_repo",
        default=os.environ.get("METADATA_REPO", "sazirarrwth99/droid_metadata_only"),
        help="HuggingFace repo containing tracks.npz and extrinsics.npz",
    )
    parser.add_argument(
        "--video_source",
        help="Local path to DROID data containing MP4 videos (e.g., /data/droid/data/droid_raw/1.0.1)",
    )
    parser.add_argument(
        "--video_repo",
        help="HuggingFace repo containing MP4 videos (alternative to --video_source)",
    )
    parser.add_argument(
        "--cache_dir",
        default=os.path.join(os.getcwd(), "hf_render_cache"),
        help="Local cache directory for downloaded data.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory for rendered videos (default: ./rendered_tracks/{episode_id})",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum number of frames to render (default: all)",
    )
    parser.add_argument(
        "--track_trail_length",
        type=int,
        default=10,
        help="Number of frames for track trail visualization (default: 10)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Output video FPS (default: 30.0)",
    )
    args = parser.parse_args()

    # Validate inputs
    if not args.video_source and not args.video_repo:
        raise ValueError("Must provide either --video_source (local path) or --video_repo (HuggingFace repo)")

    # Parse episode ID
    episode_info = parse_episode_id(args.episode_id)
    print(f"[INFO] Episode: {episode_info['full_id']}")
    print(f"[INFO] Lab: {episode_info['lab']}, Date: {episode_info['date']}")

    # Create cache directory
    os.makedirs(args.cache_dir, exist_ok=True)
    token = os.environ.get("HF_TOKEN")

    # 1. Download metadata from HuggingFace
    metadata_dir = download_metadata_from_hf(
        args.metadata_repo,
        episode_info,
        cache_dir=os.path.join(args.cache_dir, "metadata"),
        token=token,
    )

    # Load tracks and extrinsics
    tracks_path = os.path.join(metadata_dir, "tracks.npz")
    extrinsics_path = os.path.join(metadata_dir, "extrinsics.npz")
    
    if not os.path.exists(tracks_path):
        raise FileNotFoundError(f"tracks.npz not found at {tracks_path}")
    if not os.path.exists(extrinsics_path):
        raise FileNotFoundError(f"extrinsics.npz not found at {extrinsics_path}")

    print(f"[INFO] Loading tracks from: {tracks_path}")
    tracks = np.load(tracks_path, allow_pickle=True)
    extrinsics = np.load(extrinsics_path, allow_pickle=True)

    tracks_3d = tracks["tracks_3d"]
    fps = float(tracks.get("fps", args.fps))
    
    print(f"[INFO] Tracks: {tracks_3d.shape[0]} frames, {tracks_3d.shape[1]} track points")

    # 2. Get video files
    if args.video_source:
        print(f"[INFO] Looking for MP4 videos in: {args.video_source}")
        video_paths = find_local_mp4_videos(args.video_source, episode_info)
    else:
        video_paths = download_videos_from_hf(
            args.video_repo,
            episode_info,
            cache_dir=os.path.join(args.cache_dir, "videos"),
            token=token,
        )

    if not video_paths:
        raise FileNotFoundError(f"No MP4 videos found for episode {episode_info['full_id']}")
    
    print(f"[INFO] Found {len(video_paths)} MP4 videos: {list(video_paths.keys())}")

    # 3. Build camera info
    print("\n[INFO] Building camera metadata...")
    cameras = build_camera_info(extrinsics, tracks, video_paths)
    
    if not cameras:
        raise RuntimeError("No cameras with both extrinsics and video found")

    # 4. Render track overlays
    output_dir = args.output_dir or os.path.join("./rendered_tracks", episode_info['full_id'])
    
    render_tracks_on_videos(
        cameras=cameras,
        tracks_3d=tracks_3d,
        output_dir=output_dir,
        fps=fps,
        max_frames=args.max_frames,
        track_trail_length=args.track_trail_length,
        min_depth=0.01,
    )

    print(f"\n[DONE] Episode: {episode_info['full_id']}")
    print(f"  Metadata from: {args.metadata_repo}")
    print(f"  Videos from: {args.video_source or args.video_repo}")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()

