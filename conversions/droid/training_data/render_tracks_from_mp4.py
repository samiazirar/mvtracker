#!/usr/bin/env python3
"""Render track overlays on MP4 videos using metadata from HuggingFace.

This script:
1. Downloads tracks.npz and extrinsics.npz from HuggingFace
2. Downloads MP4 videos from either local source or another HuggingFace repo
3. Renders multiple visualization modes for all generated attributes:
   - tracks_overlay: 3D contact tracks projected onto video
   - contact_frames: Contact frame coordinate systems (centroid + orientation)
   - gripper_poses: End-effector poses as coordinate frames
   - normalized_flow: Distance-normalized trajectory (1mm steps)
   - all: Generate all visualization types

REQUIREMENTS:
- tracks.npz MUST contain camera intrinsics (intrinsics_{serial} and image_size_{serial})
- This means the full pipeline with extract_rgb_depth.py must have been run first
- The metadata-only pipeline does NOT generate intrinsics (no depth extraction)

Unlike render_visualize_hf.py which requires PNG+depth data, this works with
just the metadata (tracks, extrinsics with intrinsics) and source MP4 videos.

Usage:
    # Using local MP4 videos - render all visualizations
    python render_tracks_from_mp4.py \
        --episode_id "AUTOLab+84bd5053+2023-08-18-12h-01m-10s" \
        --metadata_repo "sazirarrwth99/droid_metadata_only" \
        --video_source "/data/droid/data/droid_raw/1.0.1" \
        --mode all

    # Using MP4 videos from GCS (recommended)
    python render_tracks_from_mp4.py \
        --episode_id "AUTOLab+84bd5053+2023-08-18-12h-01m-10s" \
        --metadata_repo "sazirarrwth99/droid_metadata_only" \
        --gcs_bucket "gs://gresearch/robotics/droid_raw/1.0.1"

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
import subprocess
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

# Visualization mode constants
VIS_MODES = [
    "tracks_overlay",           # Original contact points (time-based)
    "contact_frames",           # Original contact frames (time-based)
    "gripper_poses",            # End-effector poses
    "normalized_flow",          # Normalized trajectory waypoints only
    "normalized_tracks",        # Normalized contact points (distance-based)
    "normalized_frames",        # Normalized contact frames (distance-based)
    "all",                      # Generate all visualization types
]


DEFAULT_CONFIG = {
    "fps": 30.0,
    "max_frames": None,
    "track_trail_length_video": 10,
    "min_depth": 0.01,
    "output_dir": "./rendered_tracks",
}


def draw_coordinate_frame(
    image: np.ndarray,
    T_world_frame: np.ndarray,
    K: np.ndarray,
    world_T_cam: np.ndarray,
    width: int,
    height: int,
    axis_length: float = 0.05,
    line_thickness: int = 2,
    min_depth: float = 0.01,
) -> np.ndarray:
    """Draw a 3D coordinate frame projected onto the image.

    Args:
        image: Image to draw on (modified in place)
        T_world_frame: [4, 4] transform of the coordinate frame in world space
        K: [3, 3] camera intrinsic matrix
        world_T_cam: [4, 4] camera extrinsics (world to camera transform)
        width, height: Image dimensions
        axis_length: Length of each axis in meters
        line_thickness: Thickness of axis lines
        min_depth: Minimum depth for valid projection

    Returns:
        Modified image with coordinate frame drawn
    """
    # Origin and axis endpoints in world frame
    origin = T_world_frame[:3, 3]
    x_axis = origin + T_world_frame[:3, 0] * axis_length
    y_axis = origin + T_world_frame[:3, 1] * axis_length
    z_axis = origin + T_world_frame[:3, 2] * axis_length

    points = np.array([origin, x_axis, y_axis, z_axis])

    # Project to image
    uv, _ = project_points_to_image(
        points, K, world_T_cam, width, height,
        colors=None, min_depth=min_depth
    )

    if uv is None or len(uv) < 4:
        return image

    # Check if points are valid (not NaN)
    origin_uv = uv[0]
    x_uv = uv[1]
    y_uv = uv[2]
    z_uv = uv[3]

    def is_valid(pt):
        return not (np.isnan(pt[0]) or np.isnan(pt[1]))

    # Draw axes: X=Red, Y=Green, Z=Blue
    if is_valid(origin_uv):
        origin_pt = (int(origin_uv[0]), int(origin_uv[1]))

        if is_valid(x_uv):
            x_pt = (int(x_uv[0]), int(x_uv[1]))
            cv2.line(image, origin_pt, x_pt, (0, 0, 255), line_thickness)  # Red = X

        if is_valid(y_uv):
            y_pt = (int(y_uv[0]), int(y_uv[1]))
            cv2.line(image, origin_pt, y_pt, (0, 255, 0), line_thickness)  # Green = Y

        if is_valid(z_uv):
            z_pt = (int(z_uv[0]), int(z_uv[1]))
            cv2.line(image, origin_pt, z_pt, (255, 0, 0), line_thickness)  # Blue = Z

        # Draw origin point
        cv2.circle(image, origin_pt, 4, (255, 255, 255), -1)

    return image


def draw_trajectory_line(
    image: np.ndarray,
    positions: np.ndarray,
    K: np.ndarray,
    world_T_cam: np.ndarray,
    width: int,
    height: int,
    color: tuple = (255, 255, 0),
    line_thickness: int = 2,
    min_depth: float = 0.01,
) -> np.ndarray:
    """Draw a trajectory line connecting 3D positions.

    Args:
        image: Image to draw on
        positions: [N, 3] array of 3D positions
        K: Camera intrinsic matrix
        world_T_cam: Camera extrinsics
        width, height: Image dimensions
        color: Line color (BGR)
        line_thickness: Thickness of line
        min_depth: Minimum depth for valid projection

    Returns:
        Modified image with trajectory line drawn
    """
    if len(positions) < 2:
        return image

    # Project all points
    uv, _ = project_points_to_image(
        positions, K, world_T_cam, width, height,
        colors=None, min_depth=min_depth
    )

    if uv is None or len(uv) < 2:
        return image

    # Draw lines between consecutive valid points
    for i in range(len(uv) - 1):
        pt1 = uv[i]
        pt2 = uv[i + 1]

        if np.isnan(pt1[0]) or np.isnan(pt1[1]) or np.isnan(pt2[0]) or np.isnan(pt2[1]):
            continue

        cv2.line(
            image,
            (int(pt1[0]), int(pt1[1])),
            (int(pt2[0]), int(pt2[1])),
            color,
            line_thickness
        )

    return image


def draw_normalized_flow_markers(
    image: np.ndarray,
    normalized_centroids: np.ndarray,
    current_normalized_idx: int,
    K: np.ndarray,
    world_T_cam: np.ndarray,
    width: int,
    height: int,
    min_depth: float = 0.01,
) -> np.ndarray:
    """Draw normalized flow waypoints with current position highlighted.

    Args:
        image: Image to draw on
        normalized_centroids: [N, 3] normalized trajectory positions
        current_normalized_idx: Index of current position in normalized trajectory
        K: Camera intrinsic matrix
        world_T_cam: Camera extrinsics
        width, height: Image dimensions
        min_depth: Minimum depth for valid projection

    Returns:
        Modified image with normalized flow markers
    """
    # Project all normalized centroids
    uv, _ = project_points_to_image(
        normalized_centroids, K, world_T_cam, width, height,
        colors=None, min_depth=min_depth
    )

    if uv is None:
        return image

    # Draw all waypoints as small circles
    for i, pt in enumerate(uv):
        if np.isnan(pt[0]) or np.isnan(pt[1]):
            continue

        center = (int(pt[0]), int(pt[1]))

        if i == current_normalized_idx:
            # Current position: large yellow circle
            cv2.circle(image, center, 8, (0, 255, 255), -1)
            cv2.circle(image, center, 8, (0, 0, 0), 2)
        elif i < current_normalized_idx:
            # Past positions: small gray circles
            cv2.circle(image, center, 3, (128, 128, 128), -1)
        else:
            # Future positions: small white circles
            cv2.circle(image, center, 3, (255, 255, 255), -1)

    return image


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


def run_gsutil(command: str, check: bool = True) -> bool:
    """Run gsutil command."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=check,
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"[WARN] gsutil error: {e.stderr}", file=sys.stderr)
        return False


def download_videos_from_gcs(
    gcs_bucket: str,
    episode_info: dict,
    cache_dir: str,
) -> Dict[str, str]:
    """Download MP4 videos from GCS bucket.

    Args:
        gcs_bucket: GCS bucket path (e.g., gs://gresearch/robotics/droid_raw/1.0.1)
        episode_info: Parsed episode info dict
        cache_dir: Local cache directory for downloads

    Returns:
        Dict mapping camera serial to local MP4 path
    """
    timestamp_folder = episode_info["timestamp_folder"]
    base = episode_info["lab"]
    date = episode_info["date"]

    videos = {}

    # Try both success and failure paths
    for outcome in ["success", "failure"]:
        rel_path = f"{base}/{outcome}/{date}/{timestamp_folder}"
        gcs_path = f"{gcs_bucket}/{rel_path}"
        local_path = os.path.join(cache_dir, rel_path)

        # Check if trajectory.h5 exists at this path (to confirm episode location)
        check_cmd = f'gsutil -q stat "{gcs_path}/trajectory.h5"'
        if not run_gsutil(check_cmd, check=False):
            continue

        print(f"[INFO] Found episode at GCS: {rel_path}")

        # Create local directories
        mp4_local = os.path.join(local_path, "recordings", "MP4")
        os.makedirs(mp4_local, exist_ok=True)

        # Download MP4 files
        gcs_mp4_path = f"{gcs_path}/recordings/MP4"
        print(f"[INFO] Downloading MP4 videos from {gcs_mp4_path}...")

        # First list the MP4 files
        list_cmd = f'gsutil ls "{gcs_mp4_path}/*.mp4" 2>/dev/null'
        try:
            result = subprocess.run(
                list_cmd,
                shell=True,
                capture_output=True,
                text=True
            )
            mp4_files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip().endswith('.mp4')]
        except Exception:
            mp4_files = []

        if not mp4_files:
            # Try lowercase mp4 folder
            gcs_mp4_path = f"{gcs_path}/recordings/mp4"
            list_cmd = f'gsutil ls "{gcs_mp4_path}/*.mp4" 2>/dev/null'
            try:
                result = subprocess.run(
                    list_cmd,
                    shell=True,
                    capture_output=True,
                    text=True
                )
                mp4_files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip().endswith('.mp4')]
            except Exception:
                mp4_files = []

        if not mp4_files:
            print(f"[WARN] No MP4 files found at {gcs_mp4_path}")
            continue

        # Download each MP4 file
        for gcs_file in mp4_files:
            filename = os.path.basename(gcs_file)
            local_file = os.path.join(mp4_local, filename)

            if not os.path.exists(local_file):
                print(f"  Downloading {filename}...")
                cmd = f'gsutil cp "{gcs_file}" "{local_file}"'
                run_gsutil(cmd)
            else:
                print(f"  {filename} already cached")

            if os.path.exists(local_file):
                # Extract camera serial from filename
                serial = filename.replace(".mp4", "").replace("SN", "").replace("sn", "")
                videos[serial] = local_file

        if videos:
            print(f"[INFO] Downloaded {len(videos)} MP4 videos")
            return videos

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


def render_all_visualizations(
    cameras: Dict[str, dict],
    tracks_data: dict,
    output_base_dir: str,
    modes: list,
    fps: float = 30.0,
    max_frames: Optional[int] = None,
    track_trail_length: int = 10,
    min_depth: float = 0.01,
):
    """Render all visualization modes for testing generated attributes.

    Creates separate output folders for each visualization type:
    - tracks_overlay/: 3D contact track points
    - contact_frames/: Contact frame coordinate systems
    - gripper_poses/: End-effector pose coordinate frames
    - normalized_flow/: Distance-normalized trajectory waypoints

    Args:
        cameras: Camera metadata dict
        tracks_data: Dict containing all track data from npz file
        output_base_dir: Base output directory
        modes: List of visualization modes to render
        fps: Output video FPS
        max_frames: Maximum frames to render
        track_trail_length: Trail length for track visualization
        min_depth: Minimum depth for projection
    """
    # Extract data from tracks_data
    # Original (time-based) data
    tracks_3d = tracks_data.get("tracks_3d")
    contact_frames = tracks_data.get("contact_frames")
    left_contact_frames = tracks_data.get("left_contact_frames")
    right_contact_frames = tracks_data.get("right_contact_frames")
    gripper_poses = tracks_data.get("gripper_poses")
    contact_centroids = tracks_data.get("contact_centroids")

    # Normalized (distance-based) data
    normalized_centroids = tracks_data.get("normalized_centroids")
    normalized_frames_data = tracks_data.get("normalized_frames")
    normalized_tracks_3d = tracks_data.get("normalized_tracks_3d")
    normalized_left_frames = tracks_data.get("normalized_left_frames")
    normalized_right_frames = tracks_data.get("normalized_right_frames")
    frame_to_normalized_idx = tracks_data.get("frame_to_normalized_idx")
    num_normalized_steps = int(tracks_data.get("num_normalized_steps", 0))

    num_frames = int(tracks_data.get("num_frames", len(tracks_3d) if tracks_3d is not None else 0))

    # Determine actual modes to render
    if "all" in modes:
        active_modes = [
            "tracks_overlay", "contact_frames", "gripper_poses",
            "normalized_flow", "normalized_tracks", "normalized_frames"
        ]
    else:
        active_modes = modes

    print(f"\n[INFO] Rendering {len(active_modes)} visualization modes: {active_modes}")

    # Track colors
    total_track_pts = tracks_3d.shape[1] if tracks_3d is not None else 0
    num_contact_pts = total_track_pts // 2
    track_colors_rgb = np.zeros((total_track_pts, 3), dtype=np.uint8)
    if total_track_pts > 0:
        track_colors_rgb[:num_contact_pts, :] = [51, 127, 255]  # Blue-ish
        track_colors_rgb[num_contact_pts:, :] = [51, 255, 127]  # Green-ish

    # Calculate actual frame count
    actual_frames = num_frames
    for serial, cam in cameras.items():
        cap = cv2.VideoCapture(cam["video_path"])
        if cap.isOpened():
            video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            actual_frames = min(actual_frames, video_frames)
            cap.release()
        if cam["type"] == "wrist" and "transforms" in cam:
            actual_frames = min(actual_frames, cam["transforms"].shape[0])

    if max_frames is not None:
        actual_frames = min(actual_frames, max_frames)

    # Render each mode
    for mode in active_modes:
        mode_output_dir = os.path.join(output_base_dir, mode)
        os.makedirs(mode_output_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"[MODE] {mode}")
        print(f"{'='*60}")

        # Open video captures and create recorders for this mode
        captures = {}
        recorders = {}

        for serial, cam in cameras.items():
            cap = cv2.VideoCapture(cam["video_path"])
            if not cap.isOpened():
                print(f"[WARN] Could not open video: {cam['video_path']}")
                continue

            captures[serial] = cap
            recorders[serial] = VideoRecorder(
                mode_output_dir, serial, mode,
                cam["width"], cam["height"], fps=fps
            )

        print(f"[INFO] Rendering {actual_frames} frames for {mode}...")

        for frame_idx in range(actual_frames):
            if frame_idx % 50 == 0:
                print(f"  Frame {frame_idx}/{actual_frames}")

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

                K = cam["K"]
                width = cam["width"]
                height = cam["height"]

                # Apply visualization based on mode
                if mode == "tracks_overlay":
                    # Draw contact track points
                    if tracks_3d is not None and total_track_pts > 0:
                        track_points = tracks_3d[frame_idx]
                        uv_tracks, cols_tracks = project_points_to_image(
                            track_points, K, world_T_cam, width, height,
                            colors=track_colors_rgb, min_depth=min_depth,
                        )
                        frame = draw_points_on_image(
                            frame, uv_tracks, colors=cols_tracks,
                            radius=4, default_color=(0, 0, 255)
                        )

                        # Draw track trails
                        if track_trail_length > 1:
                            start_idx = max(0, frame_idx - track_trail_length + 1)
                            tracks_window = tracks_3d[start_idx:frame_idx + 1]
                            frame = draw_track_trails_on_image(
                                frame, tracks_window, K, world_T_cam,
                                width, height, track_colors_rgb, min_depth=min_depth,
                            )

                elif mode == "contact_frames":
                    # Draw contact frame coordinate systems
                    if contact_frames is not None:
                        # Combined contact frame (yellow origin)
                        frame = draw_coordinate_frame(
                            frame, contact_frames[frame_idx], K, world_T_cam,
                            width, height, axis_length=0.03, line_thickness=3,
                            min_depth=min_depth,
                        )

                    # Draw per-finger contact frames
                    if left_contact_frames is not None:
                        frame = draw_coordinate_frame(
                            frame, left_contact_frames[frame_idx], K, world_T_cam,
                            width, height, axis_length=0.02, line_thickness=2,
                            min_depth=min_depth,
                        )
                    if right_contact_frames is not None:
                        frame = draw_coordinate_frame(
                            frame, right_contact_frames[frame_idx], K, world_T_cam,
                            width, height, axis_length=0.02, line_thickness=2,
                            min_depth=min_depth,
                        )

                    # Also draw centroid trajectory
                    if contact_centroids is not None:
                        start_idx = max(0, frame_idx - 30)
                        trajectory = contact_centroids[start_idx:frame_idx + 1]
                        frame = draw_trajectory_line(
                            frame, trajectory, K, world_T_cam, width, height,
                            color=(0, 255, 255), line_thickness=2, min_depth=min_depth,
                        )

                elif mode == "gripper_poses":
                    # Draw end-effector pose coordinate frame
                    if gripper_poses is not None:
                        frame = draw_coordinate_frame(
                            frame, gripper_poses[frame_idx], K, world_T_cam,
                            width, height, axis_length=0.08, line_thickness=3,
                            min_depth=min_depth,
                        )

                        # Draw gripper trajectory
                        start_idx = max(0, frame_idx - 30)
                        trajectory = gripper_poses[start_idx:frame_idx + 1, :3, 3]
                        frame = draw_trajectory_line(
                            frame, trajectory, K, world_T_cam, width, height,
                            color=(255, 0, 255), line_thickness=2, min_depth=min_depth,
                        )

                elif mode == "normalized_flow":
                    # Draw normalized flow waypoints (trajectory overview)
                    if normalized_centroids is not None and frame_to_normalized_idx is not None:
                        current_norm_idx = int(frame_to_normalized_idx[frame_idx])

                        # Draw trajectory line through normalized waypoints
                        frame = draw_trajectory_line(
                            frame, normalized_centroids, K, world_T_cam,
                            width, height, color=(255, 255, 0), line_thickness=2,
                            min_depth=min_depth,
                        )

                        # Draw waypoint markers
                        frame = draw_normalized_flow_markers(
                            frame, normalized_centroids, current_norm_idx,
                            K, world_T_cam, width, height, min_depth=min_depth,
                        )

                        # Draw current normalized frame coordinate system
                        if normalized_frames_data is not None:
                            frame = draw_coordinate_frame(
                                frame, normalized_frames_data[current_norm_idx],
                                K, world_T_cam, width, height,
                                axis_length=0.04, line_thickness=3, min_depth=min_depth,
                            )

                        # Add text overlay with normalized step info
                        text = f"Norm step: {current_norm_idx}/{len(normalized_centroids)-1}"
                        cv2.putText(frame, text, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                elif mode == "normalized_tracks":
                    # Draw normalized contact points (distance-based)
                    if normalized_tracks_3d is not None and frame_to_normalized_idx is not None:
                        current_norm_idx = int(frame_to_normalized_idx[frame_idx])
                        norm_track_points = normalized_tracks_3d[current_norm_idx]

                        # Project and draw normalized track points
                        uv_tracks, cols_tracks = project_points_to_image(
                            norm_track_points, K, world_T_cam, width, height,
                            colors=track_colors_rgb, min_depth=min_depth,
                        )
                        frame = draw_points_on_image(
                            frame, uv_tracks, colors=cols_tracks,
                            radius=5, default_color=(0, 255, 255)
                        )

                        # Draw trajectory line through normalized centroids
                        if normalized_centroids is not None:
                            frame = draw_trajectory_line(
                                frame, normalized_centroids, K, world_T_cam,
                                width, height, color=(255, 255, 0), line_thickness=1,
                                min_depth=min_depth,
                            )

                        # Add text overlay
                        text = f"Norm step: {current_norm_idx}/{num_normalized_steps-1}"
                        cv2.putText(frame, text, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                elif mode == "normalized_frames":
                    # Draw normalized contact frames (distance-based)
                    if frame_to_normalized_idx is not None:
                        current_norm_idx = int(frame_to_normalized_idx[frame_idx])

                        # Draw combined normalized frame
                        if normalized_frames_data is not None:
                            frame = draw_coordinate_frame(
                                frame, normalized_frames_data[current_norm_idx],
                                K, world_T_cam, width, height,
                                axis_length=0.04, line_thickness=3, min_depth=min_depth,
                            )

                        # Draw per-finger normalized frames
                        if normalized_left_frames is not None:
                            frame = draw_coordinate_frame(
                                frame, normalized_left_frames[current_norm_idx],
                                K, world_T_cam, width, height,
                                axis_length=0.025, line_thickness=2, min_depth=min_depth,
                            )
                        if normalized_right_frames is not None:
                            frame = draw_coordinate_frame(
                                frame, normalized_right_frames[current_norm_idx],
                                K, world_T_cam, width, height,
                                axis_length=0.025, line_thickness=2, min_depth=min_depth,
                            )

                        # Draw trajectory through normalized centroids
                        if normalized_centroids is not None:
                            frame = draw_trajectory_line(
                                frame, normalized_centroids, K, world_T_cam,
                                width, height, color=(0, 255, 255), line_thickness=2,
                                min_depth=min_depth,
                            )

                        # Add text overlay
                        text = f"Norm step: {current_norm_idx}/{num_normalized_steps-1}"
                        cv2.putText(frame, text, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                recorders[serial].write_frame(frame)

        # Cleanup for this mode
        for cap in captures.values():
            cap.release()
        for rec in recorders.values():
            rec.close()

        print(f"[SUCCESS] {mode} videos saved to: {mode_output_dir}")

    print(f"\n{'='*60}")
    print(f"[DONE] All visualizations saved to: {output_base_dir}")
    print(f"{'='*60}")


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
        "--gcs_bucket",
        help="GCS bucket path for MP4 videos (e.g., gs://gresearch/robotics/droid_raw/1.0.1)",
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
    parser.add_argument(
        "--mode",
        type=str,
        nargs="+",
        default=["tracks_overlay"],
        choices=VIS_MODES,
        help=f"Visualization mode(s) to render. Options: {VIS_MODES}. Default: tracks_overlay",
    )
    args = parser.parse_args()

    # Validate inputs
    if not args.video_source and not args.video_repo and not args.gcs_bucket:
        raise ValueError("Must provide one of: --video_source (local path), --video_repo (HuggingFace repo), or --gcs_bucket (GCS path)")

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

    # Convert npz to dict for easier access
    tracks_data = {key: tracks[key] for key in tracks.files}

    tracks_3d = tracks_data.get("tracks_3d")
    fps = float(tracks_data.get("fps", args.fps))

    print(f"[INFO] Tracks: {tracks_3d.shape[0]} frames, {tracks_3d.shape[1]} track points")

    # Print available data for visualization
    print(f"[INFO] Available data in tracks.npz:")
    for key in sorted(tracks.files):
        val = tracks[key]
        if hasattr(val, 'shape'):
            print(f"  - {key}: {val.shape} {val.dtype}")
        else:
            print(f"  - {key}: {type(val).__name__}")

    # 2. Get video files
    if args.video_source:
        print(f"[INFO] Looking for MP4 videos in: {args.video_source}")
        video_paths = find_local_mp4_videos(args.video_source, episode_info)
    elif args.gcs_bucket:
        print(f"[INFO] Downloading MP4 videos from GCS: {args.gcs_bucket}")
        video_paths = download_videos_from_gcs(
            args.gcs_bucket,
            episode_info,
            cache_dir=os.path.join(args.cache_dir, "videos"),
        )
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

    # 4. Render visualizations
    output_dir = args.output_dir or os.path.join("./rendered_tracks", episode_info['full_id'])

    # Use comprehensive visualization if mode is not just tracks_overlay
    if args.mode == ["tracks_overlay"]:
        # Legacy single-mode rendering
        render_tracks_on_videos(
            cameras=cameras,
            tracks_3d=tracks_3d,
            output_dir=output_dir,
            fps=fps,
            max_frames=args.max_frames,
            track_trail_length=args.track_trail_length,
            min_depth=0.01,
        )
    else:
        # Multi-mode comprehensive rendering
        render_all_visualizations(
            cameras=cameras,
            tracks_data=tracks_data,
            output_base_dir=output_dir,
            modes=args.mode,
            fps=fps,
            max_frames=args.max_frames,
            track_trail_length=args.track_trail_length,
            min_depth=0.01,
        )

    print(f"\n[DONE] Episode: {episode_info['full_id']}")
    print(f"  Metadata from: {args.metadata_repo}")
    print(f"  Videos from: {args.video_source or args.gcs_bucket or args.video_repo}")
    print(f"  Output: {output_dir}")
    print(f"  Modes: {args.mode}")


if __name__ == "__main__":
    main()

