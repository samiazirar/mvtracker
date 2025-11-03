#!/usr/bin/env python3
"""
Utilities for exporting tracking results to per-camera videos.

Functions for:
- Projecting 3D tracks to 2D camera views
- Drawing tracks on video frames
- Exporting videos per camera with track overlays
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def project_3d_to_2d(
    points_3d: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points to 2D image coordinates.
    
    Args:
        points_3d: 3D points in world coordinates [..., 3]
        intrinsics: Camera intrinsics matrix [3, 3]
        extrinsics: Camera extrinsics matrix [4, 4] (world to camera)
    
    Returns:
        points_2d: 2D image coordinates [..., 2]
        depths: Depth values [...] (positive if in front of camera)
    """
    original_shape = points_3d.shape[:-1]
    points_3d_flat = points_3d.reshape(-1, 3)  # [N, 3]
    
    # Add homogeneous coordinate
    points_3d_homo = np.concatenate([
        points_3d_flat,
        np.ones((points_3d_flat.shape[0], 1))
    ], axis=-1)  # [N, 4]
    
    # Transform to camera coordinates
    points_cam = (extrinsics @ points_3d_homo.T).T  # [N, 4]
    points_cam_3d = points_cam[:, :3]  # [N, 3]
    
    # Extract depth (z-coordinate in camera frame)
    depths = points_cam_3d[:, 2]  # [N]
    
    # Project to image plane
    points_2d_homo = (intrinsics @ points_cam_3d.T).T  # [N, 3]
    
    # Normalize by depth
    points_2d = points_2d_homo[:, :2] / (points_2d_homo[:, 2:3] + 1e-8)  # [N, 2]
    
    # Reshape back to original shape
    points_2d = points_2d.reshape(*original_shape, 2)
    depths = depths.reshape(*original_shape)
    
    return points_2d, depths


def project_tracks_to_camera(
    tracks_3d: np.ndarray,
    visibilities: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    image_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D tracks to 2D camera view and update visibility.
    
    Args:
        tracks_3d: 3D track trajectories [T, N, 3]
        visibilities: Track visibility mask [T, N]
        intrinsics: Camera intrinsics [3, 3] or [T, 3, 3]
        extrinsics: Camera extrinsics [3, 4] or [4, 4] or [T, 3, 4] or [T, 4, 4]
        image_shape: Image dimensions (H, W)
    
    Returns:
        tracks_2d: 2D track coordinates [T, N, 2]
        visibilities_2d: Updated visibility (also checks if in frame) [T, N]
    """
    T, N, _ = tracks_3d.shape
    H, W = image_shape
    
    # Handle temporal dimension in intrinsics/extrinsics
    has_temporal = intrinsics.ndim == 3
    
    if has_temporal:
        # Process frame by frame
        tracks_2d = np.zeros((T, N, 2), dtype=np.float32)
        depths = np.zeros((T, N), dtype=np.float32)
        
        for t in range(T):
            K = intrinsics[t]  # [3, 3]
            E = extrinsics[t]  # [3, 4] or [4, 4]
            
            # Ensure extrinsics is [4, 4]
            if E.shape == (3, 4):
                E_4x4 = np.eye(4)
                E_4x4[:3, :] = E
                E = E_4x4
            
            tracks_2d[t], depths[t] = project_3d_to_2d(tracks_3d[t], K, E)
    else:
        # Single intrinsics/extrinsics for all frames
        K = intrinsics  # [3, 3]
        E = extrinsics  # [3, 4] or [4, 4]
        
        # Ensure extrinsics is [4, 4]
        if E.shape == (3, 4):
            E_4x4 = np.eye(4)
            E_4x4[:3, :] = E
            E = E_4x4
        
        # Project all points
        tracks_2d, depths = project_3d_to_2d(tracks_3d, K, E)
    
    # Update visibility: must be visible, in front of camera, and in frame
    in_front = depths > 0
    in_frame_x = (tracks_2d[..., 0] >= 0) & (tracks_2d[..., 0] < W)
    in_frame_y = (tracks_2d[..., 1] >= 0) & (tracks_2d[..., 1] < H)
    in_frame = in_frame_x & in_frame_y
    
    visibilities_2d = visibilities & in_front & in_frame
    
    return tracks_2d, visibilities_2d


def get_track_colors(
    n_tracks: int,
    colormap: str = "gist_rainbow",
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate colors for tracks.
    
    Args:
        n_tracks: Number of tracks
        colormap: Matplotlib colormap name
        seed: Random seed for consistency
    
    Returns:
        colors: RGB colors [N, 3] in range [0, 255]
    """
    if seed is not None:
        np.random.seed(seed)
    
    cmap = matplotlib.colormaps[colormap]
    
    # Generate evenly spaced colors
    indices = np.linspace(0, 1, n_tracks)
    colors = cmap(indices)[:, :3]  # [N, 3] in range [0, 1]
    colors = (colors * 255).astype(np.uint8)
    
    return colors


def draw_tracks_on_frame(
    frame: np.ndarray,
    tracks_2d: np.ndarray,
    visibilities: np.ndarray,
    colors: np.ndarray,
    current_frame: int,
    trail_length: int = 10,
    point_radius: int = 3,
    line_thickness: int = 2,
    show_query_frame: bool = True,
    query_frame_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Draw tracks on a video frame.
    
    Args:
        frame: Video frame [H, W, 3] (uint8)
        tracks_2d: 2D track coordinates [T, N, 2]
        visibilities: Track visibility [T, N]
        colors: Track colors [N, 3] (uint8)
        current_frame: Current frame index
        trail_length: Number of previous frames to show (0 = no trail, -1 = all)
        point_radius: Radius of track points
        line_thickness: Thickness of track lines
        show_query_frame: Whether to highlight query frame with larger circle
        query_frame_indices: Query frame index for each track [N]
    
    Returns:
        frame_with_tracks: Frame with tracks drawn [H, W, 3]
    """
    frame_with_tracks = frame.copy()
    T, N, _ = tracks_2d.shape
    
    # Determine which frames to draw
    if trail_length < 0:
        start_frame = 0
    else:
        start_frame = max(0, current_frame - trail_length)
    
    # Draw track trails (lines connecting points)
    for i in range(N):
        if not visibilities[current_frame, i]:
            continue
        
        color = tuple(int(c) for c in colors[i])
        
        # Draw trail
        for t in range(start_frame, current_frame + 1):
            if t >= T - 1:
                break
            
            if visibilities[t, i] and visibilities[t + 1, i]:
                pt1 = tuple(tracks_2d[t, i].astype(int))
                pt2 = tuple(tracks_2d[t + 1, i].astype(int))
                
                # Fade older points
                if trail_length > 0:
                    alpha = (t - start_frame + 1) / (current_frame - start_frame + 1)
                    faded_color = tuple(int(c * alpha) for c in color)
                else:
                    faded_color = color
                
                cv2.line(frame_with_tracks, pt1, pt2, faded_color, line_thickness)
    
    # Draw current points
    for i in range(N):
        if not visibilities[current_frame, i]:
            continue
        
        pt = tuple(tracks_2d[current_frame, i].astype(int))
        color = tuple(int(c) for c in colors[i])
        
        # Highlight query frame point with larger circle
        if show_query_frame and query_frame_indices is not None:
            if current_frame == query_frame_indices[i]:
                cv2.circle(frame_with_tracks, pt, point_radius + 3, color, -1)
                cv2.circle(frame_with_tracks, pt, point_radius + 5, (255, 255, 255), 2)
            else:
                cv2.circle(frame_with_tracks, pt, point_radius, color, -1)
        else:
            cv2.circle(frame_with_tracks, pt, point_radius, color, -1)
    
    return frame_with_tracks


def export_tracks_to_video(
    output_path: Path,
    frames: np.ndarray,
    tracks_2d: np.ndarray,
    visibilities: np.ndarray,
    colors: Optional[np.ndarray] = None,
    fps: float = 30.0,
    trail_length: int = 10,
    point_radius: int = 3,
    line_thickness: int = 2,
    query_frame_indices: Optional[np.ndarray] = None,
    codec: str = "mp4v",
    verbose: bool = True,
) -> Path:
    """
    Export tracks overlaid on video frames to a video file.
    
    Args:
        output_path: Output video file path
        frames: Video frames [T, H, W, 3] (uint8)
        tracks_2d: 2D track coordinates [T, N, 2]
        visibilities: Track visibility [T, N]
        colors: Track colors [N, 3] (uint8), generated if None
        fps: Frames per second
        trail_length: Number of previous frames to show in trail
        point_radius: Radius of track points
        line_thickness: Thickness of track lines
        query_frame_indices: Query frame index for each track [N]
        codec: Video codec (e.g., 'mp4v', 'avc1', 'h264')
        verbose: Print progress
    
    Returns:
        output_path: Path to created video file
    """
    T, H, W, C = frames.shape
    _, N, _ = tracks_2d.shape
    
    # Generate colors if not provided
    if colors is None:
        colors = get_track_colors(N)
    
    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))
    
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")
    
    if verbose:
        print(f"[INFO] Exporting video to {output_path}")
        print(f"[INFO]   Frames: {T}, Resolution: {W}x{H}, FPS: {fps}")
        print(f"[INFO]   Tracks: {N}, Trail length: {trail_length}")
    
    # Process each frame
    for t in range(T):
        if verbose and (t % 10 == 0 or t == T - 1):
            print(f"[INFO]   Processing frame {t+1}/{T}", end='\r')
        
        # Draw tracks on frame
        frame_with_tracks = draw_tracks_on_frame(
            frame=frames[t],
            tracks_2d=tracks_2d,
            visibilities=visibilities,
            colors=colors,
            current_frame=t,
            trail_length=trail_length,
            point_radius=point_radius,
            line_thickness=line_thickness,
            query_frame_indices=query_frame_indices,
        )
        
        # Write frame
        writer.write(cv2.cvtColor(frame_with_tracks, cv2.COLOR_RGB2BGR))
    
    writer.release()
    
    if verbose:
        print(f"\n[INFO] Video exported successfully: {output_path}")
    
    return output_path


def export_tracks_to_videos_per_camera(
    output_dir: Path,
    base_name: str,
    rgbs: np.ndarray,
    tracks_3d: np.ndarray,
    visibilities: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    camera_ids: Optional[List[str]] = None,
    query_points: Optional[np.ndarray] = None,
    fps: float = 30.0,
    trail_length: int = 10,
    colors: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Dict[str, Path]:
    """
    Export tracks to separate videos for each camera view.
    
    Args:
        output_dir: Output directory for videos
        base_name: Base name for output files
        rgbs: RGB frames [V, T, 3, H, W] (float32 in [0, 1] or uint8)
        tracks_3d: 3D track trajectories [T, N, 3]
        visibilities: Track visibility [T, N]
        intrinsics: Camera intrinsics [V, T, 3, 3] or [V, 3, 3]
        extrinsics: Camera extrinsics [V, T, 3, 4] or [V, 3, 4]
        camera_ids: Camera identifiers (optional)
        query_points: Query points [N, 4] where cols=[t, x, y, z] (optional)
        fps: Frames per second
        trail_length: Number of previous frames to show in trail
        colors: Track colors [N, 3] (uint8), generated if None
        verbose: Print progress
    
    Returns:
        video_paths: Dictionary mapping camera_id to video path
    """
    V, T, C, H, W = rgbs.shape
    _, N, _ = tracks_3d.shape
    
    # Convert RGB to uint8 if needed
    if rgbs.dtype != np.uint8:
        if rgbs.max() <= 1.0:
            rgbs = (rgbs * 255).astype(np.uint8)
        else:
            rgbs = rgbs.astype(np.uint8)
    
    # Transpose to [V, T, H, W, C]
    rgbs = rgbs.transpose(0, 1, 3, 4, 2)
    
    # Generate camera IDs if not provided
    if camera_ids is None:
        camera_ids = [f"cam_{v:02d}" for v in range(V)]
    
    # Generate colors if not provided
    if colors is None:
        colors = get_track_colors(N)
    
    # Extract query frame indices if provided
    query_frame_indices = None
    if query_points is not None:
        query_frame_indices = query_points[:, 0].astype(int)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export video for each camera
    video_paths = {}
    
    for v, cam_id in enumerate(camera_ids):
        if verbose:
            print(f"\n[INFO] ========== Camera {v+1}/{V}: {cam_id} ==========")
        
        # Project tracks to this camera
        tracks_2d, vis_2d = project_tracks_to_camera(
            tracks_3d=tracks_3d,
            visibilities=visibilities,
            intrinsics=intrinsics[v],
            extrinsics=extrinsics[v],
            image_shape=(H, W),
        )
        
        # Export video
        output_path = output_dir / f"{base_name}_{cam_id}.mp4"
        export_tracks_to_video(
            output_path=output_path,
            frames=rgbs[v],
            tracks_2d=tracks_2d,
            visibilities=vis_2d,
            colors=colors,
            fps=fps,
            trail_length=trail_length,
            query_frame_indices=query_frame_indices,
            verbose=verbose,
        )
        
        video_paths[cam_id] = output_path
    
    if verbose:
        print(f"\n[INFO] ========================================")
        print(f"[INFO] Exported {len(video_paths)} videos to {output_dir}")
        print(f"[INFO] ========================================")
    
    return video_paths
