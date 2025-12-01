"""Video utilities for rendering reprojected point clouds.

This module provides functions for:
- Creating video files from rendered frames
- Projecting 3D point clouds to 2D image coordinates
- Drawing points on images with colors
"""

import cv2
import numpy as np
import os
import imageio
from typing import Optional, Tuple
from .transforms import transform_points, invert_transform


# =============================================================================
# Video Recording
# =============================================================================

class VideoRecorder:
    """
    Robust video recorder using ImageIO (FFmpeg wrapper).
    Generates VS Code/Web compatible MP4 (H.264) files.
    
    Usage:
        recorder = VideoRecorder(output_dir, camera_name, suffix, width, height)
        for frame in frames:
            recorder.write_frame(frame)
        recorder.close()
    """
    
    def __init__(
        self, 
        output_dir: str, 
        camera_name: str, 
        suffix: str, 
        width: int, 
        height: int, 
        fps: float = 30.0,
        ext: str = "mp4",
        fourcc: str = "mp4v", # Ignored by imageio, kept for API compatibility
    ):
        """
        Initialize video recorder.
        
        Args:
            output_dir: Directory to save video files
            camera_name: Camera identifier for filename
            suffix: Suffix for filename
            width: Video width in pixels
            height: Video height in pixels
            fps: Frames per second (default: 30)
            ext: File extension (ignored, always forces mp4 for compatibility)
            fourcc: Codec string (ignored, always uses libx264)
        """
        self.output_dir = output_dir
        self.camera_name = camera_name
        self.suffix = suffix
        self.width = width
        self.height = height
        self.fps = fps
        
        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)
        
        # Force MP4 extension for VS Code compatibility
        self.filename = os.path.join(output_dir, f"{camera_name}_{suffix}.mp4")
        
        print(f"[VideoRecorder] Initializing ImageIO writer for {self.filename}...")
        
        # codec='libx264' and pixelformat='yuv420p' are CRITICAL for VS Code/QuickTime support
        self.writer = imageio.get_writer(
            self.filename, 
            fps=fps, 
            codec='libx264', 
            quality=8,
            pixelformat='yuv420p',
            macro_block_size=None # Prevents errors if resolution isn't div by 16
        )
        
        self.frame_count = 0
    
    def write_frame(self, frame: np.ndarray):
        """
        Write a frame to the video.
        
        Args:
            frame: BGR image as numpy array (H, W, 3)
        """
        if frame is None:
            return
        
        # Ensure correct size
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
            
        # OpenCV uses BGR, ImageIO (and most players) expect RGB.
        # We must convert here, otherwise people look blue.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        self.writer.append_data(frame_rgb)
        self.frame_count += 1
    
    def close(self):
        """Release the video writer."""
        if self.writer is not None:
            self.writer.close()
            print(f"[VideoRecorder] Saved {self.frame_count} frames to {self.filename}")


# =============================================================================
# Point Cloud Projection
# =============================================================================

def project_points_to_image(
    points_world: np.ndarray,
    K: np.ndarray,
    world_T_cam: np.ndarray,
    width: int,
    height: int,
    colors: Optional[np.ndarray] = None,
    min_depth: float = 0.01
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Project 3D world points to 2D image coordinates.
    
    Args:
        points_world: Nx3 array of 3D points in world frame
        K: 3x3 camera intrinsic matrix
        world_T_cam: 4x4 transformation from camera to world (camera pose)
        width: Image width
        height: Image height
        colors: Optional Nx3 array of RGB colors
        min_depth: Minimum depth for valid projection (default: 1cm)
        
    Returns:
        Tuple of (uv, colors_filtered) where:
            uv: Mx2 array of valid 2D pixel coordinates
            colors_filtered: Mx3 array of colors for valid points (or None)
    """
    if points_world.shape[0] == 0:
        return np.empty((0, 2)), None if colors is None else np.empty((0, 3))
    
    # Transform points from world to camera frame
    cam_T_world = invert_transform(world_T_cam)
    points_cam = transform_points(points_world, cam_T_world)
    
    # Filter points behind camera
    z = points_cam[:, 2]
    valid_depth = z > min_depth
    
    if not np.any(valid_depth):
        return np.empty((0, 2)), None if colors is None else np.empty((0, 3))
    
    points_cam_valid = points_cam[valid_depth]
    colors_valid = colors[valid_depth] if colors is not None else None
    
    # Project to 2D
    x = points_cam_valid[:, 0]
    y = points_cam_valid[:, 1]
    z = points_cam_valid[:, 2]
    
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    u = (x * fx / z) + cx
    v = (y * fy / z) + cy
    
    # Filter points outside image bounds
    in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    
    uv = np.stack([u[in_bounds], v[in_bounds]], axis=-1)
    colors_out = colors_valid[in_bounds] if colors_valid is not None else None
    
    return uv, colors_out


def project_points_with_depth(
    points_world: np.ndarray,
    K: np.ndarray,
    world_T_cam: np.ndarray,
    width: int,
    height: int,
    colors: Optional[np.ndarray] = None,
    min_depth: float = 0.01
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Project 3D world points to 2D with depth values for z-buffering.
    
    Args:
        points_world: Nx3 array of 3D points in world frame
        K: 3x3 camera intrinsic matrix
        world_T_cam: 4x4 transformation from camera to world
        width: Image width
        height: Image height
        colors: Optional Nx3 array of RGB colors
        min_depth: Minimum depth for valid projection
        
    Returns:
        Tuple of (uv, depths, colors_filtered)
    """
    if points_world.shape[0] == 0:
        empty_colors = None if colors is None else np.empty((0, 3))
        return np.empty((0, 2)), np.empty((0,)), empty_colors
    
    # Transform points from world to camera frame
    cam_T_world = invert_transform(world_T_cam)
    points_cam = transform_points(points_world, cam_T_world)
    
    # Filter points behind camera
    z = points_cam[:, 2]
    valid_depth = z > min_depth
    
    if not np.any(valid_depth):
        empty_colors = None if colors is None else np.empty((0, 3))
        return np.empty((0, 2)), np.empty((0,)), empty_colors
    
    points_cam_valid = points_cam[valid_depth]
    colors_valid = colors[valid_depth] if colors is not None else None
    depths_valid = points_cam_valid[:, 2]
    
    # Project to 2D
    x = points_cam_valid[:, 0]
    y = points_cam_valid[:, 1]
    z = points_cam_valid[:, 2]
    
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    u = (x * fx / z) + cx
    v = (y * fy / z) + cy
    
    # Filter points outside image bounds
    in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    
    uv = np.stack([u[in_bounds], v[in_bounds]], axis=-1)
    depths_out = depths_valid[in_bounds]
    colors_out = colors_valid[in_bounds] if colors_valid is not None else None
    
    return uv, depths_out, colors_out


# =============================================================================
# Image Drawing
# =============================================================================

def draw_points_on_image(
    image: np.ndarray,
    uv: np.ndarray,
    colors: Optional[np.ndarray] = None,
    point_size: int = 2,
    radius: Optional[int] = None,
    default_color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """
    Draw 2D points on an image.
    
    Args:
        image: BGR image as numpy array (H, W, 3)
        uv: Mx2 array of 2D pixel coordinates
        colors: Optional Mx3 array of RGB colors (0-255)
        point_size: Radius of drawn points in pixels (ignored if radius provided)
        radius: Optional alias for point_size (kept for backward compatibility)
        default_color: BGR color for points if colors not provided
        
    Returns:
        Image with points drawn
    """
    if image is None:
        return None
    
    img_out = image.copy()
    
    if uv.shape[0] == 0:
        return img_out
    
    # Round to integer pixel coordinates
    uv_int = uv.astype(np.int32)
    circle_radius = radius if radius is not None else point_size
    
    for i in range(len(uv_int)):
        u, v = uv_int[i]
        
        if colors is not None:
            # Colors may be RGB, need to convert to BGR for OpenCV
            color = colors[i].astype(int)
            if len(color) == 3:
                color_bgr = (int(color[2]), int(color[1]), int(color[0]))
            else:
                color_bgr = default_color
        else:
            color_bgr = default_color
        
        cv2.circle(img_out, (u, v), circle_radius, color_bgr, -1)
    
    return img_out


def draw_points_on_image_fast(
    image: np.ndarray,
    uv: np.ndarray,
    colors: Optional[np.ndarray] = None,
    default_color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """
    Fast point drawing using direct pixel assignment (no circles).
    
    Args:
        image: BGR image as numpy array (H, W, 3)
        uv: Mx2 array of 2D pixel coordinates
        colors: Optional Mx3 array of RGB colors (0-255)
        default_color: BGR color if colors not provided
        
    Returns:
        Image with points drawn
    """
    if image is None:
        return None
    
    img_out = image.copy()
    
    if uv.shape[0] == 0:
        return img_out
    
    # Round to integer pixel coordinates
    uv_int = uv.astype(np.int32)
    
    h, w = img_out.shape[:2]
    
    # Clip to valid range
    valid = (uv_int[:, 0] >= 0) & (uv_int[:, 0] < w) & (uv_int[:, 1] >= 0) & (uv_int[:, 1] < h)
    uv_valid = uv_int[valid]
    
    if colors is not None:
        colors_valid = colors[valid]
        # Convert RGB to BGR
        colors_bgr = colors_valid[:, [2, 1, 0]].astype(np.uint8)
        
        # Direct assignment
        img_out[uv_valid[:, 1], uv_valid[:, 0]] = colors_bgr
    else:
        img_out[uv_valid[:, 1], uv_valid[:, 0]] = default_color
    
    return img_out


def create_reprojection_video(
    output_path: str,
    frames: list,
    fps: float = 30.0
):
    """
    Create a video from a list of frames using ImageIO.
    
    Args:
        output_path: Path to save the video (forced to .mp4)
        frames: List of BGR images (numpy arrays)
        fps: Frames per second
    """
    if len(frames) == 0:
        print(f"[create_reprojection_video] No frames to save")
        return
    
    # Create directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Force MP4 extension
    base, _ = os.path.splitext(output_path)
    final_path = base + ".mp4"
    
    # ImageIO Writer (H.264)
    writer = imageio.get_writer(
        final_path, 
        fps=fps, 
        codec='libx264', 
        quality=8, 
        pixelformat='yuv420p',
        macro_block_size=None
    )
    
    h, w = frames[0].shape[:2]
    
    for frame in frames:
        # Resize if needed
        if frame.shape[1] != w or frame.shape[0] != h:
            frame = cv2.resize(frame, (w, h))
            
        # BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
    
    writer.close()
    print(f"[create_reprojection_video] Saved {len(frames)} frames to {final_path}")