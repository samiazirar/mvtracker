"""Wrist Camera Depth Scanning Script.

This script uses the wrist camera's depth for scanning static areas.
It fuses point clouds from the wrist camera with external cameras,
maintaining high quality depth from the wrist camera for static regions.

Key features:
1. Uses wrist camera depth for close-range, high-quality scanning
2. Excludes gripper region (< 15cm from camera)
3. Fuses with external camera point clouds for context
4. Keeps static areas unchanged while allowing dynamic regions to update

Usage:
    python conversions/droid/fuse_static_fields.py
"""

import numpy as np
import os
import glob
import h5py
import yaml
import cv2
from scipy.spatial.transform import Rotation as R

# Only import ZED SDK when available (not required for all functions)
try:
    import pyzed.sl as sl
    ZED_AVAILABLE = True
except ImportError:
    ZED_AVAILABLE = False
    print("[WARN] pyzed.sl not available - some features disabled")


def filter_wrist_cloud(points: np.ndarray, colors: np.ndarray, 
                       min_depth: float = 0.15, max_depth: float = 0.75):
    """
    Filter wrist camera point cloud to exclude gripper region.
    
    Args:
        points: Nx3 numpy array of 3D points in camera frame
        colors: Nx3 numpy array of RGB colors
        min_depth: Minimum depth to keep (excludes gripper at ~15cm)
        max_depth: Maximum depth to keep
        
    Returns:
        Tuple of (filtered_points, filtered_colors)
    """
    if points is None or len(points) == 0:
        return np.empty((0, 3)), np.empty((0, 3))
    
    # Filter by Z coordinate (depth in camera frame)
    z_vals = points[:, 2]
    mask = (z_vals > min_depth) & (z_vals < max_depth) & np.isfinite(points).all(axis=1)
    
    return points[mask], colors[mask] if colors is not None else None


def compute_color_change(color1: np.ndarray, color2: np.ndarray, 
                         threshold: float = 30.0) -> np.ndarray:
    """
    Compute which points have significant color changes.
    
    Args:
        color1: Nx3 array of RGB colors (first frame)
        color2: Nx3 array of RGB colors (second frame)  
        threshold: Maximum color difference for "static" classification
        
    Returns:
        Boolean mask where True = static (small color change)
    """
    if color1 is None or color2 is None:
        return np.ones(len(color1), dtype=bool)
    
    # Compute L2 distance in RGB space
    diff = np.linalg.norm(color1 - color2, axis=1)
    
    # Static = small color change
    return diff < threshold


def voxel_downsample(points: np.ndarray, colors: np.ndarray, 
                     voxel_size: float = 0.01):
    """
    Downsample point cloud using voxel grid.
    
    Args:
        points: Nx3 array of 3D points
        colors: Nx3 array of RGB colors
        voxel_size: Size of voxel grid cells
        
    Returns:
        Tuple of (downsampled_points, downsampled_colors)
    """
    if len(points) == 0:
        return points, colors
    
    # Quantize to voxel grid
    voxel_coords = np.floor(points / voxel_size).astype(np.int32)
    
    # Use dictionary to keep one point per voxel
    voxel_dict = {}
    for i, coord in enumerate(voxel_coords):
        key = tuple(coord)
        if key not in voxel_dict:
            voxel_dict[key] = i
    
    indices = list(voxel_dict.values())
    return points[indices], colors[indices] if colors is not None else None


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    Transform 3D points using a 4x4 transformation matrix.
    
    Args:
        points: Nx3 array of 3D points
        transform: 4x4 homogeneous transformation matrix
        
    Returns:
        Nx3 array of transformed points
    """
    if len(points) == 0:
        return points
    
    # Convert to homogeneous coordinates
    ones = np.ones((points.shape[0], 1))
    points_homo = np.hstack([points, ones])
    
    # Apply transformation
    points_transformed = (transform @ points_homo.T).T
    
    return points_transformed[:, :3]


def fuse_static_clouds(wrist_points: np.ndarray, wrist_colors: np.ndarray,
                       external_points: np.ndarray, external_colors: np.ndarray,
                       voxel_size: float = 0.01) -> tuple:
    """
    Fuse wrist and external camera point clouds.
    
    Prioritizes wrist camera data for overlapping regions since it
    typically has higher depth quality at close range.
    
    Args:
        wrist_points: Nx3 wrist camera points in world frame
        wrist_colors: Nx3 wrist camera colors
        external_points: Mx3 external camera points in world frame
        external_colors: Mx3 external camera colors
        voxel_size: Voxel size for fusion
        
    Returns:
        Tuple of (fused_points, fused_colors)
    """
    if len(wrist_points) == 0:
        return external_points, external_colors
    
    if len(external_points) == 0:
        return wrist_points, wrist_colors
    
    # Downsample both clouds
    wrist_down, wrist_cols = voxel_downsample(wrist_points, wrist_colors, voxel_size)
    ext_down, ext_cols = voxel_downsample(external_points, external_colors, voxel_size)
    
    # Create a voxel-based spatial hash for wrist points
    wrist_voxels = set()
    wrist_voxel_coords = np.floor(wrist_down / voxel_size).astype(np.int32)
    for coord in wrist_voxel_coords:
        wrist_voxels.add(tuple(coord))
    
    # Filter external points that don't overlap with wrist
    ext_voxel_coords = np.floor(ext_down / voxel_size).astype(np.int32)
    non_overlap_mask = np.array([tuple(coord) not in wrist_voxels 
                                  for coord in ext_voxel_coords])
    
    ext_non_overlap = ext_down[non_overlap_mask]
    ext_cols_non_overlap = ext_cols[non_overlap_mask] if ext_cols is not None else None
    
    # Combine: wrist points take priority
    fused_points = np.vstack([wrist_down, ext_non_overlap])
    
    if wrist_cols is not None and ext_cols_non_overlap is not None:
        fused_colors = np.vstack([wrist_cols, ext_cols_non_overlap])
    else:
        fused_colors = None
    
    return fused_points, fused_colors


class WristDepthScanner:
    """
    Scanner that uses wrist camera depth for high-quality static area capture.
    
    This class manages the fusion of wrist camera point clouds with external
    cameras, maintaining static areas while allowing dynamic regions to update.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the scanner.
        
        Args:
            config: Configuration dictionary with parameters
        """
        self.config = config
        self.min_depth_wrist = config.get('min_depth_wrist_icp', 0.15)  # Exclude gripper
        self.max_depth_wrist = config.get('wrist_max_depth', 0.75)
        self.voxel_size = config.get('icp_voxel_size', 0.01)
        self.color_change_threshold = config.get('color_change_threshold', 30.0)
        
        # Accumulated static cloud
        self.static_points = None
        self.static_colors = None
        
        # Previous frame data for change detection
        self.prev_points = None
        self.prev_colors = None
    
    def process_frame(self, wrist_points: np.ndarray, wrist_colors: np.ndarray,
                      wrist_transform: np.ndarray) -> tuple:
        """
        Process a single frame from the wrist camera.
        
        Args:
            wrist_points: Nx3 points in wrist camera frame
            wrist_colors: Nx3 RGB colors
            wrist_transform: 4x4 camera-to-world transformation
            
        Returns:
            Tuple of (static_points_world, static_colors)
        """
        # Filter out gripper region
        filtered_pts, filtered_cols = filter_wrist_cloud(
            wrist_points, wrist_colors,
            self.min_depth_wrist, self.max_depth_wrist
        )
        
        if len(filtered_pts) == 0:
            return self.static_points, self.static_colors
        
        # Transform to world frame
        pts_world = transform_points(filtered_pts, wrist_transform)
        
        # Detect static vs dynamic regions
        if self.prev_points is not None and self.prev_colors is not None:
            # Use color change to detect static regions
            # (This is a simplified approach - more sophisticated methods possible)
            # Ensure we compare arrays of matching size
            min_len = min(len(filtered_cols), len(self.prev_colors))
            if min_len > 0:
                static_mask = compute_color_change(
                    filtered_cols[:min_len], self.prev_colors[:min_len],
                    self.color_change_threshold
                )
                # Extend mask to cover all points (assume additional points are static)
                if len(filtered_cols) > min_len:
                    full_mask = np.ones(len(filtered_cols), dtype=bool)
                    full_mask[:min_len] = static_mask
                    static_mask = full_mask
                
                static_pts = pts_world[static_mask]
                static_cols = filtered_cols[static_mask]
            else:
                static_pts = pts_world
                static_cols = filtered_cols
        else:
            static_pts = pts_world
            static_cols = filtered_cols
        
        # Update static accumulator
        if self.static_points is None:
            self.static_points = static_pts
            self.static_colors = static_cols
        else:
            # Fuse with existing static cloud
            self.static_points, self.static_colors = fuse_static_clouds(
                self.static_points, self.static_colors,
                static_pts, static_cols,
                self.voxel_size
            )
        
        # Store for next frame
        self.prev_points = filtered_pts
        self.prev_colors = filtered_cols
        
        return self.static_points, self.static_colors
    
    def get_fused_cloud(self, external_points: np.ndarray = None,
                        external_colors: np.ndarray = None) -> tuple:
        """
        Get the final fused point cloud.
        
        Args:
            external_points: Optional external camera points in world frame
            external_colors: Optional external camera colors
            
        Returns:
            Tuple of (fused_points, fused_colors)
        """
        if self.static_points is None:
            if external_points is not None:
                return external_points, external_colors
            return np.empty((0, 3)), np.empty((0, 3))
        
        if external_points is None:
            return self.static_points, self.static_colors
        
        return fuse_static_clouds(
            self.static_points, self.static_colors,
            external_points, external_colors,
            self.voxel_size
        )
    
    def reset(self):
        """Reset the scanner state."""
        self.static_points = None
        self.static_colors = None
        self.prev_points = None
        self.prev_colors = None


def main():
    """
    Main function demonstrating the wrist depth scanner.
    """
    # Load configuration
    config_path = 'conversions/droid/config.yaml'
    
    if not os.path.exists(config_path):
        print(f"[ERROR] Config file not found: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 60)
    print("Wrist Camera Depth Scanning")
    print("=" * 60)
    print(f"[INFO] Excluding gripper region (< {config.get('min_depth_wrist_icp', 0.15)*100:.0f}cm)")
    print(f"[INFO] Wrist max depth: {config.get('wrist_max_depth', 0.75)*100:.0f}cm")
    print(f"[INFO] Voxel size: {config.get('icp_voxel_size', 0.01)*100:.1f}cm")
    print("=" * 60)
    
    # Initialize scanner
    scanner = WristDepthScanner(config)
    
    print("\n[INFO] WristDepthScanner initialized successfully")
    print("[INFO] Ready to process wrist camera frames")
    print("\n[USAGE EXAMPLE]:")
    print("  scanner = WristDepthScanner(config)")
    print("  for frame in frames:")
    print("      static_pts, static_cols = scanner.process_frame(")
    print("          wrist_points, wrist_colors, wrist_transform)")
    print("  fused_pts, fused_cols = scanner.get_fused_cloud(ext_pts, ext_cols)")
    print()
    
    # If ZED SDK is available and data paths exist, run actual processing
    if not ZED_AVAILABLE:
        print("[WARN] ZED SDK not available - skipping live processing")
        return
    
    h5_path = config.get('h5_path')
    if h5_path and os.path.exists(h5_path):
        print(f"\n[INFO] Found trajectory file: {h5_path}")
        print("[INFO] To run full processing, use icp_improved_video_and_pointcloud.py")
    else:
        print(f"[WARN] Trajectory file not found: {h5_path}")
    
    print("\n[SUCCESS] Wrist depth scanning module ready")


if __name__ == "__main__":
    main()