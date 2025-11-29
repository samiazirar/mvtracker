"""Optimization utilities for wrist camera ICP alignment.

This module provides ICP-based optimization for aligning the wrist camera
point cloud to external camera point clouds. It optimizes only the Z offset
of the wrist camera relative to the gripper (end-effector).

Key assumptions:
- Gripper/end-effector pose is assumed to be correct
- Only the Z coordinate of the wrist camera (depth direction) needs refinement
- Gripper points (<15cm from camera) are excluded from ICP to avoid self-alignment
"""

import numpy as np
import open3d as o3d
from typing import Optional, Tuple, List
from scipy.optimize import minimize_scalar


# =============================================================================
# Open3D Point Cloud Utilities
# =============================================================================

def numpy_to_o3d_pointcloud(points: np.ndarray, colors: Optional[np.ndarray] = None) -> o3d.geometry.PointCloud:
    """
    Convert numpy array to Open3D point cloud.
    
    Args:
        points: Nx3 numpy array of 3D points
        colors: Optional Nx3 numpy array of RGB colors (0-255 or 0-1)
        
    Returns:
        Open3D PointCloud object
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    
    if colors is not None:
        # Normalize colors to 0-1 if they're in 0-255 range
        colors = colors.astype(np.float64)
        if colors.max() > 1.0:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd


def o3d_to_numpy(pcd: o3d.geometry.PointCloud) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Convert Open3D point cloud to numpy arrays.
    
    Args:
        pcd: Open3D PointCloud object
        
    Returns:
        Tuple of (points, colors) where colors may be None
    """
    points = np.asarray(pcd.points)
    colors = None
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
    return points, colors


def downsample_pointcloud(pcd: o3d.geometry.PointCloud, voxel_size: float = 0.01) -> o3d.geometry.PointCloud:
    """
    Downsample point cloud using voxel grid.
    
    Args:
        pcd: Input Open3D point cloud
        voxel_size: Size of voxels in meters (default: 1cm)
        
    Returns:
        Downsampled Open3D point cloud
    """
    return pcd.voxel_down_sample(voxel_size=voxel_size)


def estimate_normals(pcd: o3d.geometry.PointCloud, radius: float = 0.05, max_nn: int = 30) -> o3d.geometry.PointCloud:
    """
    Estimate normals for point cloud (required for point-to-plane ICP).
    
    Args:
        pcd: Input Open3D point cloud
        radius: Search radius for normal estimation
        max_nn: Maximum number of neighbors to consider
        
    Returns:
        Point cloud with normals estimated
    """
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    return pcd


# =============================================================================
# Point Cloud Filtering
# =============================================================================

def filter_points_by_distance_from_camera(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    min_distance: float = 0.15
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Filter out points that are too close to the camera (gripper region).
    
    This is used to exclude the gripper from ICP alignment since we assume
    the gripper pose is correct and only the wrist camera Z offset is wrong.
    
    Args:
        points: Nx3 numpy array of 3D points in camera frame
        colors: Optional Nx3 numpy array of RGB colors
        min_distance: Minimum distance from camera origin (default: 15cm to exclude gripper)
        
    Returns:
        Tuple of (filtered_points, filtered_colors)
    """
    # Calculate distance from camera origin (0, 0, 0)
    distances = np.linalg.norm(points, axis=1)
    
    # Keep points beyond min_distance
    mask = distances > min_distance
    
    filtered_points = points[mask]
    filtered_colors = colors[mask] if colors is not None else None
    
    return filtered_points, filtered_colors


def filter_wrist_cloud_for_icp(
    points_local: np.ndarray,
    colors: Optional[np.ndarray],
    min_depth: float = 0.15,
    max_depth: float = 0.75
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Filter wrist camera point cloud for ICP alignment.
    
    Excludes gripper region (close to camera) and distant points.
    
    Args:
        points_local: Nx3 numpy array of 3D points in camera frame
        colors: Optional Nx3 numpy array of RGB colors
        min_depth: Minimum depth (Z) to exclude gripper (default: 15cm)
        max_depth: Maximum depth to include
        
    Returns:
        Tuple of (filtered_points, filtered_colors)
    """
    # Use Z coordinate (depth) for filtering
    z_vals = points_local[:, 2]
    mask = (z_vals > min_depth) & (z_vals < max_depth) & np.isfinite(points_local).all(axis=1)
    
    filtered_points = points_local[mask]
    filtered_colors = colors[mask] if colors is not None else None
    
    return filtered_points, filtered_colors


# =============================================================================
# ICP Registration
# =============================================================================

def run_icp_point_to_plane(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    max_correspondence_distance: float = 0.05,
    init_transform: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, float]:
    """
    Run point-to-plane ICP registration.
    
    Args:
        source: Source point cloud (to be aligned)
        target: Target point cloud (reference)
        max_correspondence_distance: Maximum correspondence distance in meters
        init_transform: Optional initial transformation (4x4 matrix)
        
    Returns:
        Tuple of (transformation_matrix, fitness_score)
        transformation_matrix: 4x4 numpy array
        fitness_score: ICP fitness (0-1, higher is better)
    """
    if init_transform is None:
        init_transform = np.eye(4)
    
    # Ensure normals are estimated on target
    if not target.has_normals():
        estimate_normals(target)
    
    # Run ICP
    result = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance,
        init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    
    return result.transformation, result.fitness


def run_icp_point_to_point(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    max_correspondence_distance: float = 0.05,
    init_transform: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, float]:
    """
    Run point-to-point ICP registration.
    
    Args:
        source: Source point cloud (to be aligned)
        target: Target point cloud (reference)
        max_correspondence_distance: Maximum correspondence distance in meters
        init_transform: Optional initial transformation (4x4 matrix)
        
    Returns:
        Tuple of (transformation_matrix, fitness_score)
    """
    if init_transform is None:
        init_transform = np.eye(4)
    
    result = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance,
        init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    
    return result.transformation, result.fitness


# =============================================================================
# Z-Offset Optimization (Core Functionality)
# =============================================================================

def compute_alignment_error_for_z_offset(
    z_offset: float,
    wrist_points_local: np.ndarray,
    wrist_transform: np.ndarray,
    external_points_world: np.ndarray,
    voxel_size: float = 0.01,
    max_correspondence_distance: float = 0.05
) -> float:
    """
    Compute alignment error for a given Z offset.
    
    This is the objective function for Z-offset optimization.
    
    Args:
        z_offset: Z offset to apply to wrist camera (in camera frame)
        wrist_points_local: Nx3 points in wrist camera frame
        wrist_transform: 4x4 transform from wrist camera to world
        external_points_world: Mx3 points from external cameras in world frame
        voxel_size: Voxel size for downsampling
        max_correspondence_distance: Max distance for point matching
        
    Returns:
        Negative fitness score (to minimize)
    """
    if len(wrist_points_local) < 100 or len(external_points_world) < 100:
        return 1.0  # Return high error if not enough points
    
    # Apply Z offset to local points (shift along camera Z axis)
    wrist_points_shifted = wrist_points_local.copy()
    wrist_points_shifted[:, 2] += z_offset
    
    # Transform to world frame
    ones = np.ones((wrist_points_shifted.shape[0], 1))
    wrist_homo = np.hstack([wrist_points_shifted, ones])
    wrist_world = (wrist_transform @ wrist_homo.T).T[:, :3]
    
    # Create Open3D point clouds
    pcd_wrist = numpy_to_o3d_pointcloud(wrist_world)
    pcd_external = numpy_to_o3d_pointcloud(external_points_world)
    
    # Downsample
    pcd_wrist = downsample_pointcloud(pcd_wrist, voxel_size)
    pcd_external = downsample_pointcloud(pcd_external, voxel_size)
    
    if len(pcd_wrist.points) < 50 or len(pcd_external.points) < 50:
        return 1.0
    
    # Estimate normals for point-to-plane
    estimate_normals(pcd_external)
    
    # Run ICP (just to get fitness, not to get transform)
    result = o3d.pipelines.registration.registration_icp(
        pcd_wrist, pcd_external,
        max_correspondence_distance,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    
    # Return negative fitness (we want to maximize fitness)
    return -result.fitness


def optimize_wrist_z_offset(
    wrist_points_local: np.ndarray,
    wrist_transform: np.ndarray,
    external_points_world: np.ndarray,
    z_range: Tuple[float, float] = (-0.05, 0.05),
    voxel_size: float = 0.01,
    max_correspondence_distance: float = 0.05
) -> Tuple[float, float]:
    """
    Optimize Z offset for wrist camera alignment.
    
    Args:
        wrist_points_local: Nx3 points in wrist camera frame (gripper excluded)
        wrist_transform: 4x4 transform from wrist camera to world
        external_points_world: Mx3 points from external cameras in world frame
        z_range: Tuple of (min_z, max_z) to search
        voxel_size: Voxel size for downsampling
        max_correspondence_distance: Max distance for point matching
        
    Returns:
        Tuple of (optimal_z_offset, best_fitness)
    """
    # Use scipy's bounded optimization
    result = minimize_scalar(
        compute_alignment_error_for_z_offset,
        bounds=z_range,
        method='bounded',
        args=(wrist_points_local, wrist_transform, external_points_world, 
              voxel_size, max_correspondence_distance)
    )
    
    optimal_z = result.x
    best_fitness = -result.fun  # Convert back from negative
    
    return optimal_z, best_fitness


def optimize_wrist_z_offset_multi_frame(
    frames_data: List[dict],
    z_range: Tuple[float, float] = (-0.05, 0.05),
    voxel_size: float = 0.01,
    max_correspondence_distance: float = 0.05
) -> Tuple[float, float]:
    """
    Optimize Z offset using multiple frames for robustness.
    
    Args:
        frames_data: List of dicts with keys:
            - 'wrist_points_local': Nx3 points in wrist camera frame
            - 'wrist_transform': 4x4 transform
            - 'external_points_world': Mx3 points from external cameras
        z_range: Tuple of (min_z, max_z) to search
        voxel_size: Voxel size for downsampling
        max_correspondence_distance: Max distance for point matching
        
    Returns:
        Tuple of (optimal_z_offset, average_fitness)
    """
    def multi_frame_error(z_offset):
        total_error = 0.0
        valid_frames = 0
        
        for frame in frames_data:
            error = compute_alignment_error_for_z_offset(
                z_offset,
                frame['wrist_points_local'],
                frame['wrist_transform'],
                frame['external_points_world'],
                voxel_size,
                max_correspondence_distance
            )
            if error < 0.99:  # Valid frame
                total_error += error
                valid_frames += 1
        
        if valid_frames == 0:
            return 1.0
        
        return total_error / valid_frames
    
    result = minimize_scalar(
        multi_frame_error,
        bounds=z_range,
        method='bounded'
    )
    
    optimal_z = result.x
    best_fitness = -result.fun
    
    return optimal_z, best_fitness


def apply_z_offset_to_wrist_transforms(
    transforms: List[np.ndarray],
    z_offset: float
) -> List[np.ndarray]:
    """
    Apply Z offset to all wrist camera transforms.
    
    The Z offset is applied in the camera frame, which means we need to
    translate along the camera's Z axis (viewing direction).
    
    Args:
        transforms: List of 4x4 transformation matrices (camera to world)
        z_offset: Z offset to apply (in camera frame, i.e., depth direction)
        
    Returns:
        List of modified 4x4 transformation matrices
    """
    modified_transforms = []
    
    for T in transforms:
        T_new = T.copy()
        
        # The Z axis of the camera in world frame is the third column of rotation
        # T[:3, :3] is the rotation matrix, T[:3, 2] is the Z axis direction
        z_axis_world = T[:3, 2]
        
        # Apply offset along this direction
        T_new[:3, 3] += z_offset * z_axis_world
        
        modified_transforms.append(T_new)
    
    return modified_transforms


# =============================================================================
# High-Level Optimization Functions (used by main scripts)
# =============================================================================

def optimize_wrist_camera_icp(active_cams, config):
    """
    Legacy function - calls optimize_wrist_camera_icp_z_only.
    """
    return optimize_wrist_camera_icp_z_only(active_cams, config)


def optimize_wrist_camera_icp_z_only(active_cams: dict, config: dict) -> float:
    """
    Optimize wrist camera Z offset using ICP against external cameras.
    
    This function assumes the gripper pose is correct and only optimizes
    the Z offset (depth) of the wrist camera.
    
    Args:
        active_cams: Dictionary of camera data with keys like:
            - 'type': 'wrist' or 'external'
            - 'zed': ZED camera object
            - 'runtime': RuntimeParameters
            - 'transforms': List of 4x4 matrices (for wrist)
            - 'world_T_cam': 4x4 matrix (for external)
        config: Configuration dictionary with ICP parameters
        
    Returns:
        Optimal Z offset value
    """
    import pyzed.sl as sl
    from .camera_utils import get_filtered_cloud
    
    # Find wrist and external cameras
    wrist_serial = None
    wrist_cam = None
    external_cams = {}
    
    for serial, cam in active_cams.items():
        if cam['type'] == 'wrist':
            wrist_serial = serial
            wrist_cam = cam
        else:
            external_cams[serial] = cam
    
    if wrist_cam is None or len(external_cams) == 0:
        print("[ICP] Warning: Need both wrist and external cameras for ICP optimization")
        return 0.0
    
    # Collect frames for multi-frame optimization
    num_icp_frames = config.get('icp_num_frames', 10)
    min_depth_icp = config.get('min_depth_wrist_icp', 0.15)  # Exclude gripper
    max_depth_wrist = config.get('wrist_max_depth', 0.75)
    ext_max_depth = config.get('ext_max_depth', 1.5)
    min_depth_ext = config.get('min_depth', 0.1)
    
    voxel_size = config.get('icp_voxel_size', 0.01)
    max_corr_dist = config.get('icp_max_correspondence_distance', 0.05)
    
    # Reset cameras
    wrist_cam['zed'].set_svo_position(0)
    for cam in external_cams.values():
        cam['zed'].set_svo_position(0)
    
    total_frames = len(wrist_cam['transforms'])
    frame_indices = np.linspace(0, total_frames - 1, num_icp_frames, dtype=int)
    
    frames_data = []
    
    print(f"[ICP] Collecting {num_icp_frames} frames for Z-offset optimization...")
    
    for frame_idx in frame_indices:
        # Get wrist data
        wrist_cam['zed'].set_svo_position(frame_idx)
        if wrist_cam['zed'].grab(wrist_cam['runtime']) != sl.ERROR_CODE.SUCCESS:
            continue
        
        # Get wrist points (excluding gripper)
        wrist_xyz, wrist_rgb = get_filtered_cloud(
            wrist_cam['zed'], wrist_cam['runtime'],
            max_depth=max_depth_wrist,
            min_depth=min_depth_icp  # 15cm to exclude gripper
        )
        
        if wrist_xyz is None or len(wrist_xyz) < 100:
            continue
        
        wrist_transform = wrist_cam['transforms'][frame_idx]
        
        # Collect external camera points
        external_world_points = []
        
        for ext_serial, ext_cam in external_cams.items():
            ext_cam['zed'].set_svo_position(frame_idx)
            if ext_cam['zed'].grab(ext_cam['runtime']) != sl.ERROR_CODE.SUCCESS:
                continue
            
            ext_xyz, ext_rgb = get_filtered_cloud(
                ext_cam['zed'], ext_cam['runtime'],
                max_depth=ext_max_depth,
                min_depth=min_depth_ext
            )
            
            if ext_xyz is None or len(ext_xyz) < 100:
                continue
            
            # Transform to world frame
            T_ext = ext_cam['world_T_cam']
            ones = np.ones((ext_xyz.shape[0], 1))
            ext_homo = np.hstack([ext_xyz, ones])
            ext_world = (T_ext @ ext_homo.T).T[:, :3]
            
            external_world_points.append(ext_world)
        
        if len(external_world_points) == 0:
            continue
        
        external_points_world = np.vstack(external_world_points)
        
        frames_data.append({
            'wrist_points_local': wrist_xyz,
            'wrist_transform': wrist_transform,
            'external_points_world': external_points_world
        })
    
    print(f"[ICP] Collected {len(frames_data)} valid frames")
    
    if len(frames_data) == 0:
        print("[ICP] Warning: No valid frames for ICP optimization")
        return 0.0
    
    # Optimize Z offset
    print("[ICP] Optimizing Z offset...")
    z_offset, fitness = optimize_wrist_z_offset_multi_frame(
        frames_data,
        z_range=(-0.05, 0.05),  # Search +/- 5cm
        voxel_size=voxel_size,
        max_correspondence_distance=max_corr_dist
    )
    
    print(f"[ICP] Optimal Z offset: {z_offset:.4f}m, Fitness: {fitness:.4f}")
    
    # Apply offset to wrist transforms
    wrist_cam['transforms'] = apply_z_offset_to_wrist_transforms(
        wrist_cam['transforms'], z_offset
    )
    
    return z_offset


def optimize_wrist_z_offset_icp(active_cams: dict, config: dict) -> float:
    """
    Alias for optimize_wrist_camera_icp_z_only.
    """
    return optimize_wrist_camera_icp_z_only(active_cams, config)


def optimize_external_cameras_multi_frame(active_cams: dict, config: dict):
    """
    Placeholder for external camera optimization.
    
    Currently not implemented as we assume external cameras are correctly calibrated.
    """
    print("[ICP] External camera optimization not needed (assuming correct calibration)")
    pass


def optimize_wrist_multi_frame(active_cams: dict, cartesian_positions: np.ndarray, config: dict):
    """
    Alias for optimize_wrist_camera_icp_z_only.
    """
    return optimize_wrist_camera_icp_z_only(active_cams, config)
