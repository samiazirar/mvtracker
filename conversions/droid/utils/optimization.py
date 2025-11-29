"""Optimization utilities for wrist camera ICP alignment.

This module provides ICP-based optimization for aligning the wrist camera
point cloud to external camera point clouds. It optimizes only the Z offset
of the wrist camera relative to the gripper (end-effector).

Key assumptions:
- Gripper/end-effector pose is assumed to be correct
- Only the Z coordinate of the wrist camera (depth direction) needs refinement
- Gripper points (<15cm from camera) are excluded from ICP to avoid self-alignment

Redesigned ICP Implementation:
- Uses robust multi-frame accumulation with outlier rejection
- Employs grid search followed by local refinement for better convergence
- Includes better point cloud preprocessing (statistical outlier removal)
- Provides detailed diagnostic information during optimization
"""

import numpy as np
import open3d as o3d
from typing import Optional, Tuple, List, Dict
from scipy.optimize import minimize_scalar, minimize


# =============================================================================
# Constants
# =============================================================================

# ICP optimization weights for combined fitness/RMSE scoring
ICP_FITNESS_WEIGHT = 0.7
ICP_RMSE_WEIGHT = 0.3

# Grid search parameters
GRID_SEARCH_STEPS = 21
GRID_SEARCH_REFINEMENT_DIVISOR = 20  # Divides z_range for refinement radius
LOCAL_REFINEMENT_DIVISOR = 10  # Divides z_range for local refinement

# Minimum fitness threshold for valid ICP alignment
MIN_ICP_FITNESS_THRESHOLD = 0.3

# Minimum points required for valid ICP
MIN_POINTS_FOR_ICP = 50


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
    if points is None or len(points) == 0:
        return o3d.geometry.PointCloud()
    
    # Filter out invalid points
    valid_mask = np.isfinite(points).all(axis=1)
    points = points[valid_mask]
    
    if len(points) == 0:
        return o3d.geometry.PointCloud()
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    
    if colors is not None:
        colors = colors[valid_mask]
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
    if len(pcd.points) == 0:
        return pcd
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
    if len(pcd.points) < 3:
        return pcd
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    return pcd


def remove_statistical_outliers(
    pcd: o3d.geometry.PointCloud, 
    nb_neighbors: int = 20, 
    std_ratio: float = 2.0
) -> o3d.geometry.PointCloud:
    """
    Remove statistical outliers from point cloud.
    
    Args:
        pcd: Input Open3D point cloud
        nb_neighbors: Number of neighbors to consider for each point
        std_ratio: Standard deviation ratio threshold
        
    Returns:
        Point cloud with outliers removed
    """
    if len(pcd.points) < nb_neighbors:
        return pcd
    pcd_clean, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, 
        std_ratio=std_ratio
    )
    return pcd_clean


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
    if points is None or len(points) == 0:
        return np.empty((0, 3)), None
    
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
    if points_local is None or len(points_local) == 0:
        return np.empty((0, 3)), None
    
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
    init_transform: Optional[np.ndarray] = None,
    max_iterations: int = 50
) -> Tuple[np.ndarray, float, float]:
    """
    Run point-to-plane ICP registration.
    
    Args:
        source: Source point cloud (to be aligned)
        target: Target point cloud (reference)
        max_correspondence_distance: Maximum correspondence distance in meters
        init_transform: Optional initial transformation (4x4 matrix)
        max_iterations: Maximum ICP iterations
        
    Returns:
        Tuple of (transformation_matrix, fitness_score, inlier_rmse)
        transformation_matrix: 4x4 numpy array
        fitness_score: ICP fitness (0-1, higher is better)
        inlier_rmse: Root mean square error of inliers
    """
    if len(source.points) == 0 or len(target.points) == 0:
        return np.eye(4), 0.0, float('inf')
    
    if init_transform is None:
        init_transform = np.eye(4)
    
    # Ensure normals are estimated on target
    if not target.has_normals():
        estimate_normals(target)
    
    # Run ICP with convergence criteria
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=max_iterations,
        relative_fitness=1e-6,
        relative_rmse=1e-6
    )
    
    result = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance,
        init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria
    )
    
    return result.transformation, result.fitness, result.inlier_rmse


def run_icp_point_to_point(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    max_correspondence_distance: float = 0.05,
    init_transform: Optional[np.ndarray] = None,
    max_iterations: int = 50
) -> Tuple[np.ndarray, float, float]:
    """
    Run point-to-point ICP registration.
    
    Args:
        source: Source point cloud (to be aligned)
        target: Target point cloud (reference)
        max_correspondence_distance: Maximum correspondence distance in meters
        init_transform: Optional initial transformation (4x4 matrix)
        max_iterations: Maximum ICP iterations
        
    Returns:
        Tuple of (transformation_matrix, fitness_score, inlier_rmse)
    """
    if len(source.points) == 0 or len(target.points) == 0:
        return np.eye(4), 0.0, float('inf')
    
    if init_transform is None:
        init_transform = np.eye(4)
    
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=max_iterations,
        relative_fitness=1e-6,
        relative_rmse=1e-6
    )
    
    result = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance,
        init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria
    )
    
    return result.transformation, result.fitness, result.inlier_rmse


# =============================================================================
# Z-Offset Optimization (Core Functionality - REDESIGNED)
# =============================================================================

def preprocess_pointcloud_for_icp(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float = 0.01,
    remove_outliers: bool = True
) -> o3d.geometry.PointCloud:
    """
    Preprocess point cloud for ICP registration.
    
    Args:
        pcd: Input point cloud
        voxel_size: Voxel size for downsampling
        remove_outliers: Whether to remove statistical outliers
        
    Returns:
        Preprocessed point cloud
    """
    if len(pcd.points) == 0:
        return pcd
    
    # Downsample
    pcd = downsample_pointcloud(pcd, voxel_size)
    
    # Remove outliers
    if remove_outliers and len(pcd.points) > 30:
        pcd = remove_statistical_outliers(pcd, nb_neighbors=20, std_ratio=2.0)
    
    # Estimate normals
    if len(pcd.points) > 10:
        estimate_normals(pcd, radius=voxel_size * 5, max_nn=30)
    
    return pcd


def transform_points_to_world(
    points_local: np.ndarray, 
    transform: np.ndarray,
    z_offset: float = 0.0
) -> np.ndarray:
    """
    Transform local camera points to world frame with optional Z offset.
    
    Args:
        points_local: Nx3 points in camera frame
        transform: 4x4 transformation matrix (camera to world)
        z_offset: Z offset to apply in camera frame before transformation
        
    Returns:
        Nx3 points in world frame
    """
    if len(points_local) == 0:
        return np.empty((0, 3))
    
    # Apply Z offset in camera frame
    points_shifted = points_local.copy()
    points_shifted[:, 2] += z_offset
    
    # Transform to world
    ones = np.ones((points_shifted.shape[0], 1))
    points_homo = np.hstack([points_shifted, ones])
    points_world = (transform @ points_homo.T).T[:, :3]
    
    return points_world


def compute_alignment_metric(
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    max_correspondence_distance: float = 0.05
) -> Dict[str, float]:
    """
    Compute alignment metrics between two point clouds.
    
    Args:
        source_pcd: Source point cloud
        target_pcd: Target point cloud
        max_correspondence_distance: Max distance for correspondences
        
    Returns:
        Dictionary with fitness, rmse, and correspondence_count
    """
    if len(source_pcd.points) == 0 or len(target_pcd.points) == 0:
        return {'fitness': 0.0, 'rmse': float('inf'), 'correspondence_count': 0}
    
    # Evaluate alignment without ICP transformation
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source_pcd, target_pcd,
        max_correspondence_distance,
        np.eye(4)
    )
    
    return {
        'fitness': evaluation.fitness,
        'rmse': evaluation.inlier_rmse,
        'correspondence_count': len(evaluation.correspondence_set)
    }


def compute_alignment_error_for_z_offset(
    z_offset: float,
    wrist_points_local: np.ndarray,
    wrist_transform: np.ndarray,
    external_points_world: np.ndarray,
    voxel_size: float = 0.01,
    max_correspondence_distance: float = 0.05,
    use_icp_refinement: bool = False
) -> float:
    """
    Compute alignment error for a given Z offset.
    
    This is the objective function for Z-offset optimization.
    Uses direct point cloud distance evaluation for robustness.
    
    Args:
        z_offset: Z offset to apply to wrist camera (in camera frame)
        wrist_points_local: Nx3 points in wrist camera frame
        wrist_transform: 4x4 transform from wrist camera to world
        external_points_world: Mx3 points from external cameras in world frame
        voxel_size: Voxel size for downsampling
        max_correspondence_distance: Max distance for point matching
        use_icp_refinement: Whether to use ICP for final fitness calculation
        
    Returns:
        Negative fitness score (to minimize)
    """
    if len(wrist_points_local) < MIN_POINTS_FOR_ICP or len(external_points_world) < MIN_POINTS_FOR_ICP:
        return 1.0  # Return high error if not enough points
    
    # Transform wrist points to world with Z offset
    wrist_world = transform_points_to_world(wrist_points_local, wrist_transform, z_offset)
    
    if len(wrist_world) < MIN_POINTS_FOR_ICP:
        return 1.0
    
    # Create and preprocess point clouds
    pcd_wrist = numpy_to_o3d_pointcloud(wrist_world)
    pcd_external = numpy_to_o3d_pointcloud(external_points_world)
    
    pcd_wrist = preprocess_pointcloud_for_icp(pcd_wrist, voxel_size, remove_outliers=True)
    pcd_external = preprocess_pointcloud_for_icp(pcd_external, voxel_size, remove_outliers=True)
    
    if len(pcd_wrist.points) < 30 or len(pcd_external.points) < 30:
        return 1.0
    
    if use_icp_refinement:
        # Run ICP to get refined fitness
        _, fitness, rmse = run_icp_point_to_plane(
            pcd_wrist, pcd_external,
            max_correspondence_distance
        )
        # Combine fitness and RMSE for better optimization
        if rmse == float('inf') or rmse > max_correspondence_distance:
            return 1.0
        return -(fitness * ICP_FITNESS_WEIGHT + (1.0 - min(rmse / max_correspondence_distance, 1.0)) * ICP_RMSE_WEIGHT)
    else:
        # Direct evaluation without ICP (faster, more stable for grid search)
        metrics = compute_alignment_metric(
            pcd_wrist, pcd_external, max_correspondence_distance
        )
        return -metrics['fitness']


def grid_search_z_offset(
    frames_data: List[dict],
    z_range: Tuple[float, float] = (-0.05, 0.05),
    num_steps: int = 21,
    voxel_size: float = 0.01,
    max_correspondence_distance: float = 0.05
) -> Tuple[float, float]:
    """
    Perform grid search to find approximate optimal Z offset.
    
    Args:
        frames_data: List of frame data dictionaries
        z_range: Search range (min, max)
        num_steps: Number of steps in grid search
        voxel_size: Voxel size for downsampling
        max_correspondence_distance: Max correspondence distance
        
    Returns:
        Tuple of (best_z_offset, best_fitness)
    """
    z_values = np.linspace(z_range[0], z_range[1], num_steps)
    
    best_z = 0.0
    best_fitness = -float('inf')
    
    print(f"[ICP] Grid search: testing {num_steps} Z offsets in range [{z_range[0]*100:.1f}cm, {z_range[1]*100:.1f}cm]")
    
    for z_offset in z_values:
        total_fitness = 0.0
        valid_frames = 0
        
        for frame in frames_data:
            error = compute_alignment_error_for_z_offset(
                z_offset,
                frame['wrist_points_local'],
                frame['wrist_transform'],
                frame['external_points_world'],
                voxel_size,
                max_correspondence_distance,
                use_icp_refinement=False  # Faster for grid search
            )
            
            if error < 0.99:  # Valid frame
                total_fitness += -error  # Convert back to positive fitness
                valid_frames += 1
        
        if valid_frames > 0:
            avg_fitness = total_fitness / valid_frames
            if avg_fitness > best_fitness:
                best_fitness = avg_fitness
                best_z = z_offset
    
    print(f"[ICP] Grid search result: Z offset = {best_z*100:.2f}cm, fitness = {best_fitness:.4f}")
    
    return best_z, best_fitness


def optimize_wrist_z_offset(
    wrist_points_local: np.ndarray,
    wrist_transform: np.ndarray,
    external_points_world: np.ndarray,
    z_range: Tuple[float, float] = (-0.05, 0.05),
    voxel_size: float = 0.01,
    max_correspondence_distance: float = 0.05
) -> Tuple[float, float]:
    """
    Optimize Z offset for wrist camera alignment (single frame).
    
    Uses grid search followed by local refinement.
    
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
    # Grid search
    z_values = np.linspace(z_range[0], z_range[1], GRID_SEARCH_STEPS)
    best_z = 0.0
    best_fitness = -float('inf')
    
    for z_offset in z_values:
        error = compute_alignment_error_for_z_offset(
            z_offset,
            wrist_points_local,
            wrist_transform,
            external_points_world,
            voxel_size,
            max_correspondence_distance,
            use_icp_refinement=False
        )
        fitness = -error
        if fitness > best_fitness:
            best_fitness = fitness
            best_z = z_offset
    
    # Local refinement around best Z
    search_radius = (z_range[1] - z_range[0]) / GRID_SEARCH_REFINEMENT_DIVISOR
    refined_range = (max(z_range[0], best_z - search_radius), 
                     min(z_range[1], best_z + search_radius))
    
    result = minimize_scalar(
        lambda z: compute_alignment_error_for_z_offset(
            z, wrist_points_local, wrist_transform, external_points_world,
            voxel_size, max_correspondence_distance, use_icp_refinement=True
        ),
        bounds=refined_range,
        method='bounded'
    )
    
    return result.x, -result.fun


def optimize_wrist_z_offset_multi_frame(
    frames_data: List[dict],
    z_range: Tuple[float, float] = (-0.05, 0.05),
    voxel_size: float = 0.01,
    max_correspondence_distance: float = 0.05
) -> Tuple[float, float]:
    """
    Optimize Z offset using multiple frames for robustness.
    
    Uses a two-stage approach:
    1. Grid search to find approximate optimal Z
    2. Local refinement with ICP for final optimization
    
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
    if len(frames_data) == 0:
        print("[ICP] Warning: No frame data provided")
        return 0.0, 0.0
    
    print(f"[ICP] Optimizing Z offset using {len(frames_data)} frames...")
    
    # Stage 1: Grid search
    grid_z, grid_fitness = grid_search_z_offset(
        frames_data, z_range, num_steps=GRID_SEARCH_STEPS, 
        voxel_size=voxel_size, 
        max_correspondence_distance=max_correspondence_distance
    )
    
    # Stage 2: Local refinement with ICP
    search_radius = (z_range[1] - z_range[0]) / LOCAL_REFINEMENT_DIVISOR
    refined_range = (max(z_range[0], grid_z - search_radius), 
                     min(z_range[1], grid_z + search_radius))
    
    def multi_frame_error_with_icp(z_offset):
        total_error = 0.0
        valid_frames = 0
        
        for frame in frames_data:
            error = compute_alignment_error_for_z_offset(
                z_offset,
                frame['wrist_points_local'],
                frame['wrist_transform'],
                frame['external_points_world'],
                voxel_size,
                max_correspondence_distance,
                use_icp_refinement=True  # Use ICP for final refinement
            )
            if error < 0.99:
                total_error += error
                valid_frames += 1
        
        if valid_frames == 0:
            return 1.0
        
        return total_error / valid_frames
    
    print(f"[ICP] Local refinement in range [{refined_range[0]*100:.2f}cm, {refined_range[1]*100:.2f}cm]")
    
    result = minimize_scalar(
        multi_frame_error_with_icp,
        bounds=refined_range,
        method='bounded'
    )
    
    optimal_z = result.x
    best_fitness = -result.fun
    
    print(f"[ICP] Final result: Z offset = {optimal_z*100:.3f}cm, fitness = {best_fitness:.4f}")
    
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

def collect_frames_for_icp(
    active_cams: dict,
    config: dict,
    num_frames: Optional[int] = None
) -> List[dict]:
    """
    Collect frame data from cameras for ICP optimization.
    
    Args:
        active_cams: Dictionary of active cameras
        config: Configuration dictionary
        num_frames: Number of frames to collect (default from config)
        
    Returns:
        List of frame data dictionaries
    """
    import pyzed.sl as sl
    from .camera_utils import get_filtered_cloud
    from .transforms import transform_points
    
    # Find wrist and external cameras
    wrist_cam = None
    external_cams = {}
    
    for serial, cam in active_cams.items():
        if cam['type'] == 'wrist':
            wrist_cam = cam
        else:
            external_cams[serial] = cam
    
    if wrist_cam is None or len(external_cams) == 0:
        print("[ICP] Warning: Need both wrist and external cameras")
        return []
    
    # Parameters
    if num_frames is None:
        num_frames = config.get('icp_num_frames', 10)
    min_depth_icp = config.get('min_depth_wrist_icp', 0.15)  # Exclude gripper
    max_depth_wrist = config.get('wrist_max_depth', 0.75)
    ext_max_depth = config.get('ext_max_depth', 1.5)
    min_depth_ext = config.get('min_depth', 0.1)
    
    # Reset cameras
    wrist_cam['zed'].set_svo_position(0)
    for cam in external_cams.values():
        cam['zed'].set_svo_position(0)
    
    total_frames = len(wrist_cam['transforms'])
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames_data = []
    
    print(f"[ICP] Collecting {num_frames} frames for optimization...")
    print(f"[ICP] Excluding gripper points (< {min_depth_icp*100:.0f}cm from camera)")
    
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
            ext_world = transform_points(ext_xyz, T_ext)
            external_world_points.append(ext_world)
        
        if len(external_world_points) == 0:
            continue
        
        external_points_world = np.vstack(external_world_points)
        
        frames_data.append({
            'frame_idx': frame_idx,
            'wrist_points_local': wrist_xyz,
            'wrist_transform': wrist_transform,
            'external_points_world': external_points_world
        })
    
    print(f"[ICP] Collected {len(frames_data)} valid frames")
    return frames_data


def optimize_wrist_camera_icp(active_cams: dict, config: dict) -> float:
    """
    Main ICP optimization function for wrist camera Z offset.
    
    This function:
    1. Collects point cloud data from multiple frames
    2. Excludes gripper points (< 15cm from camera)
    3. Uses grid search + local refinement to find optimal Z offset
    4. Applies the offset to all wrist camera transforms
    
    Args:
        active_cams: Dictionary of camera data
        config: Configuration dictionary
        
    Returns:
        Optimal Z offset value (meters)
    """
    # Find wrist camera
    wrist_cam = None
    for serial, cam in active_cams.items():
        if cam['type'] == 'wrist':
            wrist_cam = cam
            break
    
    if wrist_cam is None:
        print("[ICP] Error: No wrist camera found")
        return 0.0
    
    # Collect frames
    frames_data = collect_frames_for_icp(active_cams, config)
    
    if len(frames_data) == 0:
        print("[ICP] Error: No valid frames for optimization")
        return 0.0
    
    # Get optimization parameters
    voxel_size = config.get('icp_voxel_size', 0.01)
    max_corr_dist = config.get('icp_max_correspondence_distance', 0.05)
    z_range = config.get('icp_z_range', (-0.05, 0.05))  # +/- 5cm default
    
    # Run optimization
    print("\n" + "=" * 50)
    print("[ICP] Running Z-offset optimization")
    print(f"[ICP] Search range: [{z_range[0]*100:.1f}cm, {z_range[1]*100:.1f}cm]")
    print("=" * 50)
    
    z_offset, fitness = optimize_wrist_z_offset_multi_frame(
        frames_data,
        z_range=z_range,
        voxel_size=voxel_size,
        max_correspondence_distance=max_corr_dist
    )
    
    # Apply offset to wrist transforms
    wrist_cam['transforms'] = apply_z_offset_to_wrist_transforms(
        wrist_cam['transforms'], z_offset
    )
    
    print("\n" + "=" * 50)
    print(f"[ICP] RESULT: Z offset = {z_offset*100:.3f}cm, fitness = {fitness:.4f}")
    print("=" * 50 + "\n")
    
    return z_offset


def optimize_wrist_camera_icp_z_only(active_cams: dict, config: dict) -> float:
    """
    Alias for optimize_wrist_camera_icp.
    """
    return optimize_wrist_camera_icp(active_cams, config)


def optimize_wrist_z_offset_icp(active_cams: dict, config: dict) -> float:
    """
    Alias for optimize_wrist_camera_icp.
    """
    return optimize_wrist_camera_icp(active_cams, config)


def optimize_external_cameras_multi_frame(active_cams: dict, config: dict):
    """
    Placeholder for external camera optimization.
    
    Currently not implemented as we assume external cameras are correctly calibrated.
    """
    print("[ICP] External camera optimization skipped (assuming correct calibration)")


def optimize_wrist_multi_frame(active_cams: dict, cartesian_positions: np.ndarray, config: dict):
    """
    Alias for optimize_wrist_camera_icp.
    """
    return optimize_wrist_camera_icp(active_cams, config)
