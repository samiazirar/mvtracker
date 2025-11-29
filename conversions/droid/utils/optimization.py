"""High-Quality ICP Optimization for Wrist Camera Alignment.

This module provides robust ICP-based optimization for aligning the wrist camera
point cloud to external camera point clouds. It uses a multi-stage approach:

1. Point cloud preprocessing (denoising, downsampling, normal estimation)
2. Global registration using RANSAC with FPFH features for initial alignment
3. Multi-scale ICP refinement with point-to-plane for final alignment
4. Full 6-DOF transformation optimization

Key features:
- Statistical outlier removal for noise reduction
- FPFH feature extraction for global registration
- Multi-scale ICP for robustness
- Colored ICP option when color information is available
- Comprehensive fitness and RMSE metrics

Key assumptions:
- Gripper/end-effector pose is assumed to be correct as initial estimate
- Wrist camera transform is refined relative to gripper
- Gripper points (<15cm from camera) are excluded from ICP to avoid self-alignment
"""

import numpy as np
import open3d as o3d
from typing import Optional, Tuple, List, Dict
from scipy.optimize import minimize_scalar, minimize
from scipy.spatial.transform import Rotation as R
import copy


# =============================================================================
# Open3D Point Cloud Utilities
# =============================================================================

def numpy_to_o3d_pointcloud(
    points: np.ndarray, 
    colors: Optional[np.ndarray] = None
) -> o3d.geometry.PointCloud:
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
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    
    if colors is not None and len(colors) == len(points):
        colors = colors.astype(np.float64)
        if colors.max() > 1.0:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd


def o3d_to_numpy(
    pcd: o3d.geometry.PointCloud
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Convert Open3D point cloud to numpy arrays.
    
    Args:
        pcd: Open3D PointCloud object
        
    Returns:
        Tuple of (points, colors) where colors may be None
    """
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    return points, colors


def downsample_pointcloud(
    pcd: o3d.geometry.PointCloud, 
    voxel_size: float = 0.01
) -> o3d.geometry.PointCloud:
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


def estimate_normals(
    pcd: o3d.geometry.PointCloud, 
    radius: float = 0.05, 
    max_nn: int = 30
) -> o3d.geometry.PointCloud:
    """
    Estimate normals for point cloud (required for point-to-plane ICP).
    
    Args:
        pcd: Input Open3D point cloud
        radius: Search radius for normal estimation
        max_nn: Maximum number of neighbors to consider
        
    Returns:
        Point cloud with normals estimated
    """
    if len(pcd.points) == 0:
        return pcd
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    return pcd


# =============================================================================
# Point Cloud Preprocessing (NEW - High Quality)
# =============================================================================

def preprocess_pointcloud(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float = 0.01,
    remove_outliers: bool = True,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    estimate_norms: bool = True,
    normal_radius: float = 0.05
) -> o3d.geometry.PointCloud:
    """
    Comprehensive point cloud preprocessing for ICP.
    
    Applies:
    1. Voxel downsampling
    2. Statistical outlier removal
    3. Normal estimation
    
    Args:
        pcd: Input point cloud
        voxel_size: Voxel size for downsampling
        remove_outliers: Whether to remove statistical outliers
        nb_neighbors: Number of neighbors for outlier detection
        std_ratio: Standard deviation ratio for outlier removal
        estimate_norms: Whether to estimate normals
        normal_radius: Radius for normal estimation
        
    Returns:
        Preprocessed point cloud
    """
    if len(pcd.points) == 0:
        return pcd
    
    # 1. Voxel downsampling
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    if len(pcd_down.points) < 10:
        return pcd_down
    
    # 2. Statistical outlier removal
    if remove_outliers and len(pcd_down.points) > nb_neighbors:
        pcd_clean, _ = pcd_down.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )
        if len(pcd_clean.points) > 10:
            pcd_down = pcd_clean
    
    # 3. Normal estimation
    if estimate_norms:
        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=normal_radius, 
                max_nn=30
            )
        )
    
    return pcd_down


def compute_fpfh_features(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float = 0.01
) -> o3d.pipelines.registration.Feature:
    """
    Compute FPFH (Fast Point Feature Histograms) features for global registration.
    
    Args:
        pcd: Point cloud with normals
        voxel_size: Voxel size used for downsampling (affects radius)
        
    Returns:
        FPFH feature descriptor
    """
    if len(pcd.points) == 0:
        return o3d.pipelines.registration.Feature()
    
    # Ensure normals exist
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size * 2, max_nn=30
            )
        )
    
    radius_feature = voxel_size * 5
    
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_feature,
            max_nn=100
        )
    )
    
    return fpfh


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
    
    Args:
        points: Nx3 numpy array of 3D points in camera frame
        colors: Optional Nx3 numpy array of RGB colors
        min_distance: Minimum distance from camera origin (default: 15cm)
        
    Returns:
        Tuple of (filtered_points, filtered_colors)
    """
    if points is None or len(points) == 0:
        return np.empty((0, 3)), None
    
    distances = np.linalg.norm(points, axis=1)
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
    
    z_vals = points_local[:, 2]
    mask = (z_vals > min_depth) & (z_vals < max_depth) & np.isfinite(points_local).all(axis=1)
    
    filtered_points = points_local[mask]
    filtered_colors = colors[mask] if colors is not None else None
    
    return filtered_points, filtered_colors


# =============================================================================
# Global Registration (RANSAC with FPFH)
# =============================================================================

def execute_global_registration(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    source_fpfh: o3d.pipelines.registration.Feature,
    target_fpfh: o3d.pipelines.registration.Feature,
    voxel_size: float = 0.01,
    distance_threshold: float = None
) -> Tuple[np.ndarray, float]:
    """
    Perform global registration using RANSAC with FPFH features.
    
    This provides a good initial alignment for ICP refinement.
    
    Args:
        source: Source point cloud
        target: Target point cloud
        source_fpfh: FPFH features for source
        target_fpfh: FPFH features for target
        voxel_size: Voxel size used
        distance_threshold: Maximum correspondence distance (default: 1.5 * voxel_size)
        
    Returns:
        Tuple of (transformation_matrix, fitness)
    """
    if distance_threshold is None:
        distance_threshold = voxel_size * 1.5
    
    if len(source.points) < 10 or len(target.points) < 10:
        return np.eye(4), 0.0
    
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target,
        source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )
    
    return result.transformation, result.fitness


# =============================================================================
# ICP Registration (Multiple Methods)
# =============================================================================

def run_icp_point_to_plane(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    max_correspondence_distance: float = 0.05,
    init_transform: Optional[np.ndarray] = None,
    max_iteration: int = 50
) -> Tuple[np.ndarray, float, float]:
    """
    Run point-to-plane ICP registration.
    
    Args:
        source: Source point cloud (to be aligned)
        target: Target point cloud (reference)
        max_correspondence_distance: Maximum correspondence distance in meters
        init_transform: Optional initial transformation (4x4 matrix)
        max_iteration: Maximum number of ICP iterations
        
    Returns:
        Tuple of (transformation_matrix, fitness_score, rmse)
    """
    if init_transform is None:
        init_transform = np.eye(4)
    
    if len(source.points) < 10 or len(target.points) < 10:
        return np.eye(4), 0.0, float('inf')
    
    # Ensure normals are estimated on target
    if not target.has_normals():
        estimate_normals(target)
    
    result = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance,
        init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
    )
    
    return result.transformation, result.fitness, result.inlier_rmse


def run_icp_point_to_point(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    max_correspondence_distance: float = 0.05,
    init_transform: Optional[np.ndarray] = None,
    max_iteration: int = 50
) -> Tuple[np.ndarray, float, float]:
    """
    Run point-to-point ICP registration.
    
    Args:
        source: Source point cloud (to be aligned)
        target: Target point cloud (reference)
        max_correspondence_distance: Maximum correspondence distance in meters
        init_transform: Optional initial transformation (4x4 matrix)
        max_iteration: Maximum number of ICP iterations
        
    Returns:
        Tuple of (transformation_matrix, fitness_score, rmse)
    """
    if init_transform is None:
        init_transform = np.eye(4)
    
    if len(source.points) < 10 or len(target.points) < 10:
        return np.eye(4), 0.0, float('inf')
    
    result = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance,
        init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
    )
    
    return result.transformation, result.fitness, result.inlier_rmse


def run_colored_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    max_correspondence_distance: float = 0.05,
    init_transform: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, float, float]:
    """
    Run colored ICP registration (uses both geometry and color).
    
    Args:
        source: Source point cloud with colors
        target: Target point cloud with colors
        max_correspondence_distance: Maximum correspondence distance
        init_transform: Initial transformation
        
    Returns:
        Tuple of (transformation_matrix, fitness_score, rmse)
    """
    if init_transform is None:
        init_transform = np.eye(4)
    
    if len(source.points) < 10 or len(target.points) < 10:
        return np.eye(4), 0.0, float('inf')
    
    # Ensure both have normals
    if not source.has_normals():
        estimate_normals(source)
    if not target.has_normals():
        estimate_normals(target)
    
    result = o3d.pipelines.registration.registration_colored_icp(
        source, target,
        max_correspondence_distance,
        init_transform,
        o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-6,
            max_iteration=50
        )
    )
    
    return result.transformation, result.fitness, result.inlier_rmse


# =============================================================================
# Multi-Scale ICP (NEW - High Quality)
# =============================================================================

def run_multiscale_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    voxel_sizes: List[float] = [0.05, 0.025, 0.01],
    max_iterations: List[int] = [50, 30, 20],
    init_transform: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, float, float]:
    """
    Run multi-scale ICP for robust alignment.
    
    Starts with coarse alignment at large voxel size, then refines
    progressively at finer scales.
    
    Args:
        source: Source point cloud
        target: Target point cloud
        voxel_sizes: List of voxel sizes (coarse to fine)
        max_iterations: List of max iterations per scale
        init_transform: Initial transformation
        
    Returns:
        Tuple of (final_transformation, final_fitness, final_rmse)
    """
    if init_transform is None:
        current_transform = np.eye(4)
    else:
        current_transform = init_transform.copy()
    
    if len(source.points) < 10 or len(target.points) < 10:
        return np.eye(4), 0.0, float('inf')
    
    final_fitness = 0.0
    final_rmse = float('inf')
    
    for i, (voxel_size, max_iter) in enumerate(zip(voxel_sizes, max_iterations)):
        # Downsample for this scale
        source_down = source.voxel_down_sample(voxel_size)
        target_down = target.voxel_down_sample(voxel_size)
        
        if len(source_down.points) < 10 or len(target_down.points) < 10:
            continue
        
        # Estimate normals
        source_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size * 2, max_nn=30
            )
        )
        target_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size * 2, max_nn=30
            )
        )
        
        # Correspondence distance decreases with finer scales
        max_corr_dist = voxel_size * 2.0
        
        # Run ICP at this scale
        result = o3d.pipelines.registration.registration_icp(
            source_down, target_down,
            max_corr_dist,
            current_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
        )
        
        current_transform = result.transformation
        final_fitness = result.fitness
        final_rmse = result.inlier_rmse
        
        print(f"  Scale {i+1}/{len(voxel_sizes)} (voxel={voxel_size:.3f}m): "
              f"fitness={result.fitness:.4f}, rmse={result.inlier_rmse:.4f}")
    
    return current_transform, final_fitness, final_rmse


# =============================================================================
# Full 6-DOF ICP Pipeline (NEW - High Quality)
# =============================================================================

def run_full_icp_pipeline(
    source_points: np.ndarray,
    target_points: np.ndarray,
    source_colors: Optional[np.ndarray] = None,
    target_colors: Optional[np.ndarray] = None,
    voxel_size: float = 0.01,
    use_global_registration: bool = True,
    use_multiscale: bool = True,
    use_colored_icp: bool = False
) -> Dict:
    """
    Complete high-quality ICP pipeline.
    
    Performs:
    1. Preprocessing (downsampling, outlier removal, normal estimation)
    2. Optional global registration (RANSAC with FPFH)
    3. Multi-scale ICP refinement
    4. Optional colored ICP for final polish
    
    Args:
        source_points: Nx3 source points (to be aligned)
        target_points: Mx3 target points (reference)
        source_colors: Optional Nx3 source colors
        target_colors: Optional Mx3 target colors
        voxel_size: Base voxel size for processing
        use_global_registration: Whether to use RANSAC for initial alignment
        use_multiscale: Whether to use multi-scale ICP
        use_colored_icp: Whether to use colored ICP (requires colors)
        
    Returns:
        Dictionary with:
            - 'transformation': 4x4 final transformation matrix
            - 'fitness': Final fitness score (0-1)
            - 'rmse': Final RMSE
            - 'success': Whether alignment was successful
    """
    result = {
        'transformation': np.eye(4),
        'fitness': 0.0,
        'rmse': float('inf'),
        'success': False
    }
    
    if source_points is None or target_points is None:
        print("[ICP Pipeline] Error: Empty point clouds")
        return result
    
    if len(source_points) < 100 or len(target_points) < 100:
        print(f"[ICP Pipeline] Warning: Not enough points (source={len(source_points)}, target={len(target_points)})")
        return result
    
    print("[ICP Pipeline] Starting high-quality alignment...")
    
    # 1. Create Open3D point clouds
    source_pcd = numpy_to_o3d_pointcloud(source_points, source_colors)
    target_pcd = numpy_to_o3d_pointcloud(target_points, target_colors)
    
    # 2. Preprocess point clouds
    print("  [1/4] Preprocessing point clouds...")
    source_prep = preprocess_pointcloud(
        source_pcd, 
        voxel_size=voxel_size,
        remove_outliers=True,
        estimate_norms=True
    )
    target_prep = preprocess_pointcloud(
        target_pcd, 
        voxel_size=voxel_size,
        remove_outliers=True,
        estimate_norms=True
    )
    
    print(f"       Source: {len(source_pcd.points)} -> {len(source_prep.points)} points")
    print(f"       Target: {len(target_pcd.points)} -> {len(target_prep.points)} points")
    
    if len(source_prep.points) < 50 or len(target_prep.points) < 50:
        print("[ICP Pipeline] Error: Too few points after preprocessing")
        return result
    
    current_transform = np.eye(4)
    
    # 3. Global Registration (optional)
    if use_global_registration:
        print("  [2/4] Computing global registration (RANSAC + FPFH)...")
        source_fpfh = compute_fpfh_features(source_prep, voxel_size)
        target_fpfh = compute_fpfh_features(target_prep, voxel_size)
        
        global_transform, global_fitness = execute_global_registration(
            source_prep, target_prep,
            source_fpfh, target_fpfh,
            voxel_size=voxel_size
        )
        
        print(f"       Global registration fitness: {global_fitness:.4f}")
        
        if global_fitness > 0.1:  # Only use if reasonable
            current_transform = global_transform
    else:
        print("  [2/4] Skipping global registration...")
    
    # 4. ICP Refinement
    if use_multiscale:
        print("  [3/4] Running multi-scale ICP refinement...")
        icp_transform, icp_fitness, icp_rmse = run_multiscale_icp(
            source_prep, target_prep,
            voxel_sizes=[voxel_size * 4, voxel_size * 2, voxel_size],
            max_iterations=[50, 30, 20],
            init_transform=current_transform
        )
    else:
        print("  [3/4] Running single-scale ICP...")
        icp_transform, icp_fitness, icp_rmse = run_icp_point_to_plane(
            source_prep, target_prep,
            max_correspondence_distance=voxel_size * 2,
            init_transform=current_transform
        )
        print(f"       ICP fitness: {icp_fitness:.4f}, rmse: {icp_rmse:.4f}")
    
    current_transform = icp_transform
    
    # 5. Colored ICP (optional final polish)
    if use_colored_icp and source_pcd.has_colors() and target_pcd.has_colors():
        print("  [4/4] Running colored ICP for final refinement...")
        colored_transform, colored_fitness, colored_rmse = run_colored_icp(
            source_prep, target_prep,
            max_correspondence_distance=voxel_size * 1.5,
            init_transform=current_transform
        )
        
        if colored_fitness > icp_fitness:
            current_transform = colored_transform
            icp_fitness = colored_fitness
            icp_rmse = colored_rmse
            print(f"       Colored ICP improved: fitness={colored_fitness:.4f}, rmse={colored_rmse:.4f}")
    else:
        print("  [4/4] Skipping colored ICP...")
    
    result['transformation'] = current_transform
    result['fitness'] = icp_fitness
    result['rmse'] = icp_rmse
    result['success'] = icp_fitness > 0.3  # Threshold for success
    
    print(f"[ICP Pipeline] Complete: fitness={icp_fitness:.4f}, rmse={icp_rmse:.4f}, success={result['success']}")
    
    return result


# =============================================================================
# Transform Application
# =============================================================================

def apply_transform_to_points(
    points: np.ndarray,
    transform: np.ndarray
) -> np.ndarray:
    """
    Apply 4x4 transformation matrix to points.
    
    Args:
        points: Nx3 points
        transform: 4x4 transformation matrix
        
    Returns:
        Nx3 transformed points
    """
    if points is None or len(points) == 0:
        return points
    
    ones = np.ones((points.shape[0], 1))
    points_homo = np.hstack([points, ones])
    transformed = (transform @ points_homo.T).T[:, :3]
    return transformed


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
        z_axis_world = T[:3, 2]
        T_new[:3, 3] += z_offset * z_axis_world
        modified_transforms.append(T_new)
    
    return modified_transforms


def apply_6dof_correction_to_wrist_transforms(
    transforms: List[np.ndarray],
    correction_transform: np.ndarray
) -> List[np.ndarray]:
    """
    Apply a 6-DOF correction to all wrist camera transforms.
    
    The correction is applied as: T_new = T_original @ correction
    This adjusts the camera pose in its local frame.
    
    Args:
        transforms: List of 4x4 transformation matrices (camera to world)
        correction_transform: 4x4 correction transformation
        
    Returns:
        List of modified 4x4 transformation matrices
    """
    modified_transforms = []
    
    for T in transforms:
        # Apply correction in camera frame
        T_new = T @ correction_transform
        modified_transforms.append(T_new)
    
    return modified_transforms


# =============================================================================
# Legacy Z-Offset Optimization (kept for backward compatibility)
# =============================================================================

def compute_alignment_error_for_z_offset(
    z_offset: float,
    wrist_points_local: np.ndarray,
    wrist_transform: np.ndarray,
    external_points_world: np.ndarray,
    voxel_size: float = 0.01,
    max_correspondence_distance: float = 0.05
) -> float:
    """Compute alignment error for a given Z offset."""
    if len(wrist_points_local) < 100 or len(external_points_world) < 100:
        return 1.0
    
    wrist_points_shifted = wrist_points_local.copy()
    wrist_points_shifted[:, 2] += z_offset
    
    wrist_world = apply_transform_to_points(wrist_points_shifted, wrist_transform)
    
    pcd_wrist = numpy_to_o3d_pointcloud(wrist_world)
    pcd_external = numpy_to_o3d_pointcloud(external_points_world)
    
    pcd_wrist = downsample_pointcloud(pcd_wrist, voxel_size)
    pcd_external = downsample_pointcloud(pcd_external, voxel_size)
    
    if len(pcd_wrist.points) < 50 or len(pcd_external.points) < 50:
        return 1.0
    
    estimate_normals(pcd_external)
    
    result = o3d.pipelines.registration.registration_icp(
        pcd_wrist, pcd_external,
        max_correspondence_distance,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    
    return -result.fitness


def optimize_wrist_z_offset(
    wrist_points_local: np.ndarray,
    wrist_transform: np.ndarray,
    external_points_world: np.ndarray,
    z_range: Tuple[float, float] = (-0.05, 0.05),
    voxel_size: float = 0.01,
    max_correspondence_distance: float = 0.05
) -> Tuple[float, float]:
    """Optimize Z offset for wrist camera alignment."""
    result = minimize_scalar(
        compute_alignment_error_for_z_offset,
        bounds=z_range,
        method='bounded',
        args=(wrist_points_local, wrist_transform, external_points_world, 
              voxel_size, max_correspondence_distance)
    )
    
    return result.x, -result.fun


def optimize_wrist_z_offset_multi_frame(
    frames_data: List[dict],
    z_range: Tuple[float, float] = (-0.05, 0.05),
    voxel_size: float = 0.01,
    max_correspondence_distance: float = 0.05
) -> Tuple[float, float]:
    """Optimize Z offset using multiple frames for robustness."""
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
            if error < 0.99:
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
    
    return result.x, -result.fun


# =============================================================================
# High-Level Optimization Functions (Main Entry Points)
# =============================================================================

def optimize_wrist_camera_full_icp(
    active_cams: dict, 
    config: dict
) -> Tuple[np.ndarray, float]:
    """
    Optimize wrist camera using full 6-DOF ICP pipeline.
    
    This is the high-quality ICP implementation that performs:
    1. Multi-frame point cloud accumulation
    2. Preprocessing and outlier removal
    3. Optional global registration
    4. Multi-scale ICP refinement
    
    Args:
        active_cams: Dictionary of camera data
        config: Configuration dictionary
        
    Returns:
        Tuple of (correction_transform, fitness)
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
        print("[ICP] Warning: Need both wrist and external cameras")
        return np.eye(4), 0.0
    
    # Config parameters
    num_icp_frames = config.get('icp_num_frames', 10)
    min_depth_icp = config.get('min_depth_wrist_icp', 0.15)
    max_depth_wrist = config.get('wrist_max_depth', 0.75)
    ext_max_depth = config.get('ext_max_depth', 1.5)
    min_depth_ext = config.get('min_depth', 0.1)
    voxel_size = config.get('icp_voxel_size', 0.01)
    
    # Reset cameras
    wrist_cam['zed'].set_svo_position(0)
    for cam in external_cams.values():
        cam['zed'].set_svo_position(0)
    
    total_frames = len(wrist_cam['transforms'])
    frame_indices = np.linspace(0, total_frames - 1, num_icp_frames, dtype=int)
    
    # Accumulate point clouds from multiple frames
    print(f"[ICP] Accumulating points from {num_icp_frames} frames...")
    
    all_wrist_points_world = []
    all_wrist_colors = []
    all_external_points_world = []
    all_external_colors = []
    
    for frame_idx in frame_indices:
        # Get wrist data
        wrist_cam['zed'].set_svo_position(frame_idx)
        if wrist_cam['zed'].grab(wrist_cam['runtime']) != sl.ERROR_CODE.SUCCESS:
            continue
        
        wrist_xyz, wrist_rgb = get_filtered_cloud(
            wrist_cam['zed'], wrist_cam['runtime'],
            max_depth=max_depth_wrist,
            min_depth=min_depth_icp
        )
        
        if wrist_xyz is not None and len(wrist_xyz) > 100:
            # Transform to world
            wrist_world = apply_transform_to_points(wrist_xyz, wrist_cam['transforms'][frame_idx])
            all_wrist_points_world.append(wrist_world)
            if wrist_rgb is not None:
                all_wrist_colors.append(wrist_rgb)
        
        # Get external camera data
        for ext_serial, ext_cam in external_cams.items():
            ext_cam['zed'].set_svo_position(frame_idx)
            if ext_cam['zed'].grab(ext_cam['runtime']) != sl.ERROR_CODE.SUCCESS:
                continue
            
            ext_xyz, ext_rgb = get_filtered_cloud(
                ext_cam['zed'], ext_cam['runtime'],
                max_depth=ext_max_depth,
                min_depth=min_depth_ext
            )
            
            if ext_xyz is not None and len(ext_xyz) > 100:
                ext_world = apply_transform_to_points(ext_xyz, ext_cam['world_T_cam'])
                all_external_points_world.append(ext_world)
                if ext_rgb is not None:
                    all_external_colors.append(ext_rgb)
    
    if not all_wrist_points_world or not all_external_points_world:
        print("[ICP] Error: No valid point clouds collected")
        return np.eye(4), 0.0
    
    # Stack all points
    wrist_points = np.vstack(all_wrist_points_world)
    external_points = np.vstack(all_external_points_world)
    
    wrist_colors = np.vstack(all_wrist_colors) if all_wrist_colors else None
    external_colors = np.vstack(all_external_colors) if all_external_colors else None
    
    print(f"[ICP] Accumulated {len(wrist_points)} wrist points, {len(external_points)} external points")
    
    # Run full ICP pipeline
    icp_result = run_full_icp_pipeline(
        source_points=wrist_points,
        target_points=external_points,
        source_colors=wrist_colors,
        target_colors=external_colors,
        voxel_size=voxel_size,
        use_global_registration=True,
        use_multiscale=True,
        use_colored_icp=(wrist_colors is not None and external_colors is not None)
    )
    
    if icp_result['success']:
        # Apply correction to wrist transforms
        wrist_cam['transforms'] = apply_6dof_correction_to_wrist_transforms(
            wrist_cam['transforms'],
            icp_result['transformation']
        )
        print(f"[ICP] Successfully applied 6-DOF correction (fitness={icp_result['fitness']:.4f})")
    else:
        print("[ICP] Warning: ICP alignment did not meet success threshold")
    
    return icp_result['transformation'], icp_result['fitness']


def optimize_wrist_camera_icp(active_cams, config):
    """Main entry point - uses full ICP pipeline."""
    return optimize_wrist_camera_full_icp(active_cams, config)


def optimize_wrist_camera_icp_z_only(active_cams: dict, config: dict) -> float:
    """
    Optimize wrist camera Z offset using ICP against external cameras.
    
    Legacy function that only optimizes Z offset. Kept for backward compatibility.
    """
    import pyzed.sl as sl
    from .camera_utils import get_filtered_cloud
    
    wrist_cam = None
    external_cams = {}
    
    for serial, cam in active_cams.items():
        if cam['type'] == 'wrist':
            wrist_cam = cam
        else:
            external_cams[serial] = cam
    
    if wrist_cam is None or len(external_cams) == 0:
        print("[ICP] Warning: Need both wrist and external cameras")
        return 0.0
    
    num_icp_frames = config.get('icp_num_frames', 10)
    min_depth_icp = config.get('min_depth_wrist_icp', 0.15)
    max_depth_wrist = config.get('wrist_max_depth', 0.75)
    ext_max_depth = config.get('ext_max_depth', 1.5)
    min_depth_ext = config.get('min_depth', 0.1)
    voxel_size = config.get('icp_voxel_size', 0.01)
    max_corr_dist = config.get('icp_max_correspondence_distance', 0.05)
    
    wrist_cam['zed'].set_svo_position(0)
    for cam in external_cams.values():
        cam['zed'].set_svo_position(0)
    
    total_frames = len(wrist_cam['transforms'])
    frame_indices = np.linspace(0, total_frames - 1, num_icp_frames, dtype=int)
    
    frames_data = []
    
    print(f"[ICP] Collecting {num_icp_frames} frames for Z-offset optimization...")
    
    for frame_idx in frame_indices:
        wrist_cam['zed'].set_svo_position(frame_idx)
        if wrist_cam['zed'].grab(wrist_cam['runtime']) != sl.ERROR_CODE.SUCCESS:
            continue
        
        wrist_xyz, wrist_rgb = get_filtered_cloud(
            wrist_cam['zed'], wrist_cam['runtime'],
            max_depth=max_depth_wrist,
            min_depth=min_depth_icp
        )
        
        if wrist_xyz is None or len(wrist_xyz) < 100:
            continue
        
        wrist_transform = wrist_cam['transforms'][frame_idx]
        
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
            
            ext_world = apply_transform_to_points(ext_xyz, ext_cam['world_T_cam'])
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
    
    print("[ICP] Optimizing Z offset...")
    z_offset, fitness = optimize_wrist_z_offset_multi_frame(
        frames_data,
        z_range=(-0.05, 0.05),
        voxel_size=voxel_size,
        max_correspondence_distance=max_corr_dist
    )
    
    print(f"[ICP] Optimal Z offset: {z_offset:.4f}m, Fitness: {fitness:.4f}")
    
    wrist_cam['transforms'] = apply_z_offset_to_wrist_transforms(
        wrist_cam['transforms'], z_offset
    )
    
    return z_offset


def optimize_wrist_z_offset_icp(active_cams: dict, config: dict) -> float:
    """Alias for optimize_wrist_camera_icp_z_only."""
    return optimize_wrist_camera_icp_z_only(active_cams, config)


def optimize_external_cameras_multi_frame(active_cams: dict, config: dict):
    """Placeholder for external camera optimization."""
    print("[ICP] External camera optimization not needed (assuming correct calibration)")
    pass


def optimize_wrist_multi_frame(active_cams: dict, cartesian_positions: np.ndarray, config: dict):
    """Alias for optimize_wrist_camera_full_icp."""
    return optimize_wrist_camera_full_icp(active_cams, config)
