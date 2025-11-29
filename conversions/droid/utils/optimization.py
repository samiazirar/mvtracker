"""Robust ICP Optimization for Wrist Camera Alignment.

This module provides full 6-DOF ICP-based optimization for aligning the wrist camera
point cloud to external camera point clouds.

The implementation uses:
1. Multi-scale ICP with coarse-to-fine alignment
2. Robust point-to-plane ICP with outlier rejection
3. Multi-frame accumulation for stable alignment
4. Global registration (RANSAC + FPFH features) for initial alignment

Key features:
- Full 6-DOF transformation refinement (not just Z-offset)
- Gripper exclusion (points < min_depth are filtered)
- Robust to noise and partial overlaps
"""

import numpy as np
import open3d as o3d
from typing import Optional, Tuple, List, Dict
from scipy.spatial.transform import Rotation as R


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
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    
    if colors is not None and len(colors) > 0:
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
    if len(pcd.points) == 0:
        return pcd
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    return pcd


def compute_fpfh_features(pcd: o3d.geometry.PointCloud, voxel_size: float = 0.02) -> o3d.pipelines.registration.Feature:
    """
    Compute FPFH features for global registration.
    
    Args:
        pcd: Point cloud with normals
        voxel_size: Voxel size for feature computation
        
    Returns:
        FPFH feature descriptor
    """
    if not pcd.has_normals():
        estimate_normals(pcd, radius=voxel_size * 2)
    
    radius_feature = voxel_size * 5
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
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
        min_distance: Minimum distance from camera origin
        
    Returns:
        Tuple of (filtered_points, filtered_colors)
    """
    if points is None or len(points) == 0:
        return np.empty((0, 3)), None if colors is None else np.empty((0, 3))
    
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
        min_depth: Minimum depth (Z) to exclude gripper
        max_depth: Maximum depth to include
        
    Returns:
        Tuple of (filtered_points, filtered_colors)
    """
    if points_local is None or len(points_local) == 0:
        return np.empty((0, 3)), None if colors is None else np.empty((0, 3))
    
    z_vals = points_local[:, 2]
    mask = (z_vals > min_depth) & (z_vals < max_depth) & np.isfinite(points_local).all(axis=1)
    
    filtered_points = points_local[mask]
    filtered_colors = colors[mask] if colors is not None else None
    
    return filtered_points, filtered_colors


def remove_statistical_outliers(
    pcd: o3d.geometry.PointCloud,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0
) -> o3d.geometry.PointCloud:
    """
    Remove statistical outliers from point cloud.
    
    Args:
        pcd: Input point cloud
        nb_neighbors: Number of neighbors for mean distance
        std_ratio: Standard deviation ratio threshold
        
    Returns:
        Filtered point cloud
    """
    if len(pcd.points) < nb_neighbors:
        return pcd
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return cl


# =============================================================================
# ICP Registration Functions
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
    """
    if init_transform is None:
        init_transform = np.eye(4)
    
    if len(source.points) == 0 or len(target.points) == 0:
        return np.eye(4), 0.0
    
    if not target.has_normals():
        estimate_normals(target)
    
    result = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance,
        init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    
    return np.array(result.transformation), result.fitness


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
        max_correspondence_distance: Maximum correspondence distance
        init_transform: Optional initial transformation (4x4 matrix)
        
    Returns:
        Tuple of (transformation_matrix, fitness_score)
    """
    if init_transform is None:
        init_transform = np.eye(4)
    
    if len(source.points) == 0 or len(target.points) == 0:
        return np.eye(4), 0.0
    
    result = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance,
        init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    
    return np.array(result.transformation), result.fitness


def run_robust_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    max_correspondence_distance: float = 0.05,
    init_transform: Optional[np.ndarray] = None,
    robust_kernel_sigma: float = 0.01
) -> Tuple[np.ndarray, float, float]:
    """
    Run robust ICP with Tukey loss function for outlier rejection.
    
    Args:
        source: Source point cloud
        target: Target point cloud  
        max_correspondence_distance: Maximum correspondence distance
        init_transform: Initial transformation
        robust_kernel_sigma: Sigma for robust kernel (smaller = more aggressive outlier rejection)
        
    Returns:
        Tuple of (transformation, fitness, inlier_rmse)
    """
    if init_transform is None:
        init_transform = np.eye(4)
    
    if len(source.points) == 0 or len(target.points) == 0:
        return np.eye(4), 0.0, float('inf')
    
    if not target.has_normals():
        estimate_normals(target)
    
    # Use robust point-to-plane with Tukey loss
    loss = o3d.pipelines.registration.TukeyLoss(k=robust_kernel_sigma)
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
    
    result = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance,
        init_transform,
        estimation
    )
    
    return np.array(result.transformation), result.fitness, result.inlier_rmse


def run_colored_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    max_correspondence_distance: float = 0.05,
    init_transform: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, float]:
    """
    Run colored ICP which uses both geometry and color for alignment.
    
    Args:
        source: Source point cloud with colors
        target: Target point cloud with colors
        max_correspondence_distance: Maximum correspondence distance
        init_transform: Initial transformation
        
    Returns:
        Tuple of (transformation, fitness)
    """
    if init_transform is None:
        init_transform = np.eye(4)
    
    if len(source.points) == 0 or len(target.points) == 0:
        return np.eye(4), 0.0
    
    if not target.has_normals():
        estimate_normals(target)
    if not source.has_normals():
        estimate_normals(source)
    
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
    
    return np.array(result.transformation), result.fitness


def run_global_registration_ransac(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    voxel_size: float = 0.02
) -> Tuple[np.ndarray, float]:
    """
    Run global registration using RANSAC with FPFH features.
    
    This provides a rough initial alignment without requiring a good initial guess.
    
    Args:
        source: Source point cloud
        target: Target point cloud
        voxel_size: Voxel size for downsampling and feature extraction
        
    Returns:
        Tuple of (transformation, fitness)
    """
    if len(source.points) == 0 or len(target.points) == 0:
        return np.eye(4), 0.0
    
    # Downsample
    source_down = downsample_pointcloud(source, voxel_size)
    target_down = downsample_pointcloud(target, voxel_size)
    
    if len(source_down.points) < 10 or len(target_down.points) < 10:
        return np.eye(4), 0.0
    
    # Estimate normals
    estimate_normals(source_down, radius=voxel_size * 2)
    estimate_normals(target_down, radius=voxel_size * 2)
    
    # Compute FPFH features
    source_fpfh = compute_fpfh_features(source_down, voxel_size)
    target_fpfh = compute_fpfh_features(target_down, voxel_size)
    
    # RANSAC registration
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down,
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
    
    return np.array(result.transformation), result.fitness


# =============================================================================
# Multi-Scale ICP Registration
# =============================================================================

def run_multiscale_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    voxel_sizes: List[float] = [0.05, 0.02, 0.01],
    max_correspondence_multipliers: List[float] = [2.0, 1.5, 1.0],
    init_transform: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, float]:
    """
    Run multi-scale ICP for robust registration.
    
    Starts with coarse alignment and progressively refines.
    
    Args:
        source: Source point cloud
        target: Target point cloud
        voxel_sizes: List of voxel sizes from coarse to fine
        max_correspondence_multipliers: Multipliers for max correspondence distance
        init_transform: Initial transformation
        
    Returns:
        Tuple of (final_transformation, final_fitness)
    """
    if init_transform is None:
        init_transform = np.eye(4)
    
    if len(source.points) == 0 or len(target.points) == 0:
        return np.eye(4), 0.0
    
    current_transform = init_transform.copy()
    final_fitness = 0.0
    
    for i, voxel_size in enumerate(voxel_sizes):
        # Downsample at current scale
        source_down = downsample_pointcloud(source, voxel_size)
        target_down = downsample_pointcloud(target, voxel_size)
        
        if len(source_down.points) < 50 or len(target_down.points) < 50:
            continue
        
        # Estimate normals
        estimate_normals(target_down, radius=voxel_size * 2)
        
        # Run ICP
        max_corr_dist = voxel_size * max_correspondence_multipliers[i]
        transform, fitness = run_icp_point_to_plane(
            source_down, target_down,
            max_correspondence_distance=max_corr_dist,
            init_transform=current_transform
        )
        
        current_transform = transform
        final_fitness = fitness
    
    return current_transform, final_fitness


# =============================================================================
# Full 6-DOF Wrist Camera Optimization (NEW APPROACH)
# =============================================================================

def collect_multi_frame_pointclouds(
    active_cams: dict,
    config: dict,
    num_frames: int = 10
) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud, List[int]]:
    """
    Collect and accumulate point clouds from multiple frames for robust ICP.
    
    Args:
        active_cams: Dictionary of camera data
        config: Configuration dictionary
        num_frames: Number of frames to sample
        
    Returns:
        Tuple of (wrist_accumulated_cloud, external_accumulated_cloud, valid_frame_indices)
    """
    import pyzed.sl as sl
    from .camera_utils import get_filtered_cloud
    from .transforms import transform_points
    
    # Find cameras
    wrist_cam = None
    external_cams = {}
    for serial, cam in active_cams.items():
        if cam['type'] == 'wrist':
            wrist_cam = cam
        else:
            external_cams[serial] = cam
    
    if wrist_cam is None or len(external_cams) == 0:
        return o3d.geometry.PointCloud(), o3d.geometry.PointCloud(), []
    
    # Parameters
    min_depth_icp = config.get('min_depth_wrist_icp', 0.15)
    max_depth_wrist = config.get('wrist_max_depth', 0.75)
    min_depth_ext = config.get('min_depth', 0.1)
    max_depth_ext = config.get('ext_max_depth', 1.5)
    
    total_frames = len(wrist_cam['transforms'])
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    wrist_points_all = []
    wrist_colors_all = []
    ext_points_all = []
    ext_colors_all = []
    valid_indices = []
    
    # Reset cameras
    wrist_cam['zed'].set_svo_position(0)
    for cam in external_cams.values():
        cam['zed'].set_svo_position(0)
    
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
        
        if wrist_xyz is None or len(wrist_xyz) < 100:
            continue
        
        # Transform wrist points to world
        T_wrist = wrist_cam['transforms'][frame_idx]
        wrist_world = transform_points(wrist_xyz, T_wrist)
        
        wrist_points_all.append(wrist_world)
        wrist_colors_all.append(wrist_rgb)
        
        # Get external camera points
        for ext_serial, ext_cam in external_cams.items():
            ext_cam['zed'].set_svo_position(frame_idx)
            if ext_cam['zed'].grab(ext_cam['runtime']) != sl.ERROR_CODE.SUCCESS:
                continue
            
            ext_xyz, ext_rgb = get_filtered_cloud(
                ext_cam['zed'], ext_cam['runtime'],
                max_depth=max_depth_ext,
                min_depth=min_depth_ext
            )
            
            if ext_xyz is None or len(ext_xyz) < 100:
                continue
            
            ext_world = transform_points(ext_xyz, ext_cam['world_T_cam'])
            ext_points_all.append(ext_world)
            ext_colors_all.append(ext_rgb)
        
        valid_indices.append(frame_idx)
    
    # Create accumulated point clouds
    wrist_pcd = o3d.geometry.PointCloud()
    if wrist_points_all:
        all_wrist_pts = np.vstack(wrist_points_all)
        all_wrist_cols = np.vstack(wrist_colors_all)
        wrist_pcd = numpy_to_o3d_pointcloud(all_wrist_pts, all_wrist_cols)
    
    ext_pcd = o3d.geometry.PointCloud()
    if ext_points_all:
        all_ext_pts = np.vstack(ext_points_all)
        all_ext_cols = np.vstack(ext_colors_all)
        ext_pcd = numpy_to_o3d_pointcloud(all_ext_pts, all_ext_cols)
    
    return wrist_pcd, ext_pcd, valid_indices


def compute_wrist_icp_correction(
    wrist_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    config: dict
) -> Tuple[np.ndarray, float, float]:
    """
    Compute ICP correction transform for wrist camera alignment.
    
    Uses a robust multi-stage approach:
    1. Statistical outlier removal
    2. Multi-scale ICP refinement
    3. Robust ICP with outlier rejection
    
    Args:
        wrist_pcd: Wrist camera point cloud in world frame
        target_pcd: External camera point cloud (target/reference)
        config: Configuration dictionary
        
    Returns:
        Tuple of (correction_transform, fitness, rmse)
    """
    if len(wrist_pcd.points) == 0 or len(target_pcd.points) == 0:
        print("[ICP] Warning: Empty point clouds")
        return np.eye(4), 0.0, float('inf')
    
    voxel_size = config.get('icp_voxel_size', 0.01)
    max_corr_dist = config.get('icp_max_correspondence_distance', 0.05)
    
    print(f"[ICP] Wrist cloud: {len(wrist_pcd.points)} points")
    print(f"[ICP] External cloud: {len(target_pcd.points)} points")
    
    # Step 1: Downsample both clouds
    wrist_down = downsample_pointcloud(wrist_pcd, voxel_size)
    target_down = downsample_pointcloud(target_pcd, voxel_size)
    
    print(f"[ICP] After downsampling: wrist={len(wrist_down.points)}, target={len(target_down.points)}")
    
    # Step 2: Remove statistical outliers
    wrist_down = remove_statistical_outliers(wrist_down, nb_neighbors=20, std_ratio=2.0)
    target_down = remove_statistical_outliers(target_down, nb_neighbors=20, std_ratio=2.0)
    
    print(f"[ICP] After outlier removal: wrist={len(wrist_down.points)}, target={len(target_down.points)}")
    
    if len(wrist_down.points) < 100 or len(target_down.points) < 100:
        print("[ICP] Warning: Not enough points after filtering")
        return np.eye(4), 0.0, float('inf')
    
    # Step 3: Multi-scale ICP
    print("[ICP] Running multi-scale ICP...")
    transform, fitness = run_multiscale_icp(
        wrist_down, target_down,
        voxel_sizes=[voxel_size * 3, voxel_size * 2, voxel_size],
        max_correspondence_multipliers=[3.0, 2.0, 1.5],
        init_transform=np.eye(4)
    )
    
    print(f"[ICP] Multi-scale ICP: fitness={fitness:.4f}")
    
    # Step 4: Refine with robust ICP
    if fitness > 0.1:  # Only refine if initial alignment is reasonable
        print("[ICP] Refining with robust ICP...")
        transform_refined, fitness_refined, rmse = run_robust_icp(
            wrist_down, target_down,
            max_correspondence_distance=max_corr_dist,
            init_transform=transform,
            robust_kernel_sigma=voxel_size
        )
        
        if fitness_refined > fitness:
            transform = transform_refined
            fitness = fitness_refined
            print(f"[ICP] Robust ICP improved: fitness={fitness:.4f}, RMSE={rmse:.6f}")
    else:
        rmse = float('inf')
    
    return transform, fitness, rmse


def apply_transform_to_trajectory(
    transforms: List[np.ndarray],
    correction: np.ndarray
) -> List[np.ndarray]:
    """
    Apply a correction transform to all trajectory poses.
    
    The correction is applied as: T_new = correction @ T_original
    
    Args:
        transforms: List of 4x4 transformation matrices
        correction: 4x4 correction transform
        
    Returns:
        List of corrected 4x4 transformation matrices
    """
    corrected = []
    for T in transforms:
        T_new = correction @ T
        corrected.append(T_new)
    return corrected


def optimize_wrist_camera_full_icp(active_cams: dict, config: dict) -> Tuple[np.ndarray, float]:
    """
    Perform full 6-DOF ICP optimization for wrist camera alignment.
    
    This is the main entry point for the new robust ICP implementation.
    
    Args:
        active_cams: Dictionary of camera data
        config: Configuration dictionary
        
    Returns:
        Tuple of (correction_transform, fitness)
    """
    print("\n" + "=" * 60)
    print("Full 6-DOF ICP Wrist Camera Alignment")
    print("=" * 60)
    
    num_icp_frames = config.get('icp_num_frames', 10)
    
    # Find wrist camera
    wrist_cam = None
    wrist_serial = None
    for serial, cam in active_cams.items():
        if cam['type'] == 'wrist':
            wrist_cam = cam
            wrist_serial = serial
            break
    
    if wrist_cam is None:
        print("[ICP] Error: No wrist camera found")
        return np.eye(4), 0.0
    
    # Step 1: Collect multi-frame point clouds
    print(f"\n[ICP] Collecting {num_icp_frames} frames...")
    wrist_pcd, ext_pcd, valid_indices = collect_multi_frame_pointclouds(
        active_cams, config, num_icp_frames
    )
    
    if len(valid_indices) == 0:
        print("[ICP] Error: No valid frames collected")
        return np.eye(4), 0.0
    
    print(f"[ICP] Collected {len(valid_indices)} valid frames")
    
    # Step 2: Compute ICP correction
    print("\n[ICP] Computing ICP correction...")
    correction, fitness, rmse = compute_wrist_icp_correction(wrist_pcd, ext_pcd, config)
    
    # Analyze the correction
    translation = correction[:3, 3]
    rotation = R.from_matrix(correction[:3, :3])
    euler_deg = rotation.as_euler('xyz', degrees=True)
    
    print(f"\n[ICP] Correction transform:")
    print(f"  Translation: [{translation[0]*100:.2f}, {translation[1]*100:.2f}, {translation[2]*100:.2f}] cm")
    print(f"  Rotation: [{euler_deg[0]:.2f}, {euler_deg[1]:.2f}, {euler_deg[2]:.2f}] degrees")
    print(f"  Fitness: {fitness:.4f}")
    
    # Step 3: Apply correction to wrist camera trajectory
    if fitness > 0.05:  # Only apply if alignment is reasonable
        print("\n[ICP] Applying correction to wrist trajectory...")
        wrist_cam['transforms'] = apply_transform_to_trajectory(
            wrist_cam['transforms'], correction
        )
        print("[ICP] Trajectory updated")
    else:
        print("\n[ICP] Warning: Low fitness score, skipping trajectory update")
    
    return correction, fitness


# =============================================================================
# Legacy Functions (for backwards compatibility)
# =============================================================================

def optimize_wrist_z_offset(
    wrist_points_local: np.ndarray,
    wrist_transform: np.ndarray,
    external_points_world: np.ndarray,
    z_range: Tuple[float, float] = (-0.05, 0.05),
    voxel_size: float = 0.01,
    max_correspondence_distance: float = 0.05
) -> Tuple[float, float]:
    """
    Legacy: Optimize Z offset for wrist camera alignment.
    
    Note: This is kept for backwards compatibility but the new
    optimize_wrist_camera_full_icp function is recommended.
    """
    from .transforms import transform_points
    
    # Transform wrist points to world
    wrist_world = transform_points(wrist_points_local, wrist_transform)
    
    # Create point clouds
    wrist_pcd = numpy_to_o3d_pointcloud(wrist_world)
    ext_pcd = numpy_to_o3d_pointcloud(external_points_world)
    
    # Run ICP
    transform, fitness = run_multiscale_icp(wrist_pcd, ext_pcd)
    
    # Extract Z offset from transform
    z_offset = transform[2, 3]
    
    return z_offset, fitness


def optimize_wrist_z_offset_multi_frame(
    frames_data: List[dict],
    z_range: Tuple[float, float] = (-0.05, 0.05),
    voxel_size: float = 0.01,
    max_correspondence_distance: float = 0.05
) -> Tuple[float, float]:
    """
    Legacy: Optimize Z offset using multiple frames.
    
    Note: This is kept for backwards compatibility.
    """
    from .transforms import transform_points
    
    # Accumulate all points
    all_wrist_world = []
    all_ext_world = []
    
    for frame in frames_data:
        wrist_world = transform_points(
            frame['wrist_points_local'],
            frame['wrist_transform']
        )
        all_wrist_world.append(wrist_world)
        all_ext_world.append(frame['external_points_world'])
    
    if not all_wrist_world:
        return 0.0, 0.0
    
    wrist_pts = np.vstack(all_wrist_world)
    ext_pts = np.vstack(all_ext_world)
    
    wrist_pcd = numpy_to_o3d_pointcloud(wrist_pts)
    ext_pcd = numpy_to_o3d_pointcloud(ext_pts)
    
    transform, fitness = run_multiscale_icp(wrist_pcd, ext_pcd)
    z_offset = transform[2, 3]
    
    return z_offset, fitness


def apply_z_offset_to_wrist_transforms(
    transforms: List[np.ndarray],
    z_offset: float
) -> List[np.ndarray]:
    """
    Apply Z offset to all wrist camera transforms.
    
    Args:
        transforms: List of 4x4 transformation matrices
        z_offset: Z offset to apply (in camera frame)
        
    Returns:
        List of modified 4x4 transformation matrices
    """
    modified = []
    for T in transforms:
        T_new = T.copy()
        z_axis_world = T[:3, 2]
        T_new[:3, 3] += z_offset * z_axis_world
        modified.append(T_new)
    return modified


# =============================================================================
# Wrapper Functions (for compatibility with existing scripts)
# =============================================================================

def optimize_wrist_camera_icp(active_cams: dict, config: dict) -> np.ndarray:
    """
    Main ICP optimization entry point.
    
    Uses the new full 6-DOF ICP implementation.
    """
    correction, fitness = optimize_wrist_camera_full_icp(active_cams, config)
    return correction


def optimize_wrist_camera_icp_z_only(active_cams: dict, config: dict) -> float:
    """
    Legacy wrapper that returns Z offset from full ICP.
    """
    correction, fitness = optimize_wrist_camera_full_icp(active_cams, config)
    return correction[2, 3]


def optimize_wrist_z_offset_icp(active_cams: dict, config: dict) -> float:
    """
    Alias for optimize_wrist_camera_icp_z_only.
    """
    return optimize_wrist_camera_icp_z_only(active_cams, config)


def optimize_external_cameras_multi_frame(active_cams: dict, config: dict):
    """
    Placeholder for external camera optimization.
    """
    print("[ICP] External camera optimization not needed")
    pass


def optimize_wrist_multi_frame(active_cams: dict, cartesian_positions: np.ndarray, config: dict):
    """
    Alias for full ICP optimization.
    """
    return optimize_wrist_camera_full_icp(active_cams, config)
