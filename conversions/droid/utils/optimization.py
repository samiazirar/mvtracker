"""Optimization utilities for wrist camera ICP alignment.

This module provides ICP-based optimization for aligning the wrist camera
point cloud to external camera point clouds. It optimizes the 3D translation
offset of the wrist camera relative to the end-effector.
"""

import open3d as o3d
import numpy as np
import pyzed.sl as sl
from .transforms import transform_points, precompute_wrist_trajectory, invert_transform
from .camera_utils import get_filtered_cloud


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_filtered_cloud_exclude_near(zed, runtime, max_depth=2.0, min_depth=0.15):
    """
    Get filtered point cloud excluding points too close to the camera.
    
    This is used to exclude the gripper from the wrist camera point cloud.
    
    Args:
        zed: ZED Camera object
        runtime: RuntimeParameters for ZED camera
        max_depth: Maximum depth threshold in meters
        min_depth: Minimum depth threshold in meters (excludes gripper at ~15cm)
        
    Returns:
        Tuple of (xyz, rgb) where:
            xyz: Nx3 array of 3D points in camera frame
            rgb: Nx3 array of RGB colors
        Returns (None, None) if retrieval fails
    """
    return get_filtered_cloud(zed, runtime, max_depth, min_depth)


def numpy_to_o3d_pointcloud(xyz, rgb=None):
    """
    Convert numpy arrays to Open3D point cloud.
    
    Args:
        xyz: Nx3 numpy array of 3D points
        rgb: Optional Nx3 numpy array of RGB colors (0-255)
        
    Returns:
        open3d.geometry.PointCloud
    """
    pcd = o3d.geometry.PointCloud()
    if xyz is None or len(xyz) == 0:
        return pcd
    
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    if rgb is not None and len(rgb) > 0:
        # Normalize to 0-1 range for Open3D
        colors = rgb.astype(np.float64) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd


def downsample_pointcloud(pcd, voxel_size=0.01):
    """
    Downsample point cloud using voxel grid.
    
    Args:
        pcd: open3d.geometry.PointCloud
        voxel_size: Voxel size in meters
        
    Returns:
        Downsampled open3d.geometry.PointCloud
    """
    if len(pcd.points) == 0:
        return pcd
    return pcd.voxel_down_sample(voxel_size)


def estimate_normals(pcd, radius=0.03, max_nn=30):
    """
    Estimate normals for point cloud (required for point-to-plane ICP).
    
    Args:
        pcd: open3d.geometry.PointCloud
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


# ==============================================================================
# CORE ICP OPTIMIZATION
# ==============================================================================

def run_icp_point_to_plane(source_pcd, target_pcd, initial_transform=None, 
                           max_correspondence_distance=0.05, max_iterations=50):
    """
    Run point-to-plane ICP between source and target point clouds.
    
    Args:
        source_pcd: Source point cloud (to be aligned)
        target_pcd: Target point cloud (reference)
        initial_transform: Initial 4x4 transformation guess (default: identity)
        max_correspondence_distance: Maximum distance for point correspondences
        max_iterations: Maximum ICP iterations
        
    Returns:
        Tuple of (transformation, fitness, rmse) where:
            transformation: 4x4 alignment transformation
            fitness: Percentage of source points with valid correspondences
            rmse: Root mean square error of aligned points
    """
    if len(source_pcd.points) == 0 or len(target_pcd.points) == 0:
        print("[ICP] Empty point cloud(s), skipping ICP")
        return np.eye(4), 0.0, float('inf')
    
    if initial_transform is None:
        initial_transform = np.eye(4)
    
    # Ensure normals are estimated for point-to-plane ICP
    if not target_pcd.has_normals():
        estimate_normals(target_pcd)
    if not source_pcd.has_normals():
        estimate_normals(source_pcd)
    
    # Run ICP
    reg_result = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, 
        max_correspondence_distance,
        initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
    )
    
    return reg_result.transformation, reg_result.fitness, reg_result.inlier_rmse


def extract_z_offset_from_transform(transform):
    """
    Extract the Z translation component from a 4x4 transformation matrix.
    
    Args:
        transform: 4x4 transformation matrix
        
    Returns:
        Z translation value
    """
    return transform[2, 3]


def create_z_offset_transform(z_offset):
    """
    Create a 4x4 transformation matrix with only Z translation.
    
    Args:
        z_offset: Z translation value in meters
        
    Returns:
        4x4 transformation matrix
    """
    T = np.eye(4)
    T[2, 3] = z_offset
    return T


# ==============================================================================
# WRIST CAMERA ICP OPTIMIZATION
# ==============================================================================

def optimize_wrist_offset_icp(active_cams, config, num_frames_to_sample=10):
    """
    Optimize the wrist camera 3D offset using ICP against external cameras.
    
    The key insight is that we need to find a CONSTANT offset in the camera's
    local frame that, when applied, best aligns the wrist point cloud with
    external cameras across ALL frames.
    
    Approach:
    1. For each sampled frame, run ICP to get world-space correction
    2. Transform correction back to camera-local frame
    3. Average the local corrections to get a consistent offset
    
    Args:
        active_cams: Dictionary of active cameras with ZED objects and transforms
        config: Configuration dictionary
        num_frames_to_sample: Number of frames to sample for optimization
        
    Returns:
        Tuple of (xyz_offset, stats) where:
            xyz_offset: 3D offset in camera local frame [x, y, z]
            stats: Dictionary with optimization statistics
    """
    print("\n[ICP] Starting wrist camera 3D offset optimization...")
    
    # Find wrist and external cameras
    wrist_serial = None
    external_serials = []
    
    for serial, cam in active_cams.items():
        if cam['type'] == 'wrist':
            wrist_serial = serial
        else:
            external_serials.append(serial)
    
    if wrist_serial is None:
        print("[ICP] No wrist camera found, skipping optimization")
        return np.zeros(3), {}
    
    if not external_serials:
        print("[ICP] No external cameras found, skipping optimization")
        return np.zeros(3), {}
    
    wrist_cam = active_cams[wrist_serial]
    num_wrist_frames = len(wrist_cam['transforms'])
    
    # Sample frames evenly across the trajectory
    frame_indices = np.linspace(0, num_wrist_frames - 1, num_frames_to_sample, dtype=int)
    
    # Collect LOCAL frame corrections
    local_offsets = []  # Offsets in camera local frame
    
    for frame_idx in frame_indices:
        print(f"  [ICP] Processing frame {frame_idx}...")
        
        # 1. Build target point cloud from external cameras
        target_points = []
        target_colors = []
        
        for serial in external_serials:
            cam = active_cams[serial]
            zed = cam['zed']
            zed.set_svo_position(frame_idx)
            
            if zed.grab(cam['runtime']) != sl.ERROR_CODE.SUCCESS:
                continue
            
            xyz, rgb = get_filtered_cloud(zed, cam['runtime'], 
                                         config.get('ext_max_depth', 1.5),
                                         config.get('min_depth', 0.1))
            if xyz is None:
                continue
            
            # Transform to world frame
            T_world_cam = cam['world_T_cam']
            xyz_world = transform_points(xyz, T_world_cam)
            
            target_points.append(xyz_world)
            target_colors.append(rgb)
        
        if not target_points:
            print(f"  [ICP] No external points at frame {frame_idx}, skipping")
            continue
        
        target_xyz = np.vstack(target_points)
        target_rgb = np.vstack(target_colors) if target_colors else None
        target_pcd = numpy_to_o3d_pointcloud(target_xyz, target_rgb)
        target_pcd = downsample_pointcloud(target_pcd, voxel_size=0.01)
        estimate_normals(target_pcd)
        
        # 2. Build source point cloud from wrist camera (excluding gripper)
        wrist_zed = wrist_cam['zed']
        wrist_zed.set_svo_position(frame_idx)
        
        if wrist_zed.grab(wrist_cam['runtime']) != sl.ERROR_CODE.SUCCESS:
            print(f"  [ICP] Failed to grab wrist frame {frame_idx}, skipping")
            continue
        
        # Use min_depth_wrist_icp to exclude gripper for ICP optimization
        wrist_xyz_local, wrist_rgb = get_filtered_cloud_exclude_near(
            wrist_zed, wrist_cam['runtime'],
            max_depth=config.get('wrist_max_depth', 0.75),
            min_depth=config.get('min_depth_wrist_icp', 0.15)  # Exclude gripper for ICP
        )
        
        if wrist_xyz_local is None or len(wrist_xyz_local) == 0:
            print(f"  [ICP] No wrist points at frame {frame_idx}, skipping")
            continue
        
        # Get current wrist camera pose
        T_world_wrist = wrist_cam['transforms'][frame_idx]
        
        # Transform to world frame
        wrist_xyz_world = transform_points(wrist_xyz_local, T_world_wrist)
        
        source_pcd = numpy_to_o3d_pointcloud(wrist_xyz_world, wrist_rgb)
        source_pcd = downsample_pointcloud(source_pcd, voxel_size=0.01)
        estimate_normals(source_pcd)
        
        # 3. Run ICP - get world-space correction
        transformation, fitness, rmse = run_icp_point_to_plane(
            source_pcd, target_pcd,
            max_correspondence_distance=0.05,
            max_iterations=100
        )
        
        if fitness < 0.3:  # Require higher fitness for reliable result
            print(f"  [ICP] Low fitness {fitness:.3f} at frame {frame_idx}, skipping")
            continue
        
        # 4. Extract translation in WORLD frame
        world_translation = transformation[:3, 3]
        
        # 5. Convert world translation to CAMERA LOCAL frame
        # The camera's rotation matrix is R = T_world_wrist[:3, :3]
        # To convert world vector to local: local = R^T @ world
        R_world_cam = T_world_wrist[:3, :3]
        local_translation = R_world_cam.T @ world_translation
        
        local_offsets.append(local_translation)
        
        print(f"  [ICP] Frame {frame_idx}: local offset = [{local_translation[0]:.4f}, {local_translation[1]:.4f}, {local_translation[2]:.4f}]m, "
              f"fitness = {fitness:.3f}, rmse = {rmse:.4f}")
    
    if not local_offsets:
        print("[ICP] No valid offsets computed, using zeros")
        return np.zeros(3), {'valid_frames': 0}
    
    # Stack and compute robust statistics
    offsets = np.array(local_offsets)
    
    # Use median for robustness against outliers
    median_offset = np.median(offsets, axis=0)
    mean_offset = np.mean(offsets, axis=0)
    std_offset = np.std(offsets, axis=0)
    
    # Compute outlier-robust estimate using MAD (Median Absolute Deviation)
    mad = np.median(np.abs(offsets - median_offset), axis=0)
    
    print(f"\n[ICP] Offset Statistics ({len(offsets)} valid frames):")
    print(f"  Median (X, Y, Z): [{median_offset[0]:.4f}, {median_offset[1]:.4f}, {median_offset[2]:.4f}]m")
    print(f"  Mean   (X, Y, Z): [{mean_offset[0]:.4f}, {mean_offset[1]:.4f}, {mean_offset[2]:.4f}]m")
    print(f"  Std    (X, Y, Z): [{std_offset[0]:.4f}, {std_offset[1]:.4f}, {std_offset[2]:.4f}]m")
    print(f"  MAD    (X, Y, Z): [{mad[0]:.4f}, {mad[1]:.4f}, {mad[2]:.4f}]m")
    
    # Warn if high variance
    if np.any(std_offset > 0.05):
        print(f"  [WARN] High variance detected! Consider checking camera calibration or sync.")
    
    stats = {
        'valid_frames': len(offsets),
        'median': median_offset,
        'mean': mean_offset,
        'std': std_offset,
        'mad': mad,
        'all_offsets': offsets
    }
    
    return median_offset, stats


def apply_3d_offset_to_wrist(active_cams, cartesian_positions, xyz_offset):
    """
    Apply 3D offset to all wrist camera transforms.
    
    The offset is applied in the camera's LOCAL frame by modifying T_ee_cam.
    
    Args:
        active_cams: Dictionary of active cameras
        cartesian_positions: Nx6 array of end-effector poses
        xyz_offset: 3D offset in camera local frame [x, y, z]
    """
    print(f"[ICP] Applying 3D offset [{xyz_offset[0]:.4f}, {xyz_offset[1]:.4f}, {xyz_offset[2]:.4f}]m to wrist camera...")
    
    for serial, cam in active_cams.items():
        if cam['type'] != 'wrist':
            continue
        
        # Create offset transform in camera local frame
        T_offset = np.eye(4)
        T_offset[:3, 3] = xyz_offset
        
        # Modify T_ee_cam: T_ee_cam_new = T_ee_cam @ T_offset
        # This shifts the camera in its local frame
        cam['T_ee_cam'] = cam['T_ee_cam'] @ T_offset
        
        # Recompute all wrist transforms from scratch with new T_ee_cam
        cam['transforms'] = precompute_wrist_trajectory(cartesian_positions, cam['T_ee_cam'])
        print(f"[ICP] Recomputed {len(cam['transforms'])} wrist transforms with 3D offset")


def optimize_wrist_camera_icp(active_cams, cartesian_positions, config):
    """
    Main entry point for wrist camera ICP optimization (full 3D: X, Y, Z).
    
    This function optimizes the wrist camera 3D offset (X, Y, Z) and updates all transforms.
    
    Args:
        active_cams: Dictionary of active cameras
        cartesian_positions: Nx6 array of end-effector poses
        config: Configuration dictionary
        
    Returns:
        3D offset that was applied [x, y, z] in meters
    """
    # Optimize 3D offset
    xyz_offset, stats = optimize_wrist_offset_icp(
        active_cams, config,
        num_frames_to_sample=config.get('icp_num_frames', 10)
    )
    
    # Apply offset if significant (> 1mm in any dimension)
    if np.any(np.abs(xyz_offset) > 0.001):
        apply_3d_offset_to_wrist(active_cams, cartesian_positions, xyz_offset)
    else:
        print("[ICP] Offset too small, not applying")
    
    return xyz_offset


def optimize_wrist_camera_icp_z_only(active_cams, cartesian_positions, config):
    """
    Optimize wrist camera with Z-only offset (1 degree of freedom).
    
    This function optimizes only the Z offset and updates all transforms.
    
    Args:
        active_cams: Dictionary of active cameras
        cartesian_positions: Nx6 array of end-effector poses
        config: Configuration dictionary
        
    Returns:
        Z offset that was applied in meters
    """
    # Get full 3D offset but only use Z
    xyz_offset, stats = optimize_wrist_offset_icp(
        active_cams, config,
        num_frames_to_sample=config.get('icp_num_frames', 10)
    )
    
    z_offset = xyz_offset[2]
    
    print(f"\n[ICP-Z] Using Z-only offset: {z_offset:.4f}m (ignoring X={xyz_offset[0]:.4f}m, Y={xyz_offset[1]:.4f}m)")
    
    # Apply Z-only offset
    if abs(z_offset) > 0.001:
        z_only_offset = np.array([0.0, 0.0, z_offset])
        apply_3d_offset_to_wrist(active_cams, cartesian_positions, z_only_offset)
    else:
        print("[ICP-Z] Z offset too small, not applying")
    
    return z_offset


# Legacy function name for backward compatibility
def optimize_wrist_z_offset_icp(active_cams, config, num_frames_to_sample=10):
    """Legacy wrapper - now optimizes full 3D offset."""
    offset, _ = optimize_wrist_offset_icp(active_cams, config, num_frames_to_sample)
    return offset[2]  # Return Z for backward compatibility


def apply_z_offset_to_wrist_transforms(active_cams, z_offset):
    """Legacy wrapper - applies Z offset only."""
    print(f"[ICP] Applying Z offset of {z_offset:.4f}m to wrist camera...")
    for serial, cam in active_cams.items():
        if cam['type'] != 'wrist':
            continue
        for i, T_world_cam in enumerate(cam['transforms']):
            cam_z_axis = T_world_cam[:3, :3] @ np.array([0, 0, 1])
            cam['transforms'][i][:3, 3] += z_offset * cam_z_axis
