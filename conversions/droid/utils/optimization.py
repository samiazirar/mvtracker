"""Optimization utilities for wrist camera ICP alignment.

This module provides ICP-based optimization for aligning the wrist camera
point cloud to external camera point clouds. It optimizes only the Z offset
of the wrist camera relative to the gripper.
"""

import open3d as o3d
import numpy as np
import pyzed.sl as sl
from .transforms import transform_points, precompute_wrist_trajectory
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

def optimize_wrist_z_offset_icp(active_cams, config, num_frames_to_sample=10):
    """
    Optimize the wrist camera Z offset using ICP against external cameras.
    
    This function:
    1. Samples multiple frames from the trajectory
    2. For each frame, builds point clouds from external cameras (target)
    3. Builds wrist camera point cloud (source) excluding points < 15cm
    4. Runs point-to-plane ICP to find optimal Z offset
    5. Averages the Z offset across all frames
    
    Args:
        active_cams: Dictionary of active cameras with ZED objects and transforms
        config: Configuration dictionary
        num_frames_to_sample: Number of frames to sample for optimization
        
    Returns:
        Optimal Z offset in meters
    """
    print("\n[ICP] Starting wrist camera Z offset optimization...")
    
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
        return 0.0
    
    if not external_serials:
        print("[ICP] No external cameras found, skipping optimization")
        return 0.0
    
    wrist_cam = active_cams[wrist_serial]
    num_wrist_frames = len(wrist_cam['transforms'])
    
    # Sample frames evenly across the trajectory
    frame_indices = np.linspace(0, num_wrist_frames - 1, num_frames_to_sample, dtype=int)
    
    z_offsets = []
    
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
        
        # Use min_icp_depth to exclude gripper for ICP
        wrist_xyz, wrist_rgb = get_filtered_cloud_exclude_near(
            wrist_zed, wrist_cam['runtime'],
            max_depth=config.get('wrist_max_depth', 0.75),
            min_depth=config.get('min_icp_depth', 0.15)  # Exclude gripper
        )
        
        if wrist_xyz is None or len(wrist_xyz) == 0:
            print(f"  [ICP] No wrist points at frame {frame_idx}, skipping")
            continue
        
        # Transform to world frame using current wrist transform
        T_wrist_world = wrist_cam['transforms'][frame_idx]
        wrist_xyz_world = transform_points(wrist_xyz, T_wrist_world)
        
        source_pcd = numpy_to_o3d_pointcloud(wrist_xyz_world, wrist_rgb)
        source_pcd = downsample_pointcloud(source_pcd, voxel_size=0.01)
        estimate_normals(source_pcd)
        
        # 3. Run ICP
        transformation, fitness, rmse = run_icp_point_to_plane(
            source_pcd, target_pcd,
            max_correspondence_distance=0.05,
            max_iterations=100
        )
        
        if fitness < 0.1:
            print(f"  [ICP] Low fitness {fitness:.3f} at frame {frame_idx}, skipping")
            continue
        
        # Extract Z offset
        z_off = extract_z_offset_from_transform(transformation)
        z_offsets.append(z_off)
        
        print(f"  [ICP] Frame {frame_idx}: Z offset = {z_off:.4f}m, fitness = {fitness:.3f}, rmse = {rmse:.4f}")
    
    if not z_offsets:
        print("[ICP] No valid Z offsets computed, using 0")
        return 0.0
    
    # Average Z offset across frames
    avg_z_offset = np.median(z_offsets)
    print(f"[ICP] Final Z offset (median): {avg_z_offset:.4f}m")
    
    return avg_z_offset


def apply_z_offset_to_wrist_transforms(active_cams, z_offset):
    """
    Apply Z offset to all wrist camera transforms.
    
    The Z offset is applied in the end-effector frame, modifying T_ee_cam.
    
    Args:
        active_cams: Dictionary of active cameras
        z_offset: Z offset in meters
    """
    print(f"[ICP] Applying Z offset of {z_offset:.4f}m to wrist camera...")
    
    for serial, cam in active_cams.items():
        if cam['type'] != 'wrist':
            continue
        
        # Apply offset to T_ee_cam
        T_z_offset = create_z_offset_transform(z_offset)
        
        # Modify T_ee_cam: T_ee_cam_new = T_ee_cam @ T_z_offset
        # This shifts the camera along its local Z axis
        cam['T_ee_cam'] = cam['T_ee_cam'] @ T_z_offset
        
        # Recompute all wrist transforms with new offset
        # We need the original cartesian_positions for this
        # Since we don't have them here, we apply offset to each world transform directly
        for i, T_world_cam in enumerate(cam['transforms']):
            # Apply offset in world frame (approximate)
            # This shifts along the camera's Z axis in world frame
            cam_z_axis = T_world_cam[:3, :3] @ np.array([0, 0, 1])
            cam['transforms'][i][:3, 3] += z_offset * cam_z_axis


def optimize_wrist_camera_icp(active_cams, cartesian_positions, config):
    """
    Main entry point for wrist camera ICP optimization (Z-only).
    
    This function optimizes the wrist camera Z offset and updates all transforms.
    
    Args:
        active_cams: Dictionary of active cameras
        cartesian_positions: Nx6 array of end-effector poses
        config: Configuration dictionary
        
    Returns:
        Z offset that was applied (in meters)
    """
    # Optimize Z offset
    z_offset = optimize_wrist_z_offset_icp(
        active_cams, config,
        num_frames_to_sample=config.get('icp_num_frames', 10)
    )
    
    # Apply offset if significant
    if abs(z_offset) > 0.001:  # Only apply if > 1mm
        apply_z_offset_to_wrist_transforms(active_cams, z_offset)
        
        # Optionally recompute transforms from scratch with new T_ee_cam
        # This gives more accurate results
        wrist_serial = None
        for serial, cam in active_cams.items():
            if cam['type'] == 'wrist':
                wrist_serial = serial
                break
        
        if wrist_serial:
            wrist_cam = active_cams[wrist_serial]
            T_ee_cam_new = wrist_cam['T_ee_cam']
            wrist_cam['transforms'] = precompute_wrist_trajectory(cartesian_positions, T_ee_cam_new)
            print(f"[ICP] Recomputed {len(wrist_cam['transforms'])} wrist transforms with new offset")
    
    return z_offset


# ==============================================================================
# FULL 6-DOF ICP OPTIMIZATION
# ==============================================================================

def optimize_wrist_full_6dof_icp(active_cams, config, num_frames_to_sample=10):
    """
    Optimize the wrist camera using full 6-DOF ICP against external cameras.
    
    Unlike Z-only optimization, this allows rotation and translation in all axes.
    
    Args:
        active_cams: Dictionary of active cameras with ZED objects and transforms
        config: Configuration dictionary
        num_frames_to_sample: Number of frames to sample for optimization
        
    Returns:
        4x4 transformation matrix representing the average correction
    """
    print("\n[ICP-6DOF] Starting full 6-DOF wrist camera optimization...")
    
    # Find wrist and external cameras
    wrist_serial = None
    external_serials = []
    
    for serial, cam in active_cams.items():
        if cam['type'] == 'wrist':
            wrist_serial = serial
        else:
            external_serials.append(serial)
    
    if wrist_serial is None:
        print("[ICP-6DOF] No wrist camera found, skipping optimization")
        return np.eye(4)
    
    if not external_serials:
        print("[ICP-6DOF] No external cameras found, skipping optimization")
        return np.eye(4)
    
    wrist_cam = active_cams[wrist_serial]
    num_wrist_frames = len(wrist_cam['transforms'])
    
    # Sample frames evenly across the trajectory
    frame_indices = np.linspace(0, num_wrist_frames - 1, num_frames_to_sample, dtype=int)
    
    transformations = []
    
    for frame_idx in frame_indices:
        print(f"  [ICP-6DOF] Processing frame {frame_idx}...")
        
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
            print(f"  [ICP-6DOF] No external points at frame {frame_idx}, skipping")
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
            print(f"  [ICP-6DOF] Failed to grab wrist frame {frame_idx}, skipping")
            continue
        
        # Use min_icp_depth to exclude gripper for ICP
        wrist_xyz, wrist_rgb = get_filtered_cloud_exclude_near(
            wrist_zed, wrist_cam['runtime'],
            max_depth=config.get('wrist_max_depth', 0.75),
            min_depth=config.get('min_icp_depth', 0.15)  # Exclude gripper
        )
        
        if wrist_xyz is None or len(wrist_xyz) == 0:
            print(f"  [ICP-6DOF] No wrist points at frame {frame_idx}, skipping")
            continue
        
        # Transform to world frame using current wrist transform
        T_wrist_world = wrist_cam['transforms'][frame_idx]
        wrist_xyz_world = transform_points(wrist_xyz, T_wrist_world)
        
        source_pcd = numpy_to_o3d_pointcloud(wrist_xyz_world, wrist_rgb)
        source_pcd = downsample_pointcloud(source_pcd, voxel_size=0.01)
        estimate_normals(source_pcd)
        
        # 3. Run ICP - full 6-DOF
        transformation, fitness, rmse = run_icp_point_to_plane(
            source_pcd, target_pcd,
            max_correspondence_distance=0.05,
            max_iterations=100
        )
        
        if fitness < 0.1:
            print(f"  [ICP-6DOF] Low fitness {fitness:.3f} at frame {frame_idx}, skipping")
            continue
        
        transformations.append(transformation)
        
        # Extract all components for logging
        tx, ty, tz = transformation[:3, 3]
        print(f"  [ICP-6DOF] Frame {frame_idx}: T=[{tx:.4f}, {ty:.4f}, {tz:.4f}]m, fitness={fitness:.3f}, rmse={rmse:.4f}")
    
    if not transformations:
        print("[ICP-6DOF] No valid transformations computed, using identity")
        return np.eye(4)
    
    # Average transformations
    # For rotation, we use quaternion averaging; for translation, simple mean
    from scipy.spatial.transform import Rotation as R
    
    rotations = [T[:3, :3] for T in transformations]
    translations = [T[:3, 3] for T in transformations]
    
    # Average rotation using quaternion SLERP (simplified: just average quaternions)
    quats = [R.from_matrix(rot).as_quat() for rot in rotations]  # [x, y, z, w]
    quats = np.array(quats)
    
    # Handle quaternion sign ambiguity
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[0]) < 0:
            quats[i] = -quats[i]
    
    avg_quat = np.mean(quats, axis=0)
    avg_quat = avg_quat / np.linalg.norm(avg_quat)  # Normalize
    avg_rot = R.from_quat(avg_quat).as_matrix()
    
    # Average translation
    avg_trans = np.median(translations, axis=0)
    
    # Build average transformation
    avg_transform = np.eye(4)
    avg_transform[:3, :3] = avg_rot
    avg_transform[:3, 3] = avg_trans
    
    print(f"[ICP-6DOF] Final transform:")
    print(f"  Translation: [{avg_trans[0]:.4f}, {avg_trans[1]:.4f}, {avg_trans[2]:.4f}]m")
    
    # Convert rotation to Euler angles for readability
    euler = R.from_matrix(avg_rot).as_euler('xyz', degrees=True)
    print(f"  Rotation (XYZ Euler): [{euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}] deg")
    
    return avg_transform


def apply_full_transform_to_wrist(active_cams, transform):
    """
    Apply full 6-DOF transformation correction to all wrist camera transforms.
    
    Args:
        active_cams: Dictionary of active cameras
        transform: 4x4 correction transformation
    """
    print(f"[ICP-6DOF] Applying full 6-DOF transform to wrist camera...")
    
    for serial, cam in active_cams.items():
        if cam['type'] != 'wrist':
            continue
        
        # Apply transform to each world pose
        # New pose = correction @ old_pose (pre-multiply for world-frame correction)
        for i in range(len(cam['transforms'])):
            cam['transforms'][i] = transform @ cam['transforms'][i]
        
        print(f"  Updated {len(cam['transforms'])} wrist transforms")


def optimize_wrist_camera_icp_full(active_cams, cartesian_positions, config):
    """
    Main entry point for full 6-DOF wrist camera ICP optimization.
    
    This function optimizes all 6 degrees of freedom (XYZ translation + rotation)
    of the wrist camera and updates all transforms.
    
    Args:
        active_cams: Dictionary of active cameras
        cartesian_positions: Nx6 array of end-effector poses
        config: Configuration dictionary
        
    Returns:
        4x4 transformation that was applied
    """
    # Optimize full 6-DOF
    transform = optimize_wrist_full_6dof_icp(
        active_cams, config,
        num_frames_to_sample=config.get('icp_num_frames', 10)
    )
    
    # Check if transform is significant (more than 1mm translation or 0.1 deg rotation)
    from scipy.spatial.transform import Rotation as R
    
    trans_norm = np.linalg.norm(transform[:3, 3])
    rot_angle = np.abs(R.from_matrix(transform[:3, :3]).magnitude())  # radians
    rot_angle_deg = np.degrees(rot_angle)
    
    if trans_norm > 0.001 or rot_angle_deg > 0.1:
        apply_full_transform_to_wrist(active_cams, transform)
        print(f"[ICP-6DOF] Applied correction: trans={trans_norm*1000:.2f}mm, rot={rot_angle_deg:.2f}deg")
    else:
        print(f"[ICP-6DOF] Transform too small, not applying")
    
    return transform
