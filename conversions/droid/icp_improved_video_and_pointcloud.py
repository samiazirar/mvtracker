"""
High-Quality ICP-based Wrist Camera Alignment and Video Generation.

This script optimizes the wrist camera pose relative to the gripper using
a multi-stage ICP pipeline for high-quality point cloud alignment.

Key features:
1. Full 6-DOF ICP optimization (not just Z-offset)
2. Multi-scale ICP with coarse-to-fine refinement
3. FPFH-based global registration for robust initial alignment
4. Statistical outlier removal for noise reduction
5. Optional colored ICP when color information is available
6. Excludes gripper points (< 15cm from camera) during ICP
7. Generates comparison videos:
   - videos_no_icp: Before ICP optimization
   - videos_icp: After ICP optimization

Usage:
    python conversions/droid/icp_improved_video_and_pointcloud.py
"""

import numpy as np
import os
import glob
import h5py
import yaml
import cv2
import copy
from scipy.spatial.transform import Rotation as R
import pyzed.sl as sl

from utils import (
    pose6_to_T,
    rvec_tvec_to_matrix,
    transform_points,
    compute_wrist_cam_offset,
    precompute_wrist_trajectory,
    external_cam_to_world,
    find_svo_for_camera,
    find_episode_data_by_date,
    get_zed_intrinsics,
    get_filtered_cloud,
    VideoRecorder,
    project_points_to_image,
    draw_points_on_image,
    # High-quality ICP functions
    optimize_wrist_camera_full_icp,
    run_full_icp_pipeline,
    apply_6dof_correction_to_wrist_transforms,
    apply_z_offset_to_wrist_transforms,
    preprocess_pointcloud,
    numpy_to_o3d_pointcloud,
)


def copy_transforms(active_cams):
    """
    Create a deep copy of all camera transforms.
    
    Args:
        active_cams: Dictionary of camera data
        
    Returns:
        Dictionary mapping serial -> transforms (copied)
    """
    copies = {}
    for serial, cam in active_cams.items():
        if cam['type'] == 'wrist':
            copies[serial] = {
                'type': 'wrist',
                'transforms': [T.copy() for T in cam['transforms']]
            }
        else:
            copies[serial] = {
                'type': 'external',
                'world_T_cam': cam['world_T_cam'].copy()
            }
    return copies


def collect_icp_data(active_cams, config):
    """
    Collect point cloud data from multiple frames for ICP optimization.
    
    This function gathers wrist camera points (excluding gripper) and 
    external camera points for multi-frame ICP alignment.
    
    Args:
        active_cams: Dictionary of active cameras
        config: Configuration dictionary
        
    Returns:
        List of frame data dictionaries for ICP optimization
    """
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
    num_icp_frames = config.get('icp_num_frames', 10)
    min_depth_icp = config.get('min_depth_wrist_icp', 0.15)  # Exclude gripper (15cm)
    max_depth_wrist = config.get('wrist_max_depth', 0.75)
    ext_max_depth = config.get('ext_max_depth', 1.5)
    min_depth_ext = config.get('min_depth', 0.1)
    
    # Reset cameras
    wrist_cam['zed'].set_svo_position(0)
    for cam in external_cams.values():
        cam['zed'].set_svo_position(0)
    
    total_frames = len(wrist_cam['transforms'])
    frame_indices = np.linspace(0, total_frames - 1, num_icp_frames, dtype=int)
    
    frames_data = []
    
    print(f"[ICP] Collecting {num_icp_frames} frames for Z-offset optimization...")
    print(f"[ICP] Excluding gripper points (< {min_depth_icp}m from camera)")
    
    for frame_idx in frame_indices:
        # Get wrist data
        wrist_cam['zed'].set_svo_position(frame_idx)
        if wrist_cam['zed'].grab(wrist_cam['runtime']) != sl.ERROR_CODE.SUCCESS:
            continue
        
        # Get wrist points (excluding gripper - min_depth = 15cm)
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
            'wrist_points_local': wrist_xyz,
            'wrist_transform': wrist_transform,
            'external_points_world': external_points_world
        })
    
    print(f"[ICP] Collected {len(frames_data)} valid frames")
    return frames_data


def run_high_quality_icp(active_cams, config):
    """
    Run high-quality ICP optimization for wrist camera alignment.
    
    Uses multi-stage ICP pipeline:
    1. Multi-frame point cloud accumulation
    2. Preprocessing with outlier removal
    3. FPFH-based global registration
    4. Multi-scale ICP refinement
    5. Optional colored ICP
    
    Args:
        active_cams: Dictionary of active cameras
        config: Configuration dictionary
        
    Returns:
        Tuple of (correction_transform, fitness_score)
    """
    print("\n" + "=" * 60)
    print("HIGH-QUALITY ICP OPTIMIZATION")
    print("=" * 60)
    
    # Find wrist and external cameras
    wrist_cam = None
    wrist_serial = None
    external_cams = {}
    
    for serial, cam in active_cams.items():
        if cam['type'] == 'wrist':
            wrist_cam = cam
            wrist_serial = serial
        else:
            external_cams[serial] = cam
    
    if wrist_cam is None:
        print("[ICP] Error: No wrist camera found")
        return np.eye(4), 0.0
    
    if len(external_cams) == 0:
        print("[ICP] Error: No external cameras found")
        return np.eye(4), 0.0
    
    # Collect point clouds from multiple frames
    frames_data = collect_icp_data(active_cams, config)
    
    if len(frames_data) == 0:
        print("[ICP] Error: No valid frames collected")
        return np.eye(4), 0.0
    
    # Accumulate all wrist and external points in world frame
    print("\n[ICP] Accumulating point clouds for alignment...")
    all_wrist_world = []
    all_wrist_colors = []
    all_external_world = []
    all_external_colors = []
    
    for frame in frames_data:
        wrist_local = frame['wrist_points_local']
        wrist_transform = frame['wrist_transform']
        external_world = frame['external_points_world']
        
        # Transform wrist points to world
        wrist_world = transform_points(wrist_local, wrist_transform)
        all_wrist_world.append(wrist_world)
        all_external_world.append(external_world)
    
    if not all_wrist_world or not all_external_world:
        print("[ICP] Error: No points accumulated")
        return np.eye(4), 0.0
    
    wrist_points = np.vstack(all_wrist_world)
    external_points = np.vstack(all_external_world)
    
    print(f"[ICP] Total wrist points: {len(wrist_points)}")
    print(f"[ICP] Total external points: {len(external_points)}")
    
    # Run full ICP pipeline
    voxel_size = config.get('icp_voxel_size', 0.01)
    
    icp_result = run_full_icp_pipeline(
        source_points=wrist_points,
        target_points=external_points,
        source_colors=None,  # Colors not collected in frames_data
        target_colors=None,
        voxel_size=voxel_size,
        use_global_registration=True,
        use_multiscale=True,
        use_colored_icp=False
    )
    
    if icp_result['success']:
        print(f"\n[ICP] SUCCESS: fitness={icp_result['fitness']:.4f}, rmse={icp_result['rmse']:.4f}")
        
        # Apply the correction to wrist transforms
        # The ICP gives us the correction needed to align wrist to external
        # We need to apply this as a post-multiplication to all wrist transforms
        correction = icp_result['transformation']
        
        wrist_cam['transforms'] = apply_6dof_correction_to_wrist_transforms(
            wrist_cam['transforms'],
            correction
        )
        
        return correction, icp_result['fitness']
    else:
        print(f"\n[ICP] WARNING: Alignment quality below threshold (fitness={icp_result['fitness']:.4f})")
        return icp_result['transformation'], icp_result['fitness']


def optimize_wrist_z(active_cams, config):
    """
    Optimize wrist camera pose using high-quality ICP.
    
    This is the main entry point for ICP optimization.
    Uses full 6-DOF optimization by default.
    
    Args:
        active_cams: Dictionary of active cameras
        config: Configuration dictionary
        
    Returns:
        fitness_score (for backward compatibility with scripts expecting z_offset)
    """
    # Use the new high-quality ICP pipeline
    correction, fitness = run_high_quality_icp(active_cams, config)
    
    # Return fitness for backward compatibility (old code expected z_offset)
    return fitness


def generate_reprojection_videos(
    active_cams, 
    transforms_no_icp, 
    transforms_icp, 
    num_frames, 
    config
):
    """
    Generate reprojected point cloud videos before and after ICP optimization.
    
    Creates two folders:
    - videos_no_icp: Reprojections using original transforms
    - videos_icp: Reprojections using ICP-optimized transforms
    
    Args:
        active_cams: Dictionary of active cameras
        transforms_no_icp: Transforms before ICP
        transforms_icp: Transforms after ICP  
        num_frames: Total number of frames
        config: Configuration dictionary
    """
    video_base_dir = config.get('video_output_path', 'point_clouds/videos')
    
    # Create output directories
    dir_no_icp = os.path.join(video_base_dir, 'videos_no_icp')
    dir_icp = os.path.join(video_base_dir, 'videos_icp')
    os.makedirs(dir_no_icp, exist_ok=True)
    os.makedirs(dir_icp, exist_ok=True)
    
    print(f"\n[VIDEO] Generating comparison videos...")
    print(f"[VIDEO] No ICP: {dir_no_icp}")
    print(f"[VIDEO] With ICP: {dir_icp}")
    
    # Initialize video recorders for each camera
    recorders = {}
    for serial, cam in active_cams.items():
        w, h = cam['w'], cam['h']
        recorders[serial] = {
            'no_icp': VideoRecorder(dir_no_icp, serial, "reprojected", w, h),
            'icp': VideoRecorder(dir_icp, serial, "reprojected", w, h)
        }
    
    # Reset cameras
    for cam in active_cams.values():
        cam['zed'].set_svo_position(0)
    
    max_frames = min(num_frames, config.get('max_frames', 50))
    
    # Parameters for point cloud filtering (exclude gripper for wrist cam rendering)
    min_depth_wrist = config.get('min_depth_wrist', 0.01)  # Include gripper for visualization
    max_depth_wrist = config.get('wrist_max_depth', 0.75)
    min_depth_ext = config.get('min_depth', 0.1)
    max_depth_ext = config.get('ext_max_depth', 1.5)
    
    for i in range(max_frames):
        if i % 10 == 0:
            print(f"  -> Processing frame {i}/{max_frames}")
        
        # Collect frame data from all cameras
        frame_data = {}
        
        for serial, cam in active_cams.items():
            cam['zed'].set_svo_position(i)
            if cam['zed'].grab(cam['runtime']) != sl.ERROR_CODE.SUCCESS:
                continue
            
            # Get image
            mat_img = sl.Mat()
            cam['zed'].retrieve_image(mat_img, sl.VIEW.LEFT)
            img_bgra = mat_img.get_data()
            img_bgr = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)
            
            # Get points (exclude gripper for wrist cam)
            if cam['type'] == 'wrist':
                # For rendering: use min_depth_wrist (may include gripper)
                # But for better alignment visualization, we still render everything
                xyz, rgb = get_filtered_cloud(
                    cam['zed'], cam['runtime'],
                    max_depth=max_depth_wrist,
                    min_depth=min_depth_wrist
                )
            else:
                xyz, rgb = get_filtered_cloud(
                    cam['zed'], cam['runtime'],
                    max_depth=max_depth_ext,
                    min_depth=min_depth_ext
                )
            
            frame_data[serial] = {
                'image': img_bgr,
                'points': xyz,
                'colors': rgb
            }
        
        # Build world point clouds for both versions (No ICP and ICP)
        def build_world_cloud(transforms_dict):
            cloud_points = []
            cloud_colors = []
            
            for serial, data in frame_data.items():
                if data['points'] is None or len(data['points']) == 0:
                    continue
                
                cam_info = transforms_dict[serial]
                
                if cam_info['type'] == 'wrist':
                    if i < len(cam_info['transforms']):
                        T = cam_info['transforms'][i]
                        pts_world = transform_points(data['points'], T)
                        cloud_points.append(pts_world)
                        cloud_colors.append(data['colors'])
                else:
                    T = cam_info['world_T_cam']
                    pts_world = transform_points(data['points'], T)
                    cloud_points.append(pts_world)
                    cloud_colors.append(data['colors'])
            
            if cloud_points:
                return np.vstack(cloud_points), np.vstack(cloud_colors)
            return np.empty((0, 3)), np.empty((0, 3))
        
        pts_world_no_icp, cols_world_no_icp = build_world_cloud(transforms_no_icp)
        pts_world_icp, cols_world_icp = build_world_cloud(transforms_icp)
        
        # Project and render for each camera
        for serial, cam in active_cams.items():
            if serial not in frame_data:
                continue
            
            img = frame_data[serial]['image']
            K = cam['K']
            w, h = cam['w'], cam['h']
            
            # Get camera transform for this frame
            def get_cam_transform(transforms_dict):
                cam_info = transforms_dict[serial]
                if cam_info['type'] == 'wrist':
                    if i < len(cam_info['transforms']):
                        return cam_info['transforms'][i]
                    return None
                else:
                    return cam_info['world_T_cam']
            
            # No ICP version
            T_no_icp = get_cam_transform(transforms_no_icp)
            if T_no_icp is not None and len(pts_world_no_icp) > 0:
                uv, cols = project_points_to_image(
                    pts_world_no_icp, K, T_no_icp, w, h, colors=cols_world_no_icp
                )
                img_out = draw_points_on_image(img.copy(), uv, colors=cols, point_size=1)
                recorders[serial]['no_icp'].write_frame(img_out)
            
            # ICP version
            T_icp = get_cam_transform(transforms_icp)
            if T_icp is not None and len(pts_world_icp) > 0:
                uv, cols = project_points_to_image(
                    pts_world_icp, K, T_icp, w, h, colors=cols_world_icp
                )
                img_out = draw_points_on_image(img.copy(), uv, colors=cols, point_size=1)
                recorders[serial]['icp'].write_frame(img_out)
    
    # Close all recorders
    for recs in recorders.values():
        recs['no_icp'].close()
        recs['icp'].close()
    
    print("[VIDEO] Done generating comparison videos.")


def main():
    """
    Main function to run ICP optimization and generate comparison videos.
    """
    # Load configuration
    config_path = 'conversions/droid/config.yaml'
    with open(config_path, 'r') as f:
        CONFIG = yaml.safe_load(f)
    
    print("=" * 60)
    print("HIGH-QUALITY ICP WRIST CAMERA ALIGNMENT")
    print("=" * 60)
    print("[INFO] Features:")
    print("  - Full 6-DOF transformation optimization")
    print("  - Multi-scale ICP (coarse-to-fine refinement)")
    print("  - FPFH-based global registration")
    print("  - Statistical outlier removal")
    print("  - Assumes gripper pose is CORRECT")
    print("  - Excludes gripper points (< 15cm) from ICP")
    print("=" * 60)
    
    # --- 1. Load Robot Data ---
    print("\n[INFO] Loading H5 Trajectory...")
    h5_file = h5py.File(CONFIG['h5_path'], 'r')
    cartesian_positions = h5_file['observation/robot_state/cartesian_position'][:]
    gripper_positions = h5_file['observation/robot_state/gripper_position'][:]
    h5_file.close()
    num_frames = len(cartesian_positions)
    print(f"[INFO] Loaded {num_frames} frames")
    
    # --- 2. Calculate Wrist Transforms ---
    wrist_cam_transforms = []
    wrist_serial = None
    T_ee_cam = None
    
    metadata_path = CONFIG.get('metadata_path')
    if metadata_path is None:
        episode_dir = os.path.dirname(CONFIG['h5_path'])
        metadata_files = glob.glob(os.path.join(episode_dir, "metadata_*.json"))
        if metadata_files:
            metadata_path = metadata_files[0]
    
    if metadata_path and os.path.exists(metadata_path):
        import json
        with open(metadata_path, 'r') as f:
            meta = json.load(f)
        wrist_serial = str(meta.get("wrist_cam_serial", ""))
        wrist_pose_t0 = meta.get("wrist_cam_extrinsics")
        
        if wrist_pose_t0:
            # Calculate constant offset from EE to camera
            T_ee_cam = compute_wrist_cam_offset(wrist_pose_t0, cartesian_positions[0])
            # Precompute all wrist camera poses
            wrist_cam_transforms = precompute_wrist_trajectory(cartesian_positions, T_ee_cam)
            print(f"[INFO] Wrist camera serial: {wrist_serial}")
    
    # --- 3. Init Cameras ---
    cameras = {}
    
    # A. External Cameras
    ext_data = find_episode_data_by_date(CONFIG['h5_path'], CONFIG['extrinsics_json_path'])
    if ext_data:
        for cam_id, transform_list in ext_data.items():
            if not cam_id.isdigit():
                continue
            svo = find_svo_for_camera(CONFIG['recordings_dir'], cam_id)
            if svo:
                cameras[cam_id] = {
                    "type": "external",
                    "svo": svo,
                    "world_T_cam": external_cam_to_world(transform_list)
                }
    
    # B. Wrist Camera
    if wrist_serial and len(wrist_cam_transforms) > 0:
        svo = find_svo_for_camera(CONFIG['recordings_dir'], wrist_serial)
        if svo:
            print(f"[INFO] Found Wrist Camera SVO: {wrist_serial}")
            cameras[wrist_serial] = {
                "type": "wrist",
                "svo": svo,
                "transforms": wrist_cam_transforms,
                "T_ee_cam": T_ee_cam
            }
    
    # Open ZED cameras
    active_cams = {}
    for serial, data in cameras.items():
        zed = sl.Camera()
        init = sl.InitParameters()
        init.set_from_svo_file(data['svo'])
        init.svo_real_time_mode = False
        init.coordinate_units = sl.UNIT.METER
        init.depth_mode = sl.DEPTH_MODE.NEURAL
        
        if zed.open(init) == sl.ERROR_CODE.SUCCESS:
            data['zed'] = zed
            data['runtime'] = sl.RuntimeParameters()
            data['K'], data['w'], data['h'] = get_zed_intrinsics(zed)
            active_cams[serial] = data
            print(f"[INFO] Opened camera {serial} ({data['type']})")
        else:
            print(f"[ERROR] Failed to open camera {serial}")
    
    print(f"[INFO] Active cameras: {len(active_cams)}")
    
    # --- 4. Capture transforms BEFORE ICP ---
    print("\n[INFO] Saving transforms before ICP...")
    transforms_no_icp = copy_transforms(active_cams)
    
    # --- 5. Run High-Quality ICP Optimization ---
    fitness = optimize_wrist_z(active_cams, CONFIG)
    
    # --- 6. Capture transforms AFTER ICP ---
    print("\n[INFO] Saving transforms after ICP...")
    transforms_icp = copy_transforms(active_cams)
    
    # --- 7. Generate Comparison Videos ---
    print("\n" + "=" * 60)
    print("Generating Comparison Videos")
    print("=" * 60)
    generate_reprojection_videos(
        active_cams, 
        transforms_no_icp, 
        transforms_icp, 
        num_frames, 
        CONFIG
    )
    
    # --- 8. Cleanup ---
    for cam in active_cams.values():
        cam['zed'].close()
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"[RESULT] ICP Fitness Score: {fitness:.4f}")
    print(f"[RESULT] Videos saved to:")
    video_base_dir = CONFIG.get('video_output_path', 'point_clouds/videos')
    print(f"  - {os.path.join(video_base_dir, 'videos_no_icp')}")
    print(f"  - {os.path.join(video_base_dir, 'videos_icp')}")


if __name__ == "__main__":
    main()
