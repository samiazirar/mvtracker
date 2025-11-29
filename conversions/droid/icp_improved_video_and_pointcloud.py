"""
Robust ICP-based Wrist Camera Alignment and Video Generation.

This script performs full 6-DOF ICP alignment of the wrist camera relative to
external cameras and generates comparison videos before and after alignment.

Key features:
1. Full 6-DOF ICP alignment (not just Z-offset)
2. Multi-scale ICP with coarse-to-fine refinement
3. Robust outlier rejection using statistical filtering
4. Gripper exclusion (points < 15cm from camera are filtered)
5. Generates two folders:
   - videos_no_icp: Videos before ICP optimization
   - videos_icp: Videos after ICP optimization

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
    # New ICP functions
    optimize_wrist_camera_full_icp,
)


def deep_copy_transforms(active_cams):
    """
    Create a deep copy of all camera transforms.
    
    Args:
        active_cams: Dictionary of camera data
        
    Returns:
        Dictionary mapping serial -> camera info with copied transforms
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


def generate_comparison_videos(
    active_cams,
    transforms_no_icp,
    transforms_icp,
    num_frames,
    config
):
    """
    Generate comparison videos showing reprojected point clouds 
    before and after ICP optimization.
    
    Creates two folders:
    - videos_no_icp: Reprojections using original transforms
    - videos_icp: Reprojections using ICP-optimized transforms
    
    Args:
        active_cams: Dictionary of active cameras with ZED objects
        transforms_no_icp: Transforms before ICP
        transforms_icp: Transforms after ICP  
        num_frames: Total number of frames in the trajectory
        config: Configuration dictionary
    """
    video_base_dir = config.get('video_output_path', 'point_clouds/videos')
    
    # Create output directories
    dir_no_icp = os.path.join(video_base_dir, 'videos_no_icp')
    dir_icp = os.path.join(video_base_dir, 'videos_icp')
    os.makedirs(dir_no_icp, exist_ok=True)
    os.makedirs(dir_icp, exist_ok=True)
    
    print(f"\n[VIDEO] Generating comparison videos...")
    print(f"[VIDEO] No ICP folder: {dir_no_icp}")
    print(f"[VIDEO] With ICP folder: {dir_icp}")
    
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
    
    max_frames = min(num_frames, config.get('max_frames', 100))
    
    # Depth filtering parameters
    min_depth_wrist = config.get('min_depth_wrist', 0.01)
    max_depth_wrist = config.get('wrist_max_depth', 0.75)
    min_depth_ext = config.get('min_depth', 0.1)
    max_depth_ext = config.get('ext_max_depth', 1.5)
    
    for frame_idx in range(max_frames):
        if frame_idx % 10 == 0:
            print(f"  -> Processing frame {frame_idx}/{max_frames}")
        
        # Collect frame data from all cameras
        frame_data = {}
        
        for serial, cam in active_cams.items():
            cam['zed'].set_svo_position(frame_idx)
            if cam['zed'].grab(cam['runtime']) != sl.ERROR_CODE.SUCCESS:
                continue
            
            # Get image
            mat_img = sl.Mat()
            cam['zed'].retrieve_image(mat_img, sl.VIEW.LEFT)
            img_bgra = mat_img.get_data()
            img_bgr = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)
            
            # Get points with appropriate depth filtering
            if cam['type'] == 'wrist':
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
        
        # Build world point clouds for both versions
        def build_world_cloud(transforms_dict):
            cloud_points = []
            cloud_colors = []
            
            for serial, data in frame_data.items():
                if data['points'] is None or len(data['points']) == 0:
                    continue
                
                cam_info = transforms_dict[serial]
                
                if cam_info['type'] == 'wrist':
                    if frame_idx < len(cam_info['transforms']):
                        T = cam_info['transforms'][frame_idx]
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
        
        pts_no_icp, cols_no_icp = build_world_cloud(transforms_no_icp)
        pts_icp, cols_icp = build_world_cloud(transforms_icp)
        
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
                    if frame_idx < len(cam_info['transforms']):
                        return cam_info['transforms'][frame_idx]
                    return None
                else:
                    return cam_info['world_T_cam']
            
            # Render No-ICP version
            T_no_icp = get_cam_transform(transforms_no_icp)
            if T_no_icp is not None and len(pts_no_icp) > 0:
                uv, cols = project_points_to_image(
                    pts_no_icp, K, T_no_icp, w, h, colors=cols_no_icp
                )
                img_out = draw_points_on_image(img.copy(), uv, colors=cols, point_size=1)
                recorders[serial]['no_icp'].write_frame(img_out)
            
            # Render ICP version
            T_icp = get_cam_transform(transforms_icp)
            if T_icp is not None and len(pts_icp) > 0:
                uv, cols = project_points_to_image(
                    pts_icp, K, T_icp, w, h, colors=cols_icp
                )
                img_out = draw_points_on_image(img.copy(), uv, colors=cols, point_size=1)
                recorders[serial]['icp'].write_frame(img_out)
    
    # Close all recorders
    for recs in recorders.values():
        recs['no_icp'].close()
        recs['icp'].close()
    
    print("[VIDEO] Comparison videos generated successfully.")


def main():
    """
    Main function to run ICP optimization and generate comparison videos.
    """
    # Load configuration
    config_path = 'conversions/droid/config.yaml'
    with open(config_path, 'r') as f:
        CONFIG = yaml.safe_load(f)
    
    print("=" * 70)
    print("Robust ICP Wrist Camera Alignment")
    print("=" * 70)
    print("[INFO] Full 6-DOF ICP alignment (not just Z-offset)")
    print("[INFO] Gripper points (< 15cm from camera) are excluded")
    print("[INFO] Uses multi-scale ICP with robust outlier rejection")
    print("=" * 70)
    
    # --- 1. Load Robot Trajectory Data ---
    print("\n[1/6] Loading H5 Trajectory...")
    h5_file = h5py.File(CONFIG['h5_path'], 'r')
    cartesian_positions = h5_file['observation/robot_state/cartesian_position'][:]
    gripper_positions = h5_file['observation/robot_state/gripper_position'][:]
    h5_file.close()
    num_frames = len(cartesian_positions)
    print(f"[INFO] Loaded {num_frames} frames from trajectory")
    
    # --- 2. Calculate Wrist Camera Transforms ---
    print("\n[2/6] Computing wrist camera trajectory...")
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
            T_ee_cam = compute_wrist_cam_offset(wrist_pose_t0, cartesian_positions[0])
            wrist_cam_transforms = precompute_wrist_trajectory(cartesian_positions, T_ee_cam)
            print(f"[INFO] Wrist camera serial: {wrist_serial}")
            print(f"[INFO] Computed {len(wrist_cam_transforms)} wrist camera poses")
    
    # --- 3. Initialize Cameras ---
    print("\n[3/6] Initializing cameras...")
    cameras = {}
    
    # External Cameras
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
                print(f"[INFO] Found external camera: {cam_id}")
    
    # Wrist Camera
    if wrist_serial and len(wrist_cam_transforms) > 0:
        svo = find_svo_for_camera(CONFIG['recordings_dir'], wrist_serial)
        if svo:
            print(f"[INFO] Found wrist camera SVO: {wrist_serial}")
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
    
    if len(active_cams) == 0:
        print("[ERROR] No cameras available!")
        return
    
    print(f"[INFO] Total active cameras: {len(active_cams)}")
    
    # --- 4. Save Transforms BEFORE ICP ---
    print("\n[4/6] Saving transforms before ICP optimization...")
    transforms_before_icp = deep_copy_transforms(active_cams)
    
    # --- 5. Run Full 6-DOF ICP Optimization ---
    print("\n[5/6] Running ICP optimization...")
    correction_transform, icp_fitness = optimize_wrist_camera_full_icp(active_cams, CONFIG)
    
    # Analyze the correction
    translation = correction_transform[:3, 3]
    rotation = R.from_matrix(correction_transform[:3, :3])
    euler_deg = rotation.as_euler('xyz', degrees=True)
    
    print(f"\n[ICP RESULT]")
    print(f"  Correction translation: [{translation[0]*1000:.1f}, {translation[1]*1000:.1f}, {translation[2]*1000:.1f}] mm")
    print(f"  Correction rotation: [{euler_deg[0]:.2f}, {euler_deg[1]:.2f}, {euler_deg[2]:.2f}] degrees")
    print(f"  ICP fitness: {icp_fitness:.4f}")
    
    # --- 6. Save Transforms AFTER ICP ---
    print("\n[6/6] Saving transforms after ICP optimization...")
    transforms_after_icp = deep_copy_transforms(active_cams)
    
    # --- 7. Generate Comparison Videos ---
    print("\n" + "=" * 70)
    print("Generating Comparison Videos")
    print("=" * 70)
    generate_comparison_videos(
        active_cams,
        transforms_before_icp,
        transforms_after_icp,
        num_frames,
        CONFIG
    )
    
    # --- 8. Cleanup ---
    for cam in active_cams.values():
        cam['zed'].close()
    
    # --- Summary ---
    video_base_dir = CONFIG.get('video_output_path', 'point_clouds/videos')
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"[RESULT] ICP Fitness Score: {icp_fitness:.4f}")
    print(f"[RESULT] Correction (translation): {np.linalg.norm(translation)*1000:.1f} mm")
    print(f"[RESULT] Correction (rotation): {np.linalg.norm(euler_deg):.2f} degrees")
    print(f"\n[OUTPUT] Videos saved to:")
    print(f"  - {os.path.join(video_base_dir, 'videos_no_icp')} (before ICP)")
    print(f"  - {os.path.join(video_base_dir, 'videos_icp')} (after ICP)")
    print("=" * 70)


if __name__ == "__main__":
    main()
