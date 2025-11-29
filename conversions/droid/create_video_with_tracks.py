"""Create video with point cloud tracks.

This script generates videos showing reprojected point clouds from multiple
cameras. It does NOT use ICP optimization - use icp_improved_video_and_pointcloud.py
instead if you need ICP alignment.

Key features:
1. Uses original camera calibrations without ICP refinement
2. Excludes gripper region (< 15cm) from wrist camera rendering
3. Generates videos for each camera view
4. Supports Rerun visualization for 3D inspection

Usage:
    python conversions/droid/create_video_with_tracks.py
"""

import numpy as np
import os
import glob
import h5py
import yaml
import cv2
from scipy.spatial.transform import Rotation as R
import rerun as rr
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
    GripperVisualizer,
    VideoRecorder,
    project_points_to_image,
    draw_points_on_image
)


def capture_transforms(active_cams):
    """Capture the current state of camera transforms."""
    state = {}
    for serial, data in active_cams.items():
        entry = {"type": data["type"]}
        if data["type"] == "external":
            entry["world_T_cam"] = data["world_T_cam"].copy()
        elif data["type"] == "wrist":
            entry["transforms"] = [t.copy() for t in data["transforms"]]
        state[serial] = entry
    return state


def generate_videos_no_icp(active_cams, num_frames, config):
    """
    Generate MP4 videos for each camera showing reprojected point clouds.
    
    This version does NOT use ICP optimization - it uses the original
    camera calibrations directly.
    
    Args:
        active_cams: Dictionary of active cameras
        num_frames: Total number of frames
        config: Configuration dictionary
    """
    print("\n[VIDEO] Generating videos (No ICP optimization)...")
    video_dir = config.get('video_output_path', 'point_clouds/videos')
    
    # Create output directory for no-ICP videos
    no_icp_dir = os.path.join(video_dir, 'videos_no_icp')
    os.makedirs(no_icp_dir, exist_ok=True)
    
    # Initialize Video Recorders
    recorders = {}
    for serial, cam in active_cams.items():
        w, h = cam['w'], cam['h']
        recorders[serial] = VideoRecorder(no_icp_dir, serial, "reprojected", w, h)
    
    # Reset all cameras to frame 0
    for cam in active_cams.values():
        cam['zed'].set_svo_position(0)
    
    max_frames = min(num_frames, config.get('max_frames', 50))
    
    # Parameters for filtering
    min_depth_wrist = config.get('min_depth_wrist_icp', 0.15)  # Exclude gripper
    max_depth_wrist = config.get('wrist_max_depth', 0.75)
    min_depth_ext = config.get('min_depth', 0.1)
    max_depth_ext = config.get('ext_max_depth', 1.5)
    
    for i in range(max_frames):
        if i % 10 == 0:
            print(f"  -> Processing Video Frame {i}/{max_frames}")
        
        # Grab Data for this frame
        frame_data = {}
        
        for serial, cam in active_cams.items():
            zed = cam['zed']
            zed.set_svo_position(i)
            if zed.grab(cam['runtime']) != sl.ERROR_CODE.SUCCESS:
                continue
            
            # Get Image
            mat_img = sl.Mat()
            zed.retrieve_image(mat_img, sl.VIEW.LEFT)
            img_bgra = mat_img.get_data()
            img_bgr = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)
            
            # Get Points - exclude gripper for wrist camera
            if cam['type'] == 'wrist':
                # Exclude gripper region (< 15cm)
                xyz, rgb = get_filtered_cloud(
                    zed, cam['runtime'],
                    max_depth_wrist,
                    min_depth_wrist  # 15cm to exclude gripper
                )
            else:
                xyz, rgb = get_filtered_cloud(
                    zed, cam['runtime'],
                    max_depth_ext,
                    min_depth_ext
                )
            
            frame_data[serial] = {
                'image': img_bgr,
                'points': xyz,
                'colors': rgb
            }
        
        # Build World Cloud
        cloud_points = []
        cloud_colors = []
        
        for serial, data in frame_data.items():
            if data['points'] is None:
                continue
            
            cam = active_cams[serial]
            
            if cam['type'] == 'external':
                T = cam['world_T_cam']
            else:  # wrist
                if i < len(cam['transforms']):
                    T = cam['transforms'][i]
                else:
                    continue
            
            pts_world = transform_points(data['points'], T)
            cloud_points.append(pts_world)
            cloud_colors.append(data['colors'])
        
        # Stack clouds
        if cloud_points:
            pts_world = np.vstack(cloud_points)
            cols_world = np.vstack(cloud_colors)
        else:
            pts_world = np.empty((0, 3))
            cols_world = np.empty((0, 3))
        
        # Project and Write to each camera
        for serial, cam in active_cams.items():
            if serial not in frame_data:
                continue
            
            img = frame_data[serial]['image']
            if img is None:
                continue
            
            K = cam['K']
            w, h = cam['w'], cam['h']
            
            # Get camera transform
            if cam['type'] == 'external':
                T_cam = cam['world_T_cam']
            elif i < len(cam['transforms']):
                T_cam = cam['transforms'][i]
            else:
                continue
            
            if len(pts_world) > 0:
                uv, cols = project_points_to_image(
                    pts_world, K, T_cam, w, h, colors=cols_world
                )
                img_out = draw_points_on_image(img.copy(), uv, colors=cols)
            else:
                img_out = img.copy()
            
            recorders[serial].write_frame(img_out)
    
    # Cleanup
    for rec in recorders.values():
        rec.close()
    
    print(f"[VIDEO] Videos saved to: {no_icp_dir}")


def main():
    """Main function - creates videos without ICP optimization."""
    
    # Load configuration
    with open('conversions/droid/config.yaml', 'r') as f:
        CONFIG = yaml.safe_load(f)
    
    print("=" * 60)
    print("Create Video With Tracks (No ICP)")
    print("=" * 60)
    print("[INFO] This script does NOT use ICP optimization")
    print("[INFO] For ICP-optimized videos, use icp_improved_video_and_pointcloud.py")
    print("[INFO] Excluding gripper region (< 15cm from wrist camera)")
    print("=" * 60)
    
    # Initialize Rerun
    rr.init("droid_video_tracks", spawn=True)
    rrd_save_path = CONFIG['rrd_output_path']
    rrd_save_path = rrd_save_path.replace(".rrd", "")
    rrd_save_path = f"{rrd_save_path}_video_tracks.rrd"
    rr.save(rrd_save_path)
    
    # Define Up-Axis for the World (Z-up is standard for Robotics)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    
    # --- 1. Load Robot Data ---
    print("\n[INFO] Loading H5 Trajectory...")
    
    h5_file = h5py.File(CONFIG['h5_path'], 'r')
    cartesian_positions = h5_file['observation/robot_state/cartesian_position'][:]
    gripper_positions = h5_file['observation/robot_state/gripper_position'][:]
    h5_file.close()
    num_frames = len(cartesian_positions)
    print(f"[INFO] Loaded {num_frames} frames")
    
    # Init Gripper Viz
    gripper_viz = GripperVisualizer()
    gripper_viz.init_rerun()
    
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
            # Calculate constant offset
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
        else:
            print(f"[WARN] Wrist SVO not found for serial {wrist_serial}")
    
    # Open ZEDs
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
            print(f"[ERROR] Failed to open {serial}")
    
    print(f"[INFO] Active cameras: {len(active_cams)}")
    
    # --- Generate Videos (No ICP) ---
    generate_videos_no_icp(active_cams, num_frames, CONFIG)
    
    # --- 4. Render Loop for Rerun ---
    print("\n[INFO] Generating Rerun visualization...")
    
    # Reset cameras for Rerun logging
    for cam in active_cams.values():
        cam['zed'].set_svo_position(0)
    
    max_frames = CONFIG.get("max_frames", 50)
    print(f"[INFO] Processing {min(max_frames, num_frames)} frames...")
    
    # Parameters for filtering
    min_depth_wrist = CONFIG.get('min_depth_wrist_icp', 0.15)  # Exclude gripper
    max_depth_wrist = CONFIG.get('wrist_max_depth', 0.75)
    min_depth_ext = CONFIG.get('min_depth', 0.1)
    max_depth_ext = CONFIG.get('ext_max_depth', 1.5)
    
    for i in range(min(max_frames, num_frames)):
        if i % 10 == 0:
            print(f"Frame {i}")
        rr.set_time(timeline="frame_index", sequence=i)
        
        # Update Gripper (use end-effector pose directly)
        T_base_ee = pose6_to_T(cartesian_positions[i])
        # Rotate by 90 degrees to align
        R_fix = R.from_euler('z', 90, degrees=True).as_matrix()
        T_base_ee[:3, :3] = T_base_ee[:3, :3] @ R_fix
        gripper_viz.update(T_base_ee, gripper_positions[i])
        
        for serial, cam in active_cams.items():
            zed = cam['zed']
            if zed.grab(cam['runtime']) != sl.ERROR_CODE.SUCCESS:
                continue
            
            # -- WRIST CAMERA LOGIC --
            if cam['type'] == "wrist":
                if i >= len(cam['transforms']):
                    continue
                
                T_wrist = cam['transforms'][i]
                
                rr.log(
                    "world/wrist_cam",
                    rr.Transform3D(
                        translation=T_wrist[:3, 3],
                        mat3x3=T_wrist[:3, :3],
                        axis_length=0.1
                    )
                )
                
                rr.log(
                    "world/wrist_cam/pinhole",
                    rr.Pinhole(
                        image_from_camera=cam['K'],
                        width=cam['w'],
                        height=cam['h']
                    )
                )
                
                # Get Local Points - EXCLUDE GRIPPER
                xyz, rgb = get_filtered_cloud(
                    zed, cam['runtime'],
                    max_depth_wrist,
                    min_depth_wrist  # 15cm to exclude gripper
                )
                if xyz is None:
                    continue
                
                # Transform Points to World
                xyz_world = transform_points(xyz, T_wrist)
                
                # Log Points (in World Frame)
                rr.log(
                    "world/points/wrist_cam",
                    rr.Points3D(xyz_world, colors=rgb, radii=CONFIG.get('radii_size', 0.001))
                )
            
            # -- EXTERNAL CAMERA LOGIC --
            else:
                xyz, rgb = get_filtered_cloud(
                    zed, cam['runtime'],
                    max_depth_ext,
                    min_depth_ext
                )
                if xyz is None:
                    continue
                
                T = cam['world_T_cam']
                
                rr.log(
                    f"world/external_cams/{serial}",
                    rr.Transform3D(
                        translation=T[:3, 3],
                        mat3x3=T[:3, :3],
                        axis_length=0.1
                    )
                )
                
                rr.log(
                    f"world/external_cams/{serial}/pinhole",
                    rr.Pinhole(
                        image_from_camera=cam['K'],
                        width=cam['w'],
                        height=cam['h']
                    )
                )
                
                xyz_world = transform_points(xyz, T)
                
                rr.log(
                    f"world/points/external_cams/{serial}",
                    rr.Points3D(xyz_world, colors=rgb, radii=CONFIG.get('radii_size', 0.001))
                )
    
    # Cleanup
    for c in active_cams.values():
        c['zed'].close()
    
    print("\n" + "=" * 60)
    print("[SUCCESS] Done.")
    print(f"[INFO] Videos saved to: {os.path.join(CONFIG.get('video_output_path', 'point_clouds/videos'), 'videos_no_icp')}")
    print(f"[INFO] RRD saved to: {rrd_save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()