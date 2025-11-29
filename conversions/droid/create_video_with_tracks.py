"""
Video Generation with Point Cloud Tracks (No ICP).

This script generates videos showing reprojected point clouds from all cameras
without any ICP optimization. It uses the original camera transforms directly.

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


def generate_reprojection_videos(active_cams, transforms, num_frames, config):
    """
    Generate MP4 videos for each camera showing reprojected point clouds.
    
    This version does NOT use ICP - it uses the original transforms directly.
    """
    print("\n[VIDEO] Generating reprojection videos (no ICP)...")
    video_dir = config.get('video_output_path', 'point_clouds/videos')
    os.makedirs(video_dir, exist_ok=True)
    
    # Initialize Video Recorders
    recorders = {}
    for serial, cam in active_cams.items():
        w, h = cam['w'], cam['h']
        recorders[serial] = VideoRecorder(video_dir, serial, "reprojected", w, h)
    
    # Reset all cameras to frame 0
    for cam in active_cams.values():
        cam['zed'].set_svo_position(0)

    max_frames = min(num_frames, config.get('max_frames', 100))
    
    # Depth filtering parameters
    min_depth_wrist = config.get('min_depth_wrist', 0.01)
    max_depth_wrist = config.get('wrist_max_depth', 0.75)
    min_depth_ext = config.get('min_depth', 0.1)
    max_depth_ext = config.get('ext_max_depth', 2.0)
    
    for i in range(max_frames):
        if i % 10 == 0:
            print(f"  -> Processing Video Frame {i}/{max_frames}")
        
        # Collect frame data
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
            
            # Get Points with appropriate depth filtering
            if cam['type'] == 'wrist':
                xyz, rgb = get_filtered_cloud(
                    zed, cam['runtime'],
                    max_depth=max_depth_wrist,
                    min_depth=min_depth_wrist
                )
            else:
                xyz, rgb = get_filtered_cloud(
                    zed, cam['runtime'],
                    max_depth=max_depth_ext,
                    min_depth=min_depth_ext
                )
            
            frame_data[serial] = {
                'image': img_bgr,
                'points': xyz,
                'colors': rgb
            }

        # Build world cloud
        cloud_points = []
        cloud_colors = []
        
        for serial, data in frame_data.items():
            if data['points'] is None:
                continue
            
            cam_info = transforms[serial]
            T = None
            
            if cam_info['type'] == 'external':
                T = cam_info['world_T_cam']
            else:  # wrist
                if i < len(cam_info['transforms']):
                    T = cam_info['transforms'][i]
            
            if T is not None:
                cloud_points.append(transform_points(data['points'], T))
                cloud_colors.append(data['colors'])

        # Stack clouds
        if cloud_points:
            pts_world = np.vstack(cloud_points)
            cols_world = np.vstack(cloud_colors)
        else:
            pts_world = np.empty((0, 3))
            cols_world = np.empty((0, 3))
        
        # Project and write for each camera
        for serial, cam in active_cams.items():
            if serial not in frame_data:
                continue
            
            img = frame_data[serial]['image']
            if frame_data[serial]['points'] is None:
                continue
            
            K = cam['K']
            w, h = cam['w'], cam['h']
            
            # Get camera transform
            cam_info = transforms[serial]
            T_cam = None
            if cam_info['type'] == 'external':
                T_cam = cam_info['world_T_cam']
            elif i < len(cam_info['transforms']):
                T_cam = cam_info['transforms'][i]
                
            if T_cam is not None and len(pts_world) > 0:
                uv, cols = project_points_to_image(
                    pts_world, K, T_cam, w, h, colors=cols_world
                )
                img_out = draw_points_on_image(img, uv, colors=cols)
                recorders[serial].write_frame(img_out)

    # Cleanup
    for recorder in recorders.values():
        recorder.close()
    print("[VIDEO] Done.")


def main():
    # Load configuration
    with open('conversions/droid/config.yaml', 'r') as f:
        CONFIG = yaml.safe_load(f)
    
    print("=" * 60)
    print("DROID Full Fusion (Wrist + External) - No ICP")
    print("=" * 60)
    
    rr.init("droid_full_fusion", spawn=True)
    rrd_save_path = CONFIG['rrd_output_path']
    rrd_save_path = rrd_save_path.replace(".rrd", "")
    rrd_save_path = f"{rrd_save_path}_full_fusion.rrd" 
    rr.save(rrd_save_path)
    
    # Define Up-Axis for the World (Z-up is standard for Robotics)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # --- 1. Load Robot Data ---
    print("\n[1/4] Loading H5 Trajectory...")
    
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
    print("\n[2/4] Computing wrist camera trajectory...")
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
    
    # --- 3. Init Cameras ---
    print("\n[3/4] Initializing cameras...")
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

    print(f"[INFO] Total active cameras: {len(active_cams)}")

    # --- Generate Videos (No ICP) ---
    print("\n[4/4] Generating videos...")
    transforms_current = capture_transforms(active_cams)
    generate_reprojection_videos(active_cams, transforms_current, num_frames, CONFIG)

    # --- Rerun Render Loop ---
    print("\n[INFO] Logging to Rerun...")
    for cam in active_cams.values():
        cam['zed'].set_svo_position(0)

    max_frames = CONFIG.get("max_frames", 100)
    
    for i in range(min(max_frames, num_frames)):
        if i % 10 == 0:
            print(f"  -> Frame {i}")
        rr.set_time(timeline="frame_index", sequence=i)

        # Update Gripper
        T_base_ee = pose6_to_T(cartesian_positions[i])
        R_fix = R.from_euler('z', 90, degrees=True).as_matrix()
        T_base_ee[:3, :3] = T_base_ee[:3, :3] @ R_fix
        gripper_viz.update(T_base_ee, gripper_positions[i])

        for serial, cam in active_cams.items():
            zed = cam['zed']
            if zed.grab(cam['runtime']) != sl.ERROR_CODE.SUCCESS:
                continue

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
                
                xyz, rgb = get_filtered_cloud(
                    zed, cam['runtime'],
                    CONFIG['wrist_max_depth'],
                    CONFIG.get('min_depth_wrist', 0.01)
                )
                if xyz is None:
                    continue

                xyz_world = transform_points(xyz, T_wrist)

                rr.log(
                    "world/points/wrist_cam",
                    rr.Points3D(xyz_world, colors=rgb, radii=CONFIG['radii_size'])
                )

            else:  # External camera
                xyz, rgb = get_filtered_cloud(
                    zed, cam['runtime'],
                    CONFIG['ext_max_depth'],
                    CONFIG['min_depth']
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
                    rr.Points3D(xyz_world, colors=rgb, radii=CONFIG['radii_size'])
                )

    # Cleanup
    for c in active_cams.values():
        c['zed'].close()
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"[INFO] RRD saved to: {rrd_save_path}")
    video_dir = CONFIG.get('video_output_path', 'point_clouds/videos')
    print(f"[INFO] Videos saved to: {video_dir}")


if __name__ == "__main__":
    main()