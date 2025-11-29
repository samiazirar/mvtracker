import numpy as np
import os
import glob
import h5py
import yaml
import cv2
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
    VideoRecorder,
    project_points_to_image,
    draw_points_on_image
)
from utils.optimization import optimize_wrist_camera_icp, optimize_wrist_camera_icp_z_only


def capture_transforms(active_cams):
    """Capture the current state of camera transforms."""
    state = {}
    for serial, data in active_cams.items():
        entry = {"type": data["type"]}
        if data["type"] == "external":
            entry["world_T_cam"] = data["world_T_cam"].copy()
        elif data["type"] == "wrist":
            entry["transforms"] = [t.copy() for t in data["transforms"]]
            if "T_ee_cam" in data:
                entry["T_ee_cam"] = data["T_ee_cam"].copy()
        state[serial] = entry
    return state


def get_filtered_cloud_wrist(zed, runtime, max_depth=0.75, min_depth=0.15):
    """
    Get filtered point cloud from wrist camera, excluding gripper.
    Points closer than min_depth (15cm) are filtered out to exclude the gripper.
    """
    return get_filtered_cloud(zed, runtime, max_depth, min_depth)


def generate_videos_for_state(active_cams, transforms_state, num_frames, config, output_folder):
    """
    Generate MP4 videos for each camera showing reprojected point clouds.
    
    Args:
        active_cams: Dictionary of active cameras
        transforms_state: Dictionary of camera transforms to use
        num_frames: Total number of frames available
        config: Configuration dictionary
        output_folder: Folder name for output videos (e.g., "videos_no_icp")
    """
    print(f"\n[VIDEO] Generating videos to {output_folder}...")
    video_dir = os.path.join(config.get('video_output_path', 'point_clouds/videos'), output_folder)
    os.makedirs(video_dir, exist_ok=True)
    
    # 1. Initialize Video Recorders
    recorders = {}
    for serial, cam in active_cams.items():
        w, h = cam['w'], cam['h']
        recorders[serial] = VideoRecorder(video_dir, serial, "reprojection", w, h)

    # 2. Reset all cameras to frame 0
    for cam in active_cams.values():
        cam['zed'].set_svo_position(0)

    max_frames = min(num_frames, config.get('max_frames', 50))
    
    for i in range(max_frames):
        if i % 10 == 0:
            print(f"  -> Processing Video Frame {i}/{max_frames}")
        
        # A. Grab Data for this frame
        frame_data = {}  # serial -> {image, points_local, rgb}
        
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
            
            # Get Points (use min_depth_wrist for wrist camera rendering - includes gripper)
            if cam['type'] == 'wrist':
                xyz, rgb = get_filtered_cloud_wrist(
                    zed, cam['runtime'],
                    config.get('wrist_max_depth', 0.75),
                    config.get('min_depth_wrist', 0.01)  # Low threshold to include gripper
                )
            else:
                xyz, rgb = get_filtered_cloud(
                    zed, cam['runtime'],
                    config.get('ext_max_depth', 1.5),
                    config.get('min_depth', 0.1)
                )
            
            frame_data[serial] = {
                'image': img_bgr,
                'points': xyz,
                'colors': rgb
            }

        # B. Build World Cloud
        cloud_world = []
        colors_world = []
        
        for serial, data in frame_data.items():
            if data['points'] is None:
                continue
            
            # Get Transform
            T_cam = None
            if transforms_state[serial]['type'] == 'external':
                T_cam = transforms_state[serial]['world_T_cam']
            else:  # wrist
                if i < len(transforms_state[serial]['transforms']):
                    T_cam = transforms_state[serial]['transforms'][i]
            
            if T_cam is not None:
                cloud_world.append(transform_points(data['points'], T_cam))
                colors_world.append(data['colors'])

        # Stack clouds
        pts_world = np.vstack(cloud_world) if cloud_world else np.empty((0, 3))
        cols_world = np.vstack(colors_world) if colors_world else np.empty((0, 3))
        
        # C. Project and Write for each camera
        for serial, cam in active_cams.items():
            if serial not in frame_data:
                continue
            
            img = frame_data[serial]['image']
            if frame_data[serial]['points'] is None:
                recorders[serial].write_frame(img)
                continue
            
            K = cam['K']
            w, h = cam['w'], cam['h']
            
            # Get camera transform
            T_cam = None
            if transforms_state[serial]['type'] == 'external':
                T_cam = transforms_state[serial]['world_T_cam']
            elif i < len(transforms_state[serial]['transforms']):
                T_cam = transforms_state[serial]['transforms'][i]
            
            if T_cam is not None:
                # Project GLOBAL world points back to this camera
                uv, cols = project_points_to_image(pts_world, K, T_cam, w, h, colors=cols_world)
                img_out = draw_points_on_image(img, uv, colors=cols)
                recorders[serial].write_frame(img_out)
            else:
                recorders[serial].write_frame(img)

    # 3. Cleanup
    for rec in recorders.values():
        rec.close()
    
    print(f"[VIDEO] Done. Videos saved to {video_dir}")

def main():
    # Load configuration
    with open('conversions/droid/config.yaml', 'r') as f:
        CONFIG = yaml.safe_load(f)
    
    print("=== DROID Full Fusion (Wrist + External) ===")
    rr.init("droid_full_fusion", spawn=True)
    rrd_save_path = CONFIG['rrd_output_path']
    rrd_save_path = rrd_save_path.replace(".rrd", "")
    rrd_save_path = f"{rrd_save_path}_wrist_icp.rrd" 
    rr.save(rrd_save_path)
    
    # Define Up-Axis for the World (Z-up is standard for Robotics)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # --- 1. Load Robot Data ---
    print("[INFO] Loading H5 Trajectory...")
    
    h5_file = h5py.File(CONFIG['h5_path'], 'r')
    cartesian_positions = h5_file['observation/robot_state/cartesian_position'][:]
    gripper_positions = h5_file['observation/robot_state/gripper_position'][:]
    h5_file.close()
    num_frames = len(cartesian_positions)
    print(f"[INFO] Loaded {num_frames} frames")

    # NOTE: Gripper visualization excluded per user request
    # gripper_viz = GripperVisualizer()
    # gripper_viz.init_rerun()

    # --- 2. Calculate Wrist Transforms ---
    wrist_cam_transforms = []
    wrist_serial = None
    T_ee_cam = None
    
    metadata_path = CONFIG['metadata_path']
    if metadata_path is None:
        episode_dir = os.path.dirname(CONFIG['h5_path'])
        metadata_files = glob.glob(os.path.join(episode_dir, "metadata_*.json"))
        if metadata_files: metadata_path = metadata_files[0]

    if metadata_path and os.path.exists(metadata_path):
        import json
        with open(metadata_path, 'r') as f: meta = json.load(f)
        wrist_serial = str(meta.get("wrist_cam_serial", ""))
        wrist_pose_t0 = meta.get("wrist_cam_extrinsics")

        if wrist_pose_t0:
            # Calculate constant offset
            T_ee_cam = compute_wrist_cam_offset(wrist_pose_t0, cartesian_positions[0])
            
            # Precompute all wrist camera poses
            wrist_cam_transforms = precompute_wrist_trajectory(cartesian_positions, T_ee_cam)
            print(f"[INFO] Wrist camera serial: {wrist_serial}")
    
    # --- 3. Init Cameras ---
    cameras = {} # Holds all cameras (Ext + Wrist)
    
    # A. External Cameras
    ext_data = find_episode_data_by_date(CONFIG['h5_path'], CONFIG['extrinsics_json_path'])
    if ext_data:
        for cam_id, transform_list in ext_data.items():
            if not cam_id.isdigit(): continue
            svo = find_svo_for_camera(CONFIG['recordings_dir'], cam_id)
            if svo:
                cameras[cam_id] = {
                    "type": "external",
                    "svo": svo,
                    "world_T_cam": external_cam_to_world(transform_list)
                }

    # B. Wrist Camera
    if wrist_serial and T_ee_cam is not None:
        svo = find_svo_for_camera(CONFIG['recordings_dir'], wrist_serial)
        if svo:
            print(f"[INFO] Found Wrist Camera SVO: {wrist_serial}")
            cameras[wrist_serial] = {
                "type": "wrist",
                "svo": svo,
                "transforms": wrist_cam_transforms, # List of 4x4 matrices
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
        else:
            print(f"[ERROR] Failed to open {serial}")

    # ====================================================
    # WRIST CAMERA ICP OPTIMIZATION PIPELINE
    # ====================================================
    
    # Capture BEFORE state (original transforms)
    print("\n[STEP 1] Capturing initial transforms (no ICP)...")
    transforms_before = capture_transforms(active_cams)
    
    # Generate "no ICP" videos
    print("\n[STEP 2] Generating videos WITHOUT ICP optimization...")
    generate_videos_for_state(active_cams, transforms_before, num_frames, CONFIG, "videos_no_icp")

    # ====================================================
    # Z-ONLY ICP OPTIMIZATION
    # ====================================================
    
    # Run Z-only ICP Optimization
    print("\n[STEP 3] Running ICP optimization for wrist camera (Z-only)...")
    if wrist_serial and wrist_serial in active_cams:
        z_offset = optimize_wrist_camera_icp_z_only(active_cams, cartesian_positions, CONFIG)
        print(f"[ICP-Z] Applied Z offset: {z_offset:.4f}m")
    else:
        print("[ICP-Z] No wrist camera found, skipping ICP optimization")
    
    # Capture Z-only state
    print("\n[STEP 4] Capturing Z-only ICP transforms...")
    transforms_z_only = capture_transforms(active_cams)
    
    # Generate "Z-only ICP" videos
    print("\n[STEP 5] Generating videos WITH Z-only ICP optimization...")
    generate_videos_for_state(active_cams, transforms_z_only, num_frames, CONFIG, "videos_icp_z")

    # ====================================================
    # FULL 3D ICP OPTIMIZATION (X, Y, Z)
    # ====================================================
    
    # Reset wrist transforms to original state before applying 3D ICP
    print("\n[STEP 6] Resetting wrist transforms for 3D ICP...")
    for serial, cam in active_cams.items():
        if cam['type'] == 'wrist' and serial in transforms_before:
            cam['transforms'] = [t.copy() for t in transforms_before[serial]['transforms']]
            if 'T_ee_cam' in transforms_before[serial]:
                cam['T_ee_cam'] = transforms_before[serial]['T_ee_cam'].copy()
    
    # Run full 3D ICP Optimization
    print("\n[STEP 7] Running ICP optimization for wrist camera (full 3D: X, Y, Z)...")
    if wrist_serial and wrist_serial in active_cams:
        xyz_offset = optimize_wrist_camera_icp(active_cams, cartesian_positions, CONFIG)
        print(f"[ICP-3D] Applied 3D offset: X={xyz_offset[0]:.4f}m, Y={xyz_offset[1]:.4f}m, Z={xyz_offset[2]:.4f}m")
    else:
        print("[ICP-3D] No wrist camera found, skipping ICP optimization")
    
    # Capture 3D ICP state
    print("\n[STEP 8] Capturing 3D ICP transforms...")
    transforms_3d = capture_transforms(active_cams)
    
    # Generate "3D ICP" videos
    print("\n[STEP 9] Generating videos WITH 3D ICP optimization...")
    generate_videos_for_state(active_cams, transforms_3d, num_frames, CONFIG, "videos_icp_xyz")

    # ====================================================

    # --- 4. Render Loop ---
    # Reset cameras for Rerun logging
    for cam in active_cams.values():
        cam['zed'].set_svo_position(0)

    max_frames = CONFIG["max_frames"]
    print(f"\n[STEP 10] Logging to Rerun ({min(max_frames, num_frames)} frames)...")
    
    for i in range(min(max_frames, num_frames)):
        if i % 10 == 0: print(f"Frame {i}")
        rr.set_time(timeline="frame_index", sequence=i)

        # NOTE: Gripper visualization excluded per user request
        # T_base_ee = pose6_to_T(cartesian_positions[i])
        # R_fix = R.from_euler('z', 90, degrees=True).as_matrix()
        # T_base_ee[:3, :3] = T_base_ee[:3, :3] @ R_fix
        # gripper_viz.update(T_base_ee, gripper_positions[i])

        for serial, cam in active_cams.items():
            zed = cam['zed']
            if zed.grab(cam['runtime']) != sl.ERROR_CODE.SUCCESS: continue

            # -- WRIST CAMERA LOGIC --
            if cam['type'] == "wrist":
                if i >= len(cam['transforms']): continue
                
                # 1. Update the transform of the wrist camera in the World
                T_wrist = cam['transforms'][i]

                rr.log(
                    "world/wrist_cam",
                    rr.Transform3D(
                        translation=T_wrist[:3, 3],
                        mat3x3=T_wrist[:3, :3],
                        axis_length=0.1
                    )
                )

                # Log Pinhole
                rr.log(
                    "world/wrist_cam/pinhole",
                    rr.Pinhole(
                        image_from_camera=cam['K'],
                        width=cam['w'],
                        height=cam['h']
                    )
                )
                
                # 2. Get Local Points (use min_depth_wrist for rendering - includes gripper)
                xyz, rgb = get_filtered_cloud_wrist(zed, cam['runtime'], 
                                                   CONFIG.get('wrist_max_depth', 0.75), 
                                                   CONFIG.get('min_depth_wrist', 0.01))
                if xyz is None: continue

                # 3. Transform Points to World
                xyz_world = transform_points(xyz, T_wrist)

                # 4. Log Points (in World Frame)
                rr.log(
                    "world/points/wrist_cam",
                    rr.Points3D(xyz_world, colors=rgb, radii=CONFIG['radii_size'])
                )

            # -- EXTERNAL CAMERA LOGIC --
            else:
                # 1. Get Local Points
                xyz, rgb = get_filtered_cloud(zed, cam['runtime'], CONFIG['ext_max_depth'], CONFIG['min_depth'])
                if xyz is None: continue

                # 2. Transform to World (External cams are static, so we just do the math once per frame)
                T = cam['world_T_cam']
                
                # Log Transform
                rr.log(
                    f"world/external_cams/{serial}",
                    rr.Transform3D(
                        translation=T[:3, 3],
                        mat3x3=T[:3, :3],
                        axis_length=0.1
                    )
                )

                # Log Pinhole
                rr.log(
                    f"world/external_cams/{serial}/pinhole",
                    rr.Pinhole(
                        image_from_camera=cam['K'],
                        width=cam['w'],
                        height=cam['h']
                    )
                )

                # 3. Transform Points to World
                xyz_world = transform_points(xyz, T)

                # 4. Log Points (in World Frame)
                rr.log(
                    f"world/points/external_cams/{serial}",
                    rr.Points3D(xyz_world, colors=rgb, radii=CONFIG['radii_size'])
                )

    # Cleanup
    for c in active_cams.values(): c['zed'].close()
    
    print("\n" + "=" * 60)
    print("[SUCCESS] Done.")
    print(f"[INFO] RRD saved to: {rrd_save_path}")
    print(f"[INFO] Videos saved to:")
    print(f"       - {CONFIG.get('video_output_path', 'point_clouds/videos')}/videos_no_icp/   (no optimization)")
    print(f"       - {CONFIG.get('video_output_path', 'point_clouds/videos')}/videos_icp_z/   (Z-only ICP)")
    print(f"       - {CONFIG.get('video_output_path', 'point_clouds/videos')}/videos_icp_xyz/ (full 3D ICP)")
    print("=" * 60)


if __name__ == "__main__":
    main()