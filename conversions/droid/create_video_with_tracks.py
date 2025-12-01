""" Reproject the tracks only into the videos, and save full fusion RRD + tracks NPZ."""

import argparse
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
    draw_points_on_image,
    draw_points_on_image_fast,
    ContactSurfaceTracker,
    draw_track_trails_on_image,
)


def main():
    parser = argparse.ArgumentParser(description="Create videos with gripper tracks reprojected.")
    parser.add_argument(
        "--config",
        default="conversions/droid/config.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        CONFIG = yaml.safe_load(f)
    fps = CONFIG.get('fps', 30.0)
    tracks_npz_path = CONFIG.get("tracks_npz_path")
    use_precomputed_tracks = tracks_npz_path is not None and os.path.exists(tracks_npz_path)
    track_trail_length_video = CONFIG.get("track_trail_length_video", 10)
    
    print("=== DROID Full Fusion (Wrist + External) with Tracks + Reprojection ===")
    # Headless init to avoid GPU/viewer issues on servers
    rr.init("droid_full_fusion", spawn=False)
    rrd_save_path = CONFIG['rrd_output_path']
    rrd_save_path = rrd_save_path.replace(".rrd", "")
    rrd_save_path = f"{rrd_save_path}_full_fusion.rrd" 
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
    max_frames = CONFIG["max_frames"]
    actual_frames = min(max_frames, num_frames)
    track_trail_length = CONFIG.get("track_trail_length", 10)

    # Init Gripper Viz
    gripper_viz = GripperVisualizer()
    gripper_viz.init_rerun()
    # Track sources
    contact_tracker = None
    tracks_3d = None
    gripper_poses = []
    num_contact_pts = 0
    total_track_pts = 0
    track_colors_rgb = np.zeros((0, 3), dtype=np.uint8)

    if use_precomputed_tracks:
        loaded = np.load(tracks_npz_path)
        tracks_3d = loaded["tracks_3d"]
        # Align frame count with both config and loaded data
        actual_frames = min(actual_frames, tracks_3d.shape[0])
        num_contact_pts = int(loaded.get("num_points_per_finger", 0))
        if num_contact_pts == 0 and "contact_points_local" in loaded:
            num_contact_pts = len(loaded["contact_points_local"])
        if num_contact_pts == 0:
            num_contact_pts = tracks_3d.shape[1] // 2
        total_track_pts = tracks_3d.shape[1]
        # Ensure even split if possible
        if total_track_pts and num_contact_pts * 2 != total_track_pts:
            num_contact_pts = total_track_pts // 2
        track_colors_rgb = np.zeros((total_track_pts, 3), dtype=np.uint8)
        if total_track_pts > 0:
            track_colors_rgb[:num_contact_pts, :] = [51, 127, 255]
            track_colors_rgb[num_contact_pts:, :] = [51, 255, 127]
        print(f"[INFO] Loaded precomputed tracks from {tracks_npz_path} with {total_track_pts} points")
    else:
        contact_tracker = ContactSurfaceTracker(num_track_points=CONFIG.get('num_track_points', 24))
        num_contact_pts = len(contact_tracker.contact_points_local) if contact_tracker.contact_points_local is not None else 0
        total_track_pts = num_contact_pts * 2
        print(f"[INFO] Tracking {total_track_pts} contact points across both fingers")

        tracks_3d = np.zeros((actual_frames, total_track_pts, 3), dtype=np.float32)
        track_colors_rgb = np.zeros((total_track_pts, 3), dtype=np.uint8)
        if total_track_pts > 0:
            track_colors_rgb[:num_contact_pts, :] = [51, 127, 255]  # Blue-ish for left finger
            track_colors_rgb[num_contact_pts:, :] = [51, 255, 127]  # Green-ish for right finger

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
    if wrist_serial:
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

    config_tag = os.path.splitext(os.path.basename(args.config))[0]
    video_dir = os.path.join(CONFIG.get("video_output_path", "point_clouds/videos"), config_tag, "tracks_reprojection")
    os.makedirs(video_dir, exist_ok=True)
    recorders = {
        serial: VideoRecorder(video_dir, serial, "tracks", cam["w"], cam["h"], fps=fps, ext="avi", fourcc="MJPG")
        for serial, cam in active_cams.items()
    }

    # --- 4. Render Loop ---
    # Reset cameras for Rerun logging
    for cam in active_cams.values():
        cam['zed'].set_svo_position(0)

    print(f"[INFO] Processing {actual_frames} frames...")
    R_fix = R.from_euler('z', 90, degrees=True).as_matrix()

    for i in range(actual_frames):
        if i % 10 == 0: print(f"Frame {i}")
        rr.set_time(timeline="frame_index", sequence=i)

        # Update Gripper (use end-effector pose directly)
        T_base_ee = pose6_to_T(cartesian_positions[i])
        T_base_ee[:3, :3] = T_base_ee[:3, :3] @ R_fix
        gripper_viz.update(T_base_ee, gripper_positions[i])
        gripper_poses.append(T_base_ee.copy())

        # Track gripper contact points (either precomputed NPZ or on-the-fly)
        track_points_world = None
        if total_track_pts > 0:
            if use_precomputed_tracks:
                track_points_world = tracks_3d[i]
            else:
                pts_left, pts_right = contact_tracker.get_contact_points_world(T_base_ee, gripper_positions[i])
                if pts_left is not None:
                    tracks_3d[i, :num_contact_pts, :] = pts_left
                    tracks_3d[i, num_contact_pts:, :] = pts_right
                    track_points_world = np.vstack([pts_left, pts_right])

        tracks_window = None
        if total_track_pts > 0 and track_trail_length_video > 1:
            start_idx = max(0, i - track_trail_length_video + 1)
            tracks_window = tracks_3d[start_idx:i + 1]

        for serial, cam in active_cams.items():
            zed = cam['zed']
            if zed.grab(cam['runtime']) != sl.ERROR_CODE.SUCCESS: continue
            mat_img = sl.Mat()
            zed.retrieve_image(mat_img, sl.VIEW.LEFT)
            img_bgra = mat_img.get_data()
            frame = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)

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
                #TODO: are the tranformations only in rerun?..
                
                # 2. Get Local Points
                xyz, rgb = get_filtered_cloud(zed, cam['runtime'], CONFIG['wrist_max_depth'], CONFIG['min_depth_wrist'])
                if xyz is None: continue

                # 3. Transform Points to World
                xyz_world = transform_points(xyz, T_wrist)

                # 4. Log Points (in World Frame)
                rr.log(
                    "world/points/wrist_cam",
                    rr.Points3D(xyz_world, colors=rgb, radii=CONFIG['radii_size'])
                )

                # 5. Reproject points and tracks onto image for video
                uv_cloud, cols_cloud = project_points_to_image(
                    xyz_world, cam['K'], T_wrist, cam['w'], cam['h'], colors=rgb
                )
                overlay = draw_points_on_image_fast(frame, uv_cloud, colors=cols_cloud)
                if track_points_world is not None:
                    uv_tracks, cols_tracks = project_points_to_image(
                        track_points_world, cam['K'], T_wrist, cam['w'], cam['h'], colors=track_colors_rgb
                    )
                    overlay = draw_points_on_image(overlay, uv_tracks, colors=cols_tracks, radius=3, default_color=(0, 0, 255))
                if tracks_window is not None:
                    overlay = draw_track_trails_on_image(
                        overlay,
                        tracks_window,
                        cam['K'],
                        T_wrist,
                        cam['w'],
                        cam['h'],
                        track_colors_rgb,
                        min_depth=CONFIG.get('min_depth_wrist', 0.01)
                    )

                if serial in recorders:
                    recorders[serial].write_frame(overlay)

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
                
                uv_cloud, cols_cloud = project_points_to_image(
                    xyz_world, cam['K'], T, cam['w'], cam['h'], colors=rgb
                )
                overlay = draw_points_on_image_fast(frame, uv_cloud, colors=cols_cloud)
                if track_points_world is not None:
                    uv_tracks, cols_tracks = project_points_to_image(
                        track_points_world, cam['K'], T, cam['w'], cam['h'], colors=track_colors_rgb
                    )
                    overlay = draw_points_on_image(overlay, uv_tracks, colors=cols_tracks, radius=3, default_color=(0, 0, 255))
                if tracks_window is not None:
                    overlay = draw_track_trails_on_image(
                        overlay,
                        tracks_window,
                        cam['K'],
                        T,
                        cam['w'],
                        cam['h'],
                        track_colors_rgb,
                        min_depth=CONFIG.get('min_depth', 0.01)
                    )

                if serial in recorders:
                    recorders[serial].write_frame(overlay)

    # --- 5. Visualize and Save Tracks ---
    if total_track_pts > 0:
        track_colors = np.zeros((total_track_pts, 4), dtype=np.float32)
        track_colors[:num_contact_pts, :] = [0.2, 0.5, 1.0, 1.0]
        track_colors[num_contact_pts:, :] = [0.2, 1.0, 0.5, 1.0]
        for t in range(actual_frames):
            rr.set_time(timeline="frame_index", sequence=t)
            rr.log(
                "world/gripper_tracks/points",
                rr.Points3D(
                    positions=tracks_3d[t],
                    colors=(track_colors[:, :3] * 255).astype(np.uint8),
                    radii=0.003
                )
            )
            if t > 0:
                trail_len = min(t, track_trail_length)
                for n in range(total_track_pts):
                    trail_points = tracks_3d[max(0, t - trail_len):t + 1, n, :]
                    if len(trail_points) > 1:
                        segments = np.stack([trail_points[:-1], trail_points[1:]], axis=1)
                        color = track_colors[n, :3]
                        rr.log(
                            f"world/gripper_tracks/trails/track_{n:03d}",
                            rr.LineStrips3D(
                                strips=segments,
                                colors=[color] * len(segments),
                                radii=0.001
                            )
                        )

    tracks_save_path = rrd_save_path.replace(".rrd", "_gripper_tracks.npz")
    gripper_poses_array = np.stack(gripper_poses, axis=0) if len(gripper_poses) > 0 else np.empty((0, 4, 4))
    if use_precomputed_tracks:
        print(f"[INFO] Skipping track re-save (using precomputed file: {tracks_npz_path})")
    else:
        np.savez(
            tracks_save_path,
            tracks_3d=tracks_3d,
            contact_points_local=contact_tracker.contact_points_local,
            gripper_poses=gripper_poses_array,
            gripper_positions=gripper_positions[:actual_frames],
            cartesian_positions=cartesian_positions[:actual_frames],
            num_frames=actual_frames,
            num_points_per_finger=num_contact_pts,
            fps=fps,
        )

    # Cleanup
    for c in active_cams.values(): c['zed'].close()
    for rec in recorders.values(): rec.close()
    print("[SUCCESS] Done.")
    print(f"[INFO] RRD saved to: {rrd_save_path}")
    if use_precomputed_tracks:
        print(f"[INFO] Tracks loaded from: {tracks_npz_path}")
    else:
        print(f"[INFO] Tracks saved to: {tracks_save_path}")
    print(f"[INFO] Videos saved to: {video_dir}")


if __name__ == "__main__":
    main()
