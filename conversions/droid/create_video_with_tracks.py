""" Reproject the tracks only into the videos, and save full fusion RRD + tracks NPZ."""

import numpy as np
import os
import glob
import h5py
import yaml
import cv2
from scipy.spatial.transform import Rotation as R
import rerun as rr
import pyzed.sl as sl
import trimesh

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
)


class ContactSurfaceTracker:
    """Sample and track contact surface points on the gripper fingers."""

    def __init__(self, num_track_points=24):
        base_path = "/workspace/third_party/robotiq_arg85_description/meshes"
        mesh_file = os.path.join(base_path, "inner_finger_fine.STL")
        self.num_track_points = num_track_points
        self.contact_points_local = None

        if os.path.exists(mesh_file):
            finger_mesh = trimesh.load(mesh_file)
            contact_mesh = self._extract_contact_surface(finger_mesh)
            if contact_mesh is not None:
                self.contact_points_local = self._sample_contact_points(contact_mesh, num_track_points)
        else:
            print(f"[WARN] Mesh not found for contact sampling: {mesh_file}")

    def _extract_contact_surface(self, mesh):
        """Extract the inner rubber contact pad (faces pointing -Y)."""
        vertices = mesh.vertices
        contact_face_indices = []
        for i, face in enumerate(mesh.faces):
            face_verts = vertices[face]
            y_min = face_verts[:, 1].min()
            if y_min < -0.0095:
                contact_face_indices.append(i)

        if len(contact_face_indices) == 0:
            print("[WARN] Could not extract contact surface for tracking")
            return None

        contact_submesh = mesh.submesh([contact_face_indices], only_watertight=False)[0]
        print(f"[INFO] Extracted contact pad: {len(contact_submesh.vertices)} verts, {len(contact_submesh.faces)} faces")
        return contact_submesh

    def _sample_contact_points(self, mesh, num_points):
        """Sample evenly on the contact surface."""
        if len(mesh.vertices) <= num_points:
            points = mesh.vertices.copy()
        else:
            points, _ = trimesh.sample.sample_surface(mesh, num_points)
        print(f"[INFO] Sampled {len(points)} contact points for tracking")
        return points

    def _compute_finger_transforms(self, T_base_ee, gripper_pos):
        """Compute transforms for left/right finger tips."""
        val = gripper_pos[0] if isinstance(gripper_pos, (list, np.ndarray)) else gripper_pos
        theta = val * 0.8

        # Left finger chain: base_ee -> left_inner_knuckle -> left_inner_finger
        T_lik = np.eye(4)
        T_lik[:3, 3] = [0.0127, 0, 0.0693]
        T_lik[:3, :3] = R.from_rotvec([0, -theta, 0]).as_matrix()

        T_lif = np.eye(4)
        T_lif[:3, 3] = [0.03458531, 0, 0.04549702]
        T_lif[:3, :3] = R.from_rotvec([0, theta, 0]).as_matrix()
        T_world_left_finger = T_base_ee @ T_lik @ T_lif

        # Right finger chain: base_ee -> right_inner_knuckle -> right_inner_finger
        T_rik = np.eye(4)
        T_rik[:3, 3] = [-0.0127, 0, 0.0693]
        T_rik[:3, :3] = R.from_euler('z', np.pi).as_matrix() @ R.from_rotvec([0, -theta, 0]).as_matrix()

        T_rif = np.eye(4)
        T_rif[:3, 3] = [0.03410605, 0, 0.04585739]
        T_rif[:3, :3] = R.from_rotvec([0, theta, 0]).as_matrix()
        T_world_right_finger = T_base_ee @ T_rik @ T_rif

        return T_world_left_finger, T_world_right_finger

    def get_contact_points_world(self, T_base_ee, gripper_pos):
        """Return sampled contact surface points for both fingers in world frame."""
        if self.contact_points_local is None:
            return None, None

        pts_local = np.hstack([self.contact_points_local, np.ones((len(self.contact_points_local), 1))])
        T_left, T_right = self._compute_finger_transforms(T_base_ee, gripper_pos)

        pts_left_world = (T_left @ pts_local.T).T[:, :3]
        pts_right_world = (T_right @ pts_local.T).T[:, :3]
        return pts_left_world, pts_right_world


def main():
    # Load configuration
    with open('conversions/droid/config.yaml', 'r') as f:
        CONFIG = yaml.safe_load(f)
    fps = CONFIG.get('fps', 30.0)
    
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
    contact_tracker = ContactSurfaceTracker(num_track_points=CONFIG.get('num_track_points', 24))
    num_contact_pts = len(contact_tracker.contact_points_local) if contact_tracker.contact_points_local is not None else 0
    total_track_pts = num_contact_pts * 2
    print(f"[INFO] Tracking {total_track_pts} contact points across both fingers")

    tracks_3d = np.zeros((actual_frames, total_track_pts, 3), dtype=np.float32)
    gripper_poses = []
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

    video_dir = os.path.join(CONFIG.get("video_output_path", "point_clouds/videos"), "tracks_reprojection")
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

        # Track gripper contact points
        track_points_world = None
        if num_contact_pts > 0:
            pts_left, pts_right = contact_tracker.get_contact_points_world(T_base_ee, gripper_positions[i])
            if pts_left is not None:
                tracks_3d[i, :num_contact_pts, :] = pts_left
                tracks_3d[i, num_contact_pts:, :] = pts_right
                track_points_world = np.vstack([pts_left, pts_right])

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
    print(f"[INFO] Tracks saved to: {tracks_save_path}")
    print(f"[INFO] Videos saved to: {video_dir}")


if __name__ == "__main__":
    main()
