import numpy as np
import os
import glob
import h5py
import yaml
import cv2
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import DBSCAN
import rerun as rr
import pyzed.sl as sl

from utils import (
    pose6_to_T,
    transform_points,
    external_cam_to_world,
    find_svo_for_camera,
    find_episode_data_by_date,
    get_zed_intrinsics,
    get_filtered_cloud,
    project_points_to_image,
    draw_points_on_image,
    VideoRecorder,
)


# Axis-aligned box around the end-effector (expressed in gripper frame)
# Tuned to roughly fit the Robotiq 85 footprint plus a small margin so that
# nearby points are still considered part of the gripper.
GRIPPER_BOUNDS = {
    "x": (-0.08, 0.08),  # left/right
    "y": (-0.07, 0.07),  # opening direction
    "z": (-0.04, 0.18),  # along the tool z-axis (forward)
}
GRIPPER_COLOR = np.array([0, 0, 255], dtype=np.uint8)  # BGR red highlight


def classify_gripper_points(points_world, T_world_ee, bounds=GRIPPER_BOUNDS, eps=0.03, min_samples=12):
    """
    Identify points that likely belong to the gripper using a bound + clustering.

    Args:
        points_world: (N, 3) points in world frame.
        T_world_ee: 4x4 transform of end-effector in world.
        bounds: dict defining an AABB in the gripper frame.
        eps: DBSCAN neighborhood size (m).
        min_samples: DBSCAN min samples.

    Returns:
        Boolean mask over points_world where True marks gripper points.
    """
    if points_world is None or len(points_world) == 0:
        return np.zeros((0,), dtype=bool)

    # Move points into gripper frame to apply the oriented bounds.
    T_ee_world = np.linalg.inv(T_world_ee)
    points_ee = transform_points(points_world, T_ee_world)

    in_box = (
        (points_ee[:, 0] > bounds["x"][0]) & (points_ee[:, 0] < bounds["x"][1]) &
        (points_ee[:, 1] > bounds["y"][0]) & (points_ee[:, 1] < bounds["y"][1]) &
        (points_ee[:, 2] > bounds["z"][0]) & (points_ee[:, 2] < bounds["z"][1])
    )
    candidate_idx = np.where(in_box)[0]
    if len(candidate_idx) == 0:
        return np.zeros(points_world.shape[0], dtype=bool)

    # If we have only a few points, keep them all to avoid discarding the gripper entirely.
    if len(candidate_idx) < min_samples:
        mask = np.zeros(points_world.shape[0], dtype=bool)
        mask[candidate_idx] = True
        return mask

    # Cluster inside the bound to keep the densest component as the gripper.
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(points_ee[candidate_idx])
    valid = labels >= 0
    if not np.any(valid):
        mask = np.zeros(points_world.shape[0], dtype=bool)
        mask[candidate_idx] = True
        return mask

    counts = np.bincount(labels[valid])
    best_label = np.argmax(counts)
    chosen_idx = candidate_idx[valid][labels[valid] == best_label]

    mask = np.zeros(points_world.shape[0], dtype=bool)
    mask[chosen_idx] = True
    return mask


def init_video_recorders(active_cams, output_dir):
    """Create VideoRecorder objects per camera."""
    os.makedirs(output_dir, exist_ok=True)
    recorders = {}
    for serial, cam in active_cams.items():
        recorders[serial] = VideoRecorder(output_dir, serial, "gripper_points", cam["w"], cam["h"])
    return recorders


def main():
    # Load configuration
    with open('conversions/droid/config.yaml', 'r') as f:
        CONFIG = yaml.safe_load(f)

    print("=== DROID Gripper Points (No URDF mesh) ===")
    rr.init("droid_gripper_points", spawn=True)
    save_path = CONFIG['rrd_output_path']
    base = save_path.replace(".rrd", "")
    save_path = f"{base}_gripper_points_only.rrd"
    rr.save(save_path)

    # Define Up-Axis for the World (Z-up is standard for Robotics)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # --- 1. Load Robot Data ---
    print("[INFO] Loading H5 Trajectory...")

    h5_file = h5py.File(CONFIG['h5_path'], 'r')
    cartesian_positions = h5_file['observation/robot_state/cartesian_position'][:]
    gripper_positions = h5_file['observation/robot_state/gripper_position'][:]
    h5_file.close()
    num_frames = len(cartesian_positions)

    # --- 2. Calculate Wrist Transforms ---
    wrist_cam_transforms = []
    wrist_serial = None

    metadata_path = CONFIG['metadata_path']
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
            cartesian_t0 = cartesian_positions[0].copy()
            T_base_ee0 = pose6_to_T(cartesian_t0)
            R_fix = R.from_euler('z', 90, degrees=True).as_matrix()
            T_base_ee0[:3, :3] = T_base_ee0[:3, :3] @ R_fix

            T_base_cam0 = pose6_to_T(wrist_pose_t0)
            T_ee_cam = np.linalg.inv(T_base_ee0) @ T_base_cam0

            wrist_cam_transforms = []
            for cart_pos in cartesian_positions:
                T_base_ee_t = pose6_to_T(cart_pos)
                T_base_ee_t[:3, :3] = T_base_ee_t[:3, :3] @ R_fix
                wrist_cam_transforms.append(T_base_ee_t @ T_ee_cam)

    # --- 3. Init Cameras ---
    cameras = {}  # Holds all cameras (Ext + Wrist)

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
                    "world_T_cam": external_cam_to_world(transform_list),
                }

    # B. Wrist Camera
    if wrist_serial:
        svo = find_svo_for_camera(CONFIG['recordings_dir'], wrist_serial)
        if svo:
            print(f"[INFO] Found Wrist Camera SVO: {wrist_serial}")
            cameras[wrist_serial] = {
                "type": "wrist",
                "svo": svo,
                "transforms": wrist_cam_transforms,  # List of 4x4 matrices
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

    video_dir = os.path.join(CONFIG.get('video_output_path', 'point_clouds/videos'), "gripper_points_only")
    recorders = init_video_recorders(active_cams, video_dir) if active_cams else {}

    # --- 4. Render Loop ---
    max_frames = CONFIG["max_frames"]
    print(f"[INFO] Processing {min(max_frames, num_frames)} frames...")

    for i in range(min(max_frames, num_frames)):
        if i % 10 == 0:
            print(f"Frame {i}")
        rr.set_time(timeline="frame_index", sequence=i)

        # End-effector pose with 90-degree rotation fix
        T_base_ee = pose6_to_T(cartesian_positions[i])
        R_fix = R.from_euler('z', 90, degrees=True).as_matrix()
        T_base_ee[:3, :3] = T_base_ee[:3, :3] @ R_fix

        # Log only the transform (no URDF mesh)
        rr.log(
            "world/gripper_pose",
            rr.Transform3D(
                translation=T_base_ee[:3, 3],
                mat3x3=T_base_ee[:3, :3],
                axis_length=0.1,
            ),
        )

        for serial, cam in active_cams.items():
            zed = cam['zed']
            if zed.grab(cam['runtime']) != sl.ERROR_CODE.SUCCESS:
                continue

            # Get image for overlay
            mat_img = sl.Mat()
            zed.retrieve_image(mat_img, sl.VIEW.LEFT)
            img_bgra = mat_img.get_data()
            img_bgr = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)

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
                        axis_length=0.1,
                    ),
                )

                rr.log(
                    "world/wrist_cam/pinhole",
                    rr.Pinhole(
                        image_from_camera=cam['K'],
                        width=cam['w'],
                        height=cam['h'],
                    ),
                )

                xyz, rgb = get_filtered_cloud(zed, cam['runtime'], CONFIG['wrist_max_depth'], CONFIG['min_depth_wrist'])
                if xyz is None:
                    continue

                xyz_world = transform_points(xyz, T_wrist)
                gripper_mask = classify_gripper_points(xyz_world, T_base_ee)
                colors_marked = rgb.copy()
                if len(colors_marked) == len(gripper_mask):
                    colors_marked[gripper_mask] = GRIPPER_COLOR

                rr.log(
                    "world/points/wrist_cam",
                    rr.Points3D(xyz_world, colors=colors_marked, radii=CONFIG['radii_size']),
                )

                if serial in recorders:
                    uv, cols = project_points_to_image(
                        xyz_world, cam['K'], T_wrist, cam['w'], cam['h'], colors=colors_marked
                    )
                    frame = draw_points_on_image(img_bgr, uv, colors=cols, radius=2)
                    recorders[serial].write_frame(frame)

            # -- EXTERNAL CAMERA LOGIC --
            else:
                xyz, rgb = get_filtered_cloud(zed, cam['runtime'], CONFIG['ext_max_depth'], CONFIG['min_depth'])
                if xyz is None:
                    continue

                T = cam['world_T_cam']
                rr.log(
                    f"world/external_cams/{serial}",
                    rr.Transform3D(
                        translation=T[:3, 3],
                        mat3x3=T[:3, :3],
                        axis_length=0.1,
                    ),
                )

                rr.log(
                    f"world/external_cams/{serial}/pinhole",
                    rr.Pinhole(
                        image_from_camera=cam['K'],
                        width=cam['w'],
                        height=cam['h'],
                    ),
                )

                xyz_world = transform_points(xyz, T)
                gripper_mask = classify_gripper_points(xyz_world, T_base_ee)
                colors_marked = rgb.copy()
                if len(colors_marked) == len(gripper_mask):
                    colors_marked[gripper_mask] = GRIPPER_COLOR

                rr.log(
                    f"world/points/external_cams/{serial}",
                    rr.Points3D(xyz_world, colors=colors_marked, radii=CONFIG['radii_size']),
                )

                if serial in recorders:
                    uv, cols = project_points_to_image(
                        xyz_world, cam['K'], T, cam['w'], cam['h'], colors=colors_marked
                    )
                    frame = draw_points_on_image(img_bgr, uv, colors=cols, radius=2)
                    recorders[serial].write_frame(frame)

    # Cleanup
    for c in active_cams.values():
        c['zed'].close()
    for rec in recorders.values():
        rec.close()

    print("[SUCCESS] Done.")
    print(f"[INFO] RRD saved to: {save_path}")
    print(f"[INFO] Videos saved to: {video_dir}")


if __name__ == "__main__":
    main()
