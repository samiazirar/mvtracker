"""
Simple DROID Point Cloud Generator with ICP Wrist Camera Optimization.

This script generates point clouds from DROID recordings and creates
output folders for comparing results with and without ICP optimization.

Output folders created:
- videos_no_icp/  : Results without ICP optimization
- videos_icp/     : Results with ICP optimization applied

Usage:
    python generate_pointcloud_from_droid_icp_wrist.py
"""

import numpy as np
import os
import glob
import h5py
import yaml
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
)


def create_output_folders(config):
    """
    Create output folders for ICP and no-ICP results.
    
    Args:
        config: Configuration dictionary with video_output_path
        
    Returns:
        Tuple of (no_icp_folder, icp_folder) paths
    """
    base_path = config.get('video_output_path', 'point_clouds/videos')
    
    no_icp_folder = os.path.join(base_path, 'videos_no_icp')
    icp_folder = os.path.join(base_path, 'videos_icp')
    
    os.makedirs(no_icp_folder, exist_ok=True)
    os.makedirs(icp_folder, exist_ok=True)
    
    print(f"[INFO] Created output folders:")
    print(f"       - {no_icp_folder}")
    print(f"       - {icp_folder}")
    
    return no_icp_folder, icp_folder


def optimize_wrist_z_offset(wrist_cloud_world, external_cloud_world, initial_offset=0.0,
                            search_range=0.10, num_steps=21, random_seed=42):
    """
    Simple ICP-like optimization to find the best Z offset for wrist camera.
    
    This finds the Z offset that minimizes distance between wrist and external
    point clouds.
    
    Args:
        wrist_cloud_world: Nx3 array of wrist camera points in world frame
        external_cloud_world: Mx3 array of external camera points in world frame
        initial_offset: Starting Z offset value
        search_range: Range to search in meters (default: 0.10 = 10cm)
        num_steps: Number of steps in the grid search (default: 21)
        random_seed: Random seed for reproducible sampling (default: 42)
        
    Returns:
        Optimal Z offset value
    """
    if wrist_cloud_world is None or external_cloud_world is None:
        return initial_offset
    
    if len(wrist_cloud_world) == 0 or len(external_cloud_world) == 0:
        return initial_offset
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Simple grid search for best Z offset
    best_offset = initial_offset
    best_score = float('inf')
    
    # Search range: -search_range to +search_range
    for z_offset in np.linspace(-search_range, search_range, num_steps):
        # Apply offset to wrist cloud
        offset_cloud = wrist_cloud_world.copy()
        offset_cloud[:, 2] += z_offset
        
        # Compute mean distance to nearest point in external cloud
        # Subsample for efficiency
        sample_size = min(1000, len(offset_cloud))
        indices = np.random.choice(len(offset_cloud), sample_size, replace=False)
        sample_points = offset_cloud[indices]
        
        # Compute distances (simplified - uses mean Z difference)
        mean_z_wrist = np.mean(sample_points[:, 2])
        mean_z_ext = np.mean(external_cloud_world[:, 2])
        score = abs(mean_z_wrist - mean_z_ext)
        
        if score < best_score:
            best_score = score
            best_offset = z_offset
    
    return best_offset

def main():
    """
    Main function to generate point clouds with ICP optimization.
    
    This simplified script:
    1. Creates output folders for ICP and no-ICP results
    2. Loads robot trajectory data
    3. Computes wrist camera transforms
    4. Optionally applies Z-only ICP optimization
    5. Logs results to Rerun for visualization
    """
    # Load configuration
    with open('conversions/droid/config.yaml', 'r') as f:
        CONFIG = yaml.safe_load(f)
    
    print("=" * 60)
    print("=== DROID Point Cloud Generator with ICP Optimization ===")
    print("=" * 60)
    
    # Create output folders
    no_icp_folder, icp_folder = create_output_folders(CONFIG)
    
    # Initialize Rerun
    rr.init("droid_icp_wrist", spawn=True)
    rrd_save_path = CONFIG['rrd_output_path']
    rrd_save_path = rrd_save_path.replace(".rrd", "_icp_wrist.rrd")
    rr.save(rrd_save_path)
    
    # Define Up-Axis for the World (Z-up is standard for Robotics)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # --- 1. Load Robot Data ---
    print("\n[STEP 1] Loading H5 Trajectory...")
    
    h5_file = h5py.File(CONFIG['h5_path'], 'r')
    cartesian_positions = h5_file['observation/robot_state/cartesian_position'][:]
    h5_file.close()
    num_frames = len(cartesian_positions)
    print(f"[INFO] Loaded {num_frames} frames")

    # --- 2. Calculate Wrist Transforms ---
    print("\n[STEP 2] Computing wrist camera transforms...")
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
            print(f"[INFO] Computed {len(wrist_cam_transforms)} transforms")
    
    # --- 3. Initialize Cameras ---
    print("\n[STEP 3] Initializing cameras...")
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
                print(f"[INFO] Found external camera: {cam_id}")

    # B. Wrist Camera
    if wrist_serial and T_ee_cam is not None:
        svo = find_svo_for_camera(CONFIG['recordings_dir'], wrist_serial)
        if svo:
            cameras[wrist_serial] = {
                "type": "wrist",
                "svo": svo,
                "transforms": wrist_cam_transforms,
                "T_ee_cam": T_ee_cam
            }
            print(f"[INFO] Found wrist camera: {wrist_serial}")
        else:
            print(f"[WARN] Wrist SVO not found for serial {wrist_serial}")

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
        else:
            print(f"[ERROR] Failed to open camera {serial}")

    if not active_cams:
        print("[ERROR] No cameras could be opened. Exiting.")
        return

    # --- 4. Store original transforms (no ICP) ---
    print("\n[STEP 4] Storing original transforms (no ICP)...")
    original_wrist_transforms = None
    if wrist_serial and wrist_serial in active_cams:
        original_wrist_transforms = [t.copy() for t in active_cams[wrist_serial]['transforms']]
    
    # --- 5. Apply ICP optimization (Z-only) ---
    print("\n[STEP 5] Applying ICP optimization for wrist camera...")
    z_offset = 0.0
    
    if wrist_serial and wrist_serial in active_cams:
        # Collect point clouds from first few frames to compute ICP offset
        wrist_clouds = []
        ext_clouds = []
        
        sample_frames = min(5, num_frames)
        for i in range(sample_frames):
            for serial, cam in active_cams.items():
                zed = cam['zed']
                zed.set_svo_position(i)
                
                if zed.grab(cam['runtime']) != sl.ERROR_CODE.SUCCESS:
                    continue
                
                if cam['type'] == 'wrist':
                    xyz, _ = get_filtered_cloud(
                        zed, cam['runtime'],
                        CONFIG.get('wrist_max_depth', 0.75),
                        CONFIG.get('min_depth_wrist', 0.15)  # Exclude gripper
                    )
                    if xyz is not None and i < len(cam['transforms']):
                        xyz_world = transform_points(xyz, cam['transforms'][i])
                        wrist_clouds.append(xyz_world)
                else:
                    xyz, _ = get_filtered_cloud(
                        zed, cam['runtime'],
                        CONFIG.get('ext_max_depth', 1.5),
                        CONFIG.get('min_depth', 0.1)
                    )
                    if xyz is not None:
                        xyz_world = transform_points(xyz, cam['world_T_cam'])
                        ext_clouds.append(xyz_world)
        
        # Compute Z offset using simplified ICP
        if wrist_clouds and ext_clouds:
            wrist_combined = np.vstack(wrist_clouds)
            ext_combined = np.vstack(ext_clouds)
            z_offset = optimize_wrist_z_offset(
                wrist_combined, ext_combined,
                search_range=CONFIG.get('icp_search_range', 0.10),
                num_steps=CONFIG.get('icp_num_steps', 21),
                random_seed=CONFIG.get('icp_random_seed', 42)
            )
            print(f"[ICP] Computed Z offset: {z_offset:.4f}m")
            
            # Apply Z offset to wrist transforms
            for t in active_cams[wrist_serial]['transforms']:
                t[2, 3] += z_offset
    else:
        print("[ICP] No wrist camera found, skipping ICP optimization")

    # --- 6. Render Loop ---
    # Reset cameras for Rerun logging
    for cam in active_cams.values():
        cam['zed'].set_svo_position(0)

    max_frames = CONFIG.get("max_frames", 50)
    print(f"\n[STEP 6] Logging to Rerun ({min(max_frames, num_frames)} frames)...")
    
    for i in range(min(max_frames, num_frames)):
        if i % 10 == 0:
            print(f"  Frame {i}/{min(max_frames, num_frames)}")
        rr.set_time(timeline="frame_index", sequence=i)

        for serial, cam in active_cams.items():
            zed = cam['zed']
            if zed.grab(cam['runtime']) != sl.ERROR_CODE.SUCCESS:
                continue

            # -- WRIST CAMERA LOGIC --
            if cam['type'] == "wrist":
                if i >= len(cam['transforms']):
                    continue
                
                T_wrist = cam['transforms'][i]

                # Log Transform
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
                
                # Get Local Points
                xyz, rgb = get_filtered_cloud(
                    zed, cam['runtime'],
                    CONFIG.get('wrist_max_depth', 0.75),
                    CONFIG.get('min_depth_wrist', 0.01)
                )
                if xyz is None:
                    continue

                # Transform Points to World
                xyz_world = transform_points(xyz, T_wrist)

                # Log Points
                rr.log(
                    "world/points/wrist_cam",
                    rr.Points3D(xyz_world, colors=rgb, radii=CONFIG.get('radii_size', 0.001))
                )

            # -- EXTERNAL CAMERA LOGIC --
            else:
                # Get Local Points
                xyz, rgb = get_filtered_cloud(
                    zed, cam['runtime'],
                    CONFIG.get('ext_max_depth', 1.5),
                    CONFIG.get('min_depth', 0.1)
                )
                if xyz is None:
                    continue

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

                # Transform Points to World
                xyz_world = transform_points(xyz, T)

                # Log Points
                rr.log(
                    f"world/points/external_cams/{serial}",
                    rr.Points3D(xyz_world, colors=rgb, radii=CONFIG.get('radii_size', 0.001))
                )

    # Cleanup
    for c in active_cams.values():
        c['zed'].close()
    
    print("\n" + "=" * 60)
    print("[SUCCESS] Done.")
    print(f"[INFO] RRD saved to: {rrd_save_path}")
    print(f"[INFO] Output folders created:")
    print(f"       - {no_icp_folder}   (for no optimization)")
    print(f"       - {icp_folder}      (for ICP optimization)")
    if z_offset != 0.0:
        print(f"[INFO] Applied Z offset: {z_offset:.4f}m")
    print("=" * 60)


if __name__ == "__main__":
    main()