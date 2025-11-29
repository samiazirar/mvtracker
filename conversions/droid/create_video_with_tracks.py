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
from utils.optimization import optimize_external_cameras_multi_frame, optimize_wrist_multi_frame


from utils.transforms import invert_transform


def draw_trajectory_on_image(
    image: np.ndarray,
    track_points_world: np.ndarray,
    K: np.ndarray,
    world_T_cam: np.ndarray,
    width: int,
    height: int,
    color: tuple = (0, 255, 255),  # Yellow in BGR
    thickness: int = 2
) -> np.ndarray:
    """
    Draw trajectory track on image.
    
    Projects 3D world trajectory points to 2D and draws lines connecting them.
    
    Args:
        image: BGR image as numpy array
        track_points_world: Nx3 array of trajectory points in world frame
        K: 3x3 camera intrinsic matrix
        world_T_cam: 4x4 transformation from camera to world
        width: Image width
        height: Image height
        color: BGR color for trajectory line
        thickness: Line thickness
        
    Returns:
        Image with trajectory drawn
    """
    if len(track_points_world) < 2:
        return image
    
    img_out = image.copy()
    
    # Transform from world to camera frame
    cam_T_world = invert_transform(world_T_cam)
    
    # Convert to homogeneous and transform
    ones = np.ones((track_points_world.shape[0], 1))
    pts_homo = np.hstack([track_points_world, ones])
    pts_cam = (cam_T_world @ pts_homo.T).T[:, :3]
    
    # Filter points behind camera
    z = pts_cam[:, 2]
    valid = z > 0.01  # Minimum depth
    
    if not np.any(valid):
        return img_out
    
    # Project to 2D
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    u = (pts_cam[:, 0] * fx / pts_cam[:, 2]) + cx
    v = (pts_cam[:, 1] * fy / pts_cam[:, 2]) + cy
    
    # Draw connected line segments
    prev_pt = None
    for i in range(len(u)):
        if not valid[i]:
            prev_pt = None
            continue
        
        # Check if point values are finite and in bounds
        if not (np.isfinite(u[i]) and np.isfinite(v[i])):
            prev_pt = None
            continue
        
        if 0 <= u[i] < width and 0 <= v[i] < height:
            curr_pt = (int(u[i]), int(v[i]))
            
            if prev_pt is not None:
                cv2.line(img_out, prev_pt, curr_pt, color, thickness)
            
            # Draw point marker at current position (latest point gets larger marker)
            if i == len(u) - 1:
                cv2.circle(img_out, curr_pt, 5, color, -1)  # Filled circle for current pos
            else:
                cv2.circle(img_out, curr_pt, 2, color, -1)  # Small marker for history
            
            prev_pt = curr_pt
        else:
            prev_pt = None
    
    return img_out


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

def generate_comparison_videos(active_cams, transforms_before, transforms_after, num_frames, config, cartesian_positions=None):
    """
    Generate MP4 videos for each camera showing reprojected point clouds
    before and after ICP optimization, with optional trajectory tracks.
    
    Args:
        active_cams: Dictionary of active cameras
        transforms_before: Camera transforms before ICP
        transforms_after: Camera transforms after ICP
        num_frames: Total number of frames
        config: Configuration dictionary
        cartesian_positions: Optional Nx6 array of end-effector poses for trajectory tracks
    """
    print("\n[VIDEO] Generating comparison videos (Before vs After ICP)...")
    video_dir = config.get('video_output_path', 'point_clouds/videos')
    
    # 1. Initialize Video Recorders
    recorders = {}
    for serial, cam in active_cams.items():
        w, h = cam['w'], cam['h']
        recorders[serial] = {
            'before': VideoRecorder(video_dir, serial, "before_icp", w, h),
            'after': VideoRecorder(video_dir, serial, "after_icp", w, h)
        }

    # 2. Precompute trajectory points for track visualization
    track_history_length = config.get('track_history_length', 30)  # Number of past frames to show
    ee_track_points = []  # End-effector trajectory points
    
    if cartesian_positions is not None:
        for pos in cartesian_positions:
            T_ee = pose6_to_T(pos)
            ee_track_points.append(T_ee[:3, 3])  # Extract position
        ee_track_points = np.array(ee_track_points)

    # 2. Processing Loop
    # We need to iterate through frames, grab data, build world clouds, and project.
    
    # Reset all cameras to frame 0
    for cam in active_cams.values():
        cam['zed'].set_svo_position(0)

    max_frames = min(num_frames, config.get('max_frames', 50))
    
    for i in range(max_frames):
        if i % 10 == 0: print(f"  -> Processing Video Frame {i}/{max_frames}")
        
        # A. Grab Data for this frame
        frame_data = {} # serial -> {image, points_local}
        
        for serial, cam in active_cams.items():
            zed = cam['zed']
            # Ensure synchronization (simple approach: set pos)
            zed.set_svo_position(i)
            if zed.grab(cam['runtime']) != sl.ERROR_CODE.SUCCESS:
                continue
                
            # Get Image
            mat_img = sl.Mat()
            zed.retrieve_image(mat_img, sl.VIEW.LEFT)
            img_bgra = mat_img.get_data()
            img_bgr = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)
            
            # Get Points
            xyz, rgb = get_filtered_cloud(zed, cam['runtime'], 
                                      config.get('ext_max_depth', 2.0) if cam['type'] == 'external' else config.get('wrist_max_depth', 0.75),
                                      config.get('min_depth', 0.1))
            
            frame_data[serial] = {
                'image': img_bgr,
                'points': xyz,
                'colors': rgb
            }

        # B. Build World Clouds (Before & After)
        cloud_before = []
        colors_before = []
        cloud_after = []
        colors_after = []
        
        for serial, data in frame_data.items():
            if data['points'] is None: continue
            
            # Get Transform Before
            T_before = None
            if transforms_before[serial]['type'] == 'external':
                T_before = transforms_before[serial]['world_T_cam']
            else: # wrist
                if i < len(transforms_before[serial]['transforms']):
                    T_before = transforms_before[serial]['transforms'][i]
            
            if T_before is not None:
                cloud_before.append(transform_points(data['points'], T_before))
                colors_before.append(data['colors'])
                
            # Get Transform After
            T_after = None
            if transforms_after[serial]['type'] == 'external':
                T_after = transforms_after[serial]['world_T_cam']
            else: # wrist
                if i < len(transforms_after[serial]['transforms']):
                    T_after = transforms_after[serial]['transforms'][i]
            
            if T_after is not None:
                cloud_after.append(transform_points(data['points'], T_after))
                colors_after.append(data['colors'])

        # Stack clouds
        pts_world_before = np.vstack(cloud_before) if cloud_before else np.empty((0, 3))
        cols_world_before = np.vstack(colors_before) if colors_before else np.empty((0, 3))
        
        pts_world_after = np.vstack(cloud_after) if cloud_after else np.empty((0, 3))
        cols_world_after = np.vstack(colors_after) if colors_after else np.empty((0, 3))
        
        # C. Get track points for this frame (trajectory history)
        track_start = max(0, i - track_history_length)
        current_track = ee_track_points[track_start:i+1] if len(ee_track_points) > 0 else np.empty((0, 3))
        
        # C. Project and Write
        for serial, cam in active_cams.items():
            if serial not in frame_data: continue
            
            img = frame_data[serial]['image']
            local_pts = frame_data[serial]['points']
            if local_pts is None: continue
            
            K = cam['K']
            w, h = cam['w'], cam['h']
            
            # -- BEFORE --
            T_cam_before = None
            if transforms_before[serial]['type'] == 'external':
                T_cam_before = transforms_before[serial]['world_T_cam']
            elif i < len(transforms_before[serial]['transforms']):
                T_cam_before = transforms_before[serial]['transforms'][i]
                
            if T_cam_before is not None:
                # Project GLOBAL world points back to this camera
                uv, cols = project_points_to_image(pts_world_before, K, T_cam_before, w, h, colors=cols_world_before)
                img_out = draw_points_on_image(img.copy(), uv, colors=cols)
                
                # Draw trajectory track
                if len(current_track) > 1:
                    img_out = draw_trajectory_on_image(img_out, current_track, K, T_cam_before, w, h)
                
                recorders[serial]['before'].write_frame(img_out)
            
            # -- AFTER --
            T_cam_after = None
            if transforms_after[serial]['type'] == 'external':
                T_cam_after = transforms_after[serial]['world_T_cam']
            elif i < len(transforms_after[serial]['transforms']):
                T_cam_after = transforms_after[serial]['transforms'][i]
                
            if T_cam_after is not None:
                # Project GLOBAL world points back to this camera
                uv, cols = project_points_to_image(pts_world_after, K, T_cam_after, w, h, colors=cols_world_after)
                img_out = draw_points_on_image(img.copy(), uv, colors=cols)
                
                # Draw trajectory track
                if len(current_track) > 1:
                    img_out = draw_trajectory_on_image(img_out, current_track, K, T_cam_after, w, h)
                
                recorders[serial]['after'].write_frame(img_out)

    # 3. Cleanup
    for recs in recorders.values():
        recs['before'].close()
        recs['after'].close()
    print("[VIDEO] Done.")

def main():
    # Load configuration
    with open('conversions/droid/config.yaml', 'r') as f:
        CONFIG = yaml.safe_load(f)
    
    print("=== DROID Full Fusion (Wrist + External) ===")
    rr.init("droid_full_fusion", spawn=True)
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

    # Init Gripper Viz
    gripper_viz = GripperVisualizer()
    gripper_viz.init_rerun()

    # --- 2. Calculate Wrist Transforms ---
    wrist_cam_transforms = []
    wrist_serial = None
    
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

    # ====================================================
    # [NEW] MULTI-FRAME OPTIMIZATION PIPELINE
    # ====================================================
    
    # Capture BEFORE state
    transforms_before = capture_transforms(active_cams)

    # 1. Optimize External Cameras (using 20-frame accumulation)
    optimize_external_cameras_multi_frame(active_cams, CONFIG)

    # 2. Optimize Wrist Camera (using 5 different test angles)
    if wrist_serial and wrist_serial in active_cams:
        optimize_wrist_multi_frame(active_cams, cartesian_positions, CONFIG)
    
    # Capture AFTER state
    transforms_after = capture_transforms(active_cams)

    # Generate Comparison Videos WITH TRACKS
    generate_comparison_videos(
        active_cams, 
        transforms_before, 
        transforms_after, 
        num_frames, 
        CONFIG,
        cartesian_positions=cartesian_positions  # Pass trajectory data for tracks
    )

    # ====================================================

    # --- 4. Render Loop ---
    # Reset cameras for Rerun logging
    for cam in active_cams.values():
        cam['zed'].set_svo_position(0)

    max_frames = CONFIG["max_frames"]
    print(f"[INFO] Processing {min(max_frames, num_frames)} frames...")
    
    for i in range(min(max_frames, num_frames)):
        if i % 10 == 0: print(f"Frame {i}")
        rr.set_time(timeline="frame_index", sequence=i)

        # Update Gripper (use end-effector pose directly)
        T_base_ee = pose6_to_T(cartesian_positions[i])
        #rotate by 90 degrees to align
        R_fix = R.from_euler('z', 90, degrees=True).as_matrix()
        T_base_ee[:3, :3] = T_base_ee[:3, :3] @ R_fix
        gripper_viz.update(T_base_ee, gripper_positions[i])

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
    print("[SUCCESS] Done.")
    print(f"[INFO] RRD saved to: {rrd_save_path}")


if __name__ == "__main__":
    main()