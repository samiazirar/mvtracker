droid_rawimport pybullet as p
import pybullet_data
import numpy as np
import cv2
import h5py
import json
import os
import glob
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
# Try importing ZED SDK
try:
    import pyzed.sl as sl
except ImportError:
    print("\n[CRITICAL ERROR] The ZED SDK is not installed.")
    print("pip install pyzed\n")
    exit(1)

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    "h5_path": "/data/droid/data/droid_raw/1.0.1/ILIAD/success/2023-06-11/Sun_Jun_11_15:52:37_2023/trajectory.h5",
    "urdf_path": "third_party/CtRNet-X/urdfs/Panda/panda_robotiq_arg85.urdf",
    "extrinsics_json_path": "/data/droid/calib_and_annot/droid/cam2base_extrinsic_superset.json",
    "intrinsics_json_path": "/data/droid/calib_and_annot/droid/intrinsics.json",
    "recordings_dir": "/data/droid/data/droid_raw/1.0.1/ILIAD/success/2023-06-11/Sun_Jun_11_15:52:37_2023/recordings/SVO",
    "metadata_path": None,  # Will be auto-discovered
    
    # Main Output Directory
    "output_dir": "renders/debug_full_suite",
    
    "width": 1280,
    "height": 720,
    "default_fov": 60,
    "near_plane": 0.01,
    "far_plane": 5.0,
    "bg_color": [0.2, 0.2, 0.2],
    "max_frames": 100,
    "viz_depth_max": 2.0,
    
    # --- RENDERING COLORS ---
    "recolor_gripper": True,  # Set to False to keep original gripper color
    "gripper_color": [0.2, 0.5, 1.0, 1.0],  # RGBA: Light blue/cyan for gripper
    
    # --- ROTATION FIX ---
    # Fine-tune gripper rotation to match camera view
    "gripper_rot_offset": -np.pi / 4,  # -45 degrees adjustment
    
    # --- GRIPPER OFFSET ---
    # Offset gripper position in Z direction (meters) - negative moves away from wrist camera
    "gripper_offset_z": -0.014,  # 5cm down from end-effector
}# =============================================================================
# MATH & TRANSFORMS
# =============================================================================


def rvec_tvec_to_matrix(val):
    pos = np.array(val[0:3])
    euler = np.array(val[3:6])
    R_mat = R.from_euler("xyz", euler).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = pos
    return T

def get_view_matrix_from_pose(world_T_cam):
    cam_T_world = np.linalg.inv(world_T_cam)
    M_cv_gl = np.eye(4)
    M_cv_gl[1, 1] = -1
    M_cv_gl[2, 2] = -1
    view_matrix_mat = M_cv_gl @ cam_T_world
    return view_matrix_mat.T.flatten().tolist()

def get_projection_matrix_from_intrinsics(width, height, fx, fy, cx, cy, near, far):
    fov_y = 2 * np.arctan(height / (2 * fy))
    fov_degrees = np.degrees(fov_y)
    aspect = width / height
    proj_matrix = p.computeProjectionMatrixFOV(fov_degrees, aspect, near, far)
    return proj_matrix, fov_degrees

def linearize_pybullet_depth(depth_buffer, near, far):
    depth_ndc = np.array(depth_buffer)
    depth_linear = far * near / (far - (far - near) * depth_ndc)
    return depth_linear

def pose6_to_T(p):
    """Convert [x, y, z, roll, pitch, yaw] to 4x4 transformation matrix."""
    x, y, z, roll, pitch, yaw = p
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    T[:3, :3] = R.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    return T

def draw_coordinate_frame(image, origin_pixel, x_pixel, y_pixel, z_pixel, scale=1.0):
    """Draw RGB coordinate frame (X=red, Y=green, Z=blue) on image."""
    ox, oy = int(origin_pixel[0]), int(origin_pixel[1])
    
    # Check if origin is in bounds
    h, w = image.shape[:2]
    if not (0 <= ox < w and 0 <= oy < h):
        return
    
    # Draw axes with different colors
    # Z-axis (blue) - optical axis / forward
    if 0 <= int(z_pixel[0]) < w and 0 <= int(z_pixel[1]) < h:
        cv2.arrowedLine(image, (ox, oy), (int(z_pixel[0]), int(z_pixel[1])), 
                       (255, 0, 0), 2, tipLength=0.3)  # Blue
    
    # X-axis (red) - right
    if 0 <= int(x_pixel[0]) < w and 0 <= int(x_pixel[1]) < h:
        cv2.arrowedLine(image, (ox, oy), (int(x_pixel[0]), int(x_pixel[1])), 
                       (0, 0, 255), 2, tipLength=0.3)  # Red
    
    # Y-axis (green) - down
    if 0 <= int(y_pixel[0]) < w and 0 <= int(y_pixel[1]) < h:
        cv2.arrowedLine(image, (ox, oy), (int(y_pixel[0]), int(y_pixel[1])), 
                       (0, 255, 0), 2, tipLength=0.3)  # Green
    
    # Draw center point
    cv2.circle(image, (ox, oy), 5, (255, 255, 255), -1)
    cv2.circle(image, (ox, oy), 6, (0, 0, 0), 1)

# =============================================================================
# HELPERS
# =============================================================================

def find_episode_data(h5_path, json_path, data_type="extrinsics"):
    parts = h5_path.split(os.sep)
    date_str = parts[-3] 
    timestamp_folder = parts[-2]
    ts_parts = timestamp_folder.split('_')
    time_str = "00:00:00"
    for part in ts_parts:
        if ':' in part:
            time_str = part
            break
    h, m, s = time_str.split(':')
    target_suffix = f"{date_str}-{h}h-{m}m-{s}s"
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    for key in data.keys():
        if key.endswith(target_suffix):
            return data[key]
    for key in data.keys():
        if target_suffix.split('-')[-1] in key:
             return data[key]
    return None

def find_svo_for_camera(recordings_dir, cam_serial):
    patterns = [f"*{cam_serial}*.svo", f"*{cam_serial}*.svo2"]
    for pat in patterns:
        matches = glob.glob(os.path.join(recordings_dir, pat))
        if matches:
            return matches[0]
    return None

def setup_pybullet(urdf_path, gripper_color=None, recolor_gripper=False, gripper_offset_z=0.0):
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    plane_id = p.loadURDF("plane.urdf", [0, 0, 0])
    p.changeVisualShape(plane_id, -1, rgbaColor=[0.85, 0.85, 0.85, 1])
    robot_id = p.loadURDF(urdf_path, [0, 0, gripper_offset_z], useFixedBase=True)
    
    num_joints = p.getNumJoints(robot_id)
    joint_map = {}
    
    # Gripper link identifiers
    gripper_link_names = ['hand', 'finger', 'grip']
    
    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i)
        joint_name = info[1].decode('utf-8')
        joint_map[joint_name] = i
    
    # Apply gripper color if recoloring is enabled
    if recolor_gripper and gripper_color is not None:
        # Color all links that are part of the gripper
        for link_idx in range(-1, num_joints):
            try:
                # Get link name
                if link_idx == -1:
                    link_name = 'base'
                else:
                    link_name = p.getJointInfo(robot_id, link_idx)[12].decode('utf-8')
                
                # Check if this link is part of the gripper
                is_gripper = any(part in link_name.lower() for part in gripper_link_names)
                
                if is_gripper:
                    p.changeVisualShape(robot_id, link_idx, rgbaColor=gripper_color)
            except:
                pass
    
    return robot_id, plane_id, joint_map

def set_robot_state(robot_id, joint_map, joint_positions, gripper_pos, rot_offset=0.0):
    panda_joints = [f"panda_joint{i}" for i in range(1, 8)]
    if len(joint_positions) >= 7:
        for i, name in enumerate(panda_joints):
            if name in joint_map:
                angle = joint_positions[i]
                if name == "panda_joint7":
                    angle += rot_offset
                p.resetJointState(robot_id, joint_map[name], angle)
                
    # Robotiq 85 Gripper Control
    # DROID: 0.0 = Open, 1.0 = Closed
    # URDF: finger_joint 0.0 = Open, ~0.8 = Closed
    val = gripper_pos[0] if isinstance(gripper_pos, (list, np.ndarray)) else gripper_pos
    
    # Map 0-1 to 0-0.8 rad
    theta = val * 0.8
    
    # Mimic joints based on URDF
    # finger_joint drives the left_outer_knuckle
    # Correct multipliers from URDF:
    # left_inner_knuckle_joint: 1.0
    # left_inner_finger_joint: -1.0
    # right_inner_knuckle_joint: -1.0
    # right_inner_finger_joint: 1.0
    # right_outer_knuckle_joint: -1.0
    mimic_joints = {
        "finger_joint": 1.0,
        "right_outer_knuckle_joint": -1.0,
        "left_inner_knuckle_joint": 1.0,
        "right_inner_knuckle_joint": -1.0,
        "left_inner_finger_joint": -1.0,
        "right_inner_finger_joint": 1.0
    }
    
    for name, multiplier in mimic_joints.items():
        if name in joint_map:
            p.resetJointState(robot_id, joint_map[name], theta * multiplier)

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=== Starting DROID Full Suite Renderer ===")
    
    # 1. Setup Output Directories
    base_dir = CONFIG["output_dir"]
    dirs = {
        "neutral": os.path.join(base_dir, "neutral"),
        "sim_rgb": os.path.join(base_dir, "sim_rgb"),
        "real_rgb": os.path.join(base_dir, "real_rgb"),
        "overlay": os.path.join(base_dir, "overlay"),
        "depth_viz": os.path.join(base_dir, "depth_viz")
    }
    
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    
    # 2. Load H5
    try:
        h5_file = h5py.File(CONFIG['h5_path'], 'r')
        joint_pos_data = h5_file['observation/robot_state/joint_positions'][:]
        gripper_pos_data = h5_file['observation/robot_state/gripper_position'][:]
        cartesian_position = h5_file['observation/robot_state/cartesian_position'][:]
        num_frames = len(joint_pos_data)
        print(f"[INFO] Frames in H5: {num_frames}")
    except Exception as e:
        print(f"[ERROR] H5 Load Failed: {e}")
        return
    
    # 2.5. Load Metadata for Wrist Camera
    metadata = None
    wrist_cam_transforms = None
    
    # Auto-discover metadata file
    metadata_path = CONFIG['metadata_path']
    if metadata_path is None:
        episode_dir = os.path.dirname(CONFIG['h5_path'])
        metadata_files = glob.glob(os.path.join(episode_dir, "metadata_*.json"))
        if metadata_files:
            metadata_path = metadata_files[0]
            print(f"[INFO] Auto-discovered metadata: {os.path.basename(metadata_path)}")
    
    print(f"[DEBUG] Checking metadata path: {metadata_path}")
    print(f"[DEBUG] Metadata exists: {metadata_path and os.path.exists(metadata_path)}")
    
    if metadata_path and os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            wrist_cam_pose6 = metadata.get("wrist_cam_extrinsics")
            print(f"[DEBUG] Wrist cam extrinsics from metadata: {wrist_cam_pose6}")
            
            if wrist_cam_pose6:
                # Compute constant transform: end-effector → wrist camera
                T_base_cam0 = pose6_to_T(wrist_cam_pose6)  # base → camera at t=0
                T_base_ee0 = pose6_to_T(cartesian_position[0])  # base → end-effector at t=0
                T_ee_cam = np.linalg.inv(T_base_ee0) @ T_base_cam0  # constant EE→camera
                
                # Compute wrist camera position for all frames
                wrist_cam_transforms = []
                for t in range(num_frames):
                    T_base_ee_t = pose6_to_T(cartesian_position[t])
                    T_base_cam_t = T_base_ee_t @ T_ee_cam
                    wrist_cam_transforms.append(T_base_cam_t)
                
                print(f"[INFO] Loaded wrist camera transforms for {len(wrist_cam_transforms)} frames")
                print(f"[DEBUG] First wrist cam position (base frame): {wrist_cam_transforms[0][:3, 3]}")
            else:
                print(f"[WARN] No wrist_cam_extrinsics in metadata")
        except Exception as e:
            print(f"[WARN] Could not load metadata: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"[WARN] Metadata file not found at: {metadata_path}")

    # 3. Extrinsics & Intrinsics
    extrinsics = find_episode_data(CONFIG['h5_path'], CONFIG['extrinsics_json_path'], "extrinsics")
    intrinsics_data = find_episode_data(CONFIG['h5_path'], CONFIG['intrinsics_json_path'], "intrinsics")
    
    if not extrinsics: return

    cameras = {}
    for key, val in extrinsics.items():
        if key.isdigit():
            intr = None
            if intrinsics_data and key in intrinsics_data and 'cameraMatrix' in intrinsics_data[key]:
                intr = intrinsics_data[key]['cameraMatrix']
            
            cameras[key] = {
                "world_T_cam": rvec_tvec_to_matrix(val),
                "serial": key,
                "intrinsics": intr
            }
            
            # Create sub-folders per camera
            for k, path in dirs.items():
                if k != "neutral": # Neutral doesn't have camera IDs
                    os.makedirs(os.path.join(path, key), exist_ok=True)
    
    # Add wrist camera if available
    if wrist_cam_transforms is not None and len(wrist_cam_transforms) > 0:
        wrist_serial = None
        if metadata and "wrist_cam_serial" in metadata:
            wrist_serial = str(metadata["wrist_cam_serial"])
            print(f"[INFO] Adding wrist camera (serial: {wrist_serial}) to camera list")
            
            # Get wrist camera intrinsics
            wrist_intr = None
            if intrinsics_data and wrist_serial in intrinsics_data and 'cameraMatrix' in intrinsics_data[wrist_serial]:
                wrist_intr = intrinsics_data[wrist_serial]['cameraMatrix']
                print(f"[INFO] Found wrist camera intrinsics")
            
            cameras[wrist_serial] = {
                "world_T_cam": wrist_cam_transforms[0],  # Will update each frame
                "serial": wrist_serial,
                "intrinsics": wrist_intr,
                "is_wrist": True  # Mark as wrist camera
            }
            
            # Create sub-folders for wrist camera
            for k, path in dirs.items():
                if k != "neutral":
                    os.makedirs(os.path.join(path, wrist_serial), exist_ok=True)

    # 4. PyBullet Setup
    robot_id, plane_id, joint_map = setup_pybullet(CONFIG['urdf_path'], CONFIG['gripper_color'], CONFIG['recolor_gripper'], CONFIG['gripper_offset_z'])
    
    # Neutral Camera Setup
    neutral_view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0.0, 0, 0.5], distance=2.5, yaw=45, pitch=-30, roll=0, upAxisIndex=2
    )
    neutral_proj = p.computeProjectionMatrixFOV(60, CONFIG['width']/CONFIG['height'], 0.01, 10.0)
    
    # 5. ZED SVO Setup
    svo_readers = {}
    print("[INFO] Opening SVO files...")
    for cam_id in cameras.keys():
        svo_path = find_svo_for_camera(CONFIG['recordings_dir'], cam_id)
        if svo_path:
            print(f"[INFO] Cam {cam_id}: Found SVO -> {os.path.basename(svo_path)}")
            
            zed = sl.Camera()
            init_params = sl.InitParameters()
            init_params.set_from_svo_file(svo_path)
            init_params.svo_real_time_mode = False
            init_params.coordinate_units = sl.UNIT.METER
            init_params.depth_mode = sl.DEPTH_MODE.NEURAL 
            
            err = zed.open(init_params)
            if err != sl.ERROR_CODE.SUCCESS:
                print(f"[ERROR] Failed to open SVO {svo_path}: {err}")
                continue
                
            # SYNC LOGIC
            total_svo = zed.get_svo_number_of_frames()
            if total_svo >= num_frames:
                start_pos = total_svo - num_frames
                zed.set_svo_position(start_pos)
                print(f"[ALIGN] Skipped {start_pos} frames")
            else:
                print(f"[INFO] Cam {cam_id}: SVO has only {total_svo} frames, Sim has {num_frames} frames") 
                print(f"[WARN] SVO shorter than Sim")

            svo_readers[cam_id] = {
                "zed": zed,
                "mat_rgb": sl.Mat(),
                "mat_depth": sl.Mat(),
                "runtime": sl.RuntimeParameters()
            }

    # 6. Render Loop
    limit = min(CONFIG["max_frames"] or num_frames, num_frames)
    viz_max = CONFIG['viz_depth_max']
    rot_offset = CONFIG['gripper_rot_offset']
    
    for i in range(limit):
        if i % 10 == 0: print(f"Frame {i}/{limit}")
        
        set_robot_state(robot_id, joint_map, joint_pos_data[i], gripper_pos_data[i], rot_offset)
        
        # --- A. NEUTRAL RENDER (Fixed View) ---
        w, h, rgb, _, _ = p.getCameraImage(
            CONFIG['width'], CONFIG['height'], neutral_view_matrix, neutral_proj,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        neutral_rgb = np.reshape(rgb, (h, w, 4))[:, :, :3]
        neutral_rgb_bgr = cv2.cvtColor(neutral_rgb, cv2.COLOR_RGB2BGR)
        
        # Draw wrist camera coordinate system on neutral view
        if wrist_cam_transforms is not None and i < len(wrist_cam_transforms):
            T_base_wrist = wrist_cam_transforms[i]
            wrist_pos_base = T_base_wrist[:3, 3]
            wrist_R_base = T_base_wrist[:3, :3]
            
            # Define coordinate frame axes
            axis_length = 0.08
            axes_cam = np.array([
                [axis_length, 0, 0],  # X-axis
                [0, axis_length, 0],  # Y-axis
                [0, 0, axis_length]   # Z-axis
            ])
            
            # Transform axes to world frame
            axes_world = wrist_R_base @ axes_cam.T
            axes_world = axes_world.T + wrist_pos_base
            
            # Get neutral camera transform from view matrix
            # PyBullet view matrix is column-major, need to extract camera pose
            neutral_view_array = np.array(neutral_view_matrix).reshape(4, 4).T
            neutral_cam_T_world = neutral_view_array
            
            # Project origin and axes
            wrist_pos_hom = np.append(wrist_pos_base, 1.0)
            origin_cam = neutral_cam_T_world @ wrist_pos_hom
            
            if origin_cam[2] < 0:  # In PyBullet OpenGL, negative Z is forward
                # Use projection matrix to get pixel coordinates
                neutral_proj_array = np.array(neutral_proj).reshape(4, 4).T
                
                # Project to clip space then to NDC
                origin_clip = neutral_proj_array @ origin_cam
                if origin_clip[3] != 0:
                    origin_ndc = origin_clip[:3] / origin_clip[3]
                    # NDC to pixel coordinates
                    origin_pixel = np.array([
                        (origin_ndc[0] + 1) * 0.5 * w,
                        (1 - origin_ndc[1]) * 0.5 * h
                    ])
                    
                    # Project axes endpoints
                    axes_pixels = []
                    for axis_world in axes_world:
                        axis_hom = np.append(axis_world, 1.0)
                        axis_cam = neutral_cam_T_world @ axis_hom
                        if axis_cam[2] < 0:
                            axis_clip = neutral_proj_array @ axis_cam
                            if axis_clip[3] != 0:
                                axis_ndc = axis_clip[:3] / axis_clip[3]
                                axis_pixel = np.array([
                                    (axis_ndc[0] + 1) * 0.5 * w,
                                    (1 - axis_ndc[1]) * 0.5 * h
                                ])
                                axes_pixels.append(axis_pixel)
                            else:
                                axes_pixels.append(None)
                        else:
                            axes_pixels.append(None)
                    
                    # Draw coordinate frame
                    if len(axes_pixels) == 3 and all(p is not None for p in axes_pixels):
                        draw_coordinate_frame(neutral_rgb_bgr, origin_pixel, 
                                            axes_pixels[0], axes_pixels[1], axes_pixels[2])
        
        cv2.imwrite(os.path.join(dirs["neutral"], f"frame_{i:05d}.png"), neutral_rgb_bgr)
        
        # --- B. PER CAMERA RENDERS ---
        for cam_id, cam_data in cameras.items():
            # Update wrist camera pose if this is the wrist camera
            if cam_data.get("is_wrist", False) and wrist_cam_transforms is not None and i < len(wrist_cam_transforms):
                cam_data["world_T_cam"] = wrist_cam_transforms[i]
            
            # 1. Render Simulation
            view_mat = get_view_matrix_from_pose(cam_data["world_T_cam"])
            
            if cam_data["intrinsics"]:
                fx, cx, fy, cy = [float(x) for x in cam_data["intrinsics"]]
                proj_mat, _ = get_projection_matrix_from_intrinsics(
                    CONFIG['width'], CONFIG['height'], fx, fy, cx, cy, CONFIG['near_plane'], CONFIG['far_plane']
                )
            else:
                proj_mat = p.computeProjectionMatrixFOV(CONFIG['default_fov'], CONFIG['width']/CONFIG['height'], CONFIG['near_plane'], CONFIG['far_plane'])
            
            w, h, rgb, depth_buffer, seg = p.getCameraImage(
                CONFIG['width'], CONFIG['height'], view_mat, proj_mat,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )
            
            sim_rgb = np.reshape(rgb, (h, w, 4))[:, :, :3]
            sim_rgb_bgr = cv2.cvtColor(sim_rgb, cv2.COLOR_RGB2BGR)
            sim_depth_m = linearize_pybullet_depth(np.reshape(depth_buffer, (h, w)), CONFIG['near_plane'], CONFIG['far_plane'])
            
            # Draw wrist camera coordinate system on simulation (skip if this IS the wrist camera)
            if not cam_data.get("is_wrist", False) and wrist_cam_transforms is not None and i < len(wrist_cam_transforms):
                T_base_wrist = wrist_cam_transforms[i]
                wrist_pos_base = T_base_wrist[:3, 3]
                wrist_R_base = T_base_wrist[:3, :3]
                
                # Define coordinate frame axes (length in meters)
                axis_length = 0.08
                axes_cam = np.array([
                    [axis_length, 0, 0],  # X-axis (red)
                    [0, axis_length, 0],  # Y-axis (green)
                    [0, 0, axis_length]   # Z-axis (blue)
                ])
                
                # Transform axes to world frame
                axes_world = wrist_R_base @ axes_cam.T
                axes_world = axes_world.T + wrist_pos_base
                
                # Transform to camera frame
                cam_T_world = np.linalg.inv(cam_data["world_T_cam"])
                
                # Project origin and axes
                wrist_pos_hom = np.append(wrist_pos_base, 1.0)
                origin_cam = cam_T_world @ wrist_pos_hom
                
                if origin_cam[2] > 0 and cam_data["intrinsics"]:
                    fx, cx, fy, cy = [float(x) for x in cam_data["intrinsics"]]
                    intrinsics_mat = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                    
                    # Project origin
                    origin_pixel = intrinsics_mat @ origin_cam[:3]
                    origin_pixel = origin_pixel[:2] / origin_pixel[2]
                    
                    # Project axes endpoints
                    axes_pixels = []
                    for axis_world in axes_world:
                        axis_hom = np.append(axis_world, 1.0)
                        axis_cam = cam_T_world @ axis_hom
                        if axis_cam[2] > 0:
                            axis_pixel = intrinsics_mat @ axis_cam[:3]
                            axis_pixel = axis_pixel[:2] / axis_pixel[2]
                            axes_pixels.append(axis_pixel)
                        else:
                            axes_pixels.append(None)
                    
                    # Draw coordinate frame
                    if len(axes_pixels) == 3 and all(p is not None for p in axes_pixels):
                        draw_coordinate_frame(sim_rgb_bgr, origin_pixel, 
                                            axes_pixels[0], axes_pixels[1], axes_pixels[2])
            
            # SAVE: Sim Only
            cv2.imwrite(os.path.join(dirs["sim_rgb"], cam_id, f"sim_{i:05d}.png"), sim_rgb_bgr)
            
            # 2. Retrieve Real Data
            real_rgb_bgr = None
            real_depth_m = None
            
            if cam_id in svo_readers:
                reader = svo_readers[cam_id]
                zed = reader["zed"]
                if zed.grab(reader["runtime"]) == sl.ERROR_CODE.SUCCESS:
                    zed.retrieve_image(reader["mat_rgb"], sl.VIEW.LEFT, resolution=sl.Resolution(w, h))
                    real_rgb_bgr = reader["mat_rgb"].get_data()[:, :, :3] 
                    
                    zed.retrieve_measure(reader["mat_depth"], sl.MEASURE.DEPTH, resolution=sl.Resolution(w, h))
                    real_depth_m = reader["mat_depth"].get_data()
                    real_depth_m = np.nan_to_num(real_depth_m, nan=viz_max, posinf=viz_max, neginf=0.0)

                    # SAVE: Real Only
                    cv2.imwrite(os.path.join(dirs["real_rgb"], cam_id, f"real_{i:05d}.png"), real_rgb_bgr)
            
            # 3. Overlays & Viz
            if real_rgb_bgr is not None:
                # -- Overlay (Sim + Real) --
                seg_img = np.reshape(seg, (h, w))
                robot_mask = (seg_img == robot_id)
                
                overlay_rgb = real_rgb_bgr.copy()
                roi_real = real_rgb_bgr[robot_mask]
                roi_sim = sim_rgb_bgr[robot_mask]
                
                # Stronger Sim (0.9)
                blended = cv2.addWeighted(roi_real, 0.1, roi_sim, 0.9, 0)
                overlay_rgb[robot_mask] = blended
                
                # Draw wrist camera coordinate system on overlay
                if wrist_cam_transforms is not None and i < len(wrist_cam_transforms):
                    T_base_wrist = wrist_cam_transforms[i]
                    wrist_pos_base = T_base_wrist[:3, 3]
                    wrist_R_base = T_base_wrist[:3, :3]
                    
                    # Define coordinate frame axes (length in meters)
                    axis_length = 0.08
                    axes_cam = np.array([
                        [axis_length, 0, 0],  # X-axis (red)
                        [0, axis_length, 0],  # Y-axis (green)
                        [0, 0, axis_length]   # Z-axis (blue)
                    ])
                    
                    # Transform axes to world frame
                    axes_world = wrist_R_base @ axes_cam.T
                    axes_world = axes_world.T + wrist_pos_base
                    
                    # Transform to external camera frame
                    world_T_cam = cam_data["world_T_cam"]
                    cam_T_world = np.linalg.inv(world_T_cam)
                    
                    # Project origin and axes
                    wrist_pos_hom = np.append(wrist_pos_base, 1.0)
                    origin_cam = cam_T_world @ wrist_pos_hom
                    
                    if i == 0:  # Debug first frame
                        print(f"[DEBUG] Frame {i}, Cam {cam_id}:")
                        print(f"  Wrist pos (base): {wrist_pos_base}")
                        print(f"  Wrist pos (cam frame): {origin_cam[:3]}")
                        print(f"  Z (depth): {origin_cam[2]}")
                        print(f"  Has intrinsics: {cam_data['intrinsics'] is not None}")
                    
                    # Project to image plane
                    if origin_cam[2] > 0 and cam_data["intrinsics"]:
                        fx, cx, fy, cy = [float(x) for x in cam_data["intrinsics"]]
                        intrinsics_mat = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                        
                        # Project origin
                        origin_pixel = intrinsics_mat @ origin_cam[:3]
                        origin_pixel = origin_pixel[:2] / origin_pixel[2]
                        
                        # Project axes endpoints
                        axes_pixels = []
                        for axis_world in axes_world:
                            axis_hom = np.append(axis_world, 1.0)
                            axis_cam = cam_T_world @ axis_hom
                            if axis_cam[2] > 0:
                                axis_pixel = intrinsics_mat @ axis_cam[:3]
                                axis_pixel = axis_pixel[:2] / axis_pixel[2]
                                axes_pixels.append(axis_pixel)
                            else:
                                axes_pixels.append(None)
                        
                        if i == 0:  # Debug first frame
                            print(f"  Pixel coords: ({int(origin_pixel[0])}, {int(origin_pixel[1])})")
                            print(f"  Image bounds: ({w}, {h})")
                            print(f"  In bounds: {0 <= origin_pixel[0] < w and 0 <= origin_pixel[1] < h}")
                        
                        # Draw coordinate frame
                        if len(axes_pixels) == 3 and all(p is not None for p in axes_pixels):
                            draw_coordinate_frame(overlay_rgb, origin_pixel, 
                                                axes_pixels[0], axes_pixels[1], axes_pixels[2])
                            
                            # Add label
                            ox, oy = int(origin_pixel[0]), int(origin_pixel[1])
                            if 0 <= ox < w and 0 <= oy < h:
                                cv2.putText(overlay_rgb, "WRIST CAM", (ox + 15, oy - 15), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                                
                                if i == 0:
                                    print(f"  ✓ COORDINATE FRAME DRAWN!")
                        else:
                            if i == 0:
                                print(f"  ✗ Some axes behind camera or out of bounds")
                    else:
                        if i == 0:
                            print(f"  ✗ Behind camera or no intrinsics")
                
                cv2.imwrite(os.path.join(dirs["overlay"], cam_id, f"overlay_{i:05d}.png"), overlay_rgb)
                
                # -- Depth Viz --
                # Real = Grayscale (White is close)
                real_d_clipped = np.clip(real_depth_m, 0, viz_max)
                real_d_vis = 255 - (real_d_clipped / viz_max * 255).astype(np.uint8)
                depth_bg_bgr = cv2.cvtColor(real_d_vis, cv2.COLOR_GRAY2BGR)

                # Sim = JET Colormap
                sim_d_clipped = np.clip(sim_depth_m, 0, viz_max)
                sim_d_inv = 255 - (sim_d_clipped / viz_max * 255).astype(np.uint8)
                sim_d_color = cv2.applyColorMap(sim_d_inv, cv2.COLORMAP_JET)
                
                final_depth_viz = depth_bg_bgr.copy()
                final_depth_viz[robot_mask] = sim_d_color[robot_mask]
                
                # Draw wrist camera coordinate system on depth viz (skip if this IS the wrist camera)
                if not cam_data.get("is_wrist", False) and wrist_cam_transforms is not None and i < len(wrist_cam_transforms):
                    T_base_wrist = wrist_cam_transforms[i]
                    wrist_pos_base = T_base_wrist[:3, 3]
                    wrist_R_base = T_base_wrist[:3, :3]
                    
                    # Define coordinate frame axes
                    axis_length = 0.08
                    axes_cam = np.array([
                        [axis_length, 0, 0],  # X-axis
                        [0, axis_length, 0],  # Y-axis
                        [0, 0, axis_length]   # Z-axis
                    ])
                    
                    # Transform axes to world frame
                    axes_world = wrist_R_base @ axes_cam.T
                    axes_world = axes_world.T + wrist_pos_base
                    
                    # Transform to camera frame
                    world_T_cam = cam_data["world_T_cam"]
                    cam_T_world = np.linalg.inv(world_T_cam)
                    
                    # Project origin and axes
                    wrist_pos_hom = np.append(wrist_pos_base, 1.0)
                    origin_cam = cam_T_world @ wrist_pos_hom
                    
                    if origin_cam[2] > 0 and cam_data["intrinsics"]:
                        fx, cx, fy, cy = [float(x) for x in cam_data["intrinsics"]]
                        intrinsics_mat = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                        
                        # Project origin
                        origin_pixel = intrinsics_mat @ origin_cam[:3]
                        origin_pixel = origin_pixel[:2] / origin_pixel[2]
                        
                        # Project axes endpoints
                        axes_pixels = []
                        for axis_world in axes_world:
                            axis_hom = np.append(axis_world, 1.0)
                            axis_cam = cam_T_world @ axis_hom
                            if axis_cam[2] > 0:
                                axis_pixel = intrinsics_mat @ axis_cam[:3]
                                axis_pixel = axis_pixel[:2] / axis_pixel[2]
                                axes_pixels.append(axis_pixel)
                            else:
                                axes_pixels.append(None)
                        
                        # Draw coordinate frame
                        if len(axes_pixels) == 3 and all(p is not None for p in axes_pixels):
                            draw_coordinate_frame(final_depth_viz, origin_pixel, 
                                                axes_pixels[0], axes_pixels[1], axes_pixels[2])
                
                cv2.imwrite(os.path.join(dirs["depth_viz"], cam_id, f"depth_{i:05d}.png"), final_depth_viz)

    p.disconnect()
    for r in svo_readers.values():
        r["zed"].close()
    h5_file.close()
    print("[INFO] Done.")

if __name__ == "__main__":
    main()


