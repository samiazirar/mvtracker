import numpy as np
import json
import os
import glob
import sys
import h5py
import trimesh
from scipy.spatial.transform import Rotation as R
import rerun as rr
from PIL import Image
import torch
from torchvision.ops import box_convert

# ZED SDK
import pyzed.sl as sl

# Grounding DINO
sys.path.append("/workspace/third_party/groundingdino-cu128")
from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as T

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    # Inputs
    "h5_path": "/data/droid/data/droid_raw/1.0.1/ILIAD/success/2023-06-11/Sun_Jun_11_15:52:37_2023/trajectory.h5",
    "extrinsics_json_path": "/data/droid/calib_and_annot/droid/cam2base_extrinsic_superset.json",
    "recordings_dir": "/data/droid/data/droid_raw/1.0.1/ILIAD/success/2023-06-11/Sun_Jun_11_15:52:37_2023/recordings/SVO",
    "metadata_path": None, # Auto-discover

    # Output
    "rrd_output_path": "point_clouds/droid_full_fusion_gripper_with_mask.rrd",
    
    # Depth Filtering
    "wrist_max_depth": 0.75, # Meters (Close range for manipulation)
    "ext_max_depth": 1.5,   # Meters (Wider context)
    "min_depth": 0.1,      # Meters (Minimum valid depth)
    "min_depth_wrist": 0.001, # Meters (Wrist cam can see closer)
    "max_frames": 250,      # Max frames to process
    "radii_size": 0.001,    # Point size in meters
    # "gripper_wrist_offset": [0, 0, 0.014], # Meters (in the frame of the end effector)

    # Object Detection
    "object_prompt": "drill",
    "box_threshold": 0.35,
    "text_threshold": 0.25,
    "gdino_config": "/workspace/third_party/groundingdino-cu128/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "gdino_weights": "/workspace/weights/groundingdino_swint_ogc.pth",
}

# =============================================================================
# HELPERS
# =============================================================================

def pose6_to_T(p):
    """Convert [x, y, z, roll, pitch, yaw] to 4x4 matrix."""
    x, y, z, roll, pitch, yaw = p
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    T[:3, :3] = R.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    return T

def rvec_tvec_to_matrix(val):
    pos = np.array(val[0:3])
    euler = np.array(val[3:6])
    R_mat = R.from_euler("xyz", euler).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = pos
    return T

def find_svo_for_camera(recordings_dir, cam_serial):
    patterns = [f"*{cam_serial}*.svo", f"*{cam_serial}*.svo2"]
    for pat in patterns:
        matches = glob.glob(os.path.join(recordings_dir, pat))
        if matches: return matches[0]
    return None

def find_episode_data_by_date(h5_path, json_path):
    parts = h5_path.split(os.sep)
    date_str = parts[-3] 
    timestamp_folder = parts[-2]
    ts_parts = timestamp_folder.split('_')
    time_str = next((part for part in ts_parts if ':' in part), "00:00:00")
    h, m, s = time_str.split(':')
    target_suffix = f"{date_str}-{h}h-{m}m-{s}s"
    
    with open(json_path, 'r') as f: data = json.load(f)
    for key in data.keys():
        if key.endswith(target_suffix): return data[key]
    for key in data.keys():
        if target_suffix.split('-')[-1] in key: return data[key]
    
    return None

def get_zed_intrinsics(zed):
    """Extracts intrinsics for the Pinhole model."""
    # TODO: Maybe get from the intrinscs json
    info = zed.get_camera_information()
    calib = info.camera_configuration.calibration_parameters.left_cam
    fx, fy = calib.fx, calib.fy
    cx, cy = calib.cx, calib.cy
    w, h = calib.image_size.width, calib.image_size.height
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return K, w, h

def get_filtered_cloud(zed, runtime, max_depth=2.0, min_depth=0.1):   
    """
    Returns points and colors. 
    Filters out anything > max_depth or <= 0 (behind camera).
    """
    mat_cloud = sl.Mat()
    err = zed.retrieve_measure(mat_cloud, sl.MEASURE.XYZRGBA)
    if err != sl.ERROR_CODE.SUCCESS: return None, None

    # 1. Get Data
    cloud_data = mat_cloud.get_data()
    xyz = cloud_data[:, :, :3].reshape(-1, 3)
    
    # 2. Get RGB
    mat_image = sl.Mat()
    zed.retrieve_image(mat_image, sl.VIEW.LEFT)
    image_data = mat_image.get_data()
    rgb = image_data[:, :, :3].reshape(-1, 3)

    # 3. Filter Depth (Z-axis in Camera Frame)
    # ZED Camera Frame: Z is forward. 
    # We want 0 < z < max_depth
    z_vals = xyz[:, 2]
    valid_mask = np.isfinite(xyz).all(axis=1) & (z_vals > min_depth) & (z_vals < max_depth)
    
    return xyz[valid_mask], rgb[valid_mask]

# =============================================================================
# OBJECT DETECTOR
# =============================================================================

class ObjectDetector:
    def __init__(self, config_path, checkpoint_path, device="cuda"):
        self.device = device
        if not os.path.exists(checkpoint_path):
            print(f"[INFO] Downloading GroundingDINO weights to {checkpoint_path}...")
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
            os.system(f"wget {url} -O {checkpoint_path}")
        
        self.model = load_model(config_path, checkpoint_path, device=device)

    def detect(self, image_numpy, prompt, box_threshold=0.35, text_threshold=0.25):
        # image_numpy: (H, W, 3) RGB
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        image_pil = Image.fromarray(image_numpy)
        image_tensor, _ = transform(image_pil, None)
        
        boxes, logits, phrases = predict(
            self.model,
            image_tensor,
            prompt,
            box_threshold,
            text_threshold,
            device=self.device
        )
        return boxes

def boxes_to_mask(boxes, H, W):
    mask = np.zeros((H, W), dtype=bool)
    if boxes.shape[0] == 0:
        return mask
        
    # Convert to xyxy pixel coords
    boxes_xyxy = boxes * torch.Tensor([W, H, W, H])
    boxes_xyxy = box_convert(boxes=boxes_xyxy, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    
    for box in boxes_xyxy:
        x1, y1, x2, y2 = box.astype(int)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(W, x2)
        y2 = min(H, y2)
        mask[y1:y2, x1:x2] = True
    return mask

# =============================================================================
# GRIPPER VISUALIZER
# =============================================================================

class GripperVisualizer:
    def __init__(self, root_path="world/gripper"):
        self.root_path = root_path
        self.meshes = {}
        # Paths relative to workspace root
        base_path = "/workspace/third_party/robotiq_arg85_description/meshes"
        self.mesh_files = {
            "base": f"{base_path}/robotiq_85_base_link_fine.STL",
            "outer_knuckle": f"{base_path}/outer_knuckle_fine.STL",
            "outer_finger": f"{base_path}/outer_finger_fine.STL",
            "inner_knuckle": f"{base_path}/inner_knuckle_fine.STL",
            "inner_finger": f"{base_path}/inner_finger_fine.STL",
        }
        self._load_meshes()

    def _load_meshes(self):
        for name, path in self.mesh_files.items():
            if os.path.exists(path):
                self.meshes[name] = trimesh.load(path)
            else:
                print(f"[WARN] Mesh not found: {path}")

    def init_rerun(self):
        # Log meshes once as static (relative to their parent transforms)
        # Base
        self._log_mesh("base", f"{self.root_path}/base/mesh")
        
        # Left Finger
        self._log_mesh("outer_knuckle", f"{self.root_path}/left_outer_knuckle/mesh")
        self._log_mesh("outer_finger", f"{self.root_path}/left_outer_knuckle/left_outer_finger/mesh")
        self._log_mesh("inner_knuckle", f"{self.root_path}/left_inner_knuckle/mesh")
        self._log_mesh("inner_finger", f"{self.root_path}/left_inner_knuckle/left_inner_finger/mesh")

        # Right Finger
        self._log_mesh("outer_knuckle", f"{self.root_path}/right_outer_knuckle/mesh")
        self._log_mesh("outer_finger", f"{self.root_path}/right_outer_knuckle/right_outer_finger/mesh")
        self._log_mesh("inner_knuckle", f"{self.root_path}/right_inner_knuckle/mesh")
        self._log_mesh("inner_finger", f"{self.root_path}/right_inner_knuckle/right_inner_finger/mesh")

    def _log_mesh(self, name, path):
        if name in self.meshes:
            mesh = self.meshes[name]
            rr.log(path, rr.Mesh3D(vertex_positions=mesh.vertices, vertex_normals=mesh.vertex_normals, triangle_indices=mesh.faces), static=True)

    def update(self, T_base_ee, gripper_pos):
        # 1. Update Root (End Effector)
        rr.log(self.root_path, rr.Transform3D(translation=T_base_ee[:3, 3], mat3x3=T_base_ee[:3, :3], axis_length=0.1))
        
        # 2. Calculate Joint Angles
        val = gripper_pos[0] if isinstance(gripper_pos, (list, np.ndarray)) else gripper_pos
        theta = val * 0.8 

        # 3. Update Parts Transforms
        
        # Left Outer Knuckle
        # Origin: 0.03060114 0 0.06279202
        # Axis: 0 -1 0 -> Rot: [0, -theta, 0]
        T_lok = np.eye(4)
        T_lok[:3, 3] = [0.03060114, 0, 0.06279202]
        T_lok[:3, :3] = R.from_rotvec([0, -theta, 0]).as_matrix()
        rr.log(f"{self.root_path}/left_outer_knuckle", rr.Transform3D(translation=T_lok[:3, 3], mat3x3=T_lok[:3, :3]))

        # Left Outer Finger (Fixed relative to knuckle)
        # Origin: 0.03169104 0 -0.00193396
        T_lof = np.eye(4)
        T_lof[:3, 3] = [0.03169104, 0, -0.00193396]
        rr.log(f"{self.root_path}/left_outer_knuckle/left_outer_finger", rr.Transform3D(translation=T_lof[:3, 3]))

        # Left Inner Knuckle
        # Origin: 0.0127 0 0.0693
        # Axis: 0 -1 0 -> Rot: [0, -theta, 0] (Mimic 1)
        T_lik = np.eye(4)
        T_lik[:3, 3] = [0.0127, 0, 0.0693]
        T_lik[:3, :3] = R.from_rotvec([0, -theta, 0]).as_matrix()
        rr.log(f"{self.root_path}/left_inner_knuckle", rr.Transform3D(translation=T_lik[:3, 3], mat3x3=T_lik[:3, :3]))

        # Left Inner Finger
        # Origin: 0.03458531 0 0.04549702
        # Axis: 0 -1 0 -> Rot: [0, theta, 0] (Mimic -1)
        T_lif = np.eye(4)
        T_lif[:3, 3] = [0.03458531, 0, 0.04549702]
        T_lif[:3, :3] = R.from_rotvec([0, theta, 0]).as_matrix()
        rr.log(f"{self.root_path}/left_inner_knuckle/left_inner_finger", rr.Transform3D(translation=T_lif[:3, 3], mat3x3=T_lif[:3, :3]))

        # Right Outer Knuckle
        # Origin: -0.03060114 0 0.06279202
        # Base Rot: Z 180
        # Axis: 0 1 0 -> Rot: [0, -theta, 0] (Mimic -1)
        T_rok_origin = np.eye(4)
        T_rok_origin[:3, 3] = [-0.03060114, 0, 0.06279202]
        T_rok_origin[:3, :3] = R.from_euler('z', np.pi).as_matrix()
        
        R_rok_joint = R.from_rotvec([0, -theta, 0]).as_matrix()
        T_rok = T_rok_origin.copy()
        T_rok[:3, :3] = T_rok[:3, :3] @ R_rok_joint
        
        rr.log(f"{self.root_path}/right_outer_knuckle", rr.Transform3D(translation=T_rok[:3, 3], mat3x3=T_rok[:3, :3]))

        # Right Outer Finger (Fixed)
        # Origin: 0.03169104 0 -0.00193396
        T_rof = np.eye(4)
        T_rof[:3, 3] = [0.03169104, 0, -0.00193396]
        rr.log(f"{self.root_path}/right_outer_knuckle/right_outer_finger", rr.Transform3D(translation=T_rof[:3, 3]))

        # Right Inner Knuckle
        # Origin: -0.0127 0 0.0693
        # Base Rot: Z 180
        # Axis: 0 1 0 -> Rot: [0, -theta, 0] (Mimic -1)
        T_rik_origin = np.eye(4)
        T_rik_origin[:3, 3] = [-0.0127, 0, 0.0693]
        T_rik_origin[:3, :3] = R.from_euler('z', np.pi).as_matrix()
        
        R_rik_joint = R.from_rotvec([0, -theta, 0]).as_matrix()
        T_rik = T_rik_origin.copy()
        T_rik[:3, :3] = T_rik[:3, :3] @ R_rik_joint
        
        rr.log(f"{self.root_path}/right_inner_knuckle", rr.Transform3D(translation=T_rik[:3, 3], mat3x3=T_rik[:3, :3]))

        # Right Inner Finger
        # Origin: 0.03410605 0 0.04585739
        # Axis: 0 1 0 -> Rot: [0, theta, 0] (Mimic 1)
        T_rif = np.eye(4)
        T_rif[:3, 3] = [0.03410605, 0, 0.04585739]
        T_rif[:3, :3] = R.from_rotvec([0, theta, 0]).as_matrix()
        rr.log(f"{self.root_path}/right_inner_knuckle/right_inner_finger", rr.Transform3D(translation=T_rif[:3, 3], mat3x3=T_rif[:3, :3]))

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=== DROID Full Fusion (Wrist + External) ===")
    rr.init("droid_full_fusion_with_object_tracking", spawn=True)
    rr.save(CONFIG['rrd_output_path'])
    
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

    # Init Object Detector
    detector = ObjectDetector(CONFIG['gdino_config'], CONFIG['gdino_weights'])

    # --- 2. Calculate Wrist Transforms ---
    wrist_cam_transforms = []
    wrist_serial = None
    
    metadata_path = CONFIG['metadata_path']
    if metadata_path is None:
        episode_dir = os.path.dirname(CONFIG['h5_path'])
        metadata_files = glob.glob(os.path.join(episode_dir, "metadata_*.json"))
        if metadata_files: metadata_path = metadata_files[0]

    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f: meta = json.load(f)
        wrist_serial = str(meta.get("wrist_cam_serial", ""))
        wrist_pose_t0 = meta.get("wrist_cam_extrinsics")

        if wrist_pose_t0:
            T_base_cam0 = pose6_to_T(wrist_pose_t0)
            T_base_ee0 = pose6_to_T(cartesian_positions[0])
            # Constant offset: EE -> Camera
            T_ee_cam = np.linalg.inv(T_base_ee0) @ T_base_cam0

            for t in range(num_frames):
                T_base_ee_t = pose6_to_T(cartesian_positions[t])
                wrist_cam_transforms.append(T_base_ee_t @ T_ee_cam)
    
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
                    "world_T_cam": rvec_tvec_to_matrix(transform_list)
                }

    # B. Wrist Camera
    if wrist_serial:
        svo = find_svo_for_camera(CONFIG['recordings_dir'], wrist_serial)
        if svo:
            print(f"[INFO] Found Wrist Camera SVO: {wrist_serial}")
            cameras[wrist_serial] = {
                "type": "wrist",
                "svo": svo,
                "transforms": wrist_cam_transforms # List of 4x4 matrices
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
            #explain?
            data['zed'] = zed
            data['runtime'] = sl.RuntimeParameters()
            data['K'], data['w'], data['h'] = get_zed_intrinsics(zed) 
            active_cams[serial] = data
        else:
            print(f"[ERROR] Failed to open {serial}")

    # --- 4. Render Loop ---
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

                #manual transform (not needed)
                # xyz_world = (T[:3, :3] @ xyz.T).T + T[:3, 3]
                # Log the transform to Rerun
                # This moves the "frame" in the viewer
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
                
                # 2. Get Local Points
                xyz, rgb = get_filtered_cloud(zed, cam['runtime'], CONFIG['wrist_max_depth'], CONFIG['min_depth_wrist'])
                if xyz is None: continue

                # 3. Log Points as CHILD of wrist_cam
                # We do NOT transform these manually. Rerun moves them because they are children of "world/wrist_cam"
                rr.log(
                    "world/wrist_cam/points",
                    rr.Points3D(xyz, colors=rgb, radii=CONFIG['radii_size']
                )
                )

            # -- EXTERNAL CAMERA LOGIC --
            else:
                # 1. Get Image & Cloud Data
                mat_image = sl.Mat()
                zed.retrieve_image(mat_image, sl.VIEW.LEFT)
                image_data = mat_image.get_data() # (H, W, 4)
                rgb_full = image_data[:, :, :3] # RGB
                
                # Detect Object
                boxes = detector.detect(rgb_full, CONFIG['object_prompt'], CONFIG['box_threshold'], CONFIG['text_threshold'])
                obj_mask = boxes_to_mask(boxes, rgb_full.shape[0], rgb_full.shape[1])
                
                # Get Cloud
                mat_cloud = sl.Mat()
                err = zed.retrieve_measure(mat_cloud, sl.MEASURE.XYZRGBA)
                if err != sl.ERROR_CODE.SUCCESS: continue
                
                # Get point cloud data
                cloud_data = mat_cloud.get_data()
                xyz_full = cloud_data[:, :, :3].reshape(-1, 3)
                rgb_sub = image_data[:, :, :3].reshape(-1, 3)
                obj_mask_sub = obj_mask.reshape(-1)
                
                # Filter Depth
                z_vals = xyz_full[:, 2]
                valid_depth = np.isfinite(xyz_full).all(axis=1) & (z_vals > CONFIG['min_depth']) & (z_vals < CONFIG['ext_max_depth'])
                
                # Scene Points (All valid depth points)
                xyz_scene = xyz_full[valid_depth]
                rgb_scene = rgb_sub[valid_depth]
                
                # Object Points (Valid depth AND inside mask)
                valid_obj = valid_depth & obj_mask_sub
                xyz_obj = xyz_full[valid_obj]
                rgb_obj = rgb_sub[valid_obj]

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

                # 3. Log to World (as child of transform now)
                rr.log(
                    f"world/external_cams/{serial}/points",
                    rr.Points3D(xyz_scene, colors=rgb_scene, radii=CONFIG['radii_size'])
                )
                
                # 4. Log Object Points
                if len(xyz_obj) > 0:
                    rr.log(
                        f"world/external_cams/{serial}/object_points",
                        rr.Points3D(xyz_obj, colors=rgb_obj, radii=CONFIG['radii_size']*2)
                    )

    # Cleanup
    for c in active_cams.values(): c['zed'].close()
    print("[SUCCESS] Done.")
    print(f"[INFO] RRD saved to: {CONFIG['rrd_output_path']}")

if __name__ == "__main__":
    main()