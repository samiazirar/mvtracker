import numpy as np
import os
import glob
import h5py
import yaml
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
    get_filtered_cloud
)


class MinimalGripperVisualizer:
    """Simplified gripper visualizer that renders fingers with highlighted contact surfaces."""
    
    def __init__(self, root_path="world/gripper", num_track_points=50):
        self.root_path = root_path
        base_path = "/workspace/third_party/robotiq_arg85_description/meshes"
        mesh_file = f"{base_path}/inner_finger_fine.STL"
        
        # ========== GRIPPER COLOR CONFIGURATION ==========
        self.gripper_color = [255, 255, 255]  # White for the finger body
        self.contact_color = [70, 130, 180]   # Blue for contact surface
        # =================================================
        
        self.finger_mesh = None
        self.contact_mesh = None
        self.num_track_points = num_track_points
        self.contact_points_local = None  # Points sampled from contact surface in local frame
        
        if os.path.exists(mesh_file):
            self.finger_mesh = trimesh.load(mesh_file)
            # Extract the full contact surface (all faces facing -Y direction)
            self.contact_mesh = self._extract_contact_surface(self.finger_mesh)
            # Sample points from contact surface for tracking
            if self.contact_mesh is not None:
                self.contact_points_local = self._sample_contact_points(self.contact_mesh, num_track_points)
        else:
            print(f"[WARN] Mesh not found: {mesh_file}")
    
    def _extract_contact_surface(self, mesh):
        """Extract the inner rubber contact pad (at Y=-0.011, the innermost surface)."""
        vertices = mesh.vertices
        
        # Find faces at the VERY innermost surface (Y close to -0.011)
        # This is the actual rubber contact pad that touches objects
        contact_face_indices = []
        for i, face in enumerate(mesh.faces):
            face_verts = vertices[face]
            y_min = face_verts[:, 1].min()
            # The innermost flat surface at Y = -0.011
            if y_min < -0.0095:
                contact_face_indices.append(i)
        
        if len(contact_face_indices) > 0:
            contact_submesh = mesh.submesh([contact_face_indices], only_watertight=False)[0]
            print(f"[INFO] Extracted inner contact pad: {len(contact_submesh.vertices)} vertices, {len(contact_submesh.faces)} faces")
            return contact_submesh
        else:
            print("[WARN] Could not extract contact surface")
            return None
    
    def _sample_contact_points(self, mesh, num_points):
        """Sample points uniformly from the contact surface mesh."""
        # Use mesh vertices directly if we have few enough, otherwise sample
        if len(mesh.vertices) <= num_points:
            points = mesh.vertices.copy()
        else:
            # Sample points on the mesh surface
            points, _ = trimesh.sample.sample_surface(mesh, num_points)
        print(f"[INFO] Sampled {len(points)} contact points for tracking")
        return points

    def init_rerun(self):
        """Initialize Rerun visualization with finger meshes and contact surfaces."""
        # Log the full white finger meshes
        if self.finger_mesh is not None:
            num_vertices = len(self.finger_mesh.vertices)
            vertex_colors = np.tile(self.gripper_color, (num_vertices, 1)).astype(np.uint8)
            
            rr.log(
                f"{self.root_path}/left_inner_knuckle/left_inner_finger/mesh",
                rr.Mesh3D(
                    vertex_positions=self.finger_mesh.vertices,
                    vertex_normals=self.finger_mesh.vertex_normals,
                    vertex_colors=vertex_colors,
                    triangle_indices=self.finger_mesh.faces
                ),
                static=True
            )
            rr.log(
                f"{self.root_path}/right_inner_knuckle/right_inner_finger/mesh",
                rr.Mesh3D(
                    vertex_positions=self.finger_mesh.vertices,
                    vertex_normals=self.finger_mesh.vertex_normals,
                    vertex_colors=vertex_colors,
                    triangle_indices=self.finger_mesh.faces
                ),
                static=True
            )
        
        # Log the blue contact surfaces (overlaid on the fingers)
        if self.contact_mesh is not None:
            num_vertices = len(self.contact_mesh.vertices)
            vertex_colors = np.tile(self.contact_color, (num_vertices, 1)).astype(np.uint8)
            
            rr.log(
                f"{self.root_path}/left_inner_knuckle/left_inner_finger/contact",
                rr.Mesh3D(
                    vertex_positions=self.contact_mesh.vertices,
                    vertex_normals=self.contact_mesh.vertex_normals,
                    vertex_colors=vertex_colors,
                    triangle_indices=self.contact_mesh.faces
                ),
                static=True
            )
            rr.log(
                f"{self.root_path}/right_inner_knuckle/right_inner_finger/contact",
                rr.Mesh3D(
                    vertex_positions=self.contact_mesh.vertices,
                    vertex_normals=self.contact_mesh.vertex_normals,
                    vertex_colors=vertex_colors,
                    triangle_indices=self.contact_mesh.faces
                ),
                static=True
            )
    
    def _compute_finger_transforms(self, T_base_ee, gripper_pos):
        """Compute the world-space transforms for both fingers."""
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
        """Get contact surface points in world coordinates for both fingers."""
        if self.contact_points_local is None:
            return None, None
        
        T_left, T_right = self._compute_finger_transforms(T_base_ee, gripper_pos)
        
        # Transform local points to world space
        # Points are in local finger frame, transform to world
        pts_local = np.hstack([self.contact_points_local, np.ones((len(self.contact_points_local), 1))])
        
        pts_left_world = (T_left @ pts_local.T).T[:, :3]
        pts_right_world = (T_right @ pts_local.T).T[:, :3]
        
        return pts_left_world, pts_right_world

    def update(self, T_base_ee, gripper_pos):
        """Update only the two final finger transforms."""
        # Update root transform
        rr.log(
            self.root_path,
            rr.Transform3D(
                translation=T_base_ee[:3, 3],
                mat3x3=T_base_ee[:3, :3],
                axis_length=0.1
            )
        )
        
        # Calculate joint angle
        val = gripper_pos[0] if isinstance(gripper_pos, (list, np.ndarray)) else gripper_pos
        theta = val * 0.8
        
        # Left inner knuckle transform
        T_lik = np.eye(4)
        T_lik[:3, 3] = [0.0127, 0, 0.0693]
        T_lik[:3, :3] = R.from_rotvec([0, -theta, 0]).as_matrix()
        rr.log(
            f"{self.root_path}/left_inner_knuckle",
            rr.Transform3D(translation=T_lik[:3, 3], mat3x3=T_lik[:3, :3])
        )
        
        # Left inner finger transform
        T_lif = np.eye(4)
        T_lif[:3, 3] = [0.03458531, 0, 0.04549702]
        T_lif[:3, :3] = R.from_rotvec([0, theta, 0]).as_matrix()
        rr.log(
            f"{self.root_path}/left_inner_knuckle/left_inner_finger",
            rr.Transform3D(translation=T_lif[:3, 3], mat3x3=T_lif[:3, :3])
        )
        
        # Right inner knuckle transform
        T_rik_origin = np.eye(4)
        T_rik_origin[:3, 3] = [-0.0127, 0, 0.0693]
        T_rik_origin[:3, :3] = R.from_euler('z', np.pi).as_matrix()
        R_rik_joint = R.from_rotvec([0, -theta, 0]).as_matrix()
        T_rik = T_rik_origin.copy()
        T_rik[:3, :3] = T_rik[:3, :3] @ R_rik_joint
        rr.log(
            f"{self.root_path}/right_inner_knuckle",
            rr.Transform3D(translation=T_rik[:3, 3], mat3x3=T_rik[:3, :3])
        )
        
        # Right inner finger transform
        T_rif = np.eye(4)
        T_rif[:3, 3] = [0.03410605, 0, 0.04585739]
        T_rif[:3, :3] = R.from_rotvec([0, theta, 0]).as_matrix()
        rr.log(
            f"{self.root_path}/right_inner_knuckle/right_inner_finger",
            rr.Transform3D(translation=T_rif[:3, 3], mat3x3=T_rif[:3, :3])
        )


def main():
    # Load configuration
    with open('conversions/droid/config.yaml', 'r') as f:
        CONFIG = yaml.safe_load(f)
    
    print("=== DROID Full Fusion (Wrist + External) with Gripper Tracks ===")
    rr.init("droid_full_fusion", spawn=True)
    save_path = CONFIG['rrd_output_path']
    save_path = save_path.replace(".rrd", "")
    save_path = f"{save_path}_only_contact_surface_no_optimization.rrd"
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

    # Number of track points to sample from contact surface
    NUM_TRACK_POINTS = CONFIG.get('num_track_points', 24)  # Default 24 points
    
    # Init Gripper Viz (minimal version - only two final fingers)
    gripper_viz = MinimalGripperVisualizer(num_track_points=NUM_TRACK_POINTS)
    gripper_viz.init_rerun()
    
    # --- Prepare track storage ---
    # We'll store tracks for left and right finger contact points
    num_contact_pts = len(gripper_viz.contact_points_local) if gripper_viz.contact_points_local is not None else 0
    print(f"[INFO] Tracking {num_contact_pts} points per finger ({num_contact_pts * 2} total)")

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
            # Apply the same 90-degree rotation fix to the initial EE pose
            cartesian_t0 = cartesian_positions[0].copy()
            T_base_ee0 = pose6_to_T(cartesian_t0)
            R_fix = R.from_euler('z', 90, degrees=True).as_matrix()
            T_base_ee0[:3, :3] = T_base_ee0[:3, :3] @ R_fix
            
            # Calculate constant offset using the corrected pose
            T_base_cam0 = pose6_to_T(wrist_pose_t0)
            T_ee_cam = np.linalg.inv(T_base_ee0) @ T_base_cam0
            
            # Precompute all wrist camera poses with rotation fix applied
            wrist_cam_transforms = []
            for cart_pos in cartesian_positions:
                T_base_ee_t = pose6_to_T(cart_pos)
                T_base_ee_t[:3, :3] = T_base_ee_t[:3, :3] @ R_fix
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
            data['zed'] = zed
            data['runtime'] = sl.RuntimeParameters()
            data['K'], data['w'], data['h'] = get_zed_intrinsics(zed) 
            active_cams[serial] = data
        else:
            print(f"[ERROR] Failed to open {serial}")

    # --- 4. Render Loop & Collect Tracks ---
    max_frames = CONFIG["max_frames"]
    actual_frames = min(max_frames, num_frames)
    print(f"[INFO] Processing {actual_frames} frames...")
    
    # Storage for gripper tracks
    # tracks_3d: [T, N, 3] - 3D position of each point at each timestep
    # N = num_contact_pts * 2 (left + right finger)
    N_total = num_contact_pts * 2
    tracks_3d = np.zeros((actual_frames, N_total, 3), dtype=np.float32)
    gripper_poses = []  # Store end-effector poses for each frame
    
    # Also store the local points for reference
    contact_points_local = gripper_viz.contact_points_local.copy() if gripper_viz.contact_points_local is not None else None
    
    for i in range(actual_frames):
        if i % 10 == 0: print(f"Frame {i}")
        rr.set_time(timeline="frame_index", sequence=i)

        # Update Gripper (use end-effector pose directly)
        T_base_ee = pose6_to_T(cartesian_positions[i])
        # Rotate by 90 degrees to align
        R_fix = R.from_euler('z', 90, degrees=True).as_matrix()
        T_base_ee[:3, :3] = T_base_ee[:3, :3] @ R_fix
        gripper_viz.update(T_base_ee, gripper_positions[i])
        
        # Store the pose
        gripper_poses.append(T_base_ee.copy())
        
        # Get contact points in world space and store as tracks
        if num_contact_pts > 0:
            pts_left, pts_right = gripper_viz.get_contact_points_world(T_base_ee, gripper_positions[i])
            if pts_left is not None:
                tracks_3d[i, :num_contact_pts, :] = pts_left
                tracks_3d[i, num_contact_pts:, :] = pts_right

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
    
    # --- 5. Visualize Gripper Tracks ---
    print(f"[INFO] Visualizing {N_total} gripper contact tracks...")
    
    # Generate colors for tracks (left finger = blue, right finger = green)
    track_colors = np.zeros((N_total, 4), dtype=np.float32)
    track_colors[:num_contact_pts, :] = [0.2, 0.5, 1.0, 1.0]  # Blue for left
    track_colors[num_contact_pts:, :] = [0.2, 1.0, 0.5, 1.0]  # Green for right
    
    # Log tracks using rerun - per-frame visualization with line strips
    fps = CONFIG.get('fps', 30.0)
    for t in range(actual_frames):
        rr.set_time(timeline="frame_index", sequence=t)
        
        # Log track points at this frame
        rr.log(
            "world/gripper_tracks/points",
            rr.Points3D(
                positions=tracks_3d[t],
                colors=(track_colors[:, :3] * 255).astype(np.uint8),
                radii=0.003  # 3mm radius
            )
        )
        
        # Log track history as line strips (show trail)
        if t > 0:
            trail_length = min(t, 10)  # Show last 10 frames of trail
            for n in range(N_total):
                trail_points = tracks_3d[max(0, t - trail_length):t + 1, n, :]
                if len(trail_points) > 1:
                    # Create line segments
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
    
    # --- 6. Save Tracks to NPZ ---
    tracks_save_path = save_path.replace(".rrd", "_gripper_tracks.npz")
    gripper_poses_array = np.stack(gripper_poses, axis=0)  # [T, 4, 4]
    
    np.savez(
        tracks_save_path,
        # Track data
        tracks_3d=tracks_3d,  # [T, N, 3] - world-space positions
        contact_points_local=contact_points_local,  # [N/2, 3] - local mesh points
        
        # Pose data
        gripper_poses=gripper_poses_array,  # [T, 4, 4] - end-effector poses
        gripper_positions=gripper_positions[:actual_frames],  # [T] - gripper open/close
        cartesian_positions=cartesian_positions[:actual_frames],  # [T, 6] - raw pose data
        
        # Metadata
        num_frames=actual_frames,
        num_points_per_finger=num_contact_pts,
        fps=fps,
    )
    print(f"[INFO] Saved gripper tracks to: {tracks_save_path}")

    # Cleanup
    for c in active_cams.values(): c['zed'].close()
    print("[SUCCESS] Done.")
    print(f"[INFO] RRD saved to: {save_path}")


if __name__ == "__main__":
    main()
