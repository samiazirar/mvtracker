import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
from .transforms import pose6_to_T, external_cam_to_world, precompute_wrist_trajectory
from .camera_utils import get_filtered_cloud

# ==============================================================================
# CORE ALGORITHMS
# ==============================================================================

def run_point_to_plane_icp(target_points, source_points, initial_guess=np.eye(4), threshold=0.02):
    """
    Standard Point-to-Plane ICP.
    """
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_points)
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_points)

    # Pre-process
    voxel_size = 0.005
    source = source.voxel_down_sample(voxel_size)
    target = target.voxel_down_sample(voxel_size)
    
    radius_normal = voxel_size * 2
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    # Run ICP
    reg = o3d.pipelines.registration.registration_icp(
        source, target, threshold, initial_guess,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
    )
    return reg.transformation, reg.fitness

def average_transforms(transform_list):
    """
    Mathematically averages a list of 4x4 transformation matrices.
    """
    if not transform_list: return None
    
    # 1. Average Translation (Simple Mean)
    ts = np.array([T[:3, 3] for T in transform_list])
    t_mean = np.mean(ts, axis=0)
    
    # 2. Average Rotation (Using Scipy for proper SO(3) averaging)
    rs = [T[:3, :3] for T in transform_list]
    r_mean = R.from_matrix(rs).mean().as_matrix()
    
    T_final = np.eye(4)
    T_final[:3, :3] = r_mean
    T_final[:3, 3] = t_mean
    return T_final

def accumulate_frames(cam_data, num_frames=10, start_frame=10, config=None):
    """
    Helper to grab N frames from a STATIC camera and merge them.
    Returns: (N_points, 3) numpy array
    """
    points_accumulator = []
    
    # We skip a few frames between grabs to get diversity (if there's noise/flicker)
    step = 2 
    
    for i in range(num_frames):
        frame_idx = start_frame + (i * step)
        cam_data['zed'].set_svo_position(frame_idx)
        if cam_data['zed'].grab(cam_data['runtime']) != 0: continue
            
        xyz, _ = get_filtered_cloud(
            cam_data['zed'], cam_data['runtime'], 
            config.get('ext_max_depth', 2.0)
        )
        if xyz is not None:
            points_accumulator.append(xyz)
            
    if not points_accumulator: return None
    return np.vstack(points_accumulator)

# ==============================================================================
# OPTIMIZATION PIPELINES
# ==============================================================================

def optimize_external_cameras_multi_frame(active_cams, config):
    """
    Aligns all external cameras using multi-frame accumulation.
    """
    print("\n[OPTIMIZE] 1. Multi-Frame External Calibration...")
    ext_cams = [s for s, d in active_cams.items() if d['type'] == 'external']
    if len(ext_cams) < 2:
        print("[SKIP] Need >1 external camera.")
        return

    # 1. Build ANCHOR Map (Cam 0)
    anchor_serial = ext_cams[0]
    print(f"   -> Building Anchor Map from {anchor_serial} (20 frames)...")
    
    # We stack 20 frames to make the "Ground Truth" extremely dense and noise-free
    anchor_local = accumulate_frames(active_cams[anchor_serial], num_frames=20, start_frame=20, config=config)
    if anchor_local is None: return

    # Transform to World (Anchor defines the world)
    if 'world_T_cam' in active_cams[anchor_serial]:
        T_anchor = active_cams[anchor_serial]['world_T_cam']
    else:
        # Fallback if world_T_cam is not set (should not happen based on current logic)
        if 'extrinsic_params' in active_cams[anchor_serial]:
            T_anchor = external_cam_to_world(active_cams[anchor_serial]['extrinsic_params'])
        else:
            print(f"[ERROR] No transform for anchor {anchor_serial}")
            return
        
    # Apply T_anchor to points
    anchor_world = (T_anchor[:3, :3] @ anchor_local.T).T + T_anchor[:3, 3]

    # 2. Align Others
    for serial in ext_cams[1:]:
        print(f"   -> Aligning {serial} to Anchor (using 20 frames)...")
        cam = active_cams[serial]
        
        # Accumulate source frames
        source_local = accumulate_frames(cam, num_frames=20, start_frame=20, config=config)
        if source_local is None: continue

        # Get Current Guess
        T_guess = cam.get('world_T_cam')
        if T_guess is None:
             if 'extrinsic_params' in cam:
                 T_guess = external_cam_to_world(cam['extrinsic_params'])
             else:
                 print(f"[WARN] No initial transform for {serial}, skipping.")
                 continue
            
        # Run ICP (Source Local -> Target World)
        refined_T, fitness = run_point_to_plane_icp(anchor_world, source_local, T_guess)
        
        print(f"      [RESULT] Fitness: {fitness:.4f}")
        cam['world_T_cam'] = refined_T
        
    print("[SUCCESS] External cameras aligned.\n")


def optimize_wrist_multi_frame(active_cams, cartesian_positions, config):
    """
    Calculates T_ee_cam by averaging results from multiple robot poses.
    """
    print("[OPTIMIZE] 2. Multi-Frame Wrist Calibration...")
    
    # 1. Identify Wrist Cam
    wrist_serial = None
    for s, d in active_cams.items():
        if d['type'] == 'wrist': wrist_serial = s
    if not wrist_serial: return

    # 2. Build GLOBAL Map (All External Cams)
    # We grab a static chunk of the scene to use as the target
    print("   -> Building Global Target Map...")
    map_points = []
    for s, d in active_cams.items():
        if d['type'] == 'external':
            pts = accumulate_frames(d, num_frames=10, start_frame=30, config=config)
            if pts is not None:
                # Transform to World (using optimized extrinsics)
                T = d.get('world_T_cam')
                if T is None and 'extrinsic_params' in d:
                    T = external_cam_to_world(d['extrinsic_params'])
                
                if T is not None:
                    pts_world = (T[:3, :3] @ pts.T).T + T[:3, 3]
                    map_points.append(pts_world)
                
    if not map_points: return
    global_map = np.vstack(map_points)

    # 3. Test Multiple Wrist Frames
    # We pick frames where we know the robot is looking at the scene (e.g. 40, 60, 80, 100)
    test_frames = [40, 60, 80, 100, 120] 
    
    T_ee_cam_candidates = []
    
    print(f"   -> Testing {len(test_frames)} wrist frames...")
    wrist_cam = active_cams[wrist_serial]
    old_T_ee_cam = wrist_cam.get('T_ee_cam')
    
    if old_T_ee_cam is None:
        print("[WARN] No initial T_ee_cam found for wrist optimization.")
        return
    
    for f_idx in test_frames:
        if f_idx >= len(cartesian_positions): continue
        
        # Grab Wrist Cloud
        wrist_cam['zed'].set_svo_position(f_idx)
        if wrist_cam['zed'].grab(wrist_cam['runtime']) != 0: continue
        xyz_wrist, _ = get_filtered_cloud(wrist_cam['zed'], wrist_cam['runtime'], config.get('wrist_max_depth', 0.8))
        if xyz_wrist is None or len(xyz_wrist) < 1000: continue
        
        # Initial Guess: FK @ Old_Calibration
        T_base_ee = pose6_to_T(cartesian_positions[f_idx])
        guess_world_T_wrist = T_base_ee @ old_T_ee_cam
        
        # Run ICP (Find where the wrist ACTUALLY is in the world)
        refined_world_T_wrist, fitness = run_point_to_plane_icp(global_map, xyz_wrist, guess_world_T_wrist)
        
        if fitness < 0.4: # Skip bad matches
            print(f"      [WARN] Frame {f_idx} poor match ({fitness:.2f}), skipping.")
            continue
            
        # Back-calculate T_ee_cam
        # New_T_ee_cam = inv(T_base_ee) @ Refined_World_T_Wrist
        T_ee_cam_new = np.linalg.inv(T_base_ee) @ refined_world_T_wrist
        T_ee_cam_candidates.append(T_ee_cam_new)
        print(f"      [OK] Frame {f_idx} processed.")

    # 4. Average the Results
    if len(T_ee_cam_candidates) > 0:
        final_T_ee_cam = average_transforms(T_ee_cam_candidates)
        print("[SUCCESS] Calculated Robust Average Calibration.")
        
        # Apply
        wrist_cam['T_ee_cam'] = final_T_ee_cam
        
        # Re-calc trajectory
        new_transforms = precompute_wrist_trajectory(
            cartesian_positions, 
            final_T_ee_cam
        )
        wrist_cam['transforms'] = new_transforms
    else:
        print("[FAIL] Could not optimize wrist (no good frames found).")

    # Reset cameras
    for c in active_cams.values(): c['zed'].set_svo_position(0)
