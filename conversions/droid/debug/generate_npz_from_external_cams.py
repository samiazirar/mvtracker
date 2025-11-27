import numpy as np
import os
import glob
import h5py
import yaml
import sys
import cv2
import torch
from tqdm import tqdm
from dotenv import load_dotenv

# Add parent directory to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import (
    external_cam_to_world,
    find_svo_for_camera,
    find_episode_data_by_date,
    get_zed_intrinsics,
    transform_points
)
from utils.object_detector import ObjectDetector, Sam2Wrapper, get_vlm_description
import pyzed.sl as sl

load_dotenv()

def main():
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '../config.yaml')
    if not os.path.exists(config_path):
        print(f"[ERROR] Config file not found at {config_path}")
        return

    with open(config_path, 'r') as f:
        CONFIG = yaml.safe_load(f)

    # Init Detectors
    print("[INFO] Initializing Object Detectors...")
    object_detector = ObjectDetector(CONFIG['gdino_config'], CONFIG['gdino_weights'])
    sam2_wrapper = Sam2Wrapper()

    # --- Init Cameras ---
    cameras = {} # Holds external cameras
    
    # External Cameras
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

    if not active_cams:
        print("No active cameras found.")
        return

    # Sort cameras to have consistent order
    sorted_serials = sorted(active_cams.keys())
    print(f"[INFO] Active cameras: {sorted_serials}")
    
    # Data collection
    all_rgbs = []
    all_depths = []
    all_intrs = []
    all_extrs = []
    
    # Query points collection (from first frame)
    query_points_list = [] # [t, x, y, z]

    num_frames = CONFIG["max_frames"]
    
    print(f"[INFO] Processing frames...")
    
    frame_idx = 0
    vlm_prompt_text = None

    # We need to iterate until max_frames or end of stream
    pbar = tqdm(total=num_frames)
    
    while True:
        if frame_idx >= num_frames:
            break
            
        frame_rgbs = []
        frame_depths = []
        frame_intrs = []
        frame_extrs = []
        
        all_cams_success = True
        
        for serial in sorted_serials:
            cam = active_cams[serial]
            zed = cam['zed']
            
            if zed.grab(cam['runtime']) != sl.ERROR_CODE.SUCCESS:
                all_cams_success = False
                break
            
            # Retrieve RGB
            mat_image = sl.Mat()
            zed.retrieve_image(mat_image, sl.VIEW.LEFT)
            image_np = mat_image.get_data()
            image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGRA2RGB) # (H, W, 3)
            
            # Retrieve Depth
            mat_depth = sl.Mat()
            zed.retrieve_measure(mat_depth, sl.MEASURE.DEPTH)
            depth_map = mat_depth.get_data() # (H, W)
            
            # Retrieve Point Cloud (XYZ) for query points extraction
            mat_cloud = sl.Mat()
            zed.retrieve_measure(mat_cloud, sl.MEASURE.XYZ)
            xyz_map = mat_cloud.get_data() # (H, W, 4)
            xyz = xyz_map[:, :, :3]

            # Intrinsics
            K = cam['K']
            
            # Extrinsics (Camera to World)
            T_wc = cam['world_T_cam']
            
            # Store data
            frame_rgbs.append(image_rgb)
            frame_depths.append(depth_map)
            frame_intrs.append(K)
            frame_extrs.append(T_wc)
            
            # --- Query Points Extraction (First Frame Only) ---
            if frame_idx == 0:
                # Only run detection on the first camera to avoid duplicates/confusion?
                # Or run on all and aggregate?
                # Let's run on all.
                
                if vlm_prompt_text is None:
                    if CONFIG.get('debug_object_text'):
                        vlm_prompt_text = CONFIG['debug_object_text']
                        print(f"[INFO] Using debug object text: {vlm_prompt_text}")
                    else:
                        # Use VLM
                        print(f"[INFO] Querying VLM for cam {serial}...")
                        vlm_prompt_text = get_vlm_description(image_rgb, CONFIG['vlm_query_prompt'], os.getenv("OPENAI_API_KEY"), CONFIG.get('vlm_model', 'gpt-4o'))
                        print(f"[INFO] VLM Prompt: {vlm_prompt_text}")
                
                print(f"[INFO] Detecting objects for cam {serial}...")
                boxes, logits = object_detector.detect(image_rgb, vlm_prompt_text, CONFIG['box_threshold'])
                masks = sam2_wrapper.predict_mask(image_rgb, boxes) # (N_masks, H, W)
                
                # Combine masks
                if masks is not None and len(masks) > 0:
                    combined_mask = np.any(masks, axis=0) # (H, W)
                    
                    # Get points in mask
                    # Filter by depth and validity
                    valid_mask = np.isfinite(xyz).all(axis=2) & (depth_map > CONFIG['min_depth']) & (depth_map < CONFIG['ext_max_depth'])
                    final_mask = combined_mask & valid_mask
                    
                    selected_xyz = xyz[final_mask] # (N, 3)
                    
                    if len(selected_xyz) > 0:
                        # Subsample if too many
                        if len(selected_xyz) > 500:
                            indices = np.random.choice(len(selected_xyz), 500, replace=False)
                            selected_xyz = selected_xyz[indices]
                        
                        # Transform to World
                        selected_xyz_world = transform_points(selected_xyz, T_wc)
                        
                        # Add to query points list with t=0
                        # (N, 4) -> [t, x, y, z]
                        t_col = np.zeros((selected_xyz_world.shape[0], 1))
                        pts_with_t = np.hstack([t_col, selected_xyz_world])
                        query_points_list.append(pts_with_t)
                        print(f"[INFO] Added {len(pts_with_t)} query points from cam {serial}")

        if not all_cams_success:
            print(f"[WARN] Camera grab failed at frame {frame_idx}. Stopping.")
            break
            
        all_rgbs.append(np.stack(frame_rgbs)) # (V, H, W, 3)
        all_depths.append(np.stack(frame_depths)) # (V, H, W)
        all_intrs.append(np.stack(frame_intrs)) # (V, 3, 3)
        all_extrs.append(np.stack(frame_extrs)) # (V, 4, 4)
        
        frame_idx += 1
        pbar.update(1)

    pbar.close()

    # Stack everything
    # all_rgbs is list of (V, H, W, 3). Length T.
    # We want (V, T, H, W, 3)
    
    if not all_rgbs:
        print("[ERROR] No frames collected.")
        return

    print("[INFO] Stacking data...")
    rgbs_np = np.stack(all_rgbs, axis=1) # (V, T, H, W, 3)
    rgbs_np = np.transpose(rgbs_np, (0, 1, 4, 2, 3)) # (V, T, 3, H, W)
    depths_np = np.stack(all_depths, axis=1) # (V, T, H, W)
    
    # Check for NaN/Inf in depth
    nan_count = np.isnan(depths_np).sum()
    inf_count = np.isinf(depths_np).sum()
    print(f"[INFO] Depth stats: NaN={nan_count}, Inf={inf_count}, min={np.nanmin(depths_np)}, max={np.nanmax(depths_np)}")
    
    # Replace NaN/Inf with 0 (invalid depth)
    depths_np = np.nan_to_num(depths_np, nan=0.0, posinf=0.0, neginf=0.0)
    
    depths_np = np.expand_dims(depths_np, axis=2) # (V, T, 1, H, W)
    intrs_np = np.stack(all_intrs, axis=1) # (V, T, 3, 3)
    extrs_np = np.stack(all_extrs, axis=1) # (V, T, 4, 4)
    
    # Check for NaN/Inf in extrinsics
    if np.isnan(extrs_np).any() or np.isinf(extrs_np).any():
        print("[ERROR] Extrinsics contain NaN or Inf values!")
        return
    
    # Invert extrinsics (Camera-to-World -> World-to-Camera)
    extrs_np = np.linalg.inv(extrs_np)
    
    # Check again after inversion
    if np.isnan(extrs_np).any() or np.isinf(extrs_np).any():
        raise ValueError("Inverted extrinsics contain NaN or Inf values!")
    
    #TODO: check..
    extrs_np = extrs_np[:, :, :3, :] # (V, T, 3, 4)
    
    # Query points
    if query_points_list:
        query_points_np = np.vstack(query_points_list)
        
        # Check for NaN/Inf in query points
        nan_count = np.isnan(query_points_np).sum()
        inf_count = np.isinf(query_points_np).sum()
        print(f"[INFO] Query points stats: count={len(query_points_np)}, NaN={nan_count}, Inf={inf_count}")
        
        if nan_count > 0 or inf_count > 0:
            print("[WARN] Query points contain invalid values, filtering them out...")
            valid_mask = np.isfinite(query_points_np).all(axis=1)
            query_points_np = query_points_np[valid_mask]
            print(f"[INFO] After filtering: {len(query_points_np)} valid query points")
        
        # DEBUG: Verify projection for first point
        if len(query_points_np) > 0:
            print("[DEBUG] Verifying projection...")
            pt_w = query_points_np[0, 1:] # (x, y, z) world
            t = int(query_points_np[0, 0])
            
            # Check against all cameras
            for v in range(len(sorted_serials)):
                T_wc = all_extrs[t][v] # Camera-to-World
                T_cw = np.linalg.inv(T_wc) # World-to-Camera
                
                pt_c = transform_points(pt_w.reshape(1, 3), T_cw)[0]
                
                K = all_intrs[t][v]
                pt_p_homo = K @ pt_c
                pt_p = pt_p_homo[:2] / pt_p_homo[2]
                
                print(f"  Cam {v}: Pt_C={pt_c}, Pt_P={pt_p}, InFrame={0 <= pt_p[0] < cam['w'] and 0 <= pt_p[1] < cam['h']}")
            
    else:
        print("[WARN] No query points found.")
        query_points_np = np.zeros((0, 4))
        
    # Save
    output_path = "./point_clouds/npz_files/droid_external_cams.npz"
    print(f"[INFO] Saving to {output_path}...")
    np.savez(output_path, 
             rgbs=rgbs_np, 
             depths=depths_np, 
             intrs=intrs_np, 
             extrs=extrs_np, 
             camera_ids=sorted_serials, 
             query_points=query_points_np)
    print("[SUCCESS] Done.")

    # Cleanup
    for c in active_cams.values(): c['zed'].close()

if __name__ == "__main__":
    main()
