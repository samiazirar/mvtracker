#!/usr/bin/env python3
"""
Processes RH20T task folders to generate sparse, high-resolution depth maps by
reprojecting a point cloud from low-resolution depth onto high-resolution views.

This script orchestrates the main processing workflow, utilizing various utility modules
for geometry, gripper computation, point cloud processing, camera operations, data loading,
visualization, robot handling, and COLMAP integration.

Usage Examples:
-----------------

# 1. Basic Processing (Low-Resolution Data Only)
python create_sparse_depth_map.py \\
  --task-folder /path/to/low_res_data/task_0010_user_0011_scene_0010_cfg_0003 \\
  --out-dir ./output/low_res_processed

# 2. Reprojection Workflow (Low-Res Depth to High-Res RGB)
python create_sparse_depth_map.py \\
  --task-folder /path/to/low_res_data/task_0010_... \\
  --high-res-folder /path/to/high_res_data/task_0010_... \\
  --out-dir ./output/high_res_reprojected \\
  --max-frames 100

# 3. Advanced Reprojection with Color Alignment Check
python create_sparse_depth_map.py \\
  --task-folder /data/rh20t_api/data/test_data_full_rgb_upscaled_depth/uncompressed_low_res_data/task_0065_user_0010_scene_0009_cfg_0004 \\
  --high-res-folder /data/rh20t_api/data/test_data_full_rgb_upscaled_depth/rgb_data/RH20T_cfg4/task_0065_user_0010_scene_0009_cfg_0004 \\
  --out-dir ./data/high_res_filtered \\
  --max-frames 100 \\
  --color-alignment-check \\
  --color-threshold 35 \\
  --no-sharpen-edges-with-mesh

"""

# --- Standard Library Imports ---
import argparse
import json
import os
import warnings
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional

# --- Third-Party Library Imports ---
import numpy as np
import torch
import open3d as o3d
import rerun as rr
from tqdm import tqdm

# --- Project-Specific Imports ---
from mvtracker.utils.visualizer_rerun import log_pointclouds_to_rerun
from RH20T.rh20t_api.configurations import get_conf_from_dir_name, load_conf
from RH20T.rh20t_api.scene import RH20TScene

# --- Utility Module Imports ---
from utils.geometry_utils import (
    _pose_7d_to_matrix,
)
from utils.gripper_utils import (
    _compute_gripper_bbox,
    _compute_gripper_bbox_from_tcp,
    _compute_gripper_body_bbox,
    _compute_gripper_pad_points,
)
from utils.pointcloud_utils import (
    unproject_to_world_o3d,
    clean_point_cloud_radius,
    reconstruct_mesh_from_pointcloud,
    _extract_points_inside_bbox,
    _filter_points_closest_to_bbox,
    _exclude_points_inside_bbox,
    _filter_points_by_color_cluster,
    _restrict_query_points_to_frames,
    _align_bbox_with_point_cloud_com,
)
from utils.camera_utils import (
    scale_intrinsics_matrix,
    infer_calibration_resolution,
    reproject_to_sparse_depth_cv2,
    verify_camera_alignment,
    evaluate_depth_quality,
    evaluate_feature_richness,
)
from utils.data_loading_utils import (
    read_rgb,
    read_depth,
    list_frames,
    load_scene_data,
    get_synchronized_timestamps,
    select_frames,
    fix_human_metadata_calib,
    get_hand_tracked_points,
)
from utils.robot_utils import (
    _create_robot_model,
    _robot_model_to_pointcloud,
)
from utils.visualization_utils import (
    save_and_visualize,
)
from utils.colmap_utils import (
    setup_colmap_workspace,
    run_colmap_feature_extraction,
    run_colmap_matching,
    run_colmap_mapper,
    evaluate_camera_quality_from_colmap,
    select_best_cameras,
    reject_outlier_points_multiview,
)


# --- Main Workflow & Orchestration ---

def process_frames(
    args,
    scene_low,
    scene_high,
    final_cam_ids,
    cam_dirs_low,
    cam_dirs_high,
    timeline: np.ndarray,
    per_cam_low_ts: List[np.ndarray],
    per_cam_high_ts: Optional[List[np.ndarray]],
):
    """
    Iterates through timestamps to process frames, generate point clouds,
    and create the final data arrays (RGB, depth, intrinsics, extrinsics).
    """
    C, T = len(final_cam_ids), len(timeline)
    is_l515_flags = [cid.startswith('f') for cid in final_cam_ids]

    # Collect robot points if robot model will be enabled
    # We need this regardless of debug mode so we can log it to Rerun
    robot_debug_points: Optional[List[np.ndarray]] = [] if args.add_robot else None
    robot_debug_colors: Optional[List[np.ndarray]] = [] if robot_debug_points is not None else None
    robot_gripper_boxes: Optional[List[Optional[Dict[str, np.ndarray]]]] = [] if (args.add_robot and getattr(args, "gripper_bbox", False)) else None
    # empty list if defined in args, else None
    robot_gripper_body_boxes: Optional[List[Optional[Dict[str, np.ndarray]]]] = [] if (args.add_robot and getattr(args, "gripper_body_bbox", False)) else None
    robot_gripper_fingertip_boxes: Optional[List[Optional[Dict[str, np.ndarray]]]] = [] if (args.add_robot and getattr(args, "gripper_fingertip_bbox", False)) else None
    robot_gripper_pad_points: Optional[List[Optional[np.ndarray]]] = [] if (args.add_robot and getattr(args, "gripper_pad_points", False)) else None
    robot_tcp_points: Optional[List[Optional[np.ndarray]]] = [] if (args.add_robot and getattr(args, "tcp_points", False)) else None
    robot_object_points: Optional[List[Optional[np.ndarray]]] = [] if getattr(args, "object_points", False) else None
    # Query points: sensor points inside the gripper bbox (contact bbox)
    query_points: Optional[List[Optional[np.ndarray]]] = [] if (args.add_robot and getattr(args, "gripper_bbox", False)) else None
    query_colors: Optional[List[Optional[np.ndarray]]] = [] if query_points is not None else None

    color_lookup_low = [list_frames(cdir / 'color') for cdir in cam_dirs_low]
    depth_lookup_low = [list_frames(cdir / 'depth') for cdir in cam_dirs_low]

    if args.high_res_folder and cam_dirs_high:
        color_lookup_high = [list_frames(cdir / 'color') for cdir in cam_dirs_high]
    else:
        color_lookup_high = None

    # Create robot model if requested
    robot_model = None
    robot_conf: Optional[Any] = None
    robot_urdf_colors = None
    robot_mtl_colors = None
    if args.add_robot:
        configs_path = args.config.resolve()
        rh20t_root = configs_path.parent.parent if configs_path.parent.name == "configs" else configs_path.parent
        robot_bundle = _create_robot_model(
            sample_path=args.task_folder,
            configs_path=configs_path,
            rh20t_root=rh20t_root,
        )
        if robot_bundle:
            robot_model = robot_bundle.get("model")
            robot_mtl_colors = robot_bundle.get("mtl_colors")
        if robot_model:
            print("[INFO] Robot model loaded successfully.")
            # Load URDF colors for the robot (if any exist in URDF)
            try:
                confs = load_conf(str(configs_path))
                conf = get_conf_from_dir_name(str(args.task_folder), confs)
                robot_conf = conf
                robot_urdf_path = (rh20t_root / conf.robot_urdf).resolve()
                robot_urdf_colors = _load_urdf_link_colors(robot_urdf_path)
                if robot_urdf_colors:
                    print(f"[INFO] Loaded colors for {len(robot_urdf_colors)} robot links from URDF")
            except Exception as e:
                print(f"[WARN] Could not load URDF colors: {e}")
            # Report MTL colors if loaded
            if robot_mtl_colors and len(robot_mtl_colors) > 0:
                print(f"[INFO] Loaded MTL colors for {len(robot_mtl_colors)} robot parts")

    def _resolve_frame(frame_map: Dict[int, Path], ts: int, cam_name: str, label: str) -> Path:
        path = frame_map.get(ts)
        if path is not None:
            return path

        if not frame_map:
            raise KeyError(f"No frames available for {cam_name} ({label}).")

        closest_ts = min(frame_map.keys(), key=lambda k: abs(k - ts))
        delta = abs(closest_ts - ts)
        print(
            f"[WARN] Timestamp {ts} not found for {cam_name} ({label}); using closest {closest_ts} (|Δ|={delta} ms)."
        )
        return frame_map[closest_ts]

    scaled_low_intrinsics: List[np.ndarray] = []
    low_shapes: List[Tuple[int, int]] = []
    for ci in range(C):
        cid = final_cam_ids[ci]
        first_low_ts = int(per_cam_low_ts[ci][0])
        low_path = _resolve_frame(color_lookup_low[ci], first_low_ts, cid, "low-res color")
        low_img = read_rgb(low_path)
        h_low, w_low = low_img.shape[:2]
        low_shapes.append((h_low, w_low))
        base_res = infer_calibration_resolution(scene_low, cid)
        print(" base res from calib image: ", base_res)
        if base_res is None:
            base_res = (max(1, int(round(scene_low.intrinsics[cid][0, 2] * 2))),
                        max(1, int(round(scene_low.intrinsics[cid][1, 2] * 2))))
            print(
                f"[WARN] Calibration image for camera {cid} not found; "
                f"estimating base resolution as {base_res[0]}x{base_res[1]} using intrinsics."
            )
        base_w, base_h = base_res
        print("[WARN] Switch this to the original rgbd_to_point_cloud_soon")
        scaled_low_intrinsics.append(scale_intrinsics_matrix(scene_low.intrinsics[cid], w_low, h_low, base_w, base_h))

    scaled_high_intrinsics: Optional[List[np.ndarray]] = None
    high_shapes: Optional[List[Tuple[int, int]]] = None
    if args.high_res_folder and color_lookup_high:
        scaled_high_intrinsics = []
        high_shapes = []
        for ci in range(C):
            cid = final_cam_ids[ci]
            first_high_ts = int(per_cam_high_ts[ci][0]) if per_cam_high_ts else int(per_cam_low_ts[ci][0])
            high_path = _resolve_frame(color_lookup_high[ci], first_high_ts, cid, "high-res color")
            high_img = read_rgb(high_path)
            h_high, w_high = high_img.shape[:2]
            high_shapes.append((h_high, w_high))
            base_res = infer_calibration_resolution(scene_high, cid) or infer_calibration_resolution(scene_low, cid)
            if base_res is None:
                base_res = (max(1, int(round(scene_high.intrinsics[cid][0, 2] * 2))),
                            max(1, int(round(scene_high.intrinsics[cid][1, 2] * 2))))
                print(
                    f"[WARN] Calibration image for camera {cid} not found in high-res scene; "
                    f"estimating base resolution as {base_res[0]}x{base_res[1]} using intrinsics."
                )
            base_w, base_h = base_res
            scaled_high_intrinsics.append(scale_intrinsics_matrix(scene_high.intrinsics[cid], w_high, h_high, base_w, base_h))

    # Determine output resolution and initialize data containers
    if args.high_res_folder and high_shapes:
        H_out, W_out = high_shapes[0]
        print(f"[INFO] Outputting high-resolution data ({H_out}x{W_out}) with reprojected depth.")
    else:
        H_out, W_out = low_shapes[0]
        print(f"[INFO] Outputting low-resolution data ({H_out}x{W_out}).")

    rgbs_out = np.zeros((C, T, H_out, W_out, 3), dtype=np.uint8)
    depths_out = np.zeros((C, T, H_out, W_out), dtype=np.float32)
    intrs_out = np.zeros((C, T, 3, 3), dtype=np.float32)
    extrs_out = np.zeros((C, T, 3, 4), dtype=np.float32)

    for ti in tqdm(range(T), desc="Processing Frames"):
        debug_pts = None
        debug_cols = None
        bbox_entry_for_frame: Optional[Dict[str, np.ndarray]] = None
        full_bbox_for_frame: Optional[Dict[str, np.ndarray]] = None
        pad_pts_for_frame: Optional[np.ndarray] = None
        # Step 1: Create a combined point cloud for the current frame from all low-res views
        combined_pcd = o3d.geometry.PointCloud()
        if args.high_res_folder:
            pcds_per_cam = []
            for ci in range(C):
                cid = final_cam_ids[ci]
                # Load low-res data for point cloud generation
                t_low = int(per_cam_low_ts[ci][ti])
                depth_low = read_depth(_resolve_frame(depth_lookup_low[ci], t_low, cid, "low-res depth"), is_l515_flags[ci])
                rgb_low = read_rgb(_resolve_frame(color_lookup_low[ci], t_low, cid, "low-res color"))
                K_low = scaled_low_intrinsics[ci]
                E_inv = np.linalg.inv(scene_low.extrinsics_base_aligned[cid])
                
                # Create and add the point cloud for this view
                pcds_per_cam.append(unproject_to_world_o3d(depth_low, rgb_low, K_low, E_inv))
            
            # Merge all individual point clouds into a single scene representation
            for pcd in pcds_per_cam:
                # `+=` performs an in-place concatenation of geometry in Open3D, so we avoid repeated numpy conversions here.
                combined_pcd += pcd

            # Add robot point cloud with current joint state if available
            if robot_model is not None:
                current_gripper_width_mm: Optional[float] = None
                # Get the timestamp for the current frame
                t_low = int(per_cam_low_ts[0][ti])
                # Get joint angles at this timestamp
                # Use scene_high if available (it has the robot data), otherwise fall back to scene_low
                robot_scene = scene_low if scene_low is not None else scene_high
                joint_angles = robot_scene.get_joint_angles_aligned(t_low)
                # Ensure we have the correct number of joints
                # The robot model expects len(conf.robot_joint_sequence) joints
                expected_joints = len(scene_low.configuration.robot_joint_sequence)
                
                gripper_joint_angle = None
                
                if len(joint_angles) < expected_joints:
                    # Pad with zeros (neutral gripper position) if missing joints
                    padding = np.zeros(expected_joints - len(joint_angles))
                    joint_angles_padded = np.concatenate([joint_angles, padding])
                    if ti == 0:
                        print(f"[INFO] Padded joint angles from {len(joint_angles)} to {len(joint_angles_padded)}")
                    # Don't use padded zero - interpolate from gripper dictionary instead
                    gripper_joint_angle = 0.0  # Will be replaced below
                    # Use all joints for robot model (including padded gripper)
                    robot_joint_angles = joint_angles_padded
                elif len(joint_angles) > expected_joints:
                    # Truncate if we have too many joints
                    robot_joint_angles = joint_angles[:expected_joints]
                    gripper_joint_angle = robot_joint_angles[-1]
                    if ti == 0:
                        print(f"[WARN] Truncated joint angles from {len(joint_angles)} to {expected_joints}")
                else:
                    robot_joint_angles = joint_angles
                    gripper_joint_angle = joint_angles[-1]
                
                # Get actual gripper width from scene.gripper dictionary
                # The gripper dictionary has structure: scene.gripper[camera_id][timestamp] = {'gripper_command': [width, ...]}
                # We need to pick one camera's gripper data
                gripper_data_source = None
                # Use scene_high if available (it has the robot data), otherwise fall back to scene_low
                robot_scene = scene_low if scene_low is not None else scene_high
                if hasattr(robot_scene, 'gripper') and len(robot_scene.gripper) > 0:
                    # Find first camera with gripper data
                    for cam_id in sorted(robot_scene.gripper.keys()):
                        if len(robot_scene.gripper[cam_id]) > 0:
                            gripper_data_source = robot_scene.gripper[cam_id]
                            if ti == 0:
                                print(f"[INFO] Using gripper data from camera {cam_id}")
                            break
                
                if gripper_data_source is not None:
                    gripper_timestamps = sorted(gripper_data_source.keys())
                    # Find closest gripper timestamp to current frame
                    closest_idx = min(range(len(gripper_timestamps)), 
                                    key=lambda i: abs(gripper_timestamps[i] - t_low))
                    closest_ts = gripper_timestamps[closest_idx]
                    
                    # Get gripper command (first element is gripper width in mm)
                    gripper_width_mm = gripper_data_source[closest_ts]['gripper_command'][0]
                    
                    # Robotiq 2F-85 gripper: 0mm (closed) to 85mm (fully open)
                    # Convert gripper width to finger_joint angle in radians
                    # finger_joint: 0 rad (open) to ~0.8 rad (closed)
                    max_width_mm = 85.0
                    max_angle_rad = 0.8  # ~45 degrees when fully closed
                    # Invert: larger width = smaller angle (more open)
                    gripper_joint_angle = max_angle_rad * (1.0 - gripper_width_mm / max_width_mm)
                    current_gripper_width_mm = float(gripper_width_mm)
                    
                    # Update the robot_joint_angles array with the calculated angle
                    robot_joint_angles[-1] = gripper_joint_angle
                    
                    if ti == 0:
                        print(f"[INFO] Gripper width at frame 0: {gripper_width_mm:.2f} mm (finger_joint: {gripper_joint_angle:.3f} rad)")
                
                robot_pcd = _robot_model_to_pointcloud(
                    robot_model,
                    robot_joint_angles,
                    is_first_time=(ti == 0),
                    points_per_mesh=10000,
                    debug_mode=getattr(args, "debug_mode", False),
                    urdf_colors=robot_urdf_colors,
                    mtl_colors=robot_mtl_colors,
                )
                # Compute gripper bbox if ANY bbox output is requested
                if (robot_gripper_boxes is not None or 
                    robot_gripper_body_boxes is not None or 
                    robot_gripper_fingertip_boxes is not None):
                    # Determine which method to use for bbox computation
                    use_tcp = getattr(args, "use_tcp", False)
                    tcp_transform = None
                    if use_tcp:
                        # Use TCP-based computation
                        try:

                            if ti == 0:
                                print("[INFO] Using TCP-based gripper bbox computation")
                            # Get the official TCP pose from the dataset (7D: position + quaternion)

                            # robot_scene = scene_low if scene_low is not None else scene_high
                            # tcp_pose_7d = robot_scene.get_tcp_aligned(t_low) #maybe use camera serial
                            print("[DEBUG] REMOVE or CHECK THIS AFTERWARDS; USING SCENE HIGH ONLY") #todo why? find out
                            tcp_pose_7d = scene_high.get_tcp_aligned(t_low)
                            # Validate that we got valid data
                            if tcp_pose_7d is not None and len(tcp_pose_7d) == 7:
                                # Convert 7D pose [x, y, z, qx, qy, qz, qw] to 4x4 matrix
                                tcp_transform = _pose_7d_to_matrix(tcp_pose_7d)
                                if ti == 0:
                                    print(f"[INFO] Using API TCP pose: position={tcp_pose_7d[:3]}, quat={tcp_pose_7d[3:]}")
                            else:
                                if ti == 0:
                                    print(f"[WARN] API returned invalid TCP data: {tcp_pose_7d}")
                                    print(f"[WARN] Cannot compute TCP-based bbox, skipping frame")
                        except TypeError as e:
                            if ti == 0:
                                print(f"[ERROR] Could not get TCP from API: {e}")
                                print(f"[ERROR] Cannot compute TCP-based bbox, skipping frame")
                            elif ti != 0:
                                # Only log once after the first frame
                                print(f"[ERROR] Could not get TCP from API at frame {ti}: {e}")
                        
                        
                        # Compute bbox using TCP method
                        if tcp_transform is not None:
                            print("[INFO] Computing gripper bbox using TCP transform")
                            bbox_entry_for_frame, base_full_bbox, fingertip_bbox_for_frame = _compute_gripper_bbox_from_tcp(
                                tcp_transform=tcp_transform,
                                robot_conf=robot_conf,
                                gripper_width_mm=current_gripper_width_mm,
                                contact_height_m=getattr(args, "gripper_bbox_contact_height_m", None),
                                contact_length_m=getattr(args, "gripper_bbox_contact_length_m", None),
                            )
                        else:
                            print("[WARN] tcp_transform is None, cannot compute TCP-based bbox")
                            bbox_entry_for_frame, base_full_bbox, fingertip_bbox_for_frame = None, None, None
                    else:
                        print("[INFO] Not using TCP; defaulting to FK-based gripper bbox computation")  
                        # Use FK-based computation (default)
                        if ti == 0:
                            print("[INFO] Using FK-based gripper bbox computation")
                        bbox_entry_for_frame, base_full_bbox, fingertip_bbox_for_frame = _compute_gripper_bbox(
                            robot_model,
                            robot_conf,
                            current_gripper_width_mm,
                            contact_height_m=getattr(args, "gripper_bbox_contact_height_m", None),
                            contact_length_m=getattr(args, "gripper_bbox_contact_length_m", None),
                            tcp_transform=None,  # Not used in FK-based computation
                        )
                    full_bbox_for_frame = base_full_bbox
                    # if we want the griper bbox
                    if robot_gripper_body_boxes is not None and bbox_entry_for_frame is not None:
                        #if size is set the override
                        body_width_override = getattr(args, "gripper_body_width_m", None)
                        body_height_override = getattr(args, "gripper_body_height_m", None)
                        body_length_override = getattr(args, "gripper_body_length_m", None)
                        if (
                            body_width_override is not None
                            or body_height_override is not None
                            or body_length_override is not None
                        ):
                            full_bbox_for_frame = _compute_gripper_body_bbox(
                                robot_model,
                                robot_conf,
                                bbox_entry_for_frame,
                                body_width_m=body_width_override,
                                body_height_m=body_height_override,
                                body_length_m=body_length_override,
                            )
                    elif robot_gripper_body_boxes is None and full_bbox_for_frame is not None:
                        print("[WARN] Computed full gripper body bbox but no output list to store it")
                    elif robot_gripper_body_boxes is not None and full_bbox_for_frame is None:
                        print("[WARN] robot_gripper_body_boxes is not None but full_bbox_for_frame is None")
                    else:
                        print("[WARN] Could not compute gripper bbox for this frame")

                if robot_gripper_pad_points is not None:
                    pad_pts_for_frame = _compute_gripper_pad_points(robot_model, robot_conf)
                
                # Get TCP point from API if requested
                tcp_pt_for_frame: Optional[np.ndarray] = None
                if robot_tcp_points is not None:
                    try:
                        robot_scene = scene_low if scene_low is not None else scene_high
                        tcp_pose_7d = robot_scene.get_tcp_aligned(t_low)
                        #TODO: use camera serial?
                        if tcp_pose_7d is not None and len(tcp_pose_7d) >= 3:
                            # Extract position (first 3 elements: x, y, z)
                            tcp_pt_for_frame = np.array(tcp_pose_7d[:3], dtype=np.float32)
                    except Exception as e:
                        if ti == 0:
                            print(f"[WARN] Could not get TCP point from API: {e}")
                
                # Get object point from API if requested and available
                obj_pt_for_frame: Optional[np.ndarray] = None
                if robot_object_points is not None:
                    try:
                        robot_scene = scene_low if scene_low is not None else scene_high
                        # Try to get object pose if the API has it
                        # This is speculative - the exact method name may vary
                        if hasattr(robot_scene, 'get_object_pose'):
                            obj_pose = robot_scene.get_object_pose(t_low)
                            if obj_pose is not None and len(obj_pose) >= 3:
                                obj_pt_for_frame = np.array(obj_pose[:3], dtype=np.float32)
                    except Exception as e:
                        if ti == 0:
                            print(f"[INFO] Object pose not available from API: {e}")
                
                # NOTE: Removed align_mat_base transformation
                # The robot joint angles from get_joint_angles_aligned() are already in the aligned frame
                # Applying align_mat_base was causing the robot to face the wrong direction
                # if robot_pcd.has_points():
                #     align_mat = scene_low.configuration.align_mat_base
                #     robot_pcd.transform(align_mat)

                if robot_pcd.has_points():
                    if robot_debug_points is not None:
                        pts_np = np.asarray(robot_pcd.points, dtype=np.float32)
                        if robot_pcd.has_colors():
                            cols_np = (np.asarray(robot_pcd.colors) * 255).astype(np.uint8)
                        else:
                            cols_np = np.full((pts_np.shape[0], 3), [180, 180, 180], dtype=np.uint8)
                        debug_pts = pts_np
                        debug_cols = cols_np
                    combined_pcd += robot_pcd

                    
                    width_info = (
                        f"{current_gripper_width_mm:.2f} mm"
                        if current_gripper_width_mm is not None
                        else "N/A"
                    )
                    if ti == 0:
                        print(
                            f"[INFO] Added robot with {len(robot_pcd.points)} points for frame {ti} "
                            f"(gripper width: {width_info})"
                        )
                    elif ti % 5 == 0:
                        print(
                            f"[INFO] Frame {ti}: Robot has {len(robot_pcd.points)} points "
                            f"(gripper width: {width_info})"
                        )
            if robot_debug_points is not None and debug_pts is None:
                debug_pts = np.empty((0, 3), dtype=np.float32)
                debug_cols = np.empty((0, 3), dtype=np.uint8)

            if combined_pcd.has_points():
                if args.clean_pointcloud:
                    before_count = len(combined_pcd.points)
                    cleaned = clean_point_cloud_radius(
                        combined_pcd,
                        radius=args.pc_clean_radius,
                        min_points=args.pc_clean_min_points,
                    )
                    after_count = len(cleaned.points)
                    if after_count == 0:
                        print(
                            f"[WARN] Radius filter removed all points at frame index {ti}; "
                            "using raw combined point cloud instead."
                        )
                    else:
                        if ti == 0 or after_count != before_count:
                            print(
                                f"[INFO] Radius filter kept {after_count}/{before_count} points "
                                f"(radius={args.pc_clean_radius}, min_pts={args.pc_clean_min_points})."
                            )
                        combined_pcd = cleaned

                if args.sharpen_edges_with_mesh:
                    mesh = reconstruct_mesh_from_pointcloud(combined_pcd, depth=args.mesh_depth)
                    if mesh is not None and len(mesh.vertices) > 0:
                        mesh_pcd = o3d.geometry.PointCloud()
                        mesh_pcd.points = mesh.vertices
                        if mesh.has_vertex_colors() and len(mesh.vertex_colors) == len(mesh.vertices):
                            mesh_pcd.colors = mesh.vertex_colors
                        elif combined_pcd.has_colors():
                            combined_colors = np.asarray(combined_pcd.colors)
                            if combined_colors.size:
                                kdtree = o3d.geometry.KDTreeFlann(combined_pcd)
                                mesh_vertices = np.asarray(mesh.vertices)
                                remap_colors = np.zeros_like(mesh_vertices)
                                for vidx, vertex in enumerate(mesh_vertices):
                                    _, idx, _ = kdtree.search_knn_vector_3d(vertex, 1)
                                    remap_colors[vidx] = combined_colors[idx[0]]
                                mesh_pcd.colors = o3d.utility.Vector3dVector(remap_colors)
                        combined_pcd = mesh_pcd

        points_world_np: Optional[np.ndarray] = None
        colors_world_np: Optional[np.ndarray] = None
        if combined_pcd.has_points():
            points_world_np = np.asarray(combined_pcd.points, dtype=np.float32)
            if combined_pcd.has_colors():
                colors_world_np = np.asarray(combined_pcd.colors, dtype=np.float32)

        if (
            getattr(args, "align_bbox_with_points", True)
            and points_world_np is not None
            and points_world_np.size > 0
        ):
            search_radius_scale = getattr(args, "align_bbox_search_radius_scale", 2.0)
            if bbox_entry_for_frame is not None:
                aligned_contact = _align_bbox_with_point_cloud_com(
                    bbox_entry_for_frame,
                    points_world_np,
                    colors=colors_world_np,
                    search_radius_scale=search_radius_scale,
                )
                if aligned_contact is not None:
                    bbox_entry_for_frame = aligned_contact
            if full_bbox_for_frame is not None:
                aligned_body = _align_bbox_with_point_cloud_com(
                    full_bbox_for_frame,
                    points_world_np,
                    colors=colors_world_np,
                    search_radius_scale=search_radius_scale,
                )
                if aligned_body is not None:
                    full_bbox_for_frame = aligned_body
            if fingertip_bbox_for_frame is not None:
                aligned_tip = _align_bbox_with_point_cloud_com(
                    fingertip_bbox_for_frame,
                    points_world_np,
                    colors=colors_world_np,
                    search_radius_scale=search_radius_scale,
                )
                if aligned_tip is not None:
                    fingertip_bbox_for_frame = aligned_tip

        if robot_gripper_boxes is not None:
            robot_gripper_boxes.append(bbox_entry_for_frame)
        if robot_gripper_body_boxes is not None:
            robot_gripper_body_boxes.append(full_bbox_for_frame)
        if robot_gripper_fingertip_boxes is not None:
            robot_gripper_fingertip_boxes.append(fingertip_bbox_for_frame)
        if robot_gripper_pad_points is not None:
            # Ensure we append even if None to keep timeline alignment
            robot_gripper_pad_points.append(pad_pts_for_frame)
        if robot_tcp_points is not None:
            robot_tcp_points.append(tcp_pt_for_frame)
        if robot_object_points is not None:
            robot_object_points.append(obj_pt_for_frame)
        
        # Extract query points (sensor points inside the gripper body bbox)
        if query_points is not None:
            query_bbox = full_bbox_for_frame if full_bbox_for_frame is not None else bbox_entry_for_frame
            if query_bbox is not None and points_world_np is not None and points_world_np.size > 0:
                # Extract points inside the gripper body bbox (red bbox)
                inside_pts, inside_cols = _extract_points_inside_bbox(
                    points_world_np,
                    query_bbox,
                    colors=colors_world_np,
                )
                
                # Exclude points inside gripper (blue bbox) if flag is set
                exclude_inside = getattr(args, "exclude_inside_gripper", False)
                if exclude_inside and inside_pts is not None and inside_pts.size > 0:
                    if fingertip_bbox_for_frame is not None:
                        inside_pts, inside_cols = _exclude_points_inside_bbox(
                            inside_pts,
                            fingertip_bbox_for_frame,
                            colors=inside_cols,
                        )
                        if ti == 0:
                            print(f"[INFO] Excluded {len(points_world_np) - len(inside_pts) if inside_pts.size > 0 else 0} points inside gripper (blue bbox)")
                
                # Apply color clustering if specified (keep largest color cluster)
                if getattr(args, "exclude_by_cluster", False) and inside_pts is not None and inside_pts.size > 0:
                    original_count = len(inside_pts)
                    inside_pts, inside_cols = _filter_points_by_color_cluster(
                        inside_pts,
                        inside_cols,
                    )
                    if ti == 0:
                        print(f"[INFO] Filtered to largest color cluster: {len(inside_pts)} points (from {original_count})")
                
                # Apply max query points limit if specified (keep N closest to blue/fingertip bbox)
                max_query_pts = getattr(args, "max_query_points", None)
                if max_query_pts is not None and inside_pts is not None and inside_pts.size > 0:
                    # Use fingertip bbox (blue) as reference if available, otherwise use contact bbox
                    reference_bbox = fingertip_bbox_for_frame if fingertip_bbox_for_frame is not None else bbox_entry_for_frame
                    if reference_bbox is not None:
                        inside_pts, inside_cols = _filter_points_closest_to_bbox(
                            inside_pts,
                            reference_bbox,
                            max_query_pts,
                            colors=inside_cols,
                        )
                        if ti == 0:
                            print(f"[INFO] Limited query points to {len(inside_pts)} closest to fingertip bbox (max: {max_query_pts})")
                
                query_points.append(inside_pts if inside_pts is not None and inside_pts.size > 0 else None)
                if query_colors is not None:
                    query_colors.append(inside_cols if inside_cols is not None and inside_cols.size > 0 else None)
            else:
                # No bbox or no points - append None to keep timeline alignment
                query_points.append(None)
                if query_colors is not None:
                    query_colors.append(None)

        # Step 2: Generate the final output data for each camera view
        for ci in range(C):
            cid = final_cam_ids[ci]

            if args.high_res_folder and scene_high is not None:
                # Use high-res calibration when available; low-res version is only for point-cloud generation.
                E_world_to_cam = scene_high.extrinsics_base_aligned[cid]
            else:
                E_world_to_cam = scene_low.extrinsics_base_aligned[cid]

            # Persist the world-to-camera [R|t] so downstream consumers can project the world-space point cloud.
            extrs_out[ci, ti] = E_world_to_cam[:3, :4]
            
            if args.high_res_folder:
                # Reprojection workflow
                t_high = int(per_cam_high_ts[ci][ti]) if per_cam_high_ts else int(per_cam_low_ts[ci][ti])
                high_res_rgb = read_rgb(_resolve_frame(color_lookup_high[ci], t_high, cid, "high-res color"))
                K_high = scaled_high_intrinsics[ci] if scaled_high_intrinsics is not None else scaled_low_intrinsics[ci]
                rgbs_out[ci, ti] = high_res_rgb
                intrs_out[ci, ti] = K_high
                
                # If color check is disabled, use a threshold that allows all points to pass
                threshold = args.color_threshold if args.color_alignment_check else 256.0
                depths_out[ci, ti] = reproject_to_sparse_depth_cv2(
                    combined_pcd, high_res_rgb, K_high, E_world_to_cam, threshold
                )
            else:
                # Standard low-resolution workflow
                t_low = int(per_cam_low_ts[ci][ti])
                rgbs_out[ci, ti] = read_rgb(_resolve_frame(color_lookup_low[ci], t_low, cid, "low-res color"))
                depths_out[ci, ti] = read_depth(_resolve_frame(depth_lookup_low[ci], t_low, cid, "low-res depth"), is_l515_flags[ci])
                intrs_out[ci, ti] = scaled_low_intrinsics[ci]

        if robot_debug_points is not None:
            if debug_pts is None:
                debug_pts = np.empty((0, 3), dtype=np.float32)
                debug_cols = np.empty((0, 3), dtype=np.uint8)
            robot_debug_points.append(debug_pts)
            if robot_debug_colors is not None:
                robot_debug_colors.append(debug_cols)

    query_points, query_colors = _restrict_query_points_to_frames(
        query_points,
        query_colors,
        getattr(args, "frames_for_tracking", None),
    )

    return (
        rgbs_out,
        depths_out,
        intrs_out,
        extrs_out,
        robot_debug_points,
        robot_debug_colors,
        robot_gripper_boxes,
        robot_gripper_body_boxes,
        robot_gripper_fingertip_boxes,
        robot_gripper_pad_points,
        robot_tcp_points,
        robot_object_points,
        query_points,
        query_colors,
    )



def main():
    """Parses arguments and orchestrates the entire data processing workflow."""
    parser = argparse.ArgumentParser(
        description="Process RH20T data with optional color-checked reprojection using Open3D and OpenCV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--task-folder", required=True, type=Path, help="Path to the primary (low-res) RH20T task folder.")
    parser.add_argument("--high-res-folder", type=Path, default=None, help="Optional: Path to the high-resolution task folder for reprojection.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for .npz and .rrd files.")
    parser.add_argument("--config", default="RH20T/configs/configs.json", type=Path, help="Path to RH20T robot configs JSON.")
    parser.add_argument("--max-frames", type=int, default=50, help="Limit frames to process (0 for all).")
    parser.add_argument("--frame-selection", choices=["first", "last", "middle"], default="middle", help="Method for selecting frames.")
    parser.add_argument(
        "--frames-for-tracking",
        type=int,
        default=None,
        help="If provided, only export query points for the first N frames (tracking seeds).",
    )
    parser.add_argument(
        "--color-alignment-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable color-based filtering of reprojected points (disable with --no-color-alignment-check)."
    )
    parser.add_argument("--color-threshold", type=float, default=5.0, help="Max average color difference (0-255) for a point to be removed.")
    parser.add_argument(
        "--sharpen-edges-with-mesh",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Densify geometry via Poisson meshing for sharper edges (disable with --no-sharpen-edges-with-mesh)."
    )
    parser.add_argument(
        "--mesh-depth",
        type=int,
        default=9,
        help="Poisson reconstruction depth controlling mesh resolution."
    )
    parser.add_argument(
        "--clean-pointcloud",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply radius-based cleaning to the fused point cloud before reprojection (disable with --no-clean-pointcloud)."
    )
    parser.add_argument(
        "--pc-clean-radius",
        type=float,
        default=0.01,
        help="Radius (meters) for the Open3D radius outlier removal filter."
    )
    parser.add_argument(
        "--pc-clean-min-points",
        type=int,
        default=20,
        help="Minimum number of neighbors within radius to keep a point during cleaning."
    )
    parser.add_argument("--no-pointcloud", action="store_true", help="Only generate the .npz file, skip visualization.")
    parser.add_argument("--sync-fps", type=float, default=10.0, help="Target FPS for synchronization output timeline.")
    parser.add_argument("--sync-min-density", type=float, default=0.6, help="Minimum density ratio required per camera during synchronization.")
    parser.add_argument("--sync-max-drift", type=float, default=0.05, help="Maximum tolerated fractional FPS shortfall before warning.")
    parser.add_argument("--sync-tolerance-ms", type=float, default=50.0, help="Maximum timestamp deviation (ms) when matching frames; defaults to half frame period.")
    parser.add_argument(
        "--dataset-type",
        choices=["robot", "human"],
        default="robot",
        help="Quick selector that disables robot-only options when processing human demonstrations.",
    )
    parser.add_argument("--add-robot", action="store_true", help="Include robot arm model in Rerun visualization.")
    parser.add_argument(
        "--gripper-bbox",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Log an approximate gripper contact bbox in Rerun (requires --add-robot).",
    )
    parser.add_argument(
        "--gripper-bbox-contact-height-m",
        type=float,
        default=None,
        help="Override contact bbox vertical size in meters (full size, not half).",
    )
    parser.add_argument(
        "--gripper-bbox-contact-length-m",
        type=float,
        default=None,
        help="Override contact bbox length (approach axis) in meters (full size).",
    )
    parser.add_argument(
        "--gripper-body-bbox",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also log a larger gripper body bbox (fixed width).",
    )
    parser.add_argument(
        "--gripper-fingertip-bbox",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Log a blue bbox at the bottom of the body bbox (fingertip position).",
    )
    parser.add_argument(
        "--gripper-body-width-m",
        type=float,
        default=None,
        help="Body bbox width along jaw-separation axis (full size, fixed across frames).",
    )
    parser.add_argument(
        "--gripper-body-height-m",
        type=float,
        default=None,
        help="Body bbox thickness along the pad-normal axis (full size).",
    )
    parser.add_argument(
        "--gripper-body-length-m",
        type=float,
        default=None,
        help="Body bbox length along approach axis (full size).",
    )
    parser.add_argument(
        "--gripper-pad-points",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, log magenta points at the left/right gripper pad centers (FK-based).",
    )
    parser.add_argument(
        "--tcp-points",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, log cyan spheres at TCP positions from the API (requires --add-robot).",
    )
    parser.add_argument(
        "--tcp-point-radius",
        type=float,
        default=0.05,
        help="Radius of TCP point spheres in meters.",
    )
    #TODO check if used lul
    parser.add_argument(
        "--object-points",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, log yellow spheres at object positions from the API if available.",
    )
    parser.add_argument(
        "--export-bbox-video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Export per-camera RGB videos with gripper bounding boxes overlaid.",
    )
    parser.add_argument(
        "--bbox-video-fps",
        type=float,
        default=30.0,
        help="Frame rate used when exporting gripper bounding box videos (requires --export-bbox-video).",
    )
    parser.add_argument(
        "--debug-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable additional debug outputs such as red robot point overlays in Rerun.",
    )
    parser.add_argument(
        "--align-bbox-with-points",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Align gripper bboxes with nearby point cloud COM (disable with --no-align-bbox-with-points).",
    )
    parser.add_argument(
        "--align-bbox-search-radius-scale",
        type=float,
        default=2.0,
        help="Scale factor for alignment search radius relative to bbox diagonal.",
    )
    parser.add_argument(
        "--use-tcp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use TCP (Tool Center Point) pose from API for gripper bbox computation instead of FK.",
    )
    parser.add_argument(
        "--visualize-query-points",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Visualize sensor points inside gripper bbox as magenta points in Rerun.",
    )
    parser.add_argument(
        "--max-query-points",
        type=int,
        default=None,
        help="Maximum number of query points to keep. When set, keeps only the N points closest to the blue bbox (fingertip). Default: None (no limit).",
    )
    parser.add_argument(
        "--exclude-inside-gripper",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Exclude points inside the blue bbox (fingertip) from query points. Removes gripper points. Default: False (off).",
    )
    parser.add_argument(
        "--exclude-by-cluster",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use DBSCAN clustering on colors to filter query points. Keeps only largest cluster (gripper color). Default: False (off).",
    )
    # COLMAP-related arguments
    parser.add_argument(
        "--refine-colmap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use COLMAP to refine point cloud and check cross-view consistency.",
    )
    parser.add_argument(
        "--limit-num-cameras",
        type=int,
        default=None,
        help="Limit number of cameras to N best (based on combined depth quality, feature richness, and COLMAP reconstruction).",
    )
    parser.add_argument(
        "--colmap-workspace",
        type=Path,
        default=None,
        help="Reuse an existing COLMAP workspace directory instead of creating a temporary one.",
    )
    parser.add_argument(
        "--colmap-use-known-poses",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use known camera poses from calibration instead of estimating them. More robust when cameras are calibrated.",
    )
    parser.add_argument(
        "--reject-outlier-points",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use multi-view consistency to reject outlier 3D points (requires --refine-colmap).",
    )
    parser.add_argument(
        "--outlier-reprojection-threshold",
        type=float,
        default=5.0,
        help="Maximum reprojection error (pixels) for a point to be considered an inlier.",
    )
    args = parser.parse_args()

    if args.dataset_type == "human":
        robot_toggles = [
            "add_robot",
            "gripper_bbox",
            "gripper_body_bbox",
            "gripper_fingertip_bbox",
            "gripper_pad_points",
            "export_bbox_video",
            "object_points",
            "tcp_points",
            "exclude_inside_gripper",
            "exclude_by_cluster",
        ]
        disabled_flags = [flag for flag in robot_toggles if getattr(args, flag, False)]
        for flag in robot_toggles:
            setattr(args, flag, False)
        if disabled_flags:
            joined = ", ".join(sorted(disabled_flags))
            print(f"[INFO] Human dataset selected; disabling robot-only options: {joined}")
        else:
            print("[INFO] Human dataset selected; robot overlays are disabled.")

    if args.dataset_type != "human" and getattr(args, "export_bbox_video", False) and not getattr(args, "gripper_bbox", False):
        print("[INFO] --export-bbox-video requires bounding boxes; enabling --gripper-bbox.")
        args.gripper_bbox = True

    if not args.config.exists():
        print(f"[ERROR] RH20T config file not found at: {args.config}")
        return
    
    args.out_dir.mkdir(parents=True, exist_ok=True)
    robot_configs = load_conf(str(args.config))
    settings_file = "/workspace/RH20T/configs/default.yaml"
    with open(settings_file, "r") as f:
        vis_cfg_dict = yaml.load(f, Loader=yaml.FullLoader)    # --- Step 1: Load and Synchronize Data ---

    #load the metadata 
    metadata_file_path = args.task_folder / "metadata.json"
    if not metadata_file_path.exists():
        print(f"[ERROR] Metadata file not found at: {metadata_file_path}")
        raise FileNotFoundError(f"Metadata file not found at: {metadata_file_path}")
    with open(metadata_file_path, "r") as f:
        metadata_file = json.load(f)

    start_timestamp = metadata_file.get("start_time", None)
    end_timestamp = metadata_file.get("end_time", None)
    calib_timestamp = metadata_file.get("calib", None)
    calib_quality = metadata_file.get("calib_quality", None)
    if start_timestamp is None or end_timestamp is None or calib_timestamp is None:
        print(f"[ERROR] Metadata file is missing required timestamps.")
        raise ValueError(f"Metadata file is missing required timestamps.")

    print(f"[INFO] The Calib Quality is: {calib_quality}, lower is better, -1 means some cameras failed calibration. In reality means useless calib")
    #find the good cameras
        

    # For human datasets with high-res folder, load calibration from high-res folder
    # because low-res (depth) folder may not contain calibration data
    if args.dataset_type == "human" and args.high_res_folder:
        print("[INFO] Human dataset: loading calibration from high-res folder")
        # Fix metadata calibration timestamp if needed
        # fix_human_metadata_calib(args.high_res_folder)
        scene_low, cam_ids_low, cam_dirs_low = load_scene_data(args.high_res_folder, robot_configs)
        # Override cam_dirs_low to point to the actual depth data location
        if scene_low and cam_ids_low:
            cam_dirs_low = [args.task_folder / f"cam_{cid}" for cid in cam_ids_low]
            # Filter to only directories that actually exist
            valid_pairs = [(cid, cdir) for cid, cdir in zip(cam_ids_low, cam_dirs_low) if cdir.is_dir()]
            cam_ids_low = [cid for cid, _ in valid_pairs]
            cam_dirs_low = [cdir for _, cdir in valid_pairs]
            print(f"[INFO] Found {len(cam_ids_low)} cameras with depth data in {args.task_folder.name}")
    else:
        scene_low, cam_ids_low, cam_dirs_low = load_scene_data(args.task_folder, robot_configs)
    
    if not scene_low or not cam_ids_low: return

    scene_high, cam_ids_high, cam_dirs_high = (load_scene_data(args.high_res_folder, robot_configs) if args.high_res_folder else (None, None, None))
    
    # For human datasets, use scene_high for image pairs (has proper timestamps)
    # # For robot datasets, use scene_low
    # scene_for_image_pairs = scene_high if (args.dataset_type == "human" and scene_high) else scene_low
    # image_pairs = scene_for_image_pairs.get_image_path_pairs_period(vis_cfg_dict["time_interval"])
    # # # FIX FOR NOW!!
    # all_cameras = list(image_pairs[0].keys())
    # usable_cameras = [all_cameras[i] for i in (0, 2, 3)]
    # print(f"[INFO] Limiting to usable cameras: {usable_cameras}")
    # print("[INFO] FIX time alignment with the get image path function instead of manual time filtering")#TODO
    # # Filter cam_ids_low to only usable cameras
    # ##TODO: Remove this after testing
    # cam_ids_low = [cid for cid in cam_ids_low if cid in usable_cameras]

    # # Filter cam_ids_low to only usable cameras
    # ##TODO: Remove this after testing
    # cam_ids_high = [cid for cid in cam_ids_high if cid in usable_cameras]



    sync_kwargs = dict(
        frame_rate_hz=args.sync_fps,
        min_density=args.sync_min_density,
        target_fps=args.sync_fps,
        max_fps_drift=args.sync_max_drift,
        jitter_tolerance_ms=args.sync_tolerance_ms,
    )

    if args.high_res_folder:
        shared_ids = sorted(set(cam_ids_low) & set(cam_ids_high))

        id_to_dir_low = {cid: d for cid, d in zip(cam_ids_low, cam_dirs_low)}
        id_to_dir_high = {cid: d for cid, d in zip(cam_ids_high, cam_dirs_high)}

        final_cam_ids = [cid for cid in shared_ids if cid in id_to_dir_low and cid in id_to_dir_high]
        
        # Filter out cameras with letters (in-hand cameras)
        cameras_to_remove = [cid for cid in final_cam_ids if any(c.isalpha() for c in str(cid))]
        if cameras_to_remove:
            print(f"[INFO] Removing {len(cameras_to_remove)} camera(s) with non-numeric IDs (in-hand cameras): {cameras_to_remove}")
            final_cam_ids = [cid for cid in final_cam_ids if cid not in cameras_to_remove]
        
        cam_dirs_low = [id_to_dir_low[cid] for cid in final_cam_ids]
        cam_dirs_high = [id_to_dir_high[cid] for cid in final_cam_ids]

        if len(final_cam_ids) < 2:
            print("[ERROR] Fewer than 2 common cameras between low and high resolution data.")
            return

        if args.refine_colmap:
            print("[WARN] TODO: Fix the Colmap filtereing before removing camera or recalculae the frames after removing cameras. Less cameras means less frames to throw away.")

                  
        sync_low = get_synchronized_timestamps(cam_dirs_low, **sync_kwargs)
        sync_high = get_synchronized_timestamps(cam_dirs_high, require_depth=False, **sync_kwargs)

        valid_low = set(sync_low.camera_indices)
        valid_high = set(sync_high.camera_indices)
        keep_indices = sorted(valid_low & valid_high)

        if len(keep_indices) < 2:
            print("[ERROR] Synchronization rejected too many cameras; fewer than 2 remain aligned across resolutions.")
            return

        index_map_low = {idx: arr for idx, arr in zip(sync_low.camera_indices, sync_low.per_camera_timestamps)}
        index_map_high = {idx: arr for idx, arr in zip(sync_high.camera_indices, sync_high.per_camera_timestamps)}

        final_cam_ids = [final_cam_ids[i] for i in keep_indices]
        cam_dirs_low = [cam_dirs_low[i] for i in keep_indices]
        cam_dirs_high = [cam_dirs_high[i] for i in keep_indices]
        per_cam_low_full = [index_map_low[i] for i in keep_indices]
        per_cam_high_full = [index_map_high[i] for i in keep_indices]

        timeline_common = np.intersect1d(sync_low.timeline, sync_high.timeline)
        if timeline_common.size == 0:
            print("[ERROR] No overlapping synchronized timeline between low and high resolution data.")
            return

        timeline_common = np.asarray(timeline_common, dtype=np.int64)
        idx_map_low = {int(t): idx for idx, t in enumerate(sync_low.timeline)}
        idx_map_high = {int(t): idx for idx, t in enumerate(sync_high.timeline)}
        idx_low = [idx_map_low[int(t)] for t in timeline_common]
        idx_high = [idx_map_high[int(t)] for t in timeline_common]
        per_cam_low = [arr[idx_low] for arr in per_cam_low_full]
        #error here? hould be per_cam_high_full TODO: check
        per_cam_high = [arr[idx_high] for arr in per_cam_high_full] if per_cam_high_full is not None else None
    else:
        id_to_dir_low = {cid: d for cid, d in zip(cam_ids_low, cam_dirs_low)}
        final_cam_ids = [cid for cid in sorted(set(cam_ids_low)) if cid in id_to_dir_low]
        cam_dirs_low = [id_to_dir_low[cid] for cid in final_cam_ids]

        sync_low = get_synchronized_timestamps(cam_dirs_low, **sync_kwargs)
        valid_low = sorted(sync_low.camera_indices)
        if len(valid_low) < 2:
            print("[ERROR] Fewer than 2 cameras available for processing.")
            return
        final_cam_ids = [final_cam_ids[i] for i in valid_low]
        cam_dirs_low = [cam_dirs_low[i] for i in valid_low]
        index_map_low = {idx: arr for idx, arr in zip(sync_low.camera_indices, sync_low.per_camera_timestamps)}
        per_cam_low_full = [index_map_low[i] for i in valid_low]
        per_cam_low = per_cam_low_full
        per_cam_high = None
        timeline_common = np.asarray(sync_low.timeline, dtype=np.int64)

    if timeline_common.size == 0:
        print("[ERROR] Synchronization returned no frames.")
        return

    timestamps = select_frames(timeline_common, args.max_frames, args.frame_selection)
    if timestamps.size == 0:
        print("[ERROR] No frames remaining after selection.")
        return

    idx_map_common = {int(t): idx for idx, t in enumerate(timeline_common)}
    selected_idx = [idx_map_common[int(t)] for t in timestamps]
    per_cam_low_sel = [arr[selected_idx] for arr in per_cam_low]
    per_cam_high_sel = [arr[selected_idx] for arr in per_cam_high] if per_cam_high is not None else None
    # breakpoint()

    print("[DEBUG] Right now it is not using all timestampts. So for being and end it uses the closest. TODO: fix that, so that it uses all or more frames")

    closes_start_time = min(timestamps, key=lambda x: abs(x - start_timestamp))
    closes_end_time = min(timestamps, key=lambda x: abs(x - end_timestamp))
    # --- Step 2: Process Frames ---
    (rgbs,
     depths,
     intrs,
     extrs,
     robot_debug_points,
     robot_debug_colors,
     robot_gripper_boxes,
     robot_gripper_body_boxes,
     robot_gripper_fingertip_boxes,
     robot_gripper_pad_points,
     robot_tcp_points,
     robot_object_points,
     query_points,
     query_colors) = process_frames(
        args,
        scene_low,
        scene_high,
        final_cam_ids,
        cam_dirs_low,
        cam_dirs_high if args.high_res_folder else None,
        timestamps,
        per_cam_low_sel,
        per_cam_high_sel,
    )

    # --- Generate dummy tracking points for human datasets ---
    if args.dataset_type == "human":
        num_frames = len(timestamps)
        query_points = get_hand_tracked_points(num_frames, num_points=20)
        # Generate dummy colors (light blue to represent hands)
        query_colors = [np.tile([0.5, 0.7, 1.0], (20, 1)).astype(np.float32) for _ in range(num_frames)]
        print(f"[INFO] Generated {len(query_points)} frames of dummy hand tracking points")

    # --- Step 3: COLMAP Processing (if enabled) ---
    colmap_workspace = None
    if getattr(args, "refine_colmap", False):
        print("\n[INFO] ========== COLMAP Processing ==========")
        
        # Suppress verbose COLMAP logging
        import logging
        import os
        os.environ['GLOG_minloglevel'] = '2'  # Suppress INFO and WARNING logs from COLMAP
        logging.getLogger('pycolmap').setLevel(logging.ERROR)
        
        # First, verify camera alignment
        alignment_info = verify_camera_alignment(rgbs, depths, intrs, extrs, final_cam_ids)
        
        # Set up workspace
        import tempfile
        import shutil
        if args.colmap_workspace:
            colmap_workspace = Path(args.colmap_workspace).expanduser()
            colmap_workspace.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] Using provided COLMAP workspace: {colmap_workspace}")
        else:
            colmap_workspace = Path(tempfile.mkdtemp(prefix="colmap_workspace_"))
            print(f"[INFO] COLMAP workspace: {colmap_workspace}")
        
        # Define paths
        database_path = colmap_workspace / "database.db"
        images_dir = colmap_workspace / "images"
        sparse_dir = colmap_workspace / "sparse" / "0"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up COLMAP workspace
        setup_colmap_workspace(colmap_workspace, rgbs, intrs, extrs, final_cam_ids)
        
        # Run COLMAP pipeline
        if getattr(args, "refine_colmap", False):
            print("[INFO] Running COLMAP refinement pipeline...")
            
            # Always compute depth and feature quality (independent of COLMAP)
            print("[INFO] Evaluating depth quality...")
            depth_scores = evaluate_depth_quality(depths)
            print("[INFO] Depth quality scores:", {cid: f"{score:.2f}" for cid, score in zip(final_cam_ids, depth_scores)})
            
            print("[INFO] Evaluating feature richness...")
            feature_scores = evaluate_feature_richness(rgbs)
            print("[INFO] Feature richness scores:", {cid: f"{score:.2f}" for cid, score in zip(final_cam_ids, feature_scores)})
            
            # Feature extraction
            if not run_colmap_feature_extraction(colmap_workspace, database_path, images_dir):
                print("[WARN] COLMAP feature extraction failed, skipping COLMAP-based scoring")
                colmap_scores = {cid: 0.0 for cid in final_cam_ids}
            else:
                # Feature matching
                match_success, num_matches = run_colmap_matching(colmap_workspace, database_path)
                if not match_success or num_matches == 0:
                    print("[WARN] COLMAP matching failed or found no matches, skipping COLMAP-based scoring")
                    colmap_scores = {cid: 0.0 for cid in final_cam_ids}
                else:
                    # Mapping
                    map_success, map_stats = run_colmap_mapper(
                        colmap_workspace, database_path, images_dir, sparse_dir
                    )
                    if not map_success:
                        print("[WARN] COLMAP mapping failed, using only depth+feature quality for scoring")
                        colmap_scores = {cid: 0.0 for cid in final_cam_ids}
                    else:
                        # Evaluate camera quality from COLMAP reconstruction
                        colmap_scores = evaluate_camera_quality_from_colmap(
                            colmap_workspace, final_cam_ids
                        )
            
            # Filter cameras if limit is set
            if args.limit_num_cameras and args.limit_num_cameras < len(final_cam_ids):
                print(f"\n[INFO] Selecting best {args.limit_num_cameras} cameras based on combined quality scores...")
                selected_indices = select_best_cameras(
                    final_cam_ids, colmap_scores, depth_scores, feature_scores, args.limit_num_cameras
                )
                
                # Convert to numpy array for proper indexing
                selected_indices_np = np.array(selected_indices, dtype=int)
                
                # Filter all data to selected cameras
                final_cam_ids = [final_cam_ids[i] for i in selected_indices]
                cam_dirs_low = [cam_dirs_low[i] for i in selected_indices]
                if cam_dirs_high:
                    cam_dirs_high = [cam_dirs_high[i] for i in selected_indices]
                
                # Use numpy array indexing for array data
                rgbs = rgbs[selected_indices_np]
                depths = depths[selected_indices_np]
                intrs = intrs[selected_indices_np]
                extrs = extrs[selected_indices_np]
                
                per_cam_low_sel = [per_cam_low_sel[i] for i in selected_indices]
                if per_cam_high_sel:
                    per_cam_high_sel = [per_cam_high_sel[i] for i in selected_indices]
                
                print(f"[INFO] Filtered to {len(final_cam_ids)} cameras: {final_cam_ids}")
            else:
                print(f"[INFO] Keeping all {len(final_cam_ids)} cameras (no limit specified or limit >= camera count)")
        
        # Clean up temporary workspace if not specified
        if not args.colmap_workspace and colmap_workspace:
            print(f"[INFO] Cleaning up temporary COLMAP workspace: {colmap_workspace}")
            shutil.rmtree(colmap_workspace, ignore_errors=True)
        
        print("[INFO] ========== COLMAP Processing Complete ==========\n")
    
    # --- Step 3.5: Point Cloud Outlier Rejection (if enabled) ---
    if getattr(args, "reject_outlier_points", False):
        if not getattr(args, "refine_colmap", False):
            print("[WARN] --reject-outlier-points requires --refine-colmap, skipping outlier rejection")
        else:
            print("[INFO] Applying multi-view consistency check to reject outliers...")
            depths = reject_outlier_points_multiview(
                depths, intrs, extrs, 
                reprojection_threshold=args.outlier_reprojection_threshold
            )

    # --- Step 4: Save and Visualize ---
    sam2_masks = None
    sam2_fused_clouds = None
    rr.init("RH20T_Reprojection_Frameworks", spawn=False)
    # Set the desired coordinate system for the 'world' space
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # --- ADD THIS DIAGNOSTIC CODE ---
    # Log visible arrows to represent the axes of the 'world' space
    rr.log(
        "world/axes",
        rr.Arrows3D(
            origins=[[0, 0, 0]],
            vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]] # X=Red, Y=Green, Z=Blue
        ),
        static=True
    )
    # --- END OF DIAGNOSTIC CODE ---


    per_cam_for_npz = per_cam_high_sel if per_cam_high_sel is not None else per_cam_low_sel
    save_and_visualize(
        args,
        rgbs,
        depths,
        intrs,
        extrs,
        final_cam_ids,
        timestamps,
        per_cam_for_npz,
        robot_debug_points,
        robot_debug_colors,
        robot_gripper_boxes,
        robot_gripper_body_boxes,
        robot_gripper_fingertip_boxes,
        robot_gripper_pad_points,
        robot_tcp_points,
        robot_object_points,
        query_points,
        query_colors,
    )

    if not args.no_pointcloud:
        rrd_path = args.out_dir / f"{args.task_folder.name}_reprojected.rrd"
        rr.save(str(rrd_path))
        print(f"✅ [OK] Saved Rerun visualization to: {rrd_path}")

if __name__ == "__main__":
    main()
