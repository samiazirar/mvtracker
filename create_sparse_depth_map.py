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


def reject_outlier_points_multiview(
    depths: np.ndarray,
    intrs: np.ndarray,
    extrs: np.ndarray,
    reprojection_threshold: float = 5.0,
) -> np.ndarray:
    """
    Reject outlier 3D points using multi-view consistency.
    
    For each 3D point derived from depth, check if it reprojects consistently
    to other camera views. A point is kept only if it has sufficient support
    from multiple views.
    
    Args:
        depths: (C, T, H, W) depth maps
        intrs: (C, T, 3, 3) camera intrinsics
        extrs: (C, T, 3, 4) camera extrinsics (world-to-camera)
        reprojection_threshold: Maximum pixel error for a reprojection to be considered an inlier
    
    Returns:
        depths_filtered: (C, T, H, W) depth maps with outliers set to 0
    """
    print(f"\n[INFO] ========== Point Cloud Outlier Rejection ==========")
    print(f"[INFO] Reprojection threshold: {reprojection_threshold} pixels")
    
    C, T, H, W = depths.shape
    depths_filtered = depths.copy()
    
    total_points = 0
    kept_points = 0
    
    # Process each frame independently
    for t in range(T):
        # Build 3D points from all cameras for this frame
        all_points = []
        all_colors = []  # For visualization, we could add colors later
        point_sources = []  # Track which camera each point came from
        
        for c in range(C):
            depth = depths[c, t]
            K = intrs[c, t]
            E_world_to_cam = extrs[c, t]
            
            # Convert to camera-to-world for easier 3D point computation
            R_w2c = E_world_to_cam[:3, :3]
            t_w2c = E_world_to_cam[:3, 3]
            R_c2w = R_w2c.T
            t_c2w = -R_c2w @ t_w2c
            
            # Get valid depth pixels
            valid_mask = depth > 0
            v_coords, u_coords = np.where(valid_mask)
            
            if len(u_coords) == 0:
                continue
            
            # Backproject to 3D in camera frame
            z = depth[v_coords, u_coords]
            x = (u_coords - K[0, 2]) * z / K[0, 0]
            y = (v_coords - K[1, 2]) * z / K[1, 1]
            pts_cam = np.stack([x, y, z], axis=1)
            
            # Transform to world frame
            pts_world = (R_c2w @ pts_cam.T).T + t_c2w
            
            all_points.append(pts_world)
            point_sources.extend([(c, u, v) for u, v in zip(u_coords, v_coords)])
        
        if len(all_points) == 0:
            continue
        
        all_points = np.vstack(all_points)
        frame_total = len(all_points)
        total_points += frame_total
        
        # For each point, check how many cameras see it with consistent depth
        inlier_mask = np.ones(len(all_points), dtype=bool)
        
        for i, (pts_3d, src_info) in enumerate(zip(all_points, point_sources)):
            src_cam, src_u, src_v = src_info
            
            # Count how many other cameras see this point with consistent depth
            support_count = 0
            
            for c in range(C):
                if c == src_cam:
                    # Skip source camera
                    continue
                
                K = intrs[c, t]
                E_world_to_cam = extrs[c, t]
                R_w2c = E_world_to_cam[:3, :3]
                t_w2c = E_world_to_cam[:3, 3]
                
                # Project point to this camera
                pts_cam = R_w2c @ pts_3d + t_w2c
                
                # Check if point is in front of camera
                if pts_cam[2] <= 0:
                    continue
                
                # Project to image coordinates
                u_proj = K[0, 0] * pts_cam[0] / pts_cam[2] + K[0, 2]
                v_proj = K[1, 1] * pts_cam[1] / pts_cam[2] + K[1, 2]
                
                # Check if projection is within image bounds
                if u_proj < 0 or u_proj >= W or v_proj < 0 or v_proj >= H:
                    continue
                
                u_int = int(np.round(u_proj))
                v_int = int(np.round(v_proj))
                
                # Get depth at projected location
                depth_at_proj = depths[c, t, v_int, u_int]
                
                if depth_at_proj <= 0:
                    continue
                
                # Check consistency: does the depth match?
                depth_error = abs(pts_cam[2] - depth_at_proj)
                pixel_error = np.sqrt((u_proj - u_int)**2 + (v_proj - v_int)**2)
                
                # Accept if both depth and pixel location are consistent
                if pixel_error < reprojection_threshold and depth_error < 0.05:  # 5cm depth tolerance
                    support_count += 1
            
            # Require at least 1 other camera to confirm this point (for 2+ cameras)
            # For more cameras, we could require more support
            min_support = max(1, (C - 1) // 2)  # At least half of other cameras
            if support_count < min_support:
                inlier_mask[i] = False
        
        # Update depths to remove outliers
        for i, src_info in enumerate(point_sources):
            if not inlier_mask[i]:
                src_cam, src_u, src_v = src_info
                depths_filtered[src_cam, t, src_v, src_u] = 0
        
        frame_kept = inlier_mask.sum()
        kept_points += frame_kept
        
        if frame_total > 0:
            print(f"[INFO] Frame {t}: kept {frame_kept}/{frame_total} points ({100*frame_kept/frame_total:.1f}%)")
    
    if total_points > 0:
        print(f"[INFO] Overall: kept {kept_points}/{total_points} points ({100*kept_points/total_points:.1f}%)")
    print(f"[INFO] ========== Outlier Rejection Complete ==========\n")
    
    return depths_filtered


def save_and_visualize(
    args,
    rgbs,
    depths,
    intrs,
    extrs,
    final_cam_ids,
    timestamps,
    per_camera_timestamps,
    robot_debug_points=None,
    robot_debug_colors=None,
    robot_gripper_boxes=None,
    robot_gripper_body_boxes=None,
    robot_gripper_fingertip_boxes=None,
    robot_gripper_pad_points=None,
    robot_tcp_points=None,
    robot_object_points=None,
    query_points=None,
    query_colors=None,
):
    """Saves the processed data to an NPZ file and generates a Rerun visualization."""
    # Convert to channels-first format for NPZ
    rgbs_final = np.moveaxis(rgbs, -1, 2)
    depths_final = depths[:, :, None, :, :]

    per_cam_ts_arr = np.stack(per_camera_timestamps, axis=0).astype(np.int64)
    
    # Format and save query points if available
    # Query points are saved in mvtracker format: [frame_index, x, y, z]
    if query_points is not None and len(query_points) > 0:
        formatted_query_points = []
        for frame_idx, qpts in enumerate(query_points):
            if qpts is not None and qpts.size > 0:
                # Add frame index as first column: [frame_idx, x, y, z]
                n_points = len(qpts)
                frame_indices = np.full((n_points, 1), frame_idx, dtype=np.float32)
                qpts_with_time = np.concatenate([frame_indices, qpts], axis=1)
                formatted_query_points.append(qpts_with_time)
        
        if formatted_query_points:
            # Concatenate all query points from all frames: shape (N_total, 4)
            all_query_points = np.concatenate(formatted_query_points, axis=0)
            print(f"[INFO] Formatted {len(all_query_points)} query points for saving")
        else:
            all_query_points = np.empty((0, 4), dtype=np.float32)
            print("[INFO] No valid query points found")
    else:
        all_query_points = None
    
    # Persist all modalities together so downstream tools can reload the full synchronized packet.
    npz_payload = {
        'rgbs': rgbs_final,
        'depths': depths_final,
        'intrs': intrs,
        'extrs': extrs,
        'timestamps': timestamps,
        'per_camera_timestamps': per_cam_ts_arr,
        'camera_ids': np.array(final_cam_ids, dtype=object),
    }
    
    # Add query points to payload if available
    if all_query_points is not None:
        npz_payload['query_points'] = all_query_points

    out_path_npz = args.out_dir / f"{args.task_folder.name}_processed.npz"
    np.savez_compressed(out_path_npz, **npz_payload)
    print(f"✅ [OK] Wrote NPZ file to: {out_path_npz}")

    # Generate Rerun Visualization
    if not args.no_pointcloud:
        print("[INFO] Logging data to Rerun...")
        # Cast to tensors because the visualizer expects batched torch inputs.
        rgbs_tensor = torch.from_numpy(rgbs_final).float().unsqueeze(0)
        depths_tensor = torch.from_numpy(depths_final).float().unsqueeze(0)
        intrs_tensor = torch.from_numpy(intrs).float().unsqueeze(0)
        extrs_tensor = torch.from_numpy(extrs).float().unsqueeze(0)
        log_pointclouds_to_rerun(
            dataset_name="rh20t_reprojection",
            datapoint_idx=0,
            rgbs=rgbs_tensor,
            depths=depths_tensor,
            intrs=intrs_tensor,
            extrs=extrs_tensor,
            camera_ids=final_cam_ids,
            log_rgb_pointcloud=True,
            log_camera_frustrum=True,
        )
        configs_path = args.config.resolve()
        rh20t_root = configs_path.parent.parent if configs_path.parent.name == "configs" else configs_path.parent

        # Log robot points separately (they survive better than going through reprojection)
        if robot_debug_points is not None and len(robot_debug_points) > 0:
            fps = 12.0 #TODO: adjust to one fps rate later tot he real one
            print(f"[INFO] Logging {len(robot_debug_points)} robot point clouds to Rerun...")
            for idx, pts in enumerate(robot_debug_points):
                if pts is None or pts.size == 0:
                    continue
                # Keep robot color palette aligned with the optional debug stream.
                cols = robot_debug_colors[idx] if robot_debug_colors and idx < len(robot_debug_colors) else None
                rr.set_time_seconds("frame", idx / fps)
                # Log to top-level robot entity for easy toggling (NOT under sequence-0)
                rr.log(
                    f"robot",
                    rr.Points3D(pts.astype(np.float32, copy=False), colors=cols),
                )
                # Also log to debug entity if in debug mode
                if getattr(args, "debug_mode", False):
                    rr.log(
                        f"robot_debug",
                        rr.Points3D(pts.astype(np.float32, copy=False), colors=cols),
                    )
        
        # Log query points (sensor points inside gripper bbox) if requested
        if getattr(args, "visualize_query_points", False) and query_points is not None and len(query_points) > 0:
            fps = 12.0
            valid_query_count = sum(1 for qpts in query_points if qpts is not None and qpts.size > 0)
            if valid_query_count > 0:
                print(f"[INFO] Logging {valid_query_count} query point clouds to Rerun (magenta)...")
                for idx, qpts in enumerate(query_points):
                    if qpts is None or qpts.size == 0:
                        continue
                    rr.set_time_seconds("frame", idx / fps)
                    # Log query points as magenta (original sensor points inside bbox)
                    # Create magenta color array (255, 0, 255)
                    magenta_colors = np.full((len(qpts), 3), [255, 0, 255], dtype=np.uint8)
                    rr.log(
                        "query_points",
                        rr.Points3D(qpts.astype(np.float32, copy=False), colors=magenta_colors),
                    )
        
        if robot_gripper_boxes:
            valid_box_count = sum(1 for box in robot_gripper_boxes if box)
            if valid_box_count > 0:
                fps = 12.0
                print(f"[INFO] Logging {valid_box_count} gripper bounding boxes to Rerun...")
                for idx, box in enumerate(robot_gripper_boxes):
                    if not box:
                        continue
                    rr.set_time_seconds("frame", idx / fps)
                    centers = np.asarray(box["center"], dtype=np.float32)[None, :]
                    half_sizes = np.asarray(box["half_sizes"], dtype=np.float32)[None, :]
                    rr.log(
                        "robot/gripper_bbox",
                        rr.Boxes3D(
                            centers=centers,
                            half_sizes=half_sizes,
                            quaternions=np.asarray(box["quat_xyzw"], dtype=np.float32)[None, :],
                            colors=np.array([[255, 128, 0]], dtype=np.uint8),
                        ),
                    )
                    basis = _ensure_basis(box)
                    if basis is not None:
                        axes = np.asarray(basis, dtype=np.float32)
                        half_sizes_vec = np.asarray(box["half_sizes"], dtype=np.float32)
                        center_vec = np.asarray(box["center"], dtype=np.float32)

                        approach_axis = axes[:, 2]
                        approach_axis_norm = np.linalg.norm(approach_axis) + 1e-12
                        approach_axis = approach_axis / approach_axis_norm
                        # Draw helper arrows to confirm the oriented bounding box aligns with the jaw approach direction.
                        origin = center_vec - approach_axis * half_sizes_vec[2]
                        vector = approach_axis * (half_sizes_vec[2] * 2.0)
                        rr.log(
                            "robot/gripper_bbox_centerline",
                            rr.Arrows3D(
                                origins=origin[np.newaxis, :],
                                vectors=vector[np.newaxis, :],
                                colors=np.array([[255, 128, 0]], dtype=np.uint8),
                                radii=0.004,
                            ),
                        )
                        up_axis = axes[:, 1]
                        up_axis_norm = np.linalg.norm(up_axis) + 1e-12
                        up_axis = up_axis / up_axis_norm
                        rr.log(
                            "robot/gripper_bbox_axis_height",
                            rr.Arrows3D(
                                origins=center_vec[np.newaxis, :],
                                vectors=(up_axis * half_sizes_vec[1])[np.newaxis, :],
                                colors=np.array([[0, 200, 0]], dtype=np.uint8),
                                radii=0.003,
                            ),
                        )
                        width_axis = axes[:, 0]
                        width_axis_norm = np.linalg.norm(width_axis) + 1e-12
                        width_axis = width_axis / width_axis_norm
                        rr.log(
                            "robot/gripper_bbox_axis_width",
                            rr.Arrows3D(
                                origins=center_vec[np.newaxis, :],
                                vectors=(width_axis * half_sizes_vec[0])[np.newaxis, :],
                                colors=np.array([[0, 150, 255]], dtype=np.uint8),
                                radii=0.003,
                            ),
                        )
        if robot_gripper_body_boxes:
            valid_box_count = sum(1 for box in robot_gripper_body_boxes if box)
            if valid_box_count > 0:
                fps = 12.0
                print(f"[INFO] Logging {valid_box_count} gripper BODY bounding boxes to Rerun...")
                for idx, box in enumerate(robot_gripper_body_boxes):
                    if not box:
                        continue
                    rr.set_time_seconds("frame", idx / fps)
                    centers = np.asarray(box["center"], dtype=np.float32)[None, :]
                    half_sizes = np.asarray(box["half_sizes"], dtype=np.float32)[None, :]
                    # Namespace logs under "robot/..." so the inspector groups all overlays together.
                    rr.log(
                        "robot/gripper_bbox_body",
                        rr.Boxes3D(
                            centers=centers,
                            half_sizes=half_sizes,
                            quaternions=np.asarray(box["quat_xyzw"], dtype=np.float32)[None, :],
                            colors=np.array([[255, 0, 0]], dtype=np.uint8),
                        ),
                    )
                    basis = _ensure_basis(box)
                    if basis is not None:
                        axes = np.asarray(basis, dtype=np.float32)
                        half_sizes_vec = np.asarray(box["half_sizes"], dtype=np.float32)
                        center_vec = np.asarray(box["center"], dtype=np.float32)

                        approach_axis = axes[:, 2]
                        approach_axis_norm = np.linalg.norm(approach_axis) + 1e-12
                        approach_axis = approach_axis / approach_axis_norm
                        origin = center_vec - approach_axis * half_sizes_vec[2]
                        vector = approach_axis * (half_sizes_vec[2] * 2.0)
                        rr.log(
                            "robot/gripper_bbox_body_centerline",
                            rr.Arrows3D(
                                origins=origin[np.newaxis, :],
                                vectors=vector[np.newaxis, :],
                                colors=np.array([[255, 0, 0]], dtype=np.uint8),
                                radii=0.004,
                            ),
                        )
                        up_axis = axes[:, 1]
                        up_axis_norm = np.linalg.norm(up_axis) + 1e-12
                        up_axis = up_axis / up_axis_norm
                        rr.log(
                            "robot/gripper_bbox_body_axis_height",
                            rr.Arrows3D(
                                origins=center_vec[np.newaxis, :],
                                vectors=(up_axis * half_sizes_vec[1])[np.newaxis, :],
                                colors=np.array([[0, 200, 0]], dtype=np.uint8),
                                radii=0.003,
                            ),
                        )
                        width_axis = axes[:, 0]
                        width_axis_norm = np.linalg.norm(width_axis) + 1e-12
                        width_axis = width_axis / width_axis_norm
                        rr.log(
                            "robot/gripper_bbox_body_axis_width",
                            rr.Arrows3D(
                                origins=center_vec[np.newaxis, :],
                                vectors=(width_axis * half_sizes_vec[0])[np.newaxis, :],
                                colors=np.array([[0, 150, 255]], dtype=np.uint8),
                                radii=0.003,
                            ),
                        )
        if robot_gripper_fingertip_boxes:
            valid_box_count = sum(1 for box in robot_gripper_fingertip_boxes if box)
            if valid_box_count > 0:
                fps = 12.0
                print(f"[INFO] Logging {valid_box_count} gripper FINGERTIP bounding boxes to Rerun...")
                for idx, box in enumerate(robot_gripper_fingertip_boxes):
                    if not box:
                        continue
                    rr.set_time_seconds("frame", idx / fps)
                    centers = np.asarray(box["center"], dtype=np.float32)[None, :]
                    half_sizes = np.asarray(box["half_sizes"], dtype=np.float32)[None, :]
                    rr.log(
                        "robot/gripper_bbox_fingertip",
                        rr.Boxes3D(
                            centers=centers,
                            half_sizes=half_sizes,
                            quaternions=np.asarray(box["quat_xyzw"], dtype=np.float32)[None, :],
                            colors=np.array([[0, 0, 255]], dtype=np.uint8),  # Blue color
                        ),
                    )
                    basis = _ensure_basis(box)
                    if basis is not None:
                        axes = np.asarray(basis, dtype=np.float32)
                        half_sizes_vec = np.asarray(box["half_sizes"], dtype=np.float32)
                        center_vec = np.asarray(box["center"], dtype=np.float32)

                        approach_axis = axes[:, 2]
                        approach_axis_norm = np.linalg.norm(approach_axis) + 1e-12
                        approach_axis = approach_axis / approach_axis_norm
                        origin = center_vec - approach_axis * half_sizes_vec[2]
                        vector = approach_axis * (half_sizes_vec[2] * 2.0)
                        rr.log(
                            "robot/gripper_bbox_fingertip_centerline",
                            rr.Arrows3D(
                                origins=origin[np.newaxis, :],
                                vectors=vector[np.newaxis, :],
                                colors=np.array([[0, 0, 255]], dtype=np.uint8),
                                radii=0.004,
                            ),
                        )
                        up_axis = axes[:, 1]
                        up_axis_norm = np.linalg.norm(up_axis) + 1e-12
                        up_axis = up_axis / up_axis_norm
                        rr.log(
                            "robot/gripper_bbox_fingertip_axis_height",
                            rr.Arrows3D(
                                origins=center_vec[np.newaxis, :],
                                vectors=(up_axis * half_sizes_vec[1])[np.newaxis, :],
                                colors=np.array([[0, 200, 0]], dtype=np.uint8),
                                radii=0.003,
                            ),
                        )
                        width_axis = axes[:, 0]
                        width_axis_norm = np.linalg.norm(width_axis) + 1e-12
                        width_axis = width_axis / width_axis_norm
                        rr.log(
                            "robot/gripper_bbox_fingertip_axis_width",
                            rr.Arrows3D(
                                origins=center_vec[np.newaxis, :],
                                vectors=(width_axis * half_sizes_vec[0])[np.newaxis, :],
                                colors=np.array([[0, 150, 255]], dtype=np.uint8),
                                radii=0.003,
                            ),
                        )
        if robot_gripper_pad_points:
            fps = 12.0
            valid_pts = any(pts is not None and len(pts) > 0 for pts in robot_gripper_pad_points)
            if valid_pts:
                count = sum(1 for pts in robot_gripper_pad_points if pts is not None and len(pts) > 0)
                print(f"[INFO] Logging {count} gripper pad point sets to Rerun (magenta)...")
                for idx, pts in enumerate(robot_gripper_pad_points):
                    if pts is None or len(pts) == 0:
                        continue
                    rr.set_time_seconds("frame", idx / fps)
                    cols = np.tile(np.array([[255, 0, 255]], dtype=np.uint8), (len(pts), 1))
                    rr.log(
                        "robot/gripper_pad_points",
                        rr.Points3D(pts.astype(np.float32, copy=False), colors=cols),
                    )
        
        # Log TCP points from API (cyan spheres)
        if robot_tcp_points:
            fps = args.sync_fps if args.sync_fps > 0 else 30.0
            valid_pts = any(pt is not None and len(pt) == 3 for pt in robot_tcp_points)
            if valid_pts:
                count = sum(1 for pt in robot_tcp_points if pt is not None and len(pt) == 3)
                tcp_radius = getattr(args, "tcp_point_radius", 0.01)
                print(f"[INFO] Logging {count} TCP points from API to Rerun (cyan, radius={tcp_radius}m)...")
                for idx, pt in enumerate(robot_tcp_points):
                    if pt is None or len(pt) != 3:
                        continue
                    rr.set_time_seconds("frame", idx / fps)
                    # Log as a single cyan point with larger radius
                    rr.log(
                        "points/tcp_point",
                        rr.Points3D(
                            pt.reshape(1, 3).astype(np.float32, copy=False),
                            colors=np.array([[0, 255, 255]], dtype=np.uint8),  # Cyan
                            radii=np.array([0.02], dtype=np.float32),  # Increased for visibility
                        ),
                    )
        
        # Log object points from API (yellow spheres)
        if robot_object_points:
            fps = args.sync_fps if args.sync_fps > 0 else 30.0
            valid_pts = any(pt is not None and len(pt) == 3 for pt in robot_object_points)
            if valid_pts:
                count = sum(1 for pt in robot_object_points if pt is not None and len(pt) == 3)
                print(f"[INFO] Logging {count} object points from API to Rerun (yellow)...")
                for idx, pt in enumerate(robot_object_points):
                    if pt is None or len(pt) != 3:
                        continue
                    rr.set_time_seconds("frame", idx / fps)
                    # Log as a single yellow point
                    rr.log(
                        "points/object_point",
                        rr.Points3D(
                            pt.reshape(1, 3).astype(np.float32, copy=False),
                            colors=np.array([[255, 255, 0]], dtype=np.uint8),  # Yellow
                            radii=np.array([0.025], dtype=np.float32),  # Increased for visibility
                        ),
                    )


    if getattr(args, "export_bbox_video", False):
        try:
            _export_gripper_bbox_videos(
                args,
                rgbs,
                intrs,
                extrs,
                robot_gripper_boxes,
                final_cam_ids,
            )
        except Exception as exc:
            print(f"[WARN] Failed to export bounding box videos: {exc}")


# --- COLMAP Integration Functions ---

def setup_colmap_workspace(
    workspace_dir: Path,
    rgbs: np.ndarray,
    intrs: np.ndarray,
    extrs: np.ndarray,
    camera_ids: List[str],
) -> Path:
    """
    Set up COLMAP workspace with images and camera parameters.
    
    Args:
        workspace_dir: Path to COLMAP workspace directory
        rgbs: RGB images (C, T, H, W, 3)
        intrs: Camera intrinsics (C, T, 3, 3)
        extrs: Camera extrinsics (C, T, 3, 4)
        camera_ids: List of camera IDs
    
    Returns:
        Path to the created workspace
    """
    workspace_dir.mkdir(parents=True, exist_ok=True)
    images_dir = workspace_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    C, T, H, W, _ = rgbs.shape
    
    # Export images
    print(f"[INFO] Exporting {C * T} images to COLMAP workspace...")
    for c_idx in range(C):
        for t_idx in range(T):
            img = rgbs[c_idx, t_idx]
            img_path = images_dir / f"cam_{camera_ids[c_idx]}_frame_{t_idx:04d}.jpg"
            Image.fromarray(img).save(img_path, quality=95)
    
    # Create cameras.txt
    cameras_file = workspace_dir / "cameras.txt"
    with open(cameras_file, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for c_idx in range(C):
            K = intrs[c_idx, 0]  # Use first frame's intrinsics
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            # COLMAP PINHOLE model: fx, fy, cx, cy
            f.write(f"{c_idx} PINHOLE {W} {H} {fx} {fy} {cx} {cy}\n")
    
    # Create images.txt
    images_file = workspace_dir / "images.txt"
    with open(images_file, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        img_id = 1
        for c_idx in range(C):
            for t_idx in range(T):
                E = extrs[c_idx, t_idx]  # 3x4 world-to-camera
                R = E[:3, :3]
                t = E[:3, 3]
                
                # Convert rotation matrix to quaternion (w, x, y, z)
                quat = _rotation_matrix_to_quaternion(R)
                
                img_name = f"cam_{camera_ids[c_idx]}_frame_{t_idx:04d}.jpg"
                f.write(f"{img_id} {quat[0]} {quat[1]} {quat[2]} {quat[3]} ")
                f.write(f"{t[0]} {t[1]} {t[2]} {c_idx} {img_name}\n")
                f.write("\n")  # Empty line for POINTS2D
                img_id += 1
    
    print(f"[INFO] COLMAP workspace created at {workspace_dir}")
    return workspace_dir


def _rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    return np.array([w, x, y, z])



def run_colmap_feature_extraction(workspace_dir: Path, database_path: Path, images_dir: Path) -> bool:
    """Run COLMAP feature extraction using pycolmap."""
    try:
        print(f"[INFO] Running pycolmap feature extraction...")
        pycolmap.extract_features(database_path, images_dir)
        print("[INFO] Feature extraction completed successfully")
        return True
    except Exception as e:
        print(f"[ERROR] Feature extraction failed: {e}")
        return False


def run_colmap_matching(workspace_dir: Path, database_path: Path) -> tuple[bool, int]:
    """Run COLMAP feature matching using pycolmap.
    
    Returns:
        (success, num_matches): Whether matching succeeded and number of image pairs with matches
    """
    try:
        print(f"[INFO] Running pycolmap feature matching...")
        
        # Try sequential matching first (faster)
        print("[INFO] Trying sequential matching (overlap=5)...")
        pycolmap.match_sequential(
            database_path=str(database_path),
            matching_options=pycolmap.SequentialMatchingOptions(overlap=5)
        )
        
        # Query database for match statistics
        import sqlite3
        conn = sqlite3.connect(str(database_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM two_view_geometries WHERE rows > 0")
        num_matches = cursor.fetchone()[0]
        conn.close()
        
        print(f"[INFO] Sequential matching found {num_matches} image pairs with matches")
        
        # If sequential matching found very few matches, try exhaustive (slower but more thorough)
        if num_matches < 5:
            print("[INFO] Few matches found, trying exhaustive matching...")
            pycolmap.match_exhaustive(database_path)
            
            conn = sqlite3.connect(str(database_path))
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM two_view_geometries WHERE rows > 0")
            num_matches = cursor.fetchone()[0]
            conn.close()
            
            print(f"[INFO] Exhaustive matching found {num_matches} image pairs with matches")
        
        if num_matches == 0:
            print("[WARN] No feature matches found between images!")
            print("[WARN] This usually means:")
            print("[WARN]   1. Images have insufficient texture/features")
            print("[WARN]   2. Cameras don't share overlapping views (check alignment above)")
            print("[WARN]   3. Images are too different (lighting, blur, etc.)")
        
        return True, num_matches
    except Exception as e:
        print(f"[ERROR] Feature matching failed: {e}")
        import traceback
        traceback.print_exc()
        return False, 0


def run_colmap_mapper(workspace_dir: Path, database_path: Path, images_dir: Path, output_dir: Path, use_known_poses: bool = True) -> tuple[bool, dict]:
    """Run COLMAP mapper to reconstruct scene using pycolmap.
    
    Args:
        use_known_poses: Ignored for now - always uses standard incremental mapping
    
    Returns:
        (success, stats): Whether mapping succeeded and reconstruction statistics
    """
    try:
        print(f"[INFO] Running pycolmap mapper...")
        
        # Use standard incremental mapping - it's more robust and handles everything
        # The calibration is provided via cameras.txt which COLMAP will use as initial guess
        maps = pycolmap.incremental_mapping(database_path, images_dir, output_dir)
        
        if len(maps) == 0:
            print("[ERROR] No reconstruction created")
            print("[ERROR] This usually means:")
            print("[ERROR]   1. No good initial image pair found (insufficient feature matches)")
            print("[ERROR]   2. Reconstruction failed to grow beyond initial pair")
            print("[ERROR]   3. Geometric verification failed (inconsistent camera poses)")
            return False, {}
        
        # Get statistics from first (usually best) reconstruction
        reconstruction = maps[0]
        stats = {
            'num_reconstructions': len(maps),
            'num_registered_images': reconstruction.num_images(),
            'num_points3d': reconstruction.num_points3D(),
            'num_observations': sum(len(p.track.elements) for p in reconstruction.points3D.values())
        }
        
        print(f"[INFO] Mapping completed successfully:")
        print(f"[INFO]   - {stats['num_reconstructions']} reconstruction(s)")
        print(f"[INFO]   - {stats['num_registered_images']} registered images")
        print(f"[INFO]   - {stats['num_points3d']} 3D points")
        print(f"[INFO]   - {stats['num_observations']} observations")
        
        return True, stats
    except Exception as e:
        print(f"[ERROR] Mapping failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {}


def verify_camera_alignment(
    rgbs: np.ndarray,
    depths: np.ndarray,
    intrs: np.ndarray,
    extrs: np.ndarray,
    camera_ids: List[str],
) -> Dict[str, Any]:
    """
    Verify camera alignment by checking multi-view consistency.
    
    Args:
        rgbs: (C, T, H, W, 3) RGB images
        depths: (C, T, H, W) depth maps
        intrs: (C, T, 3, 3) intrinsics
        extrs: (C, T, 3, 4) extrinsics (world-to-camera)
    
    Returns:
        Dictionary with alignment diagnostics
    """
    print("\n[INFO] ========== Camera Alignment Verification ==========")
    
    C, T, H, W = depths.shape
    
    # Check 1: Are extrinsics valid?
    print("[INFO] Checking camera extrinsics...")
    for c_idx in range(C):
        E = extrs[c_idx, 0]  # First frame
        R = E[:3, :3]
        t = E[:3, 3]
        
        # Check if rotation matrix is valid (determinant should be ~1)
        det = np.linalg.det(R)
        orthogonality = np.linalg.norm(R @ R.T - np.eye(3))
        
        print(f"[INFO]   Camera {camera_ids[c_idx]}: det(R)={det:.4f}, orthogonality_error={orthogonality:.6f}")
        
        if abs(det - 1.0) > 0.1:
            print(f"[WARN]   Camera {camera_ids[c_idx]}: Rotation matrix determinant is {det:.4f} (should be ~1.0)")
        if orthogonality > 0.01:
            print(f"[WARN]   Camera {camera_ids[c_idx]}: Rotation matrix is not orthogonal (error={orthogonality:.6f})")
    
    # Check 2: Do cameras have overlapping views?
    print("\n[INFO] Checking camera view overlap...")
    frame_idx = T // 2  # Use middle frame
    
    # Project points from one camera to another
    overlap_matrix = np.zeros((C, C))
    
    for c1 in range(C):
        depth1 = depths[c1, frame_idx]
        K1 = intrs[c1, frame_idx]
        E1 = extrs[c1, frame_idx]
        
        # Get valid depth points (sample to speed up)
        valid_mask = depth1 > 0
        v_coords, u_coords = np.where(valid_mask)
        
        if len(u_coords) == 0:
            continue
        
        # Sample points (max 1000 for speed)
        if len(u_coords) > 1000:
            indices = np.random.choice(len(u_coords), 1000, replace=False)
            u_coords = u_coords[indices]
            v_coords = v_coords[indices]
        
        # Backproject to 3D in camera frame
        z1 = depth1[v_coords, u_coords]
        x1 = (u_coords - K1[0, 2]) * z1 / K1[0, 0]
        y1 = (v_coords - K1[1, 2]) * z1 / K1[1, 1]
        pts_cam1 = np.stack([x1, y1, z1], axis=1)
        
        # Transform to world frame
        R1 = E1[:3, :3]
        t1 = E1[:3, 3]
        R1_inv = R1.T
        t1_inv = -R1_inv @ t1
        pts_world = (R1_inv @ pts_cam1.T).T + t1_inv
        
        # Project to other cameras
        for c2 in range(C):
            if c1 == c2:
                overlap_matrix[c1, c2] = 1.0
                continue
            
            K2 = intrs[c2, frame_idx]
            E2 = extrs[c2, frame_idx]
            R2 = E2[:3, :3]
            t2 = E2[:3, 3]
            
            # Transform to camera 2 frame
            pts_cam2 = (R2 @ pts_world.T).T + t2
            
            # Project to image
            valid_in_front = pts_cam2[:, 2] > 0
            pts_cam2_valid = pts_cam2[valid_in_front]
            
            if len(pts_cam2_valid) == 0:
                overlap_matrix[c1, c2] = 0.0
                continue
            
            u2 = K2[0, 0] * pts_cam2_valid[:, 0] / pts_cam2_valid[:, 2] + K2[0, 2]
            v2 = K2[1, 1] * pts_cam2_valid[:, 1] / pts_cam2_valid[:, 2] + K2[1, 2]
            
            # Check how many points are visible in camera 2
            visible = (u2 >= 0) & (u2 < W) & (v2 >= 0) & (v2 < H)
            overlap_ratio = visible.sum() / len(pts_cam2_valid)
            overlap_matrix[c1, c2] = overlap_ratio
    
    print("\n[INFO] Camera overlap matrix (fraction of points visible):")
    print("[INFO]          ", end="")
    for c in range(C):
        print(f"Cam{camera_ids[c][:4]:>6}", end="  ")
    print()
    for c1 in range(C):
        print(f"[INFO] Cam{camera_ids[c1][:4]:<5}", end="")
        for c2 in range(C):
            if c1 == c2:
                print(f"  ---  ", end="  ")
            else:
                print(f" {overlap_matrix[c1, c2]:>5.2%}", end="  ")
        print()
    
    # Check if overlap is too low
    avg_overlap = np.mean(overlap_matrix[np.triu_indices(C, k=1)])
    print(f"\n[INFO] Average camera overlap: {avg_overlap:.2%}")
    
    if avg_overlap < 0.1:
        print("[WARN] Very low camera overlap (<10%)! Cameras may not see the same scene.")
        print("[WARN] This will cause:")
        print("[WARN]   - COLMAP to fail (no common features)")
        print("[WARN]   - 'Seeing everything twice' (inconsistent views)")
        print("[WARN] Possible causes:")
        print("[WARN]   - Wrong camera extrinsics (R, t)")
        print("[WARN]   - Cameras pointing at different objects")
        print("[WARN]   - Wrong world coordinate frame")
    elif avg_overlap < 0.3:
        print("[WARN] Low camera overlap (<30%). COLMAP may struggle.")
    
    print("[INFO] ========== Alignment Verification Complete ==========\n")
    
    return {
        'overlap_matrix': overlap_matrix,
        'avg_overlap': avg_overlap,
    }


def evaluate_depth_quality(depths: np.ndarray) -> np.ndarray:
    """
    Evaluate depth quality for each camera based on multiple metrics.
    
    Args:
        depths: (C, T, H, W) array of depth values
    
    Returns:
        (C,) array of quality scores (higher is better)
    """
    C, T, H, W = depths.shape
    scores = np.zeros(C)
    
    for c_idx in range(C):
        cam_depths = depths[c_idx]  # (T, H, W)
        
        # Metric 1: Coverage (% of valid depth values)
        valid_mask = cam_depths > 0
        coverage = valid_mask.sum() / (T * H * W)
        
        # Metric 2: Depth variance (higher = more 3D structure)
        # Use valid depths only
        valid_depths = cam_depths[valid_mask]
        if len(valid_depths) > 0:
            depth_variance = np.std(valid_depths)
        else:
            depth_variance = 0.0
        
        # Metric 3: Edge sharpness (gradient magnitude)
        # Compute depth gradients for each frame
        edge_scores = []
        for t in range(T):
            d = cam_depths[t]
            valid = d > 0
            if valid.sum() > 100:  # Need enough valid pixels
                # Compute gradients
                dy, dx = np.gradient(d)
                grad_mag = np.sqrt(dx**2 + dy**2)
                # Only consider gradients at valid depth locations
                valid_grads = grad_mag[valid]
                edge_scores.append(np.mean(valid_grads))
        
        edge_sharpness = np.mean(edge_scores) if edge_scores else 0.0
        
        # Combine metrics (normalized to 0-1 range approximately)
        # Higher coverage, variance, and sharpness = better depth
        score = (
            coverage * 100 +  # 0-100 range
            depth_variance * 10 +  # Scale to similar range
            edge_sharpness * 10
        )
        scores[c_idx] = score
    
    return scores


def evaluate_feature_richness(rgbs: np.ndarray) -> np.ndarray:
    """
    Evaluate feature richness for each camera (independent of COLMAP reconstruction).
    
    Args:
        rgbs: (C, T, H, W, 3) array of RGB images
    
    Returns:
        (C,) array of feature richness scores (higher is better)
    """
    C, T, H, W, _ = rgbs.shape
    scores = np.zeros(C)
    
    for c_idx in range(C):
        cam_rgbs = rgbs[c_idx]  # (T, H, W, 3)
        
        # Convert to grayscale and compute statistics
        texture_scores = []
        for t in range(min(T, 10)):  # Sample up to 10 frames
            rgb = cam_rgbs[t]
            gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
            
            # Metric 1: Intensity variance (texture)
            intensity_var = np.var(gray)
            
            # Metric 2: Edge strength (Sobel)
            dy, dx = np.gradient(gray)
            edge_strength = np.mean(np.sqrt(dx**2 + dy**2))
            
            # Metric 3: Local variance (fine texture)
            # Compute variance in 8x8 blocks
            block_vars = []
            for i in range(0, H - 8, 8):
                for j in range(0, W - 8, 8):
                    block = gray[i:i+8, j:j+8]
                    block_vars.append(np.var(block))
            local_var = np.mean(block_vars) if block_vars else 0.0
            
            # Combine metrics
            texture_score = intensity_var + edge_strength * 10 + local_var
            texture_scores.append(texture_score)
        
        scores[c_idx] = np.mean(texture_scores) if texture_scores else 0.0
    
    return scores


def evaluate_camera_quality_from_colmap(workspace_dir: Path, camera_ids: List[str]) -> Dict[str, float]:
    """
    Evaluate camera quality based on COLMAP reconstruction.
    
    Returns a dictionary mapping camera_id to quality score (higher is better).
    Quality is based on number of registered 3D points visible in each camera.
    """
    sparse_dir = workspace_dir / "sparse" / "0"
    if not sparse_dir.exists():
        print("[WARN] No COLMAP reconstruction found, cannot score cameras by reconstruction")
        return {cid: 0.0 for cid in camera_ids}
    
    # Parse images.txt to get registered images and their points
    images_file = sparse_dir / "images.txt"
    if not images_file.exists():
        print("[WARN] No images.txt found in COLMAP output")
        return {cid: 0.0 for cid in camera_ids}
    
    camera_scores = {cid: 0.0 for cid in camera_ids}
    camera_registered_count = {cid: 0 for cid in camera_ids}
    
    with open(images_file, 'r') as f:
        lines = f.readlines()
    
    # Skip header lines
    lines = [l for l in lines if not l.startswith('#')]
    
    # Parse image entries (two lines per image)
    for i in range(0, len(lines), 2):
        if i + 1 >= len(lines):
            break
        
        # First line: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        img_line = lines[i].strip().split()
        if len(img_line) < 10:
            continue
        
        img_name = img_line[9]
        
        # Second line: POINTS2D as (X, Y, POINT3D_ID) triplets
        points_line = lines[i + 1].strip()
        if not points_line:
            continue
        
        points = points_line.split()
        # Count valid 3D points (POINT3D_ID != -1)
        num_points = 0
        for j in range(2, len(points), 3):  # Every third element is POINT3D_ID
            if j < len(points):
                point3d_id = int(points[j])
                if point3d_id != -1:
                    num_points += 1
        
        # Extract camera_id from image name
        for cid in camera_ids:
            if f"cam_{cid}_" in img_name:
                camera_scores[cid] += num_points
                camera_registered_count[cid] += 1
                break
    
    print(f"[INFO] COLMAP camera scores (3D points observed):")
    for cid in camera_ids:
        reg_count = camera_registered_count[cid]
        print(f"[INFO]   Camera {cid}: {camera_scores[cid]:.0f} points in {reg_count} registered images")
    
    return camera_scores


def select_best_cameras(
    camera_ids: List[str],
    camera_scores: Dict[str, float],
    depth_scores: np.ndarray,
    feature_scores: np.ndarray,
    limit: int,
) -> List[int]:
    """
    Select indices of best N cameras based on combined quality scores.
    
    Args:
        camera_ids: List of camera IDs
        camera_scores: COLMAP reconstruction scores (0 if reconstruction failed)
        depth_scores: Depth quality scores
        feature_scores: Feature richness scores
        limit: Number of cameras to keep
    
    Returns:
        List of camera indices to keep (sorted by combined score, descending)
    """
    # Normalize scores to 0-1 range
    def normalize(arr):
        arr = np.array(arr)
        if arr.max() > 0:
            return arr / arr.max()
        return arr
    
    colmap_scores_arr = np.array([camera_scores.get(cid, 0.0) for cid in camera_ids])
    colmap_scores_norm = normalize(colmap_scores_arr)
    depth_scores_norm = normalize(depth_scores)
    feature_scores_norm = normalize(feature_scores)
    
    # Combine scores with weights
    # If COLMAP reconstruction succeeded (max score > 0), use it
    # Otherwise, rely entirely on depth and feature quality
    if colmap_scores_arr.max() > 0:
        print("[INFO] Using combined scoring: 40% COLMAP + 30% depth quality + 30% features")
        combined_scores = (
            0.4 * colmap_scores_norm +
            0.3 * depth_scores_norm +
            0.3 * feature_scores_norm
        )
    else:
        print("[INFO] COLMAP reconstruction failed, using: 60% depth quality + 40% features")
        combined_scores = (
            0.6 * depth_scores_norm +
            0.4 * feature_scores_norm
        )
    
    # Sort cameras by combined score
    scored_cameras = [(idx, cid, combined_scores[idx]) 
                      for idx, cid in enumerate(camera_ids)]
    scored_cameras.sort(key=lambda x: x[2], reverse=True)
    
    # Select top N
    selected = scored_cameras[:limit]
    selected_indices = sorted([idx for idx, _, _ in selected])
    
    print(f"[INFO] Selected {len(selected_indices)} best cameras:")
    for idx, cid, combined_score in selected:
        if idx in selected_indices:
            print(f"[INFO]   Camera {cid}: combined={combined_score:.3f} "
                  f"(COLMAP={colmap_scores_norm[idx]:.3f}, depth={depth_scores_norm[idx]:.3f}, features={feature_scores_norm[idx]:.3f})")
    
    return selected_indices



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
