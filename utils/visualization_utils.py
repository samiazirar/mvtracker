"""
Visualization Utilities for RH20T Data

This module provides visualization functions for RH20T dataset processing including:
- Exporting videos with bounding box overlays
- Saving and visualizing point clouds with Rerun
- Multi-camera gripper visualization
- Query point visualization

Functions:
    _export_gripper_bbox_videos: Export videos with gripper bounding boxes overlaid on RGB frames
    save_and_visualize: Save processed data to NPZ and generate Rerun visualization
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence
import numpy as np
import cv2
import torch
import rerun as rr

from mvtracker.utils.visualizer_rerun import log_pointclouds_to_rerun
from utils.geometry_utils import (
    _compute_bbox_corners_world,
    _ensure_basis,
)


def _project_bbox_pixels(corners_world: np.ndarray, intr: np.ndarray, extr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project 3D bounding box corners into 2D image pixels."""
    corners_world = np.asarray(corners_world, dtype=np.float32)
    intr = np.asarray(intr, dtype=np.float32)
    extr = np.asarray(extr, dtype=np.float32)
    num = corners_world.shape[0]
    pixels = np.zeros((num, 2), dtype=np.float32)
    valid = np.zeros((num,), dtype=bool)
    if num == 0:
        return pixels, valid

    corners_h = np.concatenate([corners_world, np.ones((num, 1), dtype=np.float32)], axis=1)
    # Extrinsics are world-to-camera; transform corners into the camera frame.
    cam = (extr @ corners_h.T).T
    z = cam[:, 2]
    valid = z > 1e-6
    if not np.any(valid):
        return pixels, valid
    cam_valid = cam[valid]
    proj = (intr @ cam_valid.T).T
    # Divide by depth to land in pixel coordinates using the pinhole model.
    proj_xy = proj[:, :2] / proj[:, 2:3]
    pixels[valid] = proj_xy
    return pixels, valid


def _export_gripper_bbox_videos(
    args,
    rgbs: np.ndarray,
    intrs: np.ndarray,
    extrs: np.ndarray,
    bboxes: Optional[List[Optional[Dict[str, np.ndarray]]]],
    camera_ids: Sequence[str],
) -> None:
    """Export videos with gripper bounding boxes overlaid on RGB frames."""
    if bboxes is None or len(bboxes) == 0 or all(b is None for b in bboxes):
        print("[INFO] Skipping bbox video export: no gripper boxes available.")
        return

    video_dir = Path(args.out_dir) / "bbox_videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    fps = getattr(args, "bbox_video_fps", 12.0)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    color = (0, 165, 255)

    rgbs = np.asarray(rgbs)
    intrs = np.asarray(intrs)
    extrs = np.asarray(extrs)

    num_cams, num_frames = rgbs.shape[0], rgbs.shape[1]

    for ci in range(num_cams):
        cam_id = str(camera_ids[ci]) if camera_ids is not None else str(ci)
        frame_shape = rgbs[ci, 0].shape
        if len(frame_shape) != 3:
            continue
        height, width = frame_shape[0], frame_shape[1]
        video_path = video_dir / f"{args.task_folder.name}_cam_{cam_id}_bbox.mp4"
        writer = cv2.VideoWriter(str(video_path), fourcc, float(fps), (width, height))
        if not writer.isOpened():
            print(f"[WARN] Could not open video writer for {video_path}.")
            continue

        for ti in range(num_frames):
            frame_rgb = rgbs[ci, ti]
            frame_bgr = np.ascontiguousarray(frame_rgb[:, :, ::-1])
            bbox = bboxes[ti]
            if bbox is not None:
                corners_world = _compute_bbox_corners_world(bbox)
                if corners_world is not None:
                    # Project the world-space OBB back into the current camera to draw a 2D outline.
                    pixels, valid = _project_bbox_pixels(corners_world, intrs[ci, ti], extrs[ci, ti])
                    for a, b in edges:
                        if valid[a] and valid[b]:
                            pt1 = tuple(np.round(pixels[a]).astype(int))
                            pt2 = tuple(np.round(pixels[b]).astype(int))
                            cv2.line(frame_bgr, pt1, pt2, color, 2)
            writer.write(frame_bgr)

        writer.release()
        print(f"[INFO] Wrote bbox overlay video: {video_path}")


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
