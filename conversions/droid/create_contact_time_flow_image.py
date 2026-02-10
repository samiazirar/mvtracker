#!/usr/bin/env python3
"""Create a single t0 image with full flow overlay from contact_time onward.

This mirrors the camera setup and projections used in create_video_with_tracks.py,
but collapses all flow from contact_time..end onto the initial (t0) image.
"""

import argparse
import glob
import os
import subprocess
import sys
import numpy as np
import h5py
import yaml
import cv2
import pyzed.sl as sl
from scipy.spatial.transform import Rotation as R

from utils import (
    pose6_to_T,
    compute_wrist_cam_offset,
    precompute_wrist_trajectory,
    external_cam_to_world,
    find_svo_for_camera,
    find_episode_data_by_date,
    get_zed_intrinsics,
    ContactSurfaceTracker,
    project_points_to_image,
    VideoRecorder,
)


def _load_tracks_from_config(
    config,
    cartesian_positions,
    gripper_positions,
    actual_frames,
    num_points_per_finger_override=None,
):
    tracks_npz_path = config.get("tracks_npz_path")
    use_precomputed = tracks_npz_path is not None and os.path.exists(tracks_npz_path)

    if use_precomputed:
        loaded = np.load(tracks_npz_path)
        tracks_3d = loaded["tracks_3d"]
        actual_frames = min(actual_frames, tracks_3d.shape[0])
        num_contact_pts = int(loaded.get("num_points_per_finger", 0))
        if num_contact_pts == 0 and "contact_points_local" in loaded:
            num_contact_pts = len(loaded["contact_points_local"])
        if num_contact_pts == 0:
            num_contact_pts = tracks_3d.shape[1] // 2
        total_track_pts = tracks_3d.shape[1]
        if total_track_pts and num_contact_pts * 2 != total_track_pts:
            num_contact_pts = total_track_pts // 2
        if num_points_per_finger_override:
            num_contact_pts = min(num_contact_pts, int(num_points_per_finger_override))
            left_idx = np.linspace(0, (tracks_3d.shape[1] // 2) - 1, num_contact_pts, dtype=int)
            right_offset = tracks_3d.shape[1] // 2
            right_idx = right_offset + np.linspace(0, (tracks_3d.shape[1] // 2) - 1, num_contact_pts, dtype=int)
            keep_idx = np.concatenate([left_idx, right_idx])
            tracks_3d = tracks_3d[:, keep_idx, :]
            total_track_pts = tracks_3d.shape[1]
        print(f"[INFO] Loaded precomputed tracks from {tracks_npz_path} with {total_track_pts} points")
        return tracks_3d, actual_frames, num_contact_pts, total_track_pts

    num_track_points = config.get("num_track_points", 24)
    if num_points_per_finger_override:
        num_track_points = int(num_points_per_finger_override)
    contact_tracker = ContactSurfaceTracker(num_track_points=num_track_points)
    num_contact_pts = len(contact_tracker.contact_points_local) if contact_tracker.contact_points_local is not None else 0
    total_track_pts = num_contact_pts * 2
    if total_track_pts == 0:
        print("[ERROR] No contact points available (check robotiq mesh path).")
        return None, actual_frames, 0, 0

    tracks_3d = np.zeros((actual_frames, total_track_pts, 3), dtype=np.float32)
    R_fix = R.from_euler('z', 90, degrees=True).as_matrix()
    for i in range(actual_frames):
        T_base_ee = pose6_to_T(cartesian_positions[i])
        T_base_ee[:3, :3] = T_base_ee[:3, :3] @ R_fix
        pts_left, pts_right = contact_tracker.get_contact_points_world(
            T_base_ee, gripper_positions[i]
        )
        if pts_left is not None:
            tracks_3d[i, :num_contact_pts, :] = pts_left
            tracks_3d[i, num_contact_pts:, :] = pts_right

    return tracks_3d, actual_frames, num_contact_pts, total_track_pts


def main():
    parser = argparse.ArgumentParser(
        description="Create a t0 image with full flow overlay from contact_time onward.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python conversions/droid/create_contact_time_flow_image.py \\\n"
            "    --config conversions/droid/config.yaml \\\n"
            "    --contact_time 90 \\\n"
            "    --output_dir point_clouds/contact_time_images \\\n"
            "    --num_points_per_finger 50 \\\n"
            "    --point_radius 0.5 \\\n"
            "    --alpha_start 0.8 --alpha_end 0.0 \\\n"
            "    --trail_stride 1 --trail_thickness 2 \\\n"
            "    --save_t0 --save_video\n"
            "\n"
            "  # Copy/paste for another episode\n"
            "  python conversions/droid/create_contact_time_flow_image.py \\\n"
            "    --config /path/to/episode/config.yaml \\\n"
            "    --contact_time 110 \\\n"
            "    --output_dir /path/to/output_dir\n"
        ),
    )
    parser.add_argument("--config", default="conversions/droid/config.yaml",
                        help="Path to YAML config file.")
    parser.add_argument("--contact_time", type=int, required=True,
                        help="Contact time frame index (inclusive).")
    parser.add_argument("--output_dir", default="point_clouds/contact_time_images",
                        help="Output directory for rendered images.")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Maximum frames to use (optional).")
    parser.add_argument("--num_points_per_finger", type=int, default=None,
                        help="Override number of tracked points per finger (for sparser flow).")
    parser.add_argument("--point_radius", type=float, default=2.0,
                        help="Radius of points drawn for track visualization (values <1 draw single-pixel points).")
    parser.add_argument("--alpha_start", type=float, default=0.8,
                        help="Alpha for earliest flow points (0..1).")
    parser.add_argument("--alpha_end", type=float, default=0.0,
                        help="Alpha for latest flow points (0..1).")
    parser.add_argument("--trail_stride", type=int, default=1,
                        help="Stride for trail segments (1 = every frame).")
    parser.add_argument("--trail_thickness", type=int, default=2,
                        help="Line thickness for trail segments.")
    parser.add_argument("--save_t0", action="store_true",
                        help="Also save the original t0 frame.")
    parser.add_argument("--save_video", action="store_true",
                        help="Also save the full video with tracks (same as create_video_with_tracks).")
    parser.add_argument("--save_original_video", action="store_true",
                        help="Also save the original video without overlays.")

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print("=== DROID t0 Image with Full Flow ===")

    h5_file = h5py.File(config["h5_path"], "r")
    cartesian_positions = h5_file["observation/robot_state/cartesian_position"][:]
    gripper_positions = h5_file["observation/robot_state/gripper_position"][:]
    h5_file.close()

    num_frames = len(cartesian_positions)
    actual_frames = min(config.get("max_frames", num_frames), num_frames)
    if args.max_frames:
        actual_frames = min(actual_frames, args.max_frames)

    tracks_3d, actual_frames, num_contact_pts, total_track_pts = _load_tracks_from_config(
        config,
        cartesian_positions,
        gripper_positions,
        actual_frames,
        num_points_per_finger_override=args.num_points_per_finger,
    )
    if tracks_3d is None:
        return 1

    track_colors_rgb = np.zeros((total_track_pts, 3), dtype=np.uint8)
    if total_track_pts > 0:
        track_colors_rgb[:num_contact_pts, :] = [51, 127, 255]
        track_colors_rgb[num_contact_pts:, :] = [51, 255, 127]

    # Wrist camera transforms
    wrist_cam_transforms = []
    wrist_serial = None
    T_ee_cam = None
    metadata_path = config.get("metadata_path")
    if metadata_path is None:
        episode_dir = os.path.dirname(config["h5_path"])
        metadata_files = glob.glob(os.path.join(episode_dir, "metadata_*.json"))
        if metadata_files:
            metadata_path = metadata_files[0]

    if metadata_path and os.path.exists(metadata_path):
        import json
        with open(metadata_path, "r") as f:
            meta = json.load(f)
        wrist_serial = str(meta.get("wrist_cam_serial", ""))
        wrist_pose_t0 = meta.get("wrist_cam_extrinsics")
        if wrist_pose_t0:
            T_ee_cam = compute_wrist_cam_offset(wrist_pose_t0, cartesian_positions[0])
            wrist_cam_transforms = precompute_wrist_trajectory(cartesian_positions, T_ee_cam)

    # Camera setup (external + wrist)
    cameras = {}
    ext_data = find_episode_data_by_date(config["h5_path"], config["extrinsics_json_path"])
    if ext_data:
        for cam_id, transform_list in ext_data.items():
            if not cam_id.isdigit():
                continue
            svo = find_svo_for_camera(config["recordings_dir"], cam_id)
            if svo:
                cameras[cam_id] = {
                    "type": "external",
                    "svo": svo,
                    "world_T_cam": external_cam_to_world(transform_list),
                }

    if wrist_serial:
        svo = find_svo_for_camera(config["recordings_dir"], wrist_serial)
        if svo:
            print(f"[INFO] Found Wrist Camera SVO: {wrist_serial}")
            cameras[wrist_serial] = {
                "type": "wrist",
                "svo": svo,
                "transforms": wrist_cam_transforms,
                "T_ee_cam": T_ee_cam,
            }
        else:
            print(f"[WARN] Wrist SVO not found for serial {wrist_serial}")

    if not cameras:
        print("[ERROR] No cameras found.")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    # Clamp contact_time
    contact_time = args.contact_time
    if contact_time < 0 or contact_time >= actual_frames:
        print(f"[WARN] contact_time {contact_time} out of range (0..{actual_frames-1}), clamping")
        contact_time = max(0, min(contact_time, actual_frames - 1))

    tracks_window = tracks_3d[contact_time:actual_frames]

    for serial, cam in cameras.items():
        zed = sl.Camera()
        init = sl.InitParameters()
        init.set_from_svo_file(cam["svo"])
        init.svo_real_time_mode = False
        init.coordinate_units = sl.UNIT.METER
        init.depth_mode = sl.DEPTH_MODE.NEURAL

        if zed.open(init) != sl.ERROR_CODE.SUCCESS:
            print(f"[ERROR] Failed to open {serial}")
            continue

        runtime = sl.RuntimeParameters()
        K, w, h = get_zed_intrinsics(zed)

        # Grab t0 frame
        zed.set_svo_position(0)
        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            print(f"[ERROR] Failed to grab t0 frame for {serial}")
            zed.close()
            continue

        mat_img = sl.Mat()
        zed.retrieve_image(mat_img, sl.VIEW.LEFT)
        img_bgra = mat_img.get_data()
        frame0 = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)

        if cam["type"] == "wrist":
            if not cam["transforms"]:
                print(f"[ERROR] Wrist transforms missing for {serial}")
                zed.close()
                continue
            world_T_cam = cam["transforms"][0]
            min_depth = config.get("min_depth_wrist", 0.01)
        else:
            world_T_cam = cam["world_T_cam"]
            min_depth = config.get("min_depth", 0.01)

        if args.save_t0:
            t0_path = os.path.join(args.output_dir, f"{serial}_t0.png")
            cv2.imwrite(t0_path, frame0)

        draw_radius = int(args.point_radius)
        if args.point_radius < 1:
            draw_radius = 0
        total_steps = max(1, len(tracks_window) - 1)
        alpha_start = max(0.0, min(1.0, args.alpha_start))
        alpha_end = max(0.0, min(1.0, args.alpha_end))
        trail_stride = max(1, int(args.trail_stride))

        # Build a flow-only overlay with per-point/line alpha, then composite once over the base frame.
        overlay_color = np.zeros_like(frame0, dtype=np.float32)
        overlay_alpha = np.zeros((frame0.shape[0], frame0.shape[1]), dtype=np.float32)

        for idx, pts in enumerate(tracks_window):
            t = idx / total_steps
            # Later frames become more transparent
            alpha = alpha_start + (alpha_end - alpha_start) * t
            if alpha <= 0:
                continue
            uv, cols = project_points_to_image(
                pts, K, world_T_cam, w, h, colors=track_colors_rgb, min_depth=min_depth
            )
            if uv is None or len(uv) == 0:
                continue
            uv_int = uv.astype(np.int32)
            for i, (u, v) in enumerate(uv_int):
                if u < 0 or u >= w or v < 0 or v >= h:
                    continue
                color = cols[i] if cols is not None else np.array([0, 255, 0], dtype=np.uint8)
                color_bgr = np.array([color[2], color[1], color[0]], dtype=np.float32)
                if draw_radius <= 0:
                    if alpha > overlay_alpha[v, u]:
                        overlay_alpha[v, u] = alpha
                        overlay_color[v, u, :] = color_bgr
                else:
                    u0 = max(0, u - draw_radius)
                    v0 = max(0, v - draw_radius)
                    u1 = min(w, u + draw_radius + 1)
                    v1 = min(h, v + draw_radius + 1)
                    mask = np.zeros((v1 - v0, u1 - u0), dtype=np.float32)
                    cv2.circle(mask, (u - u0, v - v0), draw_radius, 1.0, -1)
                    region_alpha = overlay_alpha[v0:v1, u0:u1]
                    update = mask * alpha > region_alpha
                    if np.any(update):
                        region_alpha[update] = mask[update] * alpha
                        overlay_alpha[v0:v1, u0:u1] = region_alpha
                        region_color = overlay_color[v0:v1, u0:u1, :]
                        region_color[update] = color_bgr
                        overlay_color[v0:v1, u0:u1, :] = region_color

            # Draw trail segments with the same time-based alpha (stride-controlled)
            if idx % trail_stride == 0 and idx + trail_stride < len(tracks_window):
                next_pts = tracks_window[idx + trail_stride]
                uv_next, cols_next = project_points_to_image(
                    next_pts, K, world_T_cam, w, h, colors=track_colors_rgb, min_depth=min_depth
                )
                if uv_next is None or len(uv_next) == 0:
                    continue
                uv_next_int = uv_next.astype(np.int32)
                n = min(len(uv_int), len(uv_next_int))
                for i in range(n):
                    u1, v1 = uv_int[i]
                    u2, v2 = uv_next_int[i]
                    if (
                        u1 < 0 or u1 >= w or v1 < 0 or v1 >= h or
                        u2 < 0 or u2 >= w or v2 < 0 or v2 >= h
                    ):
                        continue
                    color = cols_next[i] if cols_next is not None else np.array([0, 255, 0], dtype=np.uint8)
                    color_bgr = np.array([color[2], color[1], color[0]], dtype=np.float32)
                    # Draw into a mask then apply alpha/color
                    u0 = max(0, min(u1, u2) - args.trail_thickness)
                    v0 = max(0, min(v1, v2) - args.trail_thickness)
                    u1m = min(w, max(u1, u2) + args.trail_thickness + 1)
                    v1m = min(h, max(v1, v2) + args.trail_thickness + 1)
                    mask = np.zeros((v1m - v0, u1m - u0), dtype=np.float32)
                    cv2.line(mask, (u1 - u0, v1 - v0), (u2 - u0, v2 - v0), 1.0, args.trail_thickness)
                    region_alpha = overlay_alpha[v0:v1m, u0:u1m]
                    update = mask * alpha > region_alpha
                    if np.any(update):
                        region_alpha[update] = mask[update] * alpha
                        overlay_alpha[v0:v1m, u0:u1m] = region_alpha
                        region_color = overlay_color[v0:v1m, u0:u1m, :]
                        region_color[update] = color_bgr
                        overlay_color[v0:v1m, u0:u1m, :] = region_color

        # Composite overlay onto base frame using per-pixel alpha
        alpha_map = np.clip(overlay_alpha, 0.0, 1.0)[..., None]
        frame0 = (overlay_color * alpha_map + frame0.astype(np.float32) * (1.0 - alpha_map)).astype(np.uint8)

        text = f"t0 with flow f({contact_time}-{actual_frames - 1})"
        cv2.putText(frame0, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame0, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        output_path = os.path.join(args.output_dir, f"{serial}_t0_flow_from_{contact_time}.png")
        cv2.imwrite(output_path, frame0)
        print(f"[INFO] Saved: {output_path}")

        if args.save_original_video:
            # Save original video (no overlays)
            out_dir = os.path.join(args.output_dir, "original_videos")
            recorder = VideoRecorder(out_dir, serial, "original", w, h, fps=config.get("fps", 30.0))
            zed.set_svo_position(0)
            for i in range(actual_frames):
                if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                    break
                mat = sl.Mat()
                zed.retrieve_image(mat, sl.VIEW.LEFT)
                img = mat.get_data()
                frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                recorder.write_frame(frame)
            recorder.close()

        zed.close()

    if args.save_video:
        # Run the canonical pipeline for full video generation once.
        cmd = [sys.executable, "conversions/droid/create_video_with_tracks.py", "--config", args.config]
        print(f"[INFO] Running full video pipeline: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    print("[DONE] Images written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
