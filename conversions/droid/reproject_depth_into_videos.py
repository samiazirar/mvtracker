"""Reproject fused 3D point clouds back into each camera view and save videos.

This script:
1. Loads all cameras (external + wrist) and their poses
2. For each frame, fuses point clouds from ALL cameras into world frame
3. Reprojects the fused world-space points back into each camera view
4. Saves videos showing the reprojected fused point cloud for each camera
"""

import argparse
import numpy as np
import os
import glob
import h5py
import yaml
import cv2
from scipy.spatial.transform import Rotation as R
import pyzed.sl as sl
raise NotImplementedError("Please see the updated code in the latest version. This code has weird noise for some reason.")
from utils import (
    pose6_to_T,
    rvec_tvec_to_matrix,
    transform_points,
    invert_transform,
    compute_wrist_cam_offset,
    precompute_wrist_trajectory,
    external_cam_to_world,
    find_svo_for_camera,
    find_episode_data_by_date,
    get_zed_intrinsics,
    get_filtered_cloud,
    VideoRecorder,
    project_points_with_depth,
    draw_points_on_image,
)


def render_dense_pointcloud(
    width: int,
    height: int,
    points_world: np.ndarray,
    colors: np.ndarray,
    K: np.ndarray,
    world_T_cam: np.ndarray,
    min_depth: float = 0.01,
) -> np.ndarray:
    """
    Render a dense image from a point cloud using proper z-buffering.
    
    This creates a clean rendered image by:
    1. Projecting all points to image coordinates
    2. Using a depth buffer to keep only the closest point per pixel
    3. Producing a clean RGB image without gaps between points
    
    Args:
        width: Output image width
        height: Output image height
        points_world: Nx3 array of 3D points in world frame
        colors: Nx3 array of RGB colors (0-255)
        K: 3x3 camera intrinsic matrix
        world_T_cam: 4x4 camera pose (camera to world transform)
        min_depth: Minimum valid depth
        
    Returns:
        BGR image (H, W, 3) with rendered point cloud
    """
    if points_world.shape[0] == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    # Transform points from world to camera frame
    cam_T_world = invert_transform(world_T_cam)
    points_cam = transform_points(points_world, cam_T_world)
    
    # Filter points behind camera
    z = points_cam[:, 2]
    valid_mask = z > min_depth
    points_cam = points_cam[valid_mask]
    colors_valid = colors[valid_mask]
    
    if len(points_cam) == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    # Project to 2D
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    z = points_cam[:, 2]
    u = (points_cam[:, 0] * fx / z + cx).astype(np.int32)
    v = (points_cam[:, 1] * fy / z + cy).astype(np.int32)
    
    # Filter to valid image bounds
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[valid]
    v = v[valid]
    z = z[valid]
    colors_valid = colors_valid[valid]
    
    if len(u) == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create depth buffer and color buffer
    depth_buffer = np.full((height, width), np.inf, dtype=np.float32)
    color_buffer = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Sort by depth (farthest first) so closer points overwrite
    order = np.argsort(-z)
    u = u[order]
    v = v[order]
    z = z[order]
    colors_valid = colors_valid[order]
    
    # Vectorized z-buffer update
    # For each point, only update if it's closer than current depth
    for i in range(len(u)):
        if z[i] < depth_buffer[v[i], u[i]]:
            depth_buffer[v[i], u[i]] = z[i]
            # RGB to BGR for OpenCV
            color_buffer[v[i], u[i]] = colors_valid[i][::-1]
    
    return color_buffer


def render_dense_pointcloud_fast(
    width: int,
    height: int,
    points_world: np.ndarray,
    colors: np.ndarray,
    K: np.ndarray,
    world_T_cam: np.ndarray,
    min_depth: float = 0.01,
    splat_size: int = 3,
) -> np.ndarray:
    """
    Fast dense point cloud rendering with splatting to fill gaps.
    
    Uses numpy advanced indexing for speed, then dilates to fill small gaps.
    
    Args:
        width: Output image width
        height: Output image height  
        points_world: Nx3 array of 3D points in world frame
        colors: Nx3 array of RGB colors (0-255)
        K: 3x3 camera intrinsic matrix
        world_T_cam: 4x4 camera pose (camera to world transform)
        min_depth: Minimum valid depth
        splat_size: Size of kernel for filling gaps (must be odd)
        
    Returns:
        BGR image (H, W, 3) with rendered point cloud
    """
    if points_world.shape[0] == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    # Transform points from world to camera frame
    cam_T_world = invert_transform(world_T_cam)
    points_cam = transform_points(points_world, cam_T_world)
    
    # Filter points behind camera
    z = points_cam[:, 2]
    valid_mask = z > min_depth
    points_cam = points_cam[valid_mask]
    colors_valid = colors[valid_mask].copy()
    
    if len(points_cam) == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    # Project to 2D
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    z = points_cam[:, 2]
    u = (points_cam[:, 0] * fx / z + cx).astype(np.int32)
    v = (points_cam[:, 1] * fy / z + cy).astype(np.int32)
    
    # Filter to valid image bounds
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[valid]
    v = v[valid]
    z = z[valid]
    colors_valid = colors_valid[valid]
    
    if len(u) == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    # Sort by depth (closest first for proper occlusion)
    order = np.argsort(z)
    u = u[order]
    v = v[order]
    z = z[order]
    colors_valid = colors_valid[order]
    
    # Create buffers
    depth_buffer = np.full((height, width), np.inf, dtype=np.float32)
    color_buffer = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Use linear indices for faster numpy operations
    linear_idx = v * width + u
    
    # Process points - closest points will be kept due to sorting
    # We use a unique operation to keep only the first (closest) occurrence
    _, unique_indices = np.unique(linear_idx, return_index=True)
    
    u_unique = u[unique_indices]
    v_unique = v[unique_indices]
    colors_unique = colors_valid[unique_indices]
    
    # RGB to BGR for OpenCV
    color_buffer[v_unique, u_unique] = colors_unique[:, ::-1]
    
    # Optional: dilate to fill small gaps
    if splat_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (splat_size, splat_size))
        # Create mask of valid pixels
        mask = (color_buffer.sum(axis=2) > 0).astype(np.uint8)
        # Dilate each channel
        for c in range(3):
            color_buffer[:, :, c] = cv2.dilate(color_buffer[:, :, c], kernel)
    
    return color_buffer


def render_pointcloud_on_image(
    base_image: np.ndarray,
    points_world: np.ndarray,
    colors: np.ndarray,
    K: np.ndarray,
    world_T_cam: np.ndarray,
    min_depth: float = 0.01,
    splat_size: int = 1,
) -> np.ndarray:
    """
    Render point cloud on top of base image, filling gaps with original pixels.
    
    This gives a smooth result by:
    1. Starting with the original camera image
    2. Overlaying the reprojected point cloud
    3. Using the original image to fill any gaps
    
    Args:
        base_image: BGR image from the camera
        points_world: Nx3 array of 3D points in world frame
        colors: Nx3 array of RGB colors (0-255)
        K: 3x3 camera intrinsic matrix
        world_T_cam: 4x4 camera pose
        min_depth: Minimum valid depth
        splat_size: Point size for rendering
        
    Returns:
        BGR image with point cloud overlaid
    """
    height, width = base_image.shape[:2]
    
    if points_world.shape[0] == 0:
        return base_image.copy()
    
    # Transform points from world to camera frame
    cam_T_world = invert_transform(world_T_cam)
    points_cam = transform_points(points_world, cam_T_world)
    
    # Filter points behind camera
    z = points_cam[:, 2]
    valid_mask = z > min_depth
    points_cam = points_cam[valid_mask]
    colors_valid = colors[valid_mask].copy()
    
    if len(points_cam) == 0:
        return base_image.copy()
    
    # Project to 2D
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    z = points_cam[:, 2]
    u = (points_cam[:, 0] * fx / z + cx).astype(np.int32)
    v = (points_cam[:, 1] * fy / z + cy).astype(np.int32)
    
    # Filter to valid image bounds
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[valid]
    v = v[valid]
    z = z[valid]
    colors_valid = colors_valid[valid]
    
    if len(u) == 0:
        return base_image.copy()
    
    # Sort by depth (closest first)
    order = np.argsort(z)
    u = u[order]
    v = v[order]
    colors_valid = colors_valid[order]
    
    # Start with the base image
    result = base_image.copy()
    
    # Use linear indices for unique pixel selection
    linear_idx = v * width + u
    _, unique_indices = np.unique(linear_idx, return_index=True)
    
    u_unique = u[unique_indices]
    v_unique = v[unique_indices]
    colors_unique = colors_valid[unique_indices]
    
    # Draw points on image (RGB to BGR)
    if splat_size <= 1:
        result[v_unique, u_unique] = colors_unique[:, ::-1]
    else:
        # Draw circles for larger splat
        for i in range(len(u_unique)):
            color_bgr = tuple(int(c) for c in colors_unique[i][::-1])
            cv2.circle(result, (u_unique[i], v_unique[i]), splat_size, color_bgr, -1)
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Reproject fused 3D points into camera views.")
    parser.add_argument(
        "--config",
        default="conversions/droid/config.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        CONFIG = yaml.safe_load(f)
    
    fps = CONFIG.get('fps', 30.0)
    
    print("=== Reproject Fused 3D Points into Camera Views ===")

    # --- 1. Load Robot Data ---
    print("[INFO] Loading H5 Trajectory...")
    
    h5_file = h5py.File(CONFIG['h5_path'], 'r')
    cartesian_positions = h5_file['observation/robot_state/cartesian_position'][:]
    gripper_positions = h5_file['observation/robot_state/gripper_position'][:]
    h5_file.close()
    
    num_frames = len(cartesian_positions)
    max_frames = CONFIG.get("max_frames", num_frames)
    actual_frames = min(max_frames, num_frames)

    # --- 2. Calculate Wrist Transforms ---
    wrist_cam_transforms = []
    wrist_serial = None
    T_ee_cam = None
    
    metadata_path = CONFIG.get('metadata_path')
    if metadata_path is None:
        episode_dir = os.path.dirname(CONFIG['h5_path'])
        metadata_files = glob.glob(os.path.join(episode_dir, "metadata_*.json"))
        if metadata_files:
            metadata_path = metadata_files[0]

    if metadata_path and os.path.exists(metadata_path):
        import json
        with open(metadata_path, 'r') as f:
            meta = json.load(f)
        wrist_serial = str(meta.get("wrist_cam_serial", ""))
        wrist_pose_t0 = meta.get("wrist_cam_extrinsics")

        if wrist_pose_t0:
            # Calculate constant offset
            T_ee_cam = compute_wrist_cam_offset(wrist_pose_t0, cartesian_positions[0])
            # Precompute all wrist camera poses
            wrist_cam_transforms = precompute_wrist_trajectory(cartesian_positions, T_ee_cam)
    
    # --- 3. Init Cameras ---
    cameras = {}
    
    # A. External Cameras
    ext_data = find_episode_data_by_date(CONFIG['h5_path'], CONFIG['extrinsics_json_path'])
    if ext_data:
        for cam_id, transform_list in ext_data.items():
            if not cam_id.isdigit():
                continue
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
                "transforms": wrist_cam_transforms,
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
            print(f"[INFO] Opened camera {serial} ({data['type']}): {data['w']}x{data['h']}")
        else:
            print(f"[ERROR] Failed to open {serial}")

    if len(active_cams) == 0:
        print("[ERROR] No cameras opened. Exiting.")
        return

    # --- 4. Setup Video Recorders ---
    config_tag = os.path.splitext(os.path.basename(args.config))[0]
    video_dir = os.path.join(
        CONFIG.get("video_output_path", "point_clouds/videos"),
        config_tag,
        "fused_reprojection"
    )
    os.makedirs(video_dir, exist_ok=True)
    
    recorders = {
        serial: VideoRecorder(
            video_dir, serial, "fused", cam["w"], cam["h"], fps=fps
        )
        for serial, cam in active_cams.items()
    }

    # --- 5. Main Render Loop ---
    # Reset all cameras to frame 0
    for cam in active_cams.values():
        cam['zed'].set_svo_position(0)

    print(f"[INFO] Processing {actual_frames} frames...")
    R_fix = R.from_euler('z', 90, degrees=True).as_matrix()

    for frame_idx in range(actual_frames):
        if frame_idx % 10 == 0:
            print(f"  Frame {frame_idx}/{actual_frames}")

        # --- 5a. Collect fused point cloud from ALL cameras ---
        all_xyz_world = []
        all_rgb = []
        cam_images = {}  # Store images for later reprojection
        cam_poses = {}   # Store camera poses for this frame

        for serial, cam in active_cams.items():
            zed = cam['zed']
            if zed.grab(cam['runtime']) != sl.ERROR_CODE.SUCCESS:
                continue

            # Get image for video
            mat_img = sl.Mat()
            zed.retrieve_image(mat_img, sl.VIEW.LEFT)
            img_bgra = mat_img.get_data()
            frame_bgr = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)
            cam_images[serial] = frame_bgr

            # Get camera pose for this frame
            if cam['type'] == "wrist":
                if frame_idx >= len(cam['transforms']):
                    continue
                T_world_cam = cam['transforms'][frame_idx]
            else:
                T_world_cam = cam['world_T_cam']
            
            cam_poses[serial] = T_world_cam

            # Get point cloud and transform to world
            if cam['type'] == "wrist":
                max_depth = CONFIG.get('wrist_max_depth', 0.75)
                min_depth = CONFIG.get('min_depth_wrist', 0.01)
            else:
                max_depth = CONFIG.get('ext_max_depth', 1.5)
                min_depth = CONFIG.get('min_depth', 0.1)

            xyz_local, rgb = get_filtered_cloud(zed, cam['runtime'], max_depth, min_depth)
            if xyz_local is None or len(xyz_local) == 0:
                continue

            # Transform points to world frame
            xyz_world = transform_points(xyz_local, T_world_cam)
            all_xyz_world.append(xyz_world)
            all_rgb.append(rgb)

        # Fuse all points
        if len(all_xyz_world) == 0:
            # No points this frame, write black frames
            for serial in recorders:
                if serial in cam_images:
                    recorders[serial].write_frame(cam_images[serial])
            continue

        fused_xyz = np.vstack(all_xyz_world)
        fused_rgb = np.vstack(all_rgb)

        # --- 5b. Reproject fused points into each camera ---
        for serial, cam in active_cams.items():
            if serial not in cam_images or serial not in cam_poses:
                continue

            T_world_cam = cam_poses[serial]
            K = cam['K']
            w, h = cam['w'], cam['h']
            base_image = cam_images[serial]

            # Render mode: 'blend' puts points on original image, 'dense' renders pure point cloud
            render_mode = CONFIG.get('render_mode', 'blend')
            min_depth_proj = CONFIG.get('min_depth', 0.01)
            splat_size = CONFIG.get('splat_size', 1)  # Point size
            
            if render_mode == 'dense':
                # Pure point cloud render (may have gaps)
                rendered = render_dense_pointcloud_fast(
                    w, h, fused_xyz, fused_rgb, K, T_world_cam,
                    min_depth=min_depth_proj, splat_size=splat_size
                )
            else:
                # Blend: overlay point cloud on original image (smooth result)
                rendered = render_pointcloud_on_image(
                    base_image, fused_xyz, fused_rgb, K, T_world_cam,
                    min_depth=min_depth_proj, splat_size=splat_size
                )

            recorders[serial].write_frame(rendered)

    # --- 6. Cleanup ---
    for c in active_cams.values():
        c['zed'].close()
    for rec in recorders.values():
        rec.close()

    print("[SUCCESS] Done.")
    print(f"[INFO] Videos saved to: {video_dir}")


if __name__ == "__main__":
    main()