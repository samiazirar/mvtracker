#!/usr/bin/env python3
"""
Camera calibration and projection utilities.

This module provides functions for working with camera parameters including:
- scale_intrinsics_matrix: Scale intrinsic matrix for different image resolutions
- infer_calibration_resolution: Get calibration image resolution from scene
- reproject_to_sparse_depth_cv2: Reproject point cloud to sparse depth map
- _project_bbox_pixels: Project 3D bbox corners to 2D image pixels
- verify_camera_alignment: Check multi-view consistency for camera alignment
- evaluate_depth_quality: Score camera depth quality based on coverage/variance
- evaluate_feature_richness: Score camera feature richness (texture/edges)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
from PIL import Image


def scale_intrinsics_matrix(raw_K: np.ndarray, width: int, height: int, base_width: int, base_height: int) -> np.ndarray:
    """Rescales a 3x4 intrinsic matrix to match an image resolution."""
    if base_width <= 0 or base_height <= 0:
        raise ValueError("Base calibration resolution must be positive.")

    K = raw_K[:, :3].astype(np.float32, copy=True)
    scale_x = width / float(base_width)
    scale_y = height / float(base_height)
    print(f"[INFO] Scaling intrinsics by factors (x: {scale_x:.4f}, y: {scale_y:.4f}) for image size ({width}x{height})")
    K[0, 0] *= scale_x
    K[0, 2] *= scale_x
    K[1, 1] *= scale_y
    K[1, 2] *= scale_y
    return K


def infer_calibration_resolution(scene: Any, camera_id: str) -> Optional[Tuple[int, int]]:
    """Reads the calibration image resolution for a given camera if available."""
    try:
        calib_dir = Path(scene.calib_folder) / "imgs"
    except AttributeError:
        return None

    calib_path = calib_dir / f"cam_{camera_id}_c.png"
    if not calib_path.exists():
        return None

    with Image.open(calib_path) as img:
        width, height = img.size
    return width, height


def reproject_to_sparse_depth_cv2(
    pcd: o3d.geometry.PointCloud, 
    high_res_rgb: np.ndarray, 
    K: np.ndarray, 
    E: np.ndarray, 
    color_threshold: float,
) -> np.ndarray:
    """
    Projects an Open3D point cloud to a sparse depth map using OpenCV.

    Includes color alignment check:
    - If color diff > color_threshold: remove point
    - Otherwise: keep point with original color

    Args:
        pcd: The Open3D PointCloud in world coordinates.
        high_res_rgb: The target high-resolution color image.
        K: The 3x3 intrinsic matrix of the target camera.
        E: The 3x4 extrinsic matrix of the target camera (world-to-camera).
        color_threshold: Maximum color difference to keep a point.

    Returns:
        A sparse depth map of the same resolution as the high_res_rgb.
    """
    H, W, _ = high_res_rgb.shape
    sparse_depth = np.zeros((H, W), dtype=np.float32)
    
    if not pcd.has_points():
        return sparse_depth

    pts_world = np.asarray(pcd.points)
    
    # Handle point clouds with or without colors
    if pcd.has_colors():
        orig_colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    else:
        # If no colors, create a default gray color for all points
        num_points = pts_world.shape[0]
        orig_colors = np.full((num_points, 3), 128, dtype=np.uint8)  # Gray color

    # Decompose the world-to-camera matrix so OpenCV can consume its Rodrigues representation.
    R, t = E[:3, :3], E[:3, 3]
    rvec, _ = cv2.Rodrigues(R)
    tvec = t.reshape(3, 1)

    # Project all 3D points into the 2D image plane of the target camera
    projected_pts, _ = cv2.projectPoints(pts_world, rvec, tvec, K, distCoeffs=None)
    projected_pts = projected_pts.squeeze(1)
    
    # Calculate the depth of each point relative to the new camera
    # Re-use the same transform to express points in the camera frame for depth/Z buffering.
    pts_cam = (R @ pts_world.T + tvec).T
    depths = pts_cam[:, 2]

    # --- Filtering Stage ---
    # 1. Filter points behind the camera
    in_front_mask = depths > 1e-6
    # 2. Filter points outside the image boundaries
    u, v = projected_pts[in_front_mask, 0], projected_pts[in_front_mask, 1]
    bounds_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    
    # Apply combined filters
    valid_mask = np.where(in_front_mask)[0][bounds_mask]
    u_idx = np.round(u[bounds_mask]).astype(int)
    v_idx = np.round(v[bounds_mask]).astype(int)
    np.clip(u_idx, 0, W - 1, out=u_idx)
    np.clip(v_idx, 0, H - 1, out=v_idx)
    depth_final = depths[valid_mask]
    orig_colors_final = orig_colors[valid_mask]

    # 3. Color Alignment Check
    target_colors = high_res_rgb[v_idx, u_idx]
    color_diff = np.mean(np.abs(orig_colors_final.astype(float) - target_colors.astype(float)), axis=1)
    
    # Filter out points with color difference above threshold
    keep_mask = color_diff < color_threshold
    u_final = u_idx[keep_mask]
    v_final = v_idx[keep_mask]
    depth_final = depth_final[keep_mask]
    
    # 4. Z-Buffering: Handle occlusions by keeping only the closest point for each pixel
    # Sort points by depth in reverse order, so closer points are processed last and overwrite farther ones.
    sorted_indices = np.argsort(depth_final)[::-1]
    sparse_depth[v_final[sorted_indices], u_final[sorted_indices]] = depth_final[sorted_indices]
    
    return sparse_depth


def _project_bbox_pixels(corners_world: np.ndarray, intr: np.ndarray, extr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
