"""
COLMAP Utilities for Camera Calibration and Reconstruction

This module provides functions for using COLMAP to refine camera poses and
evaluate multi-view reconstruction quality. Includes:
- Setting up COLMAP workspaces
- Running COLMAP feature extraction, matching, and mapping
- Evaluating camera quality based on reconstruction
- Selecting best cameras based on combined metrics
- Multi-view point cloud outlier rejection

Functions:
    setup_colmap_workspace: Set up COLMAP workspace with images and camera parameters
    run_colmap_feature_extraction: Run COLMAP feature extraction using pycolmap
    run_colmap_matching: Run COLMAP feature matching using pycolmap
    run_colmap_mapper: Run COLMAP mapper to reconstruct scene using pycolmap
    evaluate_camera_quality_from_colmap: Evaluate camera quality based on COLMAP reconstruction
    select_best_cameras: Select indices of best N cameras based on combined quality scores
    reject_outlier_points_multiview: Reject outlier 3D points using multi-view consistency
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import pycolmap

from utils.geometry_utils import _rotation_matrix_to_quaternion


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


def run_colmap_matching(workspace_dir: Path, database_path: Path) -> Tuple[bool, int]:
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


def run_colmap_mapper(workspace_dir: Path, database_path: Path, images_dir: Path, output_dir: Path, use_known_poses: bool = True) -> Tuple[bool, dict]:
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
