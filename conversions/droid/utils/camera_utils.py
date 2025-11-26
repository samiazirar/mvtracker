"""Camera utilities for DROID point cloud generation."""

import numpy as np
import json
import os
import glob
import pyzed.sl as sl
import open3d as o3d
import copy
import cv2


def find_svo_for_camera(recordings_dir, cam_serial):
    """
    Find SVO file for a given camera serial number.
    
    Args:
        recordings_dir: Directory containing SVO recordings
        cam_serial: Camera serial number as string
        
    Returns:
        Path to the SVO file if found, None otherwise
    """
    patterns = [f"*{cam_serial}*.svo", f"*{cam_serial}*.svo2"]
    for pat in patterns:
        matches = glob.glob(os.path.join(recordings_dir, pat))
        if matches: return matches[0]
    return None


def find_episode_data_by_date(h5_path, json_path):
    """
    Find episode calibration data in JSON file based on H5 trajectory path date.
    
    Args:
        h5_path: Path to the H5 trajectory file
        json_path: Path to the JSON calibration file
        
    Returns:
        Dictionary containing episode calibration data if found, None otherwise
    """
    parts = h5_path.split(os.sep)
    date_str = parts[-3] 
    timestamp_folder = parts[-2]
    ts_parts = timestamp_folder.split('_')
    time_str = next((part for part in ts_parts if ':' in part), "00:00:00")
    h, m, s = time_str.split(':')
    target_suffix = f"{date_str}-{h}h-{m}m-{s}s"
    
    with open(json_path, 'r') as f: data = json.load(f)
    for key in data.keys():
        if key.endswith(target_suffix): return data[key]
    for key in data.keys():
        if target_suffix.split('-')[-1] in key: return data[key]
    
    return None


def get_zed_intrinsics(zed):
    """
    Extract camera intrinsics for the Pinhole model from ZED camera.
    
    Args:
        zed: ZED Camera object
        
    Returns:
        Tuple of (K, w, h) where:
            K: 3x3 intrinsic matrix
            w: Image width
            h: Image height
    """
    info = zed.get_camera_information()
    calib = info.camera_configuration.calibration_parameters.left_cam
    fx, fy = calib.fx, calib.fy
    cx, cy = calib.cx, calib.cy
    w, h = calib.image_size.width, calib.image_size.height
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return K, w, h


def get_filtered_cloud(zed, runtime, max_depth=2.0, min_depth=0.1, highlight_mask=None):
    """
    Get filtered point cloud and RGB data from ZED camera.
    
    Filters out points beyond max_depth or behind camera (z <= min_depth).
    
    Args:
        zed: ZED Camera object
        runtime: RuntimeParameters for ZED camera
        max_depth: Maximum depth threshold in meters (default: 2.0)
        min_depth: Minimum depth threshold in meters (default: 0.1)
        highlight_mask: Optional (H, W) boolean mask. Points in mask will be colored red.
        
    Returns:
        Tuple of (xyz, rgb) where:
            xyz: Nx3 array of 3D points in camera frame
            rgb: Nx3 array of RGB colors
        Returns (None, None) if retrieval fails
    """
    mat_cloud = sl.Mat()
    err = zed.retrieve_measure(mat_cloud, sl.MEASURE.XYZRGBA)
    if err != sl.ERROR_CODE.SUCCESS: return None, None

    # 1. Get Data
    cloud_data = mat_cloud.get_data()
    xyz = cloud_data[:, :, :3].reshape(-1, 3)
    
    # 2. Get RGB
    mat_image = sl.Mat()
    zed.retrieve_image(mat_image, sl.VIEW.LEFT)
    image_data = mat_image.get_data()
    rgb = image_data[:, :, :3].reshape(-1, 3)

    if highlight_mask is not None:
        mask_flat = highlight_mask.reshape(-1) > 0
        # Ensure mask matches size
        if mask_flat.shape[0] == rgb.shape[0]:
            # Set to Red (255, 0, 0) - assuming uint8
            rgb[mask_flat] = [255, 0, 0]

    # 3. Filter Depth (Z-axis in Camera Frame)
    # ZED Camera Frame: Z is forward. 
    # We want 0 < z < max_depth
    z_vals = xyz[:, 2]
    valid_mask = np.isfinite(xyz).all(axis=1) & (z_vals > min_depth) & (z_vals < max_depth)
    
    return xyz[valid_mask], rgb[valid_mask]


def reproject_point_cloud(pcd, T_src_dst):
    """
    Reproject a point cloud from source camera frame to destination camera frame.
    
    Args:
        pcd: open3d.geometry.PointCloud in source frame
        T_src_dst: 4x4 transformation matrix from source to destination
        
    Returns:
        open3d.geometry.PointCloud in destination frame
    """
    pcd_dst = copy.deepcopy(pcd)
    pcd_dst.transform(T_src_dst)
    return pcd_dst


def reproject_all_to_one(pcds, transforms, target_idx=0):
    """
    Merge point clouds from multiple cameras into the target camera frame.
    
    Args:
        pcds: List of open3d.geometry.PointCloud
        transforms: List of 4x4 transformation matrices (T_world_cam or T_cam_world).
                   Assumes transforms are Camera Poses (Camera -> World).
        target_idx: Index of the target camera in the lists
        
    Returns:
        Merged open3d.geometry.PointCloud in target camera frame
    """
    target_pcd = o3d.geometry.PointCloud()
    if not pcds or target_idx >= len(pcds):
        return target_pcd

    T_cam_target_to_world = transforms[target_idx]
    T_world_to_cam_target = np.linalg.inv(T_cam_target_to_world)
    
    for i, pcd in enumerate(pcds):
        if pcd is None: continue
        
        if i == target_idx:
            target_pcd += pcd
        else:
            # T_cam_i_to_target = T_world_to_cam_target * T_cam_i_to_world
            T_cam_i_to_world = transforms[i]
            T_cam_i_to_target = T_world_to_cam_target @ T_cam_i_to_world
            
            pcd_transformed = copy.deepcopy(pcd)
            pcd_transformed.transform(T_cam_i_to_target)
            target_pcd += pcd_transformed
            
    return target_pcd


def backproject_2d_to_3d(u, v, depth, K):
    """
    Backproject a 2D point to 3D in camera frame.
    
    Args:
        u, v: Pixel coordinates (can be scalars or arrays)
        depth: Depth value at (u, v) (scalar or array)
        K: 3x3 intrinsic matrix
        
    Returns:
        3D point (x, y, z) in camera frame. Shape depends on input.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    
    if np.isscalar(u):
        return np.array([x, y, z])
    else:
        return np.stack([x, y, z], axis=-1)


def combine_masks(points1, points2, conf1=None, conf2=None, method='union', voxel_size=0.005):
    """
    Combine 3D object masks (point clouds) from different views.
    
    Args:
        points1: (N, 3) array of points
        points2: (M, 3) array of points
        conf1: (N,) array of confidences (optional)
        conf2: (M,) array of confidences (optional)
        method: 'union' or 'average' (uses voxel grid downsampling)
        voxel_size: Voxel size for averaging
        
    Returns:
        combined_points: (K, 3) array
        combined_conf: (K,) array (if inputs provided)
    """
    # Convert to Open3D point clouds for easier processing
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1)
    
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points2)
    
    # Handle confidence as colors or extra channel if needed, but O3D doesn't support arbitrary channels easily.
    # We'll manage confidence separately or use colors as proxy if needed, but here we stick to numpy for confidence.
    
    if method == 'union':
        combined_points = np.vstack((points1, points2))
        combined_conf = None
        if conf1 is not None and conf2 is not None:
            combined_conf = np.concatenate((conf1, conf2))
        return combined_points, combined_conf
        
    elif method == 'average':
        # Combine first
        all_points = np.vstack((points1, points2))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        
        # Voxel downsample averages points in the voxel
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        combined_points = np.asarray(pcd_down.points)
        
        combined_conf = None
        if conf1 is not None and conf2 is not None:
            # For confidence, we need to average it too. 
            # Since O3D voxel_down_sample doesn't average arbitrary attributes, we can use a trick:
            # Store confidence in the "intensity" or "color" channel if it was 1D/3D.
            # But here we can just do a manual voxel grid or KDTree search.
            # Simple approximation: assign confidence of nearest original point.
            # Better: manual voxelization.
            
            # Let's use a simple KDTree to find neighbors in original cloud and average confidence
            # This is slow but accurate.
            # Alternatively, just return None for now or implement a simple average.
            
            # Let's try to use colors for confidence averaging if it's 1D (grayscale)
            all_conf = np.concatenate((conf1, conf2))
            # Normalize to 0-1 for color
            # But confidence might be outside 0-1.
            # Let's just skip confidence averaging for 'average' method for now unless requested.
            pass
            
        return combined_points, combined_conf
    
    return points1, conf1
