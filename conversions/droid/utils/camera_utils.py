"""Camera utilities for DROID point cloud generation."""

import numpy as np
import json
import os
import glob
import pyzed.sl as sl


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


def get_filtered_cloud(zed, runtime, max_depth=2.0, min_depth=0.1):
    """
    Get filtered point cloud and RGB data from ZED camera.
    
    Filters out points beyond max_depth or behind camera (z <= min_depth).
    
    Args:
        zed: ZED Camera object
        runtime: RuntimeParameters for ZED camera
        max_depth: Maximum depth threshold in meters (default: 2.0)
        min_depth: Minimum depth threshold in meters (default: 0.1)
        
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

    # 3. Filter Depth (Z-axis in Camera Frame)
    # ZED Camera Frame: Z is forward. 
    # We want 0 < z < max_depth
    z_vals = xyz[:, 2]
    valid_mask = np.isfinite(xyz).all(axis=1) & (z_vals > min_depth) & (z_vals < max_depth)
    
    return xyz[valid_mask], rgb[valid_mask]
