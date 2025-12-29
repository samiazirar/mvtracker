#!/usr/bin/env python3
"""
Stage 2: Classify sensor points using robot mesh and reproject to RGB.

Instead of projecting the URDF/mesh directly to 2D:
1. Get depth sensor points as 3D point cloud
2. Use 3D robot mesh to classify which points belong to the robot
3. Reproject classified sensor points to RGB image

This gives a more accurate mask because it uses actual sensor data.

Usage:
    python robot_mask_from_sensor_points.py \
        --h5_path /data/droid/data/droid_raw/1.0.1/IRIS/success/2023-05-12/Fri_May_12_10:51:00_2023/trajectory.h5 \
        --processed_dir /workspace/droid_processed/IRIS/success/2023-05-12/Fri_May_12_10:51:00_2023 \
        --output_dir ./sensor_robot_mask_output \
        --distance_threshold 0.03
"""

import argparse
import numpy as np
import os
import sys
import json
import h5py
import cv2
import trimesh
from typing import Dict, List, Tuple, Optional
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree
import glob

# Import from our other script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robot_mask_from_depth import (
    panda_forward_kinematics,
    robotiq_gripper_transforms,
    load_robot_meshes,
    load_gripper_meshes,
    sample_points_from_mesh,
    overlay_mask
)


# =============================================================================
# DEPTH LOADING
# =============================================================================

def load_depth_from_npz(npz_path: str, frame_idx: int) -> Optional[np.ndarray]:
    """Load depth from npz file if available."""
    try:
        data = np.load(npz_path, allow_pickle=True)
        if 'depths' in data:
            return data['depths'][frame_idx]
        return None
    except:
        return None


def try_load_depth(processed_dir: str, camera_serial: str, frame_idx: int) -> Optional[np.ndarray]:
    """Try to load depth from various sources."""
    # Try depth directory
    depth_dir = os.path.join(processed_dir, 'recordings', camera_serial, 'depth')
    if os.path.exists(depth_dir):
        depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith(('.png', '.npy'))])
        if frame_idx < len(depth_files):
            depth_path = os.path.join(depth_dir, depth_files[frame_idx])
            if depth_path.endswith('.npy'):
                return np.load(depth_path)
            else:
                # PNG depth (assume 16-bit, scale to meters)
                depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                if depth is not None:
                    return depth.astype(float) / 1000.0  # mm to m
    return None


# =============================================================================
# POINT CLOUD FROM DEPTH
# =============================================================================

def depth_to_pointcloud(
    depth: np.ndarray,
    K: np.ndarray,
    T_world_cam: np.ndarray,
    rgb: np.ndarray = None,
    max_depth: float = 3.0,
    downsample: int = 2
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Convert depth image to 3D point cloud in world frame.
    
    Args:
        depth: (H, W) depth image in meters
        K: 3x3 camera intrinsics
        T_world_cam: 4x4 camera-to-world transform (camera pose in world)
        rgb: Optional (H, W, 3) RGB image for colors
        max_depth: Maximum valid depth
        downsample: Downsample factor for speed
        
    Returns:
        points_world: (N, 3) points in world frame
        colors: (N, 3) RGB colors if rgb provided, else None
    """
    h, w = depth.shape[:2]
    
    # Downsample for speed
    if downsample > 1:
        depth = depth[::downsample, ::downsample]
        if rgb is not None:
            rgb = rgb[::downsample, ::downsample]
        h, w = depth.shape[:2]
        # Adjust intrinsics
        K = K.copy()
        K[0, 0] /= downsample
        K[1, 1] /= downsample
        K[0, 2] /= downsample
        K[1, 2] /= downsample
    
    # Create pixel coordinates
    u = np.arange(w)
    v = np.arange(h)
    u, v = np.meshgrid(u, v)
    
    # Unproject to camera frame
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    z = depth.flatten()
    x = ((u.flatten() - cx) * z / fx)
    y = ((v.flatten() - cy) * z / fy)
    
    # Stack and filter valid points
    points_cam = np.stack([x, y, z], axis=-1)
    valid = (z > 0.01) & (z < max_depth)
    points_cam = points_cam[valid]
    
    if len(points_cam) == 0:
        return np.array([]).reshape(0, 3), None
    
    # Transform to world frame
    points_hom = np.hstack([points_cam, np.ones((len(points_cam), 1))])
    points_world = (T_world_cam @ points_hom.T).T[:, :3]
    
    colors = None
    if rgb is not None:
        colors = rgb.reshape(-1, 3)[valid]
    
    return points_world, colors


# =============================================================================
# ROBOT MESH POINT SAMPLING
# =============================================================================

def get_robot_surface_points(
    joint_angles: np.ndarray,
    gripper_pos: float,
    arm_meshes: Dict[str, trimesh.Trimesh],
    gripper_meshes: Dict[str, trimesh.Trimesh],
    num_arm_points: int = 20000,
    num_gripper_points: int = 5000,
    T_world_base: np.ndarray = None
) -> np.ndarray:
    """
    Sample points from robot surface for collision detection.
    
    Args:
        joint_angles: 7-element joint angles
        gripper_pos: Gripper position (0-1)
        arm_meshes: Pre-loaded arm meshes
        gripper_meshes: Pre-loaded gripper meshes
        num_arm_points: Points to sample from arm
        num_gripper_points: Points to sample from gripper
        T_world_base: Robot base transform
        
    Returns:
        (N, 3) sampled points on robot surface in world frame
    """
    if T_world_base is None:
        T_world_base = np.eye(4)
    
    all_points = []
    
    # Get link transforms
    link_transforms = panda_forward_kinematics(joint_angles, T_world_base)
    
    # Sample from arm
    pts_per_link = num_arm_points // len(arm_meshes) if arm_meshes else 0
    for link_name, mesh in arm_meshes.items():
        if link_name in link_transforms:
            transformed = mesh.copy()
            transformed.apply_transform(link_transforms[link_name])
            if len(transformed.vertices) > 0:
                pts, _ = trimesh.sample.sample_surface(transformed, pts_per_link)
                all_points.append(pts)
    
    # Sample from gripper
    if gripper_meshes:
        T_world_ee = link_transforms.get("hand", np.eye(4))
        
        # Apply R_fix rotation to EE frame (90° Z rotation)
        # This matches the convention used in gripper_poses stored in tracks.npz
        from scipy.spatial.transform import Rotation as R
        R_fix = R.from_euler('z', 90, degrees=True).as_matrix()
        T_world_ee_rotated = T_world_ee.copy()
        T_world_ee_rotated[:3, :3] = T_world_ee[:3, :3] @ R_fix
        
        gripper_transforms = robotiq_gripper_transforms(T_world_ee_rotated, gripper_pos)
        
        mesh_to_component = {
            "base": ["base"],
            "outer_knuckle": ["left_outer_knuckle", "right_outer_knuckle"],
            "outer_finger": ["left_outer_finger", "right_outer_finger"],
            "inner_knuckle": ["left_inner_knuckle", "right_inner_knuckle"],
            "inner_finger": ["left_inner_finger", "right_inner_finger"],
        }
        
        total_gripper_meshes = sum(len(v) for v in mesh_to_component.values())
        pts_per_gripper = num_gripper_points // total_gripper_meshes if total_gripper_meshes > 0 else 0
        
        for mesh_name, mesh in gripper_meshes.items():
            if mesh_name in mesh_to_component:
                for comp_name in mesh_to_component[mesh_name]:
                    if comp_name in gripper_transforms:
                        transformed = mesh.copy()
                        transformed.apply_transform(gripper_transforms[comp_name])
                        if len(transformed.vertices) > 0:
                            pts, _ = trimesh.sample.sample_surface(transformed, pts_per_gripper)
                            all_points.append(pts)
    
    if all_points:
        return np.vstack(all_points)
    return np.array([]).reshape(0, 3)


# =============================================================================
# SENSOR POINT CLASSIFICATION
# =============================================================================

def classify_sensor_points(
    sensor_points: np.ndarray,
    robot_points: np.ndarray,
    distance_threshold: float = 0.03
) -> np.ndarray:
    """
    Classify which sensor points belong to the robot.
    
    Args:
        sensor_points: (N, 3) sensor points in world frame
        robot_points: (M, 3) robot mesh surface points in world frame
        distance_threshold: Max distance to classify as robot (meters)
        
    Returns:
        Boolean mask (N,) where True = robot point
    """
    if len(sensor_points) == 0 or len(robot_points) == 0:
        return np.zeros(len(sensor_points), dtype=bool)
    
    # Build KD-tree for robot points
    tree = KDTree(robot_points)
    
    # Find distance to nearest robot point
    distances, _ = tree.query(sensor_points, k=1)
    
    # Classify as robot if within threshold
    return distances < distance_threshold


def project_points_to_image(
    points_world: np.ndarray,
    K: np.ndarray,
    T_cam_world: np.ndarray,
    width: int,
    height: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points to 2D image.
    
    Args:
        points_world: (N, 3) points in world frame
        K: 3x3 camera intrinsics
        T_cam_world: 4x4 world-to-camera transform
        width, height: Image dimensions
        
    Returns:
        uv: (M, 2) valid pixel coordinates
        valid_indices: (M,) indices of valid points
    """
    if len(points_world) == 0:
        return np.array([]).reshape(0, 2), np.array([], dtype=int)
    
    # Transform to camera frame
    points_hom = np.hstack([points_world, np.ones((len(points_world), 1))])
    points_cam = (T_cam_world @ points_hom.T).T[:, :3]
    
    # Get depths
    depths = points_cam[:, 2]
    
    # Project to 2D
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        u = (fx * points_cam[:, 0] / depths) + cx
        v = (fy * points_cam[:, 1] / depths) + cy
    
    # Valid if in front of camera and within bounds
    valid = (depths > 0.01) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    valid_indices = np.where(valid)[0]
    
    uv = np.stack([u[valid], v[valid]], axis=1)
    
    return uv, valid_indices


def create_mask_from_classified_points(
    sensor_points: np.ndarray,
    is_robot: np.ndarray,
    K: np.ndarray,
    T_cam_world: np.ndarray,
    width: int,
    height: int,
    point_radius: int = 2
) -> np.ndarray:
    """
    Create mask by reprojecting classified sensor points.
    
    Args:
        sensor_points: (N, 3) sensor points in world frame
        is_robot: (N,) boolean mask of robot points
        K: Camera intrinsics
        T_cam_world: World-to-camera transform
        width, height: Image dimensions
        point_radius: Radius for each point in mask
        
    Returns:
        Binary mask (height, width) as uint8
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Get robot points only
    robot_sensor_pts = sensor_points[is_robot]
    
    if len(robot_sensor_pts) == 0:
        return mask
    
    # Project to image
    uv, _ = project_points_to_image(robot_sensor_pts, K, T_cam_world, width, height)
    
    # Draw points on mask
    for pt in uv.astype(int):
        cv2.circle(mask, (pt[0], pt[1]), point_radius, 255, -1)
    
    # Optional: fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)
    
    return mask


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_episode_with_depth(
    h5_path: str,
    processed_dir: str,
    output_dir: str,
    max_frames: int = 500,
    distance_threshold: float = 0.03,
    downsample: int = 4
):
    """
    Process episode using depth sensor data.
    
    Args:
        h5_path: Path to trajectory.h5 with joint_positions
        processed_dir: Path to processed episode
        output_dir: Output directory
        max_frames: Max frames to process
        distance_threshold: Distance threshold for robot classification (meters)
        downsample: Depth image downsample factor
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("SENSOR POINT ROBOT MASK GENERATION")
    print("=" * 60)
    print(f"H5 path: {h5_path}")
    print(f"Processed dir: {processed_dir}")
    print(f"Output: {output_dir}")
    print(f"Distance threshold: {distance_threshold}m")
    print()
    
    # 1. Load robot data
    print("[1/6] Loading robot state...")
    with h5py.File(h5_path, 'r') as f:
        joint_positions = f['observation/robot_state/joint_positions'][:]
        gripper_positions = f['observation/robot_state/gripper_position'][:]
    
    num_frames = min(len(joint_positions), max_frames)
    print(f"  Joint positions: {joint_positions.shape}")
    print(f"  Frames: {num_frames}")
    
    # 2. Load extrinsics
    print("\n[2/6] Loading camera extrinsics...")
    extr_path = os.path.join(processed_dir, 'extrinsics.npz')
    extr = np.load(extr_path, allow_pickle=True)
    
    cameras = {}
    recordings_dir = os.path.join(processed_dir, 'recordings')
    
    for key in extr.files:
        if key.startswith('external_'):
            serial = key.replace('external_', '')
            cam_dir = os.path.join(recordings_dir, serial)
            if os.path.exists(cam_dir):
                intr_path = os.path.join(cam_dir, 'intrinsics.json')
                if os.path.exists(intr_path):
                    with open(intr_path) as f:
                        intr = json.load(f)
                    cameras[serial] = {
                        'world_T_cam': extr[key],
                        'intrinsics': intr,
                        'rgb_dir': os.path.join(cam_dir, 'rgb'),
                        'depth_dir': os.path.join(cam_dir, 'depth')
                    }
                    print(f"  Camera: {serial}")
    
    # 3. Load meshes
    print("\n[3/6] Loading meshes...")
    arm_meshes = load_robot_meshes()
    gripper_meshes = load_gripper_meshes()
    print(f"  Arm links: {list(arm_meshes.keys())}")
    print(f"  Gripper parts: {list(gripper_meshes.keys())}")
    
    # 4. Check for depth availability
    print("\n[4/6] Checking depth availability...")
    has_depth = False
    for serial, cam_info in cameras.items():
        depth_dir = cam_info['depth_dir']
        if os.path.exists(depth_dir):
            depth_files = [f for f in os.listdir(depth_dir) if f.endswith(('.png', '.npy'))]
            if depth_files:
                has_depth = True
                print(f"  {serial}: {len(depth_files)} depth files")
            else:
                print(f"  {serial}: No depth files")
        else:
            print(f"  {serial}: No depth directory")
    
    if not has_depth:
        print("\n  [WARN] No depth data found. Falling back to direct mesh projection.")
        print("  Run robot_mask_from_depth.py instead for mesh-based masks.")
        return
    
    # 5. Process cameras
    print("\n[5/6] Processing cameras...")
    
    for serial, cam_info in cameras.items():
        depth_dir = cam_info['depth_dir']
        if not os.path.exists(depth_dir):
            continue
        
        depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith(('.png', '.npy'))])
        if not depth_files:
            continue
        
        print(f"\n  Camera: {serial}")
        
        # Get camera params
        world_T_cam = cam_info['world_T_cam']
        cam_T_world = np.linalg.inv(world_T_cam)
        intr = cam_info['intrinsics']
        
        fx = intr.get('focal_length_x', intr.get('fx', 640))
        fy = intr.get('focal_length_y', intr.get('fy', 640))
        cx = intr.get('cx', intr.get('principal_point_x', 640))
        cy = intr.get('cy', intr.get('principal_point_y', 360))
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        
        # Get image size
        rgb_dir = cam_info['rgb_dir']
        rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')])
        first_img = cv2.imread(os.path.join(rgb_dir, rgb_files[0]))
        height, width = first_img.shape[:2]
        
        # Setup output
        cam_output_dir = os.path.join(output_dir, serial)
        os.makedirs(cam_output_dir, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            os.path.join(cam_output_dir, 'sensor_robot_mask.mp4'),
            fourcc, 30, (width, height)
        )
        
        # Process frames
        for frame_idx in range(min(num_frames, len(rgb_files), len(depth_files))):
            if frame_idx % 10 == 0:
                print(f"    Frame {frame_idx}/{num_frames}")
            
            # Load RGB
            rgb = cv2.imread(os.path.join(rgb_dir, rgb_files[frame_idx]))
            
            # Load depth
            depth_path = os.path.join(depth_dir, depth_files[frame_idx])
            if depth_path.endswith('.npy'):
                depth = np.load(depth_path)
            else:
                depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                depth = depth.astype(float) / 1000.0  # mm to m
            
            # Get robot state
            joints = joint_positions[frame_idx][:7]
            gripper_pos = gripper_positions[frame_idx]
            
            # Get robot surface points
            robot_points = get_robot_surface_points(
                joints, gripper_pos, arm_meshes, gripper_meshes,
                num_arm_points=30000, num_gripper_points=10000
            )
            
            # Get sensor points from depth
            sensor_points, _ = depth_to_pointcloud(
                depth, K, world_T_cam, downsample=downsample
            )
            
            if len(sensor_points) == 0 or len(robot_points) == 0:
                mask = np.zeros((height, width), dtype=np.uint8)
            else:
                # Classify sensor points
                is_robot = classify_sensor_points(
                    sensor_points, robot_points, distance_threshold
                )
                
                # Create mask from classified points
                # Need to transform back with original K (not downsampled)
                K_full = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                mask = create_mask_from_classified_points(
                    sensor_points, is_robot, K_full, cam_T_world, width, height,
                    point_radius=max(2, downsample)
                )
            
            # Overlay
            overlay = overlay_mask(rgb, mask, (0, 255, 255), 0.5)  # Yellow
            
            # Add stats
            robot_pts = np.sum(is_robot) if 'is_robot' in dir() else 0
            cv2.putText(overlay, f"Robot pts: {robot_pts}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            writer.write(overlay)
        
        writer.release()
        print(f"    Saved to {cam_output_dir}/")
    
    print("\n[6/6] Done!")


def main():
    parser = argparse.ArgumentParser(description="Generate robot masks from sensor depth")
    parser.add_argument('--h5_path', required=True, help='Path to trajectory.h5')
    parser.add_argument('--processed_dir', required=True, help='Path to processed episode')
    parser.add_argument('--output_dir', default='./sensor_robot_mask_output', help='Output directory')
    parser.add_argument('--max_frames', type=int, default=500, help='Max frames')
    parser.add_argument('--distance_threshold', type=float, default=0.03,
                       help='Distance threshold for robot classification (meters)')
    parser.add_argument('--downsample', type=int, default=4, help='Depth downsample factor')
    
    args = parser.parse_args()
    
    process_episode_with_depth(
        h5_path=args.h5_path,
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        distance_threshold=args.distance_threshold,
        downsample=args.downsample
    )


if __name__ == "__main__":
    main()
