"""
Example: Load and use converted BEHAVE .npz data for tracking.

This script demonstrates how to load the converted BEHAVE dataset
and prepare it for use with tracking algorithms.
"""

import argparse
from pathlib import Path

import numpy as np
import torch


def load_behave_scene(npz_path: Path):
    """
    Load a converted BEHAVE scene.
    
    Returns:
        dict with:
            - rgbs: (C, T, 3, H, W) RGB images
            - depths: (C, T, H, W) depth maps
            - masks: (C, T, H, W) masks
            - intrinsics: (C, 3, 3) camera intrinsics
            - extrinsics: (C, 3, 4) camera extrinsics
            - query_points: (T, C, N, 2) query points
            - metadata: dict with scene info
    """
    data = np.load(npz_path)
    
    return {
        'rgbs': data['rgbs'],
        'depths': data['depths'],
        'masks': data['masks'],
        'intrinsics': data['intrinsics'],
        'extrinsics': data['extrinsics'],
        'query_points': data['query_points'],
        'metadata': {
            'scene_name': str(data['scene_name']),
            'mask_type': str(data['mask_type']),
            'num_frames': int(data['num_frames']),
            'num_cameras': int(data['num_cameras']),
            'image_height': int(data['image_height']),
            'image_width': int(data['image_width']),
            'downscale_factor': int(data['downscale_factor']),
        }
    }


def prepare_for_single_camera_tracking(scene_data, camera_id=0):
    """
    Prepare data for single-camera tracking (e.g., TAP-Vid, CoTracker).
    
    Args:
        scene_data: dict from load_behave_scene()
        camera_id: camera to use (0-3)
    
    Returns:
        dict with:
            - rgbs: (T, 3, H, W) RGB frames
            - query_points: (N, 2) query points from first frame
            - intrinsics: (3, 3) camera intrinsics
    """
    rgbs = scene_data['rgbs'][camera_id]  # (T, 3, H, W)
    query_points = scene_data['query_points'][0, camera_id]  # (N, 2) from first frame
    intrinsics = scene_data['intrinsics'][camera_id]  # (3, 3)
    
    return {
        'rgbs': rgbs,
        'query_points': query_points,
        'intrinsics': intrinsics,
    }


def prepare_for_multi_view_tracking(scene_data):
    """
    Prepare data for multi-view tracking.
    
    Args:
        scene_data: dict from load_behave_scene()
    
    Returns:
        dict with:
            - rgbs: (C, T, 3, H, W) RGB frames
            - query_points: (T, C, N, 2) query points
            - intrinsics: (C, 3, 3) camera intrinsics
            - extrinsics: (C, 3, 4) camera extrinsics
            - depths: (C, T, H, W) depth maps
    """
    return {
        'rgbs': scene_data['rgbs'],
        'query_points': scene_data['query_points'],
        'intrinsics': scene_data['intrinsics'],
        'extrinsics': scene_data['extrinsics'],
        'depths': scene_data['depths'],
    }


def convert_to_torch(data_dict):
    """Convert numpy arrays to torch tensors."""
    return {
        k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
        for k, v in data_dict.items()
    }


def unproject_to_3d(points_2d, depth_map, intrinsics):
    """
    Unproject 2D points to 3D using depth.
    
    Args:
        points_2d: (N, 2) [x, y] coordinates
        depth_map: (H, W) depth map
        intrinsics: (3, 3) camera intrinsics
    
    Returns:
        points_3d: (N, 3) [X, Y, Z] in camera coordinates
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    points_3d = []
    for x, y in points_2d:
        x_int, y_int = int(x), int(y)
        if 0 <= y_int < depth_map.shape[0] and 0 <= x_int < depth_map.shape[1]:
            z = depth_map[y_int, x_int]
            if z > 0:  # Valid depth
                X = (x - cx) * z / fx
                Y = (y - cy) * z / fy
                Z = z
                points_3d.append([X, Y, Z])
            else:
                points_3d.append([0, 0, 0])
        else:
            points_3d.append([0, 0, 0])
    
    return np.array(points_3d, dtype=np.float32)


def example_single_camera_tracking(npz_path: Path):
    """Example: Single-camera tracking setup."""
    print("=" * 70)
    print("Example: Single-Camera Tracking")
    print("=" * 70)
    
    # Load scene
    scene_data = load_behave_scene(npz_path)
    print(f"Loaded scene: {scene_data['metadata']['scene_name']}")
    
    # Prepare for tracking
    camera_id = 0
    track_data = prepare_for_single_camera_tracking(scene_data, camera_id)
    
    rgbs = track_data['rgbs']  # (T, 3, H, W)
    query_points = track_data['query_points']  # (N, 2)
    
    print(f"\nCamera {camera_id}:")
    print(f"  RGB frames: {rgbs.shape}")
    print(f"  Query points: {query_points.shape}")
    print(f"  Intrinsics:\n{track_data['intrinsics']}")
    
    # Filter valid query points
    valid_mask = (query_points[:, 0] != 0) | (query_points[:, 1] != 0)
    valid_query_points = query_points[valid_mask]
    print(f"  Valid query points: {len(valid_query_points)}/{len(query_points)}")
    
    print("\nReady for tracking!")
    print("Example usage with CoTracker:")
    print("  model = torch.hub.load('facebookresearch/co-tracker', 'cotracker2')")
    print("  rgbs_tensor = torch.from_numpy(rgbs).float() / 255.0")
    print("  queries = torch.from_numpy(valid_query_points).float()")
    print("  tracks = model(rgbs_tensor, queries)")


def example_multi_view_tracking(npz_path: Path):
    """Example: Multi-view tracking setup."""
    print("\n" + "=" * 70)
    print("Example: Multi-View Tracking")
    print("=" * 70)
    
    # Load scene
    scene_data = load_behave_scene(npz_path)
    metadata = scene_data['metadata']
    
    # Prepare for multi-view tracking
    mv_data = prepare_for_multi_view_tracking(scene_data)
    
    print(f"\nScene: {metadata['scene_name']}")
    print(f"  Cameras: {metadata['num_cameras']}")
    print(f"  Frames: {metadata['num_frames']}")
    print(f"  Resolution: {metadata['image_height']}×{metadata['image_width']}")
    
    print(f"\nData shapes:")
    print(f"  RGBs: {mv_data['rgbs'].shape}")
    print(f"  Query points: {mv_data['query_points'].shape}")
    print(f"  Intrinsics: {mv_data['intrinsics'].shape}")
    print(f"  Extrinsics: {mv_data['extrinsics'].shape}")
    print(f"  Depths: {mv_data['depths'].shape}")
    
    print("\nCamera extrinsics (camera-to-world):")
    for cam_id in range(metadata['num_cameras']):
        extr = mv_data['extrinsics'][cam_id]
        t = extr[:, 3]
        print(f"  Camera {cam_id} position: [{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}]")
    
    print("\nReady for multi-view tracking!")
    print("Example: MVTracker, SpatialTracker, etc.")


def example_3d_reconstruction(npz_path: Path):
    """Example: 3D point reconstruction from depth."""
    print("\n" + "=" * 70)
    print("Example: 3D Reconstruction from Depth")
    print("=" * 70)
    
    # Load scene
    scene_data = load_behave_scene(npz_path)
    
    # Use first camera, first frame
    camera_id = 0
    frame_id = 0
    
    depth_map = scene_data['depths'][camera_id, frame_id]
    query_points_2d = scene_data['query_points'][frame_id, camera_id]
    intrinsics = scene_data['intrinsics'][camera_id]
    extrinsics = scene_data['extrinsics'][camera_id]
    
    print(f"\nReconstrucing 3D points from camera {camera_id}, frame {frame_id}")
    
    # Unproject to 3D
    points_3d_cam = unproject_to_3d(query_points_2d, depth_map, intrinsics)
    
    # Filter valid points
    valid_mask = points_3d_cam[:, 2] > 0
    valid_points_3d = points_3d_cam[valid_mask]
    
    print(f"Valid 3D points: {len(valid_points_3d)}/{len(points_3d_cam)}")
    
    if len(valid_points_3d) > 0:
        print(f"3D point cloud bounds:")
        print(f"  X: [{valid_points_3d[:, 0].min():.2f}, {valid_points_3d[:, 0].max():.2f}]")
        print(f"  Y: [{valid_points_3d[:, 1].min():.2f}, {valid_points_3d[:, 1].max():.2f}]")
        print(f"  Z: [{valid_points_3d[:, 2].min():.2f}, {valid_points_3d[:, 2].max():.2f}]")
        
        # Transform to world coordinates
        R = extrinsics[:, :3]
        t = extrinsics[:, 3]
        points_3d_world = (R @ valid_points_3d.T).T + t
        
        print(f"\nWorld coordinates bounds:")
        print(f"  X: [{points_3d_world[:, 0].min():.2f}, {points_3d_world[:, 0].max():.2f}]")
        print(f"  Y: [{points_3d_world[:, 1].min():.2f}, {points_3d_world[:, 1].max():.2f}]")
        print(f"  Z: [{points_3d_world[:, 2].min():.2f}, {points_3d_world[:, 2].max():.2f}]")


def main():
    parser = argparse.ArgumentParser(description="Example usage of converted BEHAVE data")
    parser.add_argument(
        "npz_path",
        type=str,
        help="Path to converted .npz file"
    )
    parser.add_argument(
        "--example",
        type=str,
        choices=["single", "multi", "3d", "all"],
        default="all",
        help="Which example to run"
    )
    
    args = parser.parse_args()
    npz_path = Path(args.npz_path)
    
    if not npz_path.exists():
        print(f"Error: File not found: {npz_path}")
        return
    
    if args.example in ["single", "all"]:
        example_single_camera_tracking(npz_path)
    
    if args.example in ["multi", "all"]:
        example_multi_view_tracking(npz_path)
    
    if args.example in ["3d", "all"]:
        example_3d_reconstruction(npz_path)
    
    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
