"""
Convert BEHAVE dataset to .npz format for tracking.

BEHAVE Dataset Structure:
- behave_all/
  - Date##_Sub##_object_action/  (scene)
    - info.json
    - t####.000/  (timestamp)
      - k#.color.jpg
      - k#.depth.png
      - k#.person_mask.jpg
      - k#.obj_rend_mask.jpg
      - k#.color.json (2D pose keypoints)

This script converts a BEHAVE scene into .npz format with:
- RGB images per camera and timestamp
- Camera intrinsics and extrinsics
- Body/hand masks
- Query points derived from masks
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm


def load_camera_intrinsics(calibs_root: Path, camera_id: int) -> np.ndarray:
    """
    Load camera intrinsics from BEHAVE calibration files.
    
    Args:
        calibs_root: Path to calibs folder
        camera_id: Camera ID (0-3)
    
    Returns:
        intrinsics: 3x3 intrinsics matrix [fx, 0, cx; 0, fy, cy; 0, 0, 1]
    """
    calib_file = calibs_root / "intrinsics" / str(camera_id) / "calibration.json"
    
    with open(calib_file, 'r') as f:
        calib = json.load(f)
    
    color_calib = calib['color']
    fx = color_calib['fx']
    fy = color_calib['fy']
    cx = color_calib['cx']
    cy = color_calib['cy']
    
    intrinsics = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    return intrinsics


def load_camera_extrinsics(calibs_root: Path, date: str, camera_id: int) -> np.ndarray:
    """
    Load camera extrinsics (camera-to-world transform) from BEHAVE calibration files.
    
    Args:
        calibs_root: Path to calibs folder
        date: Date identifier (e.g., "Date01")
        camera_id: Camera ID (0-3)
    
    Returns:
        extrinsics: 3x4 camera-to-world transformation [R|t]
    """
    config_file = calibs_root / date / "config" / str(camera_id) / "config.json"
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Rotation matrix (3x3) stored in row-major order
    rotation = np.array(config['rotation'], dtype=np.float32).reshape(3, 3)
    # Translation vector (3,)
    translation = np.array(config['translation'], dtype=np.float32)
    
    # Construct 3x4 extrinsics [R|t]
    extrinsics = np.column_stack([rotation, translation])
    
    return extrinsics


def extract_query_points_from_mask(mask: np.ndarray, num_points: int = 256, 
                                   method: str = 'grid') -> np.ndarray:
    """
    Extract query points from a binary mask.
    
    Args:
        mask: Binary mask (H, W) where non-zero pixels are foreground
        num_points: Number of query points to extract
        method: 'grid' for grid sampling, 'random' for random sampling
    
    Returns:
        query_points: (num_points, 2) array of [x, y] coordinates
    """
    # Get foreground pixels
    ys, xs = np.where(mask > 0)
    
    if len(xs) == 0:
        # No foreground pixels, return zeros
        return np.zeros((num_points, 2), dtype=np.float32)
    
    if method == 'random':
        # Random sampling
        if len(xs) >= num_points:
            indices = np.random.choice(len(xs), num_points, replace=False)
        else:
            indices = np.random.choice(len(xs), num_points, replace=True)
        query_points = np.stack([xs[indices], ys[indices]], axis=1).astype(np.float32)
    
    elif method == 'grid':
        # Grid sampling in the bounding box of the mask
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        
        # Create a grid
        grid_size = int(np.sqrt(num_points))
        x_grid = np.linspace(x_min, x_max, grid_size)
        y_grid = np.linspace(y_min, y_max, grid_size)
        xx, yy = np.meshgrid(x_grid, y_grid)
        
        grid_points = np.stack([xx.flatten(), yy.flatten()], axis=1)
        
        # Keep only points inside the mask
        valid_points = []
        for point in grid_points:
            x, y = int(point[0]), int(point[1])
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                if mask[y, x] > 0:
                    valid_points.append(point)
        
        if len(valid_points) >= num_points:
            indices = np.random.choice(len(valid_points), num_points, replace=False)
            query_points = np.array([valid_points[i] for i in indices], dtype=np.float32)
        elif len(valid_points) > 0:
            # Not enough valid points, sample with replacement
            indices = np.random.choice(len(valid_points), num_points, replace=True)
            query_points = np.array([valid_points[i] for i in indices], dtype=np.float32)
        else:
            # Fall back to random sampling from all foreground pixels
            indices = np.random.choice(len(xs), num_points, replace=True)
            query_points = np.stack([xs[indices], ys[indices]], axis=1).astype(np.float32)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return query_points


def convert_behave_scene(
    scene_path: Path,
    calibs_root: Path,
    output_path: Path,
    mask_type: str = 'person',  # 'person' or 'hand'
    num_cameras: int = 4,
    num_query_points: int = 256,
    downscale_factor: int = 1,
    max_frames: Optional[int] = None,
    save_rerun_viz: bool = False,
):
    """
    Convert a BEHAVE scene to .npz format.
    
    Args:
        scene_path: Path to the scene directory
        calibs_root: Path to calibs directory
        output_path: Path to save the .npz file
        mask_type: Type of mask to use ('person' or 'hand')
        num_cameras: Number of cameras (default 4)
        num_query_points: Number of query points to extract from masks
        downscale_factor: Factor to downscale images
        max_frames: Maximum number of frames to process (None for all)
        save_rerun_viz: Save rerun visualization
    """
    print(f"Converting scene: {scene_path.name}")
    
    # Load scene info
    info_file = scene_path / "info.json"
    with open(info_file, 'r') as f:
        info = json.load(f)
    
    # Extract date from info (e.g., "../../calibs/Date01/config")
    config_path = info['config']
    date = config_path.split('/')[-2]  # Extract "Date01"
    
    # Get all timestamp directories
    timestamp_dirs = sorted([d for d in scene_path.iterdir() if d.is_dir() and d.name.startswith('t')])
    
    if max_frames is not None:
        timestamp_dirs = timestamp_dirs[:max_frames]
    
    num_frames = len(timestamp_dirs)
    print(f"Found {num_frames} frames")
    
    # Load camera parameters
    intrinsics = {}
    extrinsics = {}
    
    for cam_id in range(num_cameras):
        intrinsics[cam_id] = load_camera_intrinsics(calibs_root, cam_id)
        extrinsics[cam_id] = load_camera_extrinsics(calibs_root, date, cam_id)
    
    # Load first image to get dimensions
    first_timestamp = timestamp_dirs[0]
    first_image = cv2.imread(str(first_timestamp / "k0.color.jpg"))
    h, w = first_image.shape[:2]
    
    if downscale_factor > 1:
        h = h // downscale_factor
        w = w // downscale_factor
    
    print(f"Image dimensions: {h}x{w}")
    
    # Initialize data arrays
    rgbs = np.zeros((num_cameras, num_frames, 3, h, w), dtype=np.uint8)
    depths = np.zeros((num_cameras, num_frames, h, w), dtype=np.float32)
    masks = np.zeros((num_cameras, num_frames, h, w), dtype=np.uint8)
    query_points_per_frame = []  # List of query points per frame per camera
    
    # Process each timestamp
    for frame_idx, timestamp_dir in enumerate(tqdm(timestamp_dirs, desc="Loading frames")):
        frame_query_points = []
        
        for cam_id in range(num_cameras):
            # Load color image
            color_file = timestamp_dir / f"k{cam_id}.color.jpg"
            if color_file.exists():
                img = cv2.imread(str(color_file))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                if downscale_factor > 1:
                    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
                
                rgbs[cam_id, frame_idx] = img.transpose(2, 0, 1)  # (3, H, W)
            
            # Load depth image
            depth_file = timestamp_dir / f"k{cam_id}.depth.png"
            if depth_file.exists():
                depth = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
                
                if downscale_factor > 1:
                    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
                
                # Convert depth to meters (BEHAVE uses mm)
                depths[cam_id, frame_idx] = depth.astype(np.float32) / 1000.0
            
            # Load mask
            if mask_type == 'person':
                mask_file = timestamp_dir / f"k{cam_id}.person_mask.jpg"
            else:
                # For hand masks, we might need to extract from person mask
                # or use a different file if available
                mask_file = timestamp_dir / f"k{cam_id}.person_mask.jpg"
            
            if mask_file.exists():
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                
                if downscale_factor > 1:
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                
                masks[cam_id, frame_idx] = mask
                
                # Extract query points from mask
                query_pts = extract_query_points_from_mask(mask, num_query_points, method='random')
                frame_query_points.append(query_pts)
            else:
                frame_query_points.append(np.zeros((num_query_points, 2), dtype=np.float32))
        
        query_points_per_frame.append(frame_query_points)
    
    # Convert query points to numpy array: (num_frames, num_cameras, num_query_points, 2)
    query_points = np.array(query_points_per_frame, dtype=np.float32)
    
    # Prepare intrinsics and extrinsics arrays
    intrinsics_array = np.stack([intrinsics[i] for i in range(num_cameras)], axis=0)  # (num_cameras, 3, 3)
    extrinsics_array = np.stack([extrinsics[i] for i in range(num_cameras)], axis=0)  # (num_cameras, 3, 4)
    
    # Save to .npz
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        output_path,
        rgbs=rgbs,  # (num_cameras, num_frames, 3, H, W)
        depths=depths,  # (num_cameras, num_frames, H, W)
        masks=masks,  # (num_cameras, num_frames, H, W)
        intrinsics=intrinsics_array,  # (num_cameras, 3, 3)
        extrinsics=extrinsics_array,  # (num_cameras, 3, 4)
        query_points=query_points,  # (num_frames, num_cameras, num_query_points, 2)
        scene_name=scene_path.name,
        mask_type=mask_type,
        num_frames=num_frames,
        num_cameras=num_cameras,
        image_height=h,
        image_width=w,
        downscale_factor=downscale_factor,
    )
    
    print(f"Saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Optional: Save rerun visualization
    if save_rerun_viz:
        save_rerun_visualization(
            output_path.with_suffix('.rrd'),
            rgbs, masks, intrinsics_array, extrinsics_array, query_points, scene_path.name
        )


def save_rerun_visualization(
    rrd_path: Path,
    rgbs: np.ndarray,
    masks: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    query_points: np.ndarray,
    scene_name: str,
):
    """Save rerun visualization of the converted scene."""
    try:
        import rerun as rr
        
        rr.init(scene_name, spawn=False)
        rr.save(str(rrd_path))
        
        num_cameras, num_frames = rgbs.shape[:2]
        
        for frame_idx in range(num_frames):
            rr.set_time_sequence("frame", frame_idx)
            
            for cam_id in range(num_cameras):
                # Log RGB image
                rr.log(
                    f"camera_{cam_id}/image",
                    rr.Image(rgbs[cam_id, frame_idx].transpose(1, 2, 0))
                )
                
                # Log mask
                rr.log(
                    f"camera_{cam_id}/mask",
                    rr.SegmentationImage(masks[cam_id, frame_idx])
                )
                
                # Log query points
                qpts = query_points[frame_idx, cam_id]
                rr.log(
                    f"camera_{cam_id}/query_points",
                    rr.Points2D(qpts, radii=3.0)
                )
                
                # Log camera
                # Convert extrinsics to world-to-camera
                R = extrinsics[cam_id, :3, :3]
                t = extrinsics[cam_id, :3, 3]
                # Camera-to-world -> World-to-camera
                R_inv = R.T
                t_inv = -R_inv @ t
                
                rr.log(
                    f"world/camera_{cam_id}",
                    rr.Transform3D(
                        translation=t,
                        mat3x3=R,
                    )
                )
        
        print(f"Saved rerun visualization: {rrd_path}")
    except ImportError:
        print("Warning: rerun not installed, skipping visualization")


def main():
    parser = argparse.ArgumentParser(description="Convert BEHAVE dataset to .npz format")
    parser.add_argument(
        "--behave_root",
        type=str,
        default="/data/behave-dataset/behave_all",
        help="Path to BEHAVE dataset root"
    )
    parser.add_argument(
        "--scene",
        type=str,
        required=True,
        help="Scene name (e.g., Date01_Sub01_backpack_back)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./conversions/behave_converted",
        help="Output directory for .npz files"
    )
    parser.add_argument(
        "--mask_type",
        type=str,
        default="person",
        choices=["person", "hand"],
        help="Type of mask to use for query points"
    )
    parser.add_argument(
        "--num_cameras",
        type=int,
        default=4,
        help="Number of cameras to process"
    )
    parser.add_argument(
        "--num_query_points",
        type=int,
        default=256,
        help="Number of query points to extract per mask"
    )
    parser.add_argument(
        "--downscale_factor",
        type=int,
        default=2,
        help="Factor to downscale images (1=no downscale)"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (None for all)"
    )
    parser.add_argument(
        "--save_rerun",
        action="store_true",
        help="Save rerun visualization"
    )
    
    args = parser.parse_args()
    
    behave_root = Path(args.behave_root)
    scene_path = behave_root / args.scene
    calibs_root = behave_root / "calibs"
    output_dir = Path(args.output_dir)
    
    if not scene_path.exists():
        print(f"Error: Scene not found: {scene_path}")
        return
    
    if not calibs_root.exists():
        print(f"Error: Calibration directory not found: {calibs_root}")
        return
    
    output_path = output_dir / f"{args.scene}.npz"
    
    convert_behave_scene(
        scene_path=scene_path,
        calibs_root=calibs_root,
        output_path=output_path,
        mask_type=args.mask_type,
        num_cameras=args.num_cameras,
        num_query_points=args.num_query_points,
        downscale_factor=args.downscale_factor,
        max_frames=args.max_frames,
        save_rerun_viz=args.save_rerun,
    )


if __name__ == "__main__":
    main()
