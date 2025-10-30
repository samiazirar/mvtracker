#!/usr/bin/env python3
"""
Lift 2D Tracks to 3D and Visualize in Rerun

This script takes 2D tracks from SpatialTrackerV2 (per camera) and lifts them to 3D
world coordinates using depth maps and camera parameters. It then visualizes all tracks
in Rerun as 4D trajectories (3D + time).

Handles RGB/depth resolution mismatch by automatically scaling coordinates.

Workflow:
1. Load NPZ with tracks_2d dict {camera_id: [T, N, 2]} and camera data
2. Detect RGB and depth resolutions
3. Scale track coordinates from RGB space to depth space if needed
4. For each camera and each track:
   - Lift 2D pixel coordinates to 3D world coordinates using depth
   - Handle visibility flags (skip invalid/invisible points)
5. Combine all tracks across cameras
6. Visualize in Rerun with:
   - RGB point clouds (togglable)
   - Camera frustums
   - Track trajectories as colored lines
   - Track points with temporal animation

Usage:
    python lift_and_visualize_tracks.py \
        --npz sam_tracks_per_camera.npz \
        --output sam_tracks_3d.rrd \
        --spawn \
        --fps 10
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import rerun as rr
from tqdm import tqdm
import matplotlib.pyplot as plt

from mvtracker.utils.visualizer_rerun import log_pointclouds_to_rerun
from utils.camera_utils import scale_intrinsics_matrix, infer_calibration_resolution


def unproject_pixel_to_3d(
    pixel: np.ndarray,  # [2] - (x, y)
    depth: float,
    intr: np.ndarray,  # [3, 3]
    extr: np.ndarray,  # [3, 4] or [4, 4]
) -> Optional[np.ndarray]:
    """
    Unproject a 2D pixel to 3D world coordinates.
    
    Args:
        pixel: [2] pixel coordinates (x, y)
        depth: Depth value at pixel
        intr: Intrinsic matrix [3, 3]
        extr: Extrinsic matrix [3, 4] or [4, 4] (world-to-camera)
    
    Returns:
        point_3d: [3] world coordinates (x, y, z) or None if invalid depth
    """
    if depth <= 0 or not np.isfinite(depth):
        return None
    
    # Pixel to camera coordinates
    x, y = pixel
    fx, fy = intr[0, 0], intr[1, 1]
    cx, cy = intr[0, 2], intr[1, 2]
    
    x_cam = (x - cx) * depth / fx
    y_cam = (y - cy) * depth / fy
    z_cam = depth
    
    point_cam = np.array([x_cam, y_cam, z_cam, 1.0])
    
    # Camera to world coordinates
    # extr is world-to-camera, so invert to get camera-to-world
    if extr.shape[0] == 3:
        # [3, 4] -> [4, 4]
        extr_4x4 = np.eye(4)
        extr_4x4[:3] = extr
    else:
        extr_4x4 = extr
    
    c2w = np.linalg.inv(extr_4x4)
    point_world = c2w @ point_cam
    
    return point_world[:3]


def lift_tracks_to_3d(
    tracks_2d: Dict[str, np.ndarray],  # {camera_id: [T, N, 2]}
    visibility: Dict[str, np.ndarray],  # {camera_id: [T, N]}
    depths: np.ndarray,  # [C, T, H, W]
    intrs: np.ndarray,  # [C, T, 3, 3] or [C, 3, 3]
    extrs: np.ndarray,  # [C, T, 3, 4] or [C, 3, 4]
    camera_ids: List[str],
    track_instance_ids: Optional[Dict[str, np.ndarray]] = None,  # {camera_id: [N]}
    rgbs: Optional[np.ndarray] = None,  # [C, T, H, W, 3] - for resolution inference
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Lift 2D tracks to 3D world coordinates.
    
    Handles resolution mismatch between RGB (where tracks were computed) and depth maps
    by scaling pixel coordinates from RGB space to depth space.
    
    Args:
        tracks_2d: Dict {camera_id: [T, N, 2]} - 2D tracks in RGB space
        visibility: Dict {camera_id: [T, N]} - visibility flags
        depths: Depth maps [C, T, H, W]
        intrs: Intrinsics [C, T, 3, 3] or [C, 3, 3] - calibrated for base resolution
        extrs: Extrinsics [C, T, 3, 4] or [C, 3, 4]
        camera_ids: List of camera ID strings
        track_instance_ids: Optional dict {camera_id: [N]} - instance IDs
        rgbs: Optional RGB images [C, T, H, W, 3] - for inferring RGB resolution
    
    Returns:
        tracks_3d: Dict with structure:
            {
                camera_id: {
                    "positions": [T, N, 3],  # 3D positions
                    "visibility": [T, N],     # visibility flags
                    "instance_ids": [N],      # instance IDs (if provided)
                }
            }
    """
    print(f"[INFO] ========== Lifting Tracks to 3D ==========")
    
    C = len(camera_ids)
    T = depths.shape[1]
    
    # Get RGB and depth dimensions
    if rgbs is not None:
        # RGB can be [C, T, H, W, 3] or [C, T, 3, H, W]
        if rgbs.ndim == 5 and rgbs.shape[2] == 3:
            # [C, T, 3, H, W]
            H_rgb, W_rgb = rgbs.shape[3], rgbs.shape[4]
        elif rgbs.ndim == 5 and rgbs.shape[-1] == 3:
            # [C, T, H, W, 3]
            H_rgb, W_rgb = rgbs.shape[2], rgbs.shape[3]
        else:
            print(f"[WARN] Unexpected RGB shape: {rgbs.shape}")
            H_rgb, W_rgb = None, None
    else:
        # Assume tracks are in depth space if no RGB provided
        H_rgb, W_rgb = None, None
    
    H_depth, W_depth = depths.shape[2], depths.shape[3]  # [C, T, H, W]
    
    print(f"[INFO] RGB dimensions: {W_rgb}x{H_rgb}" if rgbs is not None else "[INFO] No RGB provided, assuming tracks in depth space")
    print(f"[INFO] Depth dimensions: {W_depth}x{H_depth}")
    
    # Compute scaling factors if there's a resolution mismatch
    if rgbs is not None and (W_rgb != W_depth or H_rgb != H_depth):
        scale_x = W_depth / W_rgb
        scale_y = H_depth / H_rgb
        print(f"[INFO] Resolution mismatch detected! Scaling coordinates: x={scale_x:.4f}, y={scale_y:.4f}")
    else:
        scale_x = 1.0
        scale_y = 1.0
        print(f"[INFO] No resolution scaling needed (RGB and depth match)")
    
    # Create camera ID to index mapping
    camera_id_to_index = {str(cam_id): c for c, cam_id in enumerate(camera_ids)}
    
    tracks_3d = {}
    
    for cam_id in tqdm(camera_ids, desc="Lifting tracks"):
        if cam_id not in tracks_2d:
            print(f"[WARN] No tracks for camera {cam_id}, skipping")
            continue
        
        c = camera_id_to_index[cam_id]
        tracks_2d_cam = tracks_2d[cam_id]  # [T, N, 2]
        vis_cam = visibility[cam_id]  # [T, N] or [T, N, 1]
        
        # Handle visibility format
        if vis_cam.ndim == 3 and vis_cam.shape[2] == 1:
            vis_cam = vis_cam[:, :, 0]  # [T, N, 1] -> [T, N]
        
        T_cam, N_cam = tracks_2d_cam.shape[0], tracks_2d_cam.shape[1]
        
        # Handle intrinsics/extrinsics format
        if intrs.ndim == 3 and intrs.shape[1] == 3:
            # [C, 3, 3] -> expand to [T, 3, 3]
            intrs_cam = np.tile(intrs[c][None], (T, 1, 1))
        else:
            intrs_cam = intrs[c]  # [T, 3, 3]
        
        if extrs.ndim == 3 and extrs.shape[1] in [3, 4]:
            # [C, 3, 4] -> expand to [T, 3, 4]
            extrs_cam = np.tile(extrs[c][None], (T, 1, 1))
        else:
            extrs_cam = extrs[c]  # [T, 3, 4]
        
        depths_cam = depths[c]  # [T, H, W]
        
        # Lift each track
        positions_3d = np.full((T_cam, N_cam, 3), np.nan, dtype=np.float32)
        for t in range(T_cam):
            for n in range(N_cam):
                if not vis_cam[t, n]:
                    continue
                
                # Get 2D pixel in RGB space
                pixel_rgb = tracks_2d_cam[t, n]  # [2] or [3] if confidence included
                x_rgb, y_rgb = pixel_rgb[0], pixel_rgb[1]
                
                # Scale to depth space
                x_depth = x_rgb * scale_x
                y_depth = y_rgb * scale_y
                
                # Round to nearest pixel in depth map
                x_int = int(round(x_depth))
                y_int = int(round(y_depth))
                
                H, W = depths_cam.shape[1], depths_cam.shape[2]
                if x_int < 0 or x_int >= W or y_int < 0 or y_int >= H:
                    continue
                
                # Get depth value
                depth = depths_cam[t, y_int, x_int]
                
                # Unproject to 3D using depth-space coordinates
                pixel_depth = np.array([x_depth, y_depth])
                point_3d = unproject_pixel_to_3d(
                    pixel=pixel_depth,
                    depth=depth,
                    intr=intrs_cam[t],
                    extr=extrs_cam[t],
                )
                
                if point_3d is not None:
                    positions_3d[t, n] = point_3d
        
        # Store results
        tracks_3d[cam_id] = {
            "positions": positions_3d,  # [T, N, 3]
            "visibility": vis_cam,  # [T, N]
        }
        
        if track_instance_ids is not None and cam_id in track_instance_ids:
            tracks_3d[cam_id]["instance_ids"] = track_instance_ids[cam_id]
        
        valid_points = np.isfinite(positions_3d).all(axis=2).sum()
        print(f"[INFO] Camera {cam_id}: lifted {N_cam} tracks, {valid_points}/{T_cam*N_cam} valid 3D points")
    
    return tracks_3d


def visualize_tracks_rerun(
    tracks_3d: Dict[str, Dict[str, np.ndarray]],
    npz_path: Path,
    output_path: Optional[Path] = None,
    spawn: bool = False,
    fps: float = 10.0,
    max_frames: Optional[int] = None,
):
    """
    Visualize 3D tracks in Rerun with scene context.
    
    Args:
        tracks_3d: Dict with camera tracks
        npz_path: Path to original NPZ (to load RGB, cameras)
        output_path: Output .rrd file path
        spawn: Whether to spawn viewer
        fps: Frame rate for animation
        max_frames: Optional frame limit
    """
    print(f"\n[INFO] ========== Visualizing in Rerun ==========")
    
    # Load scene data
    data = np.load(npz_path, allow_pickle=True)
    rgbs = data.get("rgbs", None)
    depths = data["depths"]
    intrs = data["intrs"]
    extrs = data["extrs"]
    camera_ids = [str(cid) for cid in data["camera_ids"]]
    
    # Handle depth format
    if depths.ndim == 5 and depths.shape[2] == 1:
        depths = depths[:, :, 0]  # [C, T, 1, H, W] -> [C, T, H, W]
    
    T = depths.shape[1]
    if max_frames is not None:
        T = min(T, max_frames)
    
    # Initialize Rerun
    recording_name = f"sam_tracks_{npz_path.stem}"
    rr.init(recording_name, recording_id=npz_path.stem, spawn=spawn)
    
    # Set coordinate system
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    
    # Log world axes
    rr.log(
        "world/axes",
        rr.Arrows3D(
            origins=[[0, 0, 0]],
            vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            radii=0.01,
        ),
        static=True,
    )
    
    # Log RGB point clouds and cameras
    if rgbs is not None:
        print(f"[INFO] Logging RGB point clouds and camera frustums...")
        
        # Handle RGB format
        if rgbs.ndim == 5 and rgbs.shape[-1] == 3:
            # [C, T, H, W, 3] -> [C, T, 3, H, W]
            rgbs_viz = np.moveaxis(rgbs, -1, 2)
        else:
            rgbs_viz = rgbs
        
        # Limit frames
        rgbs_viz = rgbs_viz[:, :T]
        depths_viz = depths[:, :T]
        
        # Add channel dimension to depth if needed
        if depths_viz.ndim == 4:
            depths_viz = depths_viz[:, :, None, :, :]  # [C, T, H, W] -> [C, T, 1, H, W]
        
        # Handle intrs/extrs - they may need time dimension
        intrs_viz = intrs[:, :T] if intrs.ndim == 4 else intrs
        extrs_viz = extrs[:, :T] if extrs.ndim == 4 else extrs
        
        # Convert to torch tensors
        rgbs_tensor = torch.from_numpy(rgbs_viz).float().unsqueeze(0)  # [1, C, T, 3, H, W]
        depths_tensor = torch.from_numpy(depths_viz).float().unsqueeze(0)  # [1, C, T, 1, H, W]
        intrs_tensor = torch.from_numpy(intrs_viz).float().unsqueeze(0)  # [1, C, T, 3, 3]
        extrs_tensor = torch.from_numpy(extrs_viz).float().unsqueeze(0)  # [1, C, T, 3, 4]
        
        # Log
        log_pointclouds_to_rerun(
            dataset_name="sam_tracks_scene",
            datapoint_idx=0,
            rgbs=rgbs_tensor,
            depths=depths_tensor,
            intrs=intrs_tensor,
            extrs=extrs_tensor,
            camera_ids=camera_ids,
            log_rgb_pointcloud=True,
            log_camera_frustrum=True,
            fps=fps,
            radii=-0.95,
        )
        print(f"[INFO] Logged RGB point clouds and cameras")
    
    # Log tracks
    print(f"[INFO] Logging 3D tracks...")
    
    # Combine all tracks from all cameras
    all_positions = []
    all_visibility = []
    all_camera_indices = []
    all_instance_ids = []
    
    for cam_idx, cam_id in enumerate(camera_ids):
        if cam_id not in tracks_3d:
            continue
        
        cam_data = tracks_3d[cam_id]
        positions = cam_data["positions"][:T]  # [T, N, 3]
        visibility = cam_data["visibility"][:T]  # [T, N]
        
        N = positions.shape[1]
        
        all_positions.append(positions)
        all_visibility.append(visibility)
        all_camera_indices.extend([cam_idx] * N)
        
        if "instance_ids" in cam_data:
            all_instance_ids.extend(cam_data["instance_ids"])
        else:
            all_instance_ids.extend([-1] * N)
    
    if not all_positions:
        print("[WARN] No tracks to visualize!")
        return
    
    all_positions = np.concatenate(all_positions, axis=1)  # [T, N_total, 3]
    all_visibility = np.concatenate(all_visibility, axis=1)  # [T, N_total]
    all_camera_indices = np.array(all_camera_indices)
    all_instance_ids = np.array(all_instance_ids)
    
    print(f"[INFO] Total tracks: {all_positions.shape[1]}")
    print(f"[INFO] Frames: {T}")
    
    # Generate colors for tracks (by camera)
    num_cameras = len(camera_ids)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_cameras))[:, :3] * 255
    colors = colors.astype(np.uint8)
    
    track_colors = colors[all_camera_indices]  # [N_total, 3]
    
    # Log tracks frame by frame
    for t in tqdm(range(T), desc="Logging tracks"):
        time_seconds = t / fps
        rr.set_time_seconds("frame", time_seconds)
        
        # Get visible tracks at this frame - ensure boolean arrays
        visibility_bool = all_visibility[t].astype(bool)
        positions_finite = np.isfinite(all_positions[t]).all(axis=1)
        visible_mask = visibility_bool & positions_finite
        
        if visible_mask.any():
            positions_t = all_positions[t, visible_mask]  # [N_vis, 3]
            colors_t = track_colors[visible_mask]  # [N_vis, 3]
            
            # Log as points
            rr.log(
                "world/tracks/points",
                rr.Points3D(
                    positions=positions_t,
                    colors=colors_t,
                    radii=0.01,
                ),
            )
            
            # Log trajectories (lines from previous positions)
            if t > 0:
                for n in range(all_positions.shape[1]):
                    if not visible_mask[n]:
                        continue
                    
                    # Get trajectory up to current frame
                    vis_traj = all_visibility[:t+1, n].astype(bool)
                    pos_finite = np.isfinite(all_positions[:t+1, n]).all(axis=1)
                    traj_mask = vis_traj & pos_finite
                    
                    if traj_mask.sum() < 2:
                        continue
                    
                    traj_positions = all_positions[:t+1, n][traj_mask]  # [T_valid, 3]
                    
                    # Log as line strip
                    rr.log(
                        f"world/tracks/trajectories/track_{n}",
                        rr.LineStrips3D(
                            [traj_positions],
                            colors=[track_colors[n]],
                            radii=0.003,
                        ),
                    )
    
    print(f"[INFO] Logged {all_positions.shape[1]} tracks")
    
    # Save recording
    if output_path is None:
        output_path = npz_path.parent / f"{npz_path.stem}_tracks_3d.rrd"
    
    print(f"\n[INFO] Saving to {output_path}...")
    rr.save(str(output_path))
    
    print(f"[INFO] Done! View with: rerun {output_path}")
    if not spawn:
        print(f"[INFO] Or use --spawn flag to open viewer automatically")


def main():
    parser = argparse.ArgumentParser(
        description="Lift 2D tracks to 3D and visualize in Rerun",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--npz",
        type=Path,
        required=True,
        help="Path to NPZ file with tracks_2d and camera data",
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .rrd file path (default: based on input name)",
    )
    
    parser.add_argument(
        "--spawn",
        action="store_true",
        help="Spawn Rerun viewer automatically",
    )
    
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Frame rate for visualization (default: 10.0)",
    )
    
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit number of frames to visualize",
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.npz.exists():
        raise FileNotFoundError(f"Input file not found: {args.npz}")
    
    # Load data
    print(f"[INFO] Loading data from {args.npz}...")
    data = np.load(args.npz, allow_pickle=True)
    
    tracks_2d = data["tracks_2d"].item()  # Dict
    visibility = data["visibility"].item()  # Dict
    camera_ids = [str(cid) for cid in data["camera_ids"]]
    
    track_instance_ids = None
    if "track_instance_ids" in data:
        track_instance_ids = data["track_instance_ids"].item()
    
    # Handle depth format for lifting
    depths_for_lifting = data["depths"]
    if depths_for_lifting.ndim == 5 and depths_for_lifting.shape[2] == 1:
        depths_for_lifting = depths_for_lifting[:, :, 0]  # [C, T, 1, H, W] -> [C, T, H, W]
    
    # Lift to 3D
    tracks_3d = lift_tracks_to_3d(
        tracks_2d=tracks_2d,
        visibility=visibility,
        depths=depths_for_lifting,
        intrs=data["intrs"],
        extrs=data["extrs"],
        camera_ids=camera_ids,
        track_instance_ids=track_instance_ids,
        rgbs=data.get("rgbs", None),  # Pass RGB for resolution detection
    )
    
    # Visualize
    visualize_tracks_rerun(
        tracks_3d=tracks_3d,
        npz_path=args.npz,
        output_path=args.output,
        spawn=args.spawn,
        fps=args.fps,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
