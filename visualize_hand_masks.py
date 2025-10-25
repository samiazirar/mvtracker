#!/usr/bin/env python3
"""
Visualize hand tracking results from _hand_tracked.npz files.
Creates a Rerun visualization showing:
1. RGB point clouds from depth maps
2. Lifted SAM hand masks as 3D point clouds
3. Query points derived from hand keypoints
4. Original query points for debugging alignment

This helps debug why query points may not align in 3D even when they look correct in 2D.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import rerun as rr
from typing import Optional, List
import warnings


def lift_mask_to_3d(mask, depth, intr, extr):
    """
    Lift a binary mask to 3D world coordinates using depth map.
    
    Args:
        mask: Binary mask [H, W]
        depth: Depth map [H, W]
        intr: Intrinsic matrix [3, 3]
        extr: Extrinsic matrix [3, 4] (world_T_cam)
    
    Returns:
        points_3d: 3D points in world coordinates [N, 3]
    """
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float32)
    
    # Get depth values at mask locations
    z = depth[ys, xs].astype(np.float32)
    valid = z > 0
    
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)
    
    xs = xs[valid]
    ys = ys[valid]
    z = z[valid]
    
    # Create homogeneous pixel coordinates
    pixels = np.stack([xs, ys, np.ones_like(xs)], axis=0).astype(np.float32) * z
    
    # Backproject to camera space
    cam_points = np.linalg.inv(intr) @ pixels
    
    # Transform to world space
    world_points = (extr[:3, :3] @ cam_points + extr[:3, 3:4]).T
    
    return world_points.astype(np.float32)


def visualize_hand_tracking(npz_path: Path, output_rrd: Path, max_frames: Optional[int] = None):
    """
    Create a Rerun visualization from a _hand_tracked.npz file.
    
    Args:
        npz_path: Path to the _hand_tracked.npz file
        output_rrd: Path to save the .rrd recording
        max_frames: Optional limit on number of frames to visualize
    """
    print(f"[INFO] Loading {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)
    
    # Load required data
    rgbs = data["rgbs"]              # [C, T, 3, H, W]
    depths = data["depths"][:, :, 0]  # [C, T, H, W]
    intrs = data["intrs"]             # [C, T, 3, 3] or [C, 3, 3]
    extrs = data["extrs"]             # [C, T, 3, 4] or [C, 3, 4]
    
    # Load hand tracking data
    sam_hand_masks = data["sam_hand_masks"]        # [C, T, H, W]
    sam_hand_mask_scores = data["sam_hand_mask_scores"]  # [C, T, H, W]
    hand_points = data["hand_points"]              # object array of [N, 3] arrays per frame
    query_points = data["query_points"]            # [N, 4] where cols are [t, x, y, z]
    query_colors = data.get("query_colors", None)  # [N, 3]
    
    # Optional camera IDs
    camera_ids = data.get("camera_ids", None)
    
    C, T, _, H, W = rgbs.shape
    print(f"[INFO] Data shape: {C} cameras, {T} frames, {H}x{W} resolution")
    print(f"[INFO] Query points: {len(query_points)}")
    print(f"[INFO] Hand points per frame: {[len(hp) if isinstance(hp, np.ndarray) else 0 for hp in hand_points[:5]]}...")
    
    # Limit frames if requested
    if max_frames is not None and T > max_frames:
        print(f"[INFO] Limiting visualization to {max_frames} frames")
        T = max_frames
        rgbs = rgbs[:, :max_frames]
        depths = depths[:, :max_frames]
        sam_hand_masks = sam_hand_masks[:, :max_frames]
        sam_hand_mask_scores = sam_hand_mask_scores[:, :max_frames]
        hand_points = hand_points[:max_frames]
        if intrs.ndim == 3 and intrs.shape[0] == C:
            pass  # Static intrinsics
        else:
            intrs = intrs[:, :max_frames]
        if extrs.ndim == 3 and extrs.shape[0] == C:
            pass  # Static extrinsics
        else:
            extrs = extrs[:, :max_frames]
    
    # Initialize Rerun
    rr.init("hand_mask_visualization", recording_id=npz_path.stem)
    
    fps = 12.0
    
    # Process each camera
    for c in range(C):
        cam_id = str(camera_ids[c]) if camera_ids is not None else f"cam_{c:03d}"
        print(f"[INFO] Processing camera {cam_id}...")
        
        intr = intrs[c]
        extr = extrs[c]
        
        for t in range(T):
            rr.set_time_seconds("frame", t / fps)
            
            # Get data for this frame
            rgb = np.moveaxis(rgbs[c, t], 0, 2)  # [H, W, 3]
            depth = depths[c, t]  # [H, W]
            mask = sam_hand_masks[c, t]  # [H, W]
            
            # Get intrinsics and extrinsics for this frame
            intr_t = intr[t] if intr.ndim == 3 else intr
            extr_t = extr[t] if extr.ndim == 3 else extr
            
            # Compute world_T_cam for camera visualization
            world_T_cam = np.eye(4, dtype=np.float32)
            world_T_cam[:3, :4] = extr_t
            
            # Log camera
            rr.log(
                f"world/cameras/{cam_id}",
                rr.Pinhole(
                    image_from_camera=intr_t,
                    width=W,
                    height=H
                )
            )
            rr.log(
                f"world/cameras/{cam_id}",
                rr.Transform3D(
                    translation=world_T_cam[:3, 3],
                    mat3x3=world_T_cam[:3, :3]
                )
            )
            
            # Log RGB image to camera
            rr.log(f"world/cameras/{cam_id}/rgb", rr.Image(rgb))
            
            # Create full RGB point cloud from depth
            y, x = np.indices((H, W))
            homo_pixel_coords = np.stack([x.ravel(), y.ravel(), np.ones_like(x).ravel()], axis=1).T
            depth_values = depth.ravel()
            
            valid = depth_values > 0
            if valid.sum() > 0:
                cam_coords = (np.linalg.inv(intr_t) @ homo_pixel_coords) * depth_values
                cam_coords_hom = np.vstack((cam_coords, np.ones((1, cam_coords.shape[1]))))
                world_coords = (world_T_cam @ cam_coords_hom)[:3].T
                
                rgb_colors = rgb.reshape(-1, 3).astype(np.uint8)
                
                # Filter valid points
                world_coords_valid = world_coords[valid]
                rgb_colors_valid = rgb_colors[valid]
                
                # Log full point cloud
                rr.log(
                    f"world/point_clouds/full/{cam_id}",
                    rr.Points3D(
                        positions=world_coords_valid,
                        colors=rgb_colors_valid,
                        radii=-2.5  # Adaptive radius
                    )
                )
            
            # Lift hand mask to 3D and visualize
            if mask.any():
                hand_mask_points = lift_mask_to_3d(mask, depth, intr_t, extr_t)
                
                if len(hand_mask_points) > 0:
                    # Color hand mask points in magenta
                    magenta = np.array([255, 0, 255], dtype=np.uint8)
                    hand_colors = np.tile(magenta, (len(hand_mask_points), 1))
                    
                    rr.log(
                        f"world/hand_masks/{cam_id}",
                        rr.Points3D(
                            positions=hand_mask_points,
                            colors=hand_colors,
                            radii=-1.5  # Slightly larger for visibility
                        )
                    )
            
            # Log the pre-computed hand points (if any)
            if isinstance(hand_points[t], np.ndarray) and len(hand_points[t]) > 0:
                hp = hand_points[t]
                cyan = np.array([0, 255, 255], dtype=np.uint8)
                hp_colors = np.tile(cyan, (len(hp), 1))
                
                rr.log(
                    f"world/hand_points_precomputed/frame_{t}",
                    rr.Points3D(
                        positions=hp,
                        colors=hp_colors,
                        radii=-1.0
                    )
                )
    
    # Log query points (these are the ones we want to debug)
    # Query points format: [frame_idx, x, y, z]
    print(f"[INFO] Logging {len(query_points)} query points...")
    
    if len(query_points) > 0:
        # Group query points by frame for temporal logging
        for qp_idx, qp in enumerate(query_points):
            frame_idx = int(qp[0])
            xyz = qp[1:4]
            
            if frame_idx >= T:
                continue
            
            rr.set_time_seconds("frame", frame_idx / fps)
            
            # Use the provided color or default to purple
            if query_colors is not None:
                color = (query_colors[qp_idx] * 255).astype(np.uint8)
            else:
                color = np.array([128, 0, 255], dtype=np.uint8)
            
            rr.log(
                f"world/query_points/point_{qp_idx}",
                rr.Points3D(
                    positions=[xyz],
                    colors=[color],
                    radii=0.02  # Fixed size for visibility
                )
            )
    
    # Add a debug view: show query points as persistent across all frames
    for qp_idx, qp in enumerate(query_points[:100]):  # Limit to 100 for performance
        xyz = qp[1:4]
        if query_colors is not None:
            color = (query_colors[qp_idx] * 255).astype(np.uint8)
        else:
            color = np.array([128, 0, 255], dtype=np.uint8)
        
        rr.set_time_seconds("frame", 0)
        rr.log(
            f"world/query_points_persistent/point_{qp_idx}",
            rr.Points3D(
                positions=[xyz],
                colors=[color],
                radii=0.015
            )
        )
    
    # Save recording
    print(f"[INFO] Saving visualization to {output_rrd}...")
    rr.save(str(output_rrd))
    print(f"[INFO] Done! View with: rerun {output_rrd}")


def main():
    parser = argparse.ArgumentParser(description="Visualize hand tracking with SAM masks in Rerun")
    parser.add_argument("--npz", required=True, help="Path to *_hand_tracked.npz file")
    parser.add_argument("--output", default=None, help="Output .rrd file path (default: based on input name)")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames to visualize")
    args = parser.parse_args()
    
    npz_path = Path(args.npz)
    if not npz_path.exists():
        raise FileNotFoundError(f"Input file not found: {npz_path}")
    
    if args.output:
        output_rrd = Path(args.output)
    else:
        output_rrd = npz_path.parent / f"{npz_path.stem}_visualization.rrd"
    
    visualize_hand_tracking(npz_path, output_rrd, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
