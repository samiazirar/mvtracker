#!/usr/bin/env python3
"""
Create Query Points from HOISTFormer Masks Around Contact Frames

This script processes NPZ files with HOISTFormer predictions to create query points
for MVTracker by lifting masks to 3D around contact frames.

Workflow:
1. Load NPZ file with hoist_masks and hoist_contact_frames
2. For each instance, find contact frame per camera (or use frame 0 with --use-first-frame)
3. Select frames: [contact - before, ..., contact, ..., contact + after]
4. Lift all mask pixels to 3D points in world coordinates
5. Create query points in format (frame_idx, x, y, z) for demo.py

Output NPZ Format:
    - All original data from input file
    - query_points: [N, 4] array with columns [frame_idx, x, y, z]
      where frame_idx is the frame number and (x, y, z) are world coordinates
      

python create_query_points_from_masks.py --npz third_party/HOISTFormer/hoist_output/task_0045_user_0020_scene_0004_cfg_0006_human_processed_hand_tracked_hoist.npz --key hoist_masks --frames-before 3 --frames-after 1 --output third_party/HOISTFormer/hoist_output/task_0045_user_0020_scene_0004_cfg_0006_human_processed_hand_tracked_hoist_query.npz

Usage:
    python create_query_points_from_masks.py 
        --npz path/to/file_hoist.npz 
        --key hoist_masks 
        --frames-before 3 
        --frames-after 1 
        --output path/to/output.npz
        
Example:
    # Use 3 frames before and 1 frame after contact (default)
    python create_query_points_from_masks.py \
        --npz third_party/HOISTFormer/hoist_output/task_0045_hoist.npz \
        --key hoist_masks
        
    # Custom frame range: 5 before, 2 after
    python create_query_points_from_masks.py \
        --npz third_party/HOISTFormer/hoist_output/task_0045_hoist.npz \
        --key hoist_masks \
        --frames-before 5 \
        --frames-after 2
        
    # Use FIRST FRAME for all cameras (ignore contact frames) - RECOMMENDED for tracking
    python create_query_points_from_masks.py \
        --npz third_party/HOISTFormer/sam2_output/task_0045_sam2.npz \
        --key sam2_masks \
        --use-first-frame \
        --frames-before 0 \
        --frames-after 0
        --frames-after 2
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import rerun as rr

from utils.mask_lifting_utils import lift_mask_to_3d
from mvtracker.utils.visualizer_rerun import log_pointclouds_to_rerun
from utils.camera_utils import scale_intrinsics_matrix, infer_calibration_resolution


def get_frame_range_around_contact(
    contact_frame: int,
    frames_before: int,
    frames_after: int,
    total_frames: int,
) -> List[int]:
    """
    Get list of frame indices around a contact frame.
    
    Args:
        contact_frame: Frame index where contact occurred
        frames_before: Number of frames to include before contact
        frames_after: Number of frames to include after contact
        total_frames: Total number of frames in sequence
        
    Returns:
        List of frame indices, clipped to valid range [0, total_frames-1]
        
    Example:
        >>> get_frame_range_around_contact(22, 3, 1, 100)
        [19, 20, 21, 22, 23]
    """
    start_frame = max(0, contact_frame - frames_before)
    end_frame = min(total_frames - 1, contact_frame + frames_after)
    
    # Generate inclusive range
    return list(range(start_frame, end_frame + 1))


def lift_instance_masks_to_query_points(
    instance_name: str,
    masks: np.ndarray,  # [C, T, H, W] for this instance
    contact_frames: Dict[str, int],  # {camera_id: contact_frame}
    depths: np.ndarray,  # [C, T, H, W]
    intrs: np.ndarray,  # [C, T, 3, 3] or [C, 3, 3]
    extrs: np.ndarray,  # [C, T, 3, 4] or [C, 3, 4]
    camera_ids: List[str],
    frames_before: int,
    frames_after: int,
    min_depth: float = 1e-6,
    max_depth: float = 10.0,
    max_points_per_frame: Optional[int] = None,
    use_first_frame: bool = False,
    rgbs: Optional[np.ndarray] = None,  # [C, T, H, W, 3] or [C, T, 3, H, W] for resolution detection
) -> np.ndarray:
    """
    Lift instance masks to 3D query points around contact frames.
    
    For each camera where contact was detected, lifts mask pixels to 3D
    in the frame range [contact - before, ..., contact + after].
    
    Args:
        instance_name: Name of the instance (e.g., "instance_0")
        masks: Binary masks for this instance [C, T, H, W]
        contact_frames: Dictionary mapping camera_id to contact frame number
        depths: Depth maps [C, T, H, W]
        intrs: Camera intrinsics [C, T, 3, 3] or [C, 3, 3]
        extrs: Camera extrinsics [C, T, 3, 4] or [C, 3, 4]
        camera_ids: List of camera ID strings [C]
        frames_before: Number of frames before contact to include
        frames_after: Number of frames after contact to include
        min_depth: Minimum valid depth threshold
        max_depth: Maximum valid depth threshold
        max_points_per_frame: Optional limit on points per frame (random sampling if exceeded)
        use_first_frame: If True, use frame 0 for ALL cameras regardless of contact_frames
        
    Returns:
        query_points: [N, 4] array with columns [frame_idx, x, y, z]
                     Empty array if no valid points found
    """
    C, T, H_mask, W_mask = masks.shape
    H_depth, W_depth = depths.shape[2], depths.shape[3]
    
    # Detect resolution mismatch
    if rgbs is not None:
        # Determine RGB resolution
        if rgbs.ndim == 5 and rgbs.shape[2] == 3:
            # [C, T, 3, H, W]
            H_rgb, W_rgb = rgbs.shape[3], rgbs.shape[4]
        elif rgbs.ndim == 5 and rgbs.shape[-1] == 3:
            # [C, T, H, W, 3]
            H_rgb, W_rgb = rgbs.shape[2], rgbs.shape[3]
        else:
            H_rgb, W_rgb = None, None
        
        if H_rgb is not None and (H_mask, W_mask) == (H_rgb, W_rgb) and (H_rgb, W_rgb) != (H_depth, W_depth):
            print(f"[INFO] Resolution mismatch detected:")
            print(f"[INFO]   Masks: {W_mask}x{H_mask} (matches RGB)")
            print(f"[INFO]   Depth: {W_depth}x{H_depth}")
            print(f"[INFO]   Will scale coordinates from mask space to depth space")
            mask_resolution = (W_mask, H_mask)
        else:
            mask_resolution = None
    else:
        mask_resolution = None
    
    # Handle intrinsics/extrinsics format
    if intrs.ndim == 3 and intrs.shape[1] == 3:
        # [C, 3, 3] -> [C, T, 3, 3]
        intrs_expanded = np.tile(intrs[:, None], (1, T, 1, 1))
    else:
        intrs_expanded = intrs
        
    if extrs.ndim == 3 and extrs.shape[1] in [3, 4]:
        # [C, 3, 4] -> [C, T, 3, 4]
        extrs_expanded = np.tile(extrs[:, None], (1, T, 1, 1))
    else:
        extrs_expanded = extrs
    
    all_query_points = []
    
    print(f"[INFO] Processing instance: {instance_name}")
    if use_first_frame:
        print(f"[INFO]   Using FIRST FRAME mode: ignoring contact frames, using frame 0 for all cameras")
    else:
        print(f"[INFO]   Contact frames: {contact_frames}")
    
    # Create mapping from camera_id string to camera index
    camera_id_to_index = {str(cam_id): c for c, cam_id in enumerate(camera_ids)}
    
    # Determine which cameras to process
    if use_first_frame:
        # Use ALL cameras with frame 0
        cameras_to_process = {cam_id: 0 for cam_id in camera_ids}
    else:
        # Use only cameras where contact was detected
        cameras_to_process = contact_frames
    
    # Process each camera
    for cam_id_str, contact_frame in cameras_to_process.items():
        # Find camera index for this camera_id
        if cam_id_str not in camera_id_to_index:
            print(f"[WARN]   Camera {cam_id_str} not found in camera_ids list, skipping")
            continue
        
        c = camera_id_to_index[cam_id_str]
        
        frame_range = get_frame_range_around_contact(
            contact_frame, frames_before, frames_after, T
        )
        
        print(f"[INFO]   Camera {cam_id_str} (index {c}): contact at frame {contact_frame}, using frames {frame_range}")
        
        # Lift masks for each frame in range
        for t in frame_range:
            mask = masks[c, t]
            
            if not mask.any():
                print(f"[INFO]     Frame {t}: Empty mask, skipping")
                continue
            
            # Lift to 3D (with resolution handling)
            points_3d, _ = lift_mask_to_3d(
                mask=mask,
                depth=depths[c, t],
                intr=intrs_expanded[c, t],
                extr=extrs_expanded[c, t],
                min_depth=min_depth,
                max_depth=max_depth,
                rgb=None,
                mask_resolution=mask_resolution,  # Pass resolution info
            )
            
            if len(points_3d) == 0:
                # Debug: check why no points
                mask_count = mask.sum()
                depth_at_mask = depths[c, t][mask]
                valid_depths = ((depth_at_mask > min_depth) & (depth_at_mask < max_depth)).sum()
                print(f"[INFO]     Frame {t}: No valid 3D points (mask pixels: {mask_count}, valid depths: {valid_depths})")
                continue
            
            # Subsample if too many points
            if max_points_per_frame is not None and len(points_3d) > max_points_per_frame:
                indices = np.random.choice(len(points_3d), max_points_per_frame, replace=False)
                points_3d = points_3d[indices]
                print(f"[INFO]     Frame {t}: Subsampled to {max_points_per_frame} points (from {len(points_3d)})")
            
            # Create query points with frame index
            # Format: [frame_idx, x, y, z]
            frame_indices = np.full((len(points_3d), 1), t, dtype=np.float32)
            query_points_t = np.hstack([frame_indices, points_3d])  # [N, 4]
            
            all_query_points.append(query_points_t)
            print(f"[INFO]     Frame {t}: Added {len(points_3d)} query points")
    
    if not all_query_points:
        print(f"[WARN]   No query points generated for {instance_name}")
        return np.empty((0, 4), dtype=np.float32)
    
    # Concatenate all points
    query_points = np.vstack(all_query_points)
    print(f"[INFO]   Total query points for {instance_name}: {len(query_points)}")
    
    return query_points


def create_query_points_from_hoist_masks(
    npz_path: Path,
    mask_key: str = "hoist_masks",
    frames_before: int = 3,
    frames_after: int = 1,
    min_depth: float = 1e-6,
    max_depth: float = 10.0,
    max_points_per_frame: Optional[int] = None,
    output_path: Optional[Path] = None,
    use_first_frame: bool = False,
    rgbs: Optional[np.ndarray] = None,  # Will be loaded from NPZ if not provided
):
    """
    Create query points from HOISTFormer masks around contact frames.
    
    Args:
        npz_path: Path to NPZ file with masks and contact_frames
        mask_key: Key for masks in NPZ file (default: 'hoist_masks')
        frames_before: Number of frames before contact to include (default: 3)
        frames_after: Number of frames after contact to include (default: 1)
        min_depth: Minimum valid depth threshold
        max_depth: Maximum valid depth threshold
        max_points_per_frame: Optional limit on points per frame (random sampling)
        output_path: Optional output path (default: input_path with _query suffix)
        use_first_frame: If True, use frame 0 for ALL cameras (ignores contact frames)
    """
    print(f"[INFO] ========== Loading NPZ File ==========")
    print(f"[INFO] Input: {npz_path}")
    
    # Load data
    data = np.load(npz_path, allow_pickle=True)
    
    # Check required keys
    required_keys = [mask_key, "depths", "intrs", "extrs", "camera_ids"]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Required key '{key}' not found in NPZ. Available: {list(data.keys())}")
    
    # Load arrays
    hoist_masks = data[mask_key].item()  # Dict of masks
    # Infer contact frames key from mask key
    contact_key = mask_key.replace("_masks", "_contact_frames")
    hoist_contact_frames = data.get(contact_key, None)
    
    if hoist_contact_frames is not None:
        hoist_contact_frames = hoist_contact_frames.item()  # Dict
    else:
        print(f"[WARN] No {contact_key} found, will use first frame for all instances")
        hoist_contact_frames = {}
    
    depths = data["depths"]
    intrs = data["intrs"]
    extrs = data["extrs"]
    camera_ids = [str(cid) for cid in data["camera_ids"]]
    
    # Load RGBs if not provided (for resolution detection)
    if rgbs is None:
        rgbs = data.get("rgbs", None)
    
    # Handle depth format: [C, T, 1, H, W] -> [C, T, H, W]
    if depths.ndim == 5 and depths.shape[2] == 1:
        depths = depths[:, :, 0]
        print(f"[INFO] Reshaped depths from 5D to 4D: {depths.shape}")
    
    print(f"[INFO] Found {len(hoist_masks)} instance(s): {list(hoist_masks.keys())}")
    print(f"[INFO] Contact frames info: {hoist_contact_frames}")
    print(f"[INFO] Frame range: contact - {frames_before} to contact + {frames_after}")
    
    # Get dimensions
    first_mask = list(hoist_masks.values())[0]
    C, T, H, W = first_mask.shape
    print(f"[INFO] Data dimensions: {C} cameras, {T} frames, {H}x{W} resolution")
    
    print(f"\n[INFO] ========== Lifting Masks to 3D Query Points ==========")
    
    all_query_points = []
    
    # Process each instance
    for instance_name, masks in hoist_masks.items():
        # Get contact frames for this instance (if not using first frame mode)
        if use_first_frame:
            contact_frames = {cam_id: 0 for cam_id in camera_ids}
        elif instance_name in hoist_contact_frames:
            contact_frames = hoist_contact_frames[instance_name]
        else:
            # Fallback: use frame 0 for all cameras
            print(f"[WARN] No contact frames for {instance_name}, using frame 0")
            contact_frames = {cam_id: 0 for cam_id in camera_ids}
        
        # Lift to query points (with RGB for resolution detection)
        query_points = lift_instance_masks_to_query_points(
            instance_name=instance_name,
            masks=masks,
            contact_frames=contact_frames,
            depths=depths,
            intrs=intrs,
            extrs=extrs,
            camera_ids=camera_ids,
            frames_before=frames_before,
            frames_after=frames_after,
            min_depth=min_depth,
            max_depth=max_depth,
            max_points_per_frame=max_points_per_frame,
            use_first_frame=use_first_frame,
            rgbs=rgbs,  # Pass RGB for resolution detection
        )
        
        if len(query_points) > 0:
            all_query_points.append(query_points)
    
    if not all_query_points:
        raise ValueError("No query points generated from any instance!")
    
    # Combine all query points
    query_points_final = np.vstack(all_query_points)
    
    print(f"\n[INFO] ========== Query Points Summary ==========")
    print(f"[INFO] Total query points: {len(query_points_final)}")
    print(f"[INFO] Shape: {query_points_final.shape}")
    print(f"[INFO] Frame range in query points: [{query_points_final[:, 0].min():.0f}, {query_points_final[:, 0].max():.0f}]")
    print(f"[INFO] Spatial range:")
    print(f"[INFO]   X: [{query_points_final[:, 1].min():.3f}, {query_points_final[:, 1].max():.3f}]")
    print(f"[INFO]   Y: [{query_points_final[:, 2].min():.3f}, {query_points_final[:, 2].max():.3f}]")
    print(f"[INFO]   Z: [{query_points_final[:, 3].min():.3f}, {query_points_final[:, 3].max():.3f}]")
    
    # Determine output path
    if output_path is None:
        output_path = npz_path.parent / f"{npz_path.stem}_query.npz"
    
    print(f"\n[INFO] ========== Saving Output NPZ ==========")
    print(f"[INFO] Output: {output_path}")
    
    # Prepare output payload
    payload = dict(data)
    payload["query_points"] = query_points_final
    payload["query_generation_config"] = {
        "frames_before": frames_before,
        "frames_after": frames_after,
        "min_depth": min_depth,
        "max_depth": max_depth,
        "max_points_per_frame": max_points_per_frame,
    }
    
    # Save
    np.savez_compressed(output_path, **payload)
    
    print(f"[INFO] Saved successfully!")
    print(f"[INFO] NPZ contains:")
    print(f"[INFO]   - All original data")
    print(f"[INFO]   - query_points: {query_points_final.shape}")
    print(f"[INFO]   - query_generation_config: metadata about generation")
    print(f"\n[INFO] ========== Done! ==========")
    print(f"[INFO] Use this NPZ with demo.py by setting --sample-path {output_path}")
    
    return query_points_final


def visualize_query_points_rerun(
    query_points: np.ndarray,
    npz_path: Path,
    output_path: Optional[Path] = None,
    spawn: bool = False,
    fps: float = 12.0,
):
    """
    Visualize query points in Rerun along with RGB point clouds and cameras.
    
    Args:
        query_points: [N, 4] array with columns [frame_idx, x, y, z]
        npz_path: Path to original NPZ file (to load RGB, depth, cameras)
        output_path: Optional path to save .rrd file
        spawn: Whether to spawn Rerun viewer
        fps: Frame rate for temporal visualization
    """
    print(f"\n[INFO] ========== Visualizing Query Points in Rerun ==========")
    
    # Load data for visualization
    data = np.load(npz_path, allow_pickle=True)
    
    rgbs = data.get("rgbs", None)
    depths = data["depths"]
    intrs = data["intrs"]
    extrs = data["extrs"]
    camera_ids = [str(cid) for cid in data["camera_ids"]]
    
    # Handle depth format
    if depths.ndim == 5 and depths.shape[2] == 1:
        depths = depths[:, :, 0]
    
    # Initialize Rerun
    recording_name = f"query_points_{npz_path.stem}"
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
    
    print(f"[INFO] Logging RGB point clouds and camera frustums...")
    
    # Prepare RGB data for visualization
    if rgbs is not None:
        # Handle channel format
        if rgbs.ndim == 5 and rgbs.shape[-1] == 3:
            # [C, T, H, W, 3] -> [C, T, 3, H, W]
            rgbs_viz = np.moveaxis(rgbs, -1, 2)
        else:
            rgbs_viz = rgbs
        
        # Add channel dimension to depth if needed
        depths_viz = depths if depths.ndim == 5 else depths[:, :, None, :, :]
        
        # Convert to torch tensors with batch dimension
        rgbs_tensor = torch.from_numpy(rgbs_viz).float().unsqueeze(0)
        depths_tensor = torch.from_numpy(depths_viz).float().unsqueeze(0)
        intrs_tensor = torch.from_numpy(intrs).float().unsqueeze(0)
        extrs_tensor = torch.from_numpy(extrs).float().unsqueeze(0)
        
        # Log RGB point clouds and camera frustums
        log_pointclouds_to_rerun(
            dataset_name="query_points_scene",
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
    else:
        print(f"[WARN] No RGB data available, skipping point cloud visualization")
    
    # Log query points
    print(f"[INFO] Logging {len(query_points)} query points...")
    
    # Group query points by frame
    unique_frames = np.unique(query_points[:, 0]).astype(int)
    
    for frame_idx in unique_frames:
        # Get points for this frame
        frame_mask = query_points[:, 0] == frame_idx
        frame_points = query_points[frame_mask, 1:4]  # [N, 3] - x, y, z
        
        # Set time
        time_seconds = frame_idx / fps
        rr.set_time_seconds("frame", time_seconds)
        
        # Log query points with distinctive color (bright yellow)
        rr.log(
            "world/query_points",
            rr.Points3D(
                positions=frame_points,
                colors=[255, 255, 0],  # Bright yellow
                radii=0.015,  # Larger than scene points for visibility
            ),
        )
    
    print(f"[INFO] Logged query points across {len(unique_frames)} frames")
    print(f"[INFO] Query points appear in YELLOW with larger radius for visibility")
    
    # Save recording if output path specified
    if output_path is None:
        output_path = npz_path.parent / f"{npz_path.stem}_query_visualization.rrd"
    
    print(f"\n[INFO] Saving Rerun recording to {output_path}...")
    rr.save(str(output_path))
    
    print(f"[INFO] Done! View with: rerun {output_path}")
    if not spawn:
        print(f"[INFO] Or use --spawn flag to open viewer automatically")



def main():
    parser = argparse.ArgumentParser(
        description="Create query points from HOISTFormer masks around contact frames",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--npz",
        type=Path,
        required=True,
        help="Path to NPZ file with masks and contact_frames",
    )
    
    parser.add_argument(
        "--key",
        type=str,
        default="hoist_masks",
        help="Key for masks in NPZ file (default: 'hoist_masks')",
    )
    
    parser.add_argument(
        "--frames-before",
        type=int,
        default=3,
        help="Number of frames before contact to include (default: 3)",
    )
    
    parser.add_argument(
        "--frames-after",
        type=int,
        default=1,
        help="Number of frames after contact to include (default: 1)",
    )
    
    parser.add_argument(
        "--min-depth",
        type=float,
        default=1e-6,
        help="Minimum valid depth in meters (default: 1e-6 to exclude only zero/invalid depths for sparse depth)",
    )
    
    parser.add_argument(
        "--max-depth",
        type=float,
        default=10.0,
        help="Maximum valid depth in meters (default: 10.0)",
    )
    
    parser.add_argument(
        "--max-points-per-frame",
        type=int,
        default=None,
        help="Maximum number of points per frame (random sampling if exceeded). Default: no limit",
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output NPZ path (default: input_path with _query suffix)",
    )
    
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate Rerun visualization of query points with scene context",
    )
    
    parser.add_argument(
        "--viz-output",
        type=Path,
        default=None,
        help="Output path for Rerun recording (default: input_path with _query_visualization.rrd suffix)",
    )
    
    parser.add_argument(
        "--spawn",
        action="store_true",
        help="Spawn Rerun viewer automatically when visualizing",
    )
    
    parser.add_argument(
        "--fps",
        type=float,
        default=12.0,
        help="Frame rate for visualization (default: 12.0)",
    )
    
    parser.add_argument(
        "--use-first-frame",
        action="store_true",
        help="Use frame 0 for ALL cameras (ignores contact frames). Useful for tracking initialization.",
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.npz.exists():
        raise FileNotFoundError(f"Input file not found: {args.npz}")
    
    # Create query points
    query_points = create_query_points_from_hoist_masks(
        npz_path=args.npz,
        mask_key=args.key,
        frames_before=args.frames_before,
        frames_after=args.frames_after,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        max_points_per_frame=args.max_points_per_frame,
        output_path=args.output,
        use_first_frame=args.use_first_frame,
    )
    
    # Visualize if requested
    if args.visualize:
        visualize_query_points_rerun(
            query_points=query_points,
            npz_path=args.npz,
            output_path=args.viz_output,
            spawn=args.spawn,
            fps=args.fps,
        )


if __name__ == "__main__":
    main()
