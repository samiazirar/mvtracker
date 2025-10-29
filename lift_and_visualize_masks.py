#!/usr/bin/env python3
"""
Lift and Visualize 2D Masks in 3D using Rerun

This script visualizes the full scene including RGB point clouds, camera frustums,
and 2D masks lifted to 3D. The masks appear as a togglable layer in Rerun, allowing
you to compare them with the RGB point clouds.

Global ID Mode (NEW):
---------------------
Use --use-global-ids to match masks across cameras and assign consistent IDs.
This is useful when the same object is tracked by multiple cameras but has
different local IDs in each camera's output.

Example: Camera 0 has IDs [0, 1, 2] and Camera 1 has IDs [0, 1]. The global ID
assignment will match them based on 3D position, so if Camera 1's ID 0 is close
to Camera 0's ID 2 in 3D space, they'll both get the same global ID.

Usage Examples:
---------------

# 1. Visualize with GLOBAL IDs (match masks across cameras)
python lift_and_visualize_masks.py \
    --npz third_party/HOISTFormer/hoist_output/task_0045_user_0020_scene_0004_cfg_0006_human_processed_hand_tracked_hoist.npz \
    --mask-key hoist_masks \
    --use-global-ids \
    --distance-threshold 0.15 \
    --max-frames 111

# 2. Visualize SAM2 tracking output with global IDs
python lift_and_visualize_masks.py \
    --npz third_party/HOISTFormer/sam2_tracking_output/task_0045_user_0020_scene_0004_cfg_0006_human_processed_hand_tracked_hoist_sam2.npz \
    --mask-key sam2_predictions \
    --use-global-ids \
    --distance-threshold 0.20 \
    --max-frames 111

# 3. Original per-camera mode (without global IDs)
python lift_and_visualize_masks.py \
    --npz data/human_high_res_filtered/task_0045_user_0020_scene_0004_cfg_0006_human_processed_hand_tracked.npz \
    --mask-key sam_hand_masks \
    --color 255 0 255 \
    --max-frames 50

# 4. Visualize with RGB colors from images (match mask colors to scene)
python lift_and_visualize_masks.py \
    --npz data/human_high_res_filtered/task_0045_user_0020_scene_0004_cfg_0006_human_processed_hand_tracked.npz \
    --mask-key sam_hand_masks \
    --use-rgb-colors \
    --max-frames 50

Features:
---------
- RGB point clouds from depth maps (togglable)
- Camera frustums showing camera poses
- 3D mask visualization as point clouds (togglable)
- World coordinate axes (X=Red, Y=Green, Z=Blue)
- Temporal playback with configurable FPS
- Global ID assignment across cameras with distance threshold
"""

import argparse
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import rerun as rr

from utils.mask_lifting_utils import visualize_masks_batch, visualize_masks_with_global_ids
from mvtracker.utils.visualizer_rerun import log_pointclouds_to_rerun


def load_data_from_npz(npz_path: Path, mask_key: str = "masks"):
    """
    Load mask data and camera parameters from NPZ file.
    
    Expected NPZ format:
        - masks or {mask_key}: [C, T, H, W] binary masks OR dict of masks {name: [C, T, H, W]}
        - depths: [C, T, H, W] depth maps
        - intrs: [C, T, 3, 3] or [C, 3, 3] intrinsics
        - extrs: [C, T, 3, 4] or [C, 3, 4] extrinsics
        - rgbs: [C, T, 3, H, W] or [C, T, H, W, 3] optional RGB images
        - camera_ids: [C] optional camera identifiers
    
    Returns:
        Dictionary with loaded data. 'masks' will be a dict {mask_name: mask_array}
    """
    print(f"[INFO] Loading data from {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)
    # Load masks
    if mask_key not in data:
        available_keys = list(data.keys())
        raise KeyError(
            f"Mask key '{mask_key}' not found in NPZ. "
            f"Available keys: {available_keys}"
        )
    
    masks_raw = data[mask_key]
    
    # Handle both single mask array and dictionary of masks
    if isinstance(masks_raw, dict) or (isinstance(masks_raw, np.ndarray) and masks_raw.dtype == object and isinstance(masks_raw.item(), dict)):
        # Dictionary of masks
        if isinstance(masks_raw, np.ndarray):
            masks_raw = masks_raw.item()
        masks_dict = {}
        for name, mask_array in masks_raw.items():
            if mask_array.dtype != bool:
                print(f"[INFO] Converting mask '{name}' from {mask_array.dtype} to bool")
                mask_array = mask_array > 0
            masks_dict[name] = mask_array
            print(f"[INFO] Loaded mask '{name}' with shape: {mask_array.shape}")
        masks = masks_dict
    else:
        # Single mask array
        print(f"[INFO] Loaded masks with shape: {masks_raw.shape}")
        # Ensure masks are boolean
        if masks_raw.dtype != bool:
            print(f"[INFO] Converting masks from {masks_raw.dtype} to bool")
            masks_raw = masks_raw > 0
        masks = {"default": masks_raw}
    
    # Load required data
    required_keys = ["depths", "intrs", "extrs"]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Required key '{key}' not found in NPZ")
    
    depths = data["depths"]
    intrs = data["intrs"]
    extrs = data["extrs"]
    
    # Handle different depth formats
    if depths.ndim == 5 and depths.shape[2] == 1:
        # [C, T, 1, H, W] -> [C, T, H, W]
        depths = depths[:, :, 0]
        print(f"[INFO] Reshaped depths to {depths.shape}")
    
    # Load optional data
    rgbs = None
    if "rgbs" in data:
        rgbs = data["rgbs"]
        # Handle channel-first format [C, T, 3, H, W]
        if rgbs.ndim == 5 and rgbs.shape[2] == 3:
            rgbs = np.moveaxis(rgbs, 2, -1)  # -> [C, T, H, W, 3]
            print(f"[INFO] Converted RGB to channel-last format: {rgbs.shape}")
    
    camera_ids = None
    if "camera_ids" in data:
        camera_ids = [str(cid) for cid in data["camera_ids"]]
        print(f"[INFO] Found camera IDs: {camera_ids}")
    
    # Validate shapes
    first_mask = list(masks.values())[0]
    C, T = first_mask.shape[0], first_mask.shape[1]
    print(f"[INFO] Data summary:")
    print(f"[INFO]   Number of mask types: {len(masks)}")
    print(f"[INFO]   Mask names: {list(masks.keys())}")
    print(f"[INFO]   Cameras: {C}")
    print(f"[INFO]   Frames: {T}")
    print(f"[INFO]   Resolution: {first_mask.shape[2]}x{first_mask.shape[3]}")
    print(f"[INFO]   Masks dtype: {first_mask.dtype}")
    print(f"[INFO]   Depths range: [{depths.min():.3f}, {depths.max():.3f}]")
    
    return {
        "masks": masks,
        "depths": depths,
        "intrs": intrs,
        "extrs": extrs,
        "rgbs": rgbs,
        "camera_ids": camera_ids,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Lift 2D masks to 3D and visualize in Rerun",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Input/output
    parser.add_argument(
        "--npz",
        type=Path,
        required=True,
        help="Path to NPZ file containing masks and camera data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .rrd file path (default: based on input name)",
    )
    
    # Data selection
    parser.add_argument(
        "--mask-key",
        type=str,
        default="masks",
        help="Key for masks in NPZ file (default: 'masks')",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit number of frames to visualize",
    )
    
    # Visualization options
    parser.add_argument(
        "--entity-path",
        type=str,
        default="world/masks",
        help="Base entity path in Rerun (default: 'world/masks')",
    )
    parser.add_argument(
        "--color",
        type=int,
        nargs=3,
        default=None,
        metavar=("R", "G", "B"),
        help="Fixed RGB color for mask points (0-255). Example: 255 0 255 for magenta",
    )
    parser.add_argument(
        "--use-rgb-colors",
        action="store_true",
        help="Use RGB image colors for mask points (if available)",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.005,
        help="Point radius for visualization (default: 0.005)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frame rate for temporal visualization (default: 30.0)",
    )
    
    # Depth filtering
    parser.add_argument(
        "--min-depth",
        type=float,
        default=0.0,
        help="Minimum valid depth in meters (default: 0.0)",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=10.0,
        help="Maximum valid depth in meters (default: 10.0)",
    )
    
    # Rerun options
    parser.add_argument(
        "--spawn",
        action="store_true",
        help="Spawn Rerun viewer automatically",
    )
    
    # Global ID options
    parser.add_argument(
        "--use-global-ids",
        action="store_true",
        help="Assign global IDs to masks across cameras based on 3D proximity",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=0.15,
        help="Max distance (meters) to match masks across cameras (default: 0.15)",
    )
    parser.add_argument(
        "--frame-for-matching",
        type=int,
        default=0,
        help="Which frame to use for computing global ID assignments (default: 0)",
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.npz.exists():
        raise FileNotFoundError(f"Input file not found: {args.npz}")
    
    # Set output path
    if args.output is None:
        args.output = args.npz.parent / f"{args.npz.stem}_masks_3d.rrd"
    
    # Load data
    data = load_data_from_npz(args.npz, mask_key=args.mask_key)
    # Prepare color
    color = None
    if args.color is not None:
        color = np.array(args.color, dtype=np.uint8)
        print(f"[INFO] Using fixed color: RGB{tuple(color)}")
    
    # Prepare RGB
    rgbs = None
    if args.use_rgb_colors:
        if data["rgbs"] is not None:
            rgbs = data["rgbs"]
            print(f"[INFO] Using RGB colors from images")
        else:
            print(f"[WARN] --use-rgb-colors specified but no RGB data in NPZ")
    
    # Initialize Rerun
    rr.init("mask_3d_visualization", recording_id=args.npz.stem, spawn=args.spawn)
    
    # Set coordinate system
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    
    # Visualize world axes
    rr.log(
        "world/axes",
        rr.Arrows3D(
            origins=[[0, 0, 0]],
            vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],  # X=Red, Y=Green, Z=Blue
            radii=0.01,
        ),
        static=True,
    )
    
    # Log RGB point clouds and camera frustums (like demo.py)
    print(f"\n[INFO] ========== Logging RGB Point Clouds and Cameras ==========")
    
    # Prepare data for visualization (convert to torch tensors as expected by visualizer)
    rgbs_viz = data["rgbs"] if data["rgbs"] is not None else None
    if rgbs_viz is not None:
        # Handle channel format
        if rgbs_viz.ndim == 5 and rgbs_viz.shape[-1] == 3:
            # [C, T, H, W, 3] -> [C, T, 3, H, W]
            rgbs_viz = np.moveaxis(rgbs_viz, -1, 2)
    else:
        print("[WARN] No RGB data available, skipping RGB point cloud visualization")
    
    depths_viz = data["depths"]
    if depths_viz.ndim == 4:
        # [C, T, H, W] -> [C, T, 1, H, W]
        depths_viz = depths_viz[:, :, None, :, :]
    
    if rgbs_viz is not None:
        # Convert to torch tensors (add batch dimension as expected by visualizer)
        rgbs_tensor = torch.from_numpy(rgbs_viz).float().unsqueeze(0)
        depths_tensor = torch.from_numpy(depths_viz).float().unsqueeze(0)
        intrs_tensor = torch.from_numpy(data["intrs"]).float().unsqueeze(0)
        extrs_tensor = torch.from_numpy(data["extrs"]).float().unsqueeze(0)
        
        # Log point clouds and camera frustums
        log_pointclouds_to_rerun(
            dataset_name="rh20t_mask_visualization",
            datapoint_idx=0,
            rgbs=rgbs_tensor,
            depths=depths_tensor,
            intrs=intrs_tensor,
            extrs=extrs_tensor,
            camera_ids=data["camera_ids"] if data["camera_ids"] is not None else None,
            log_rgb_pointcloud=True,
            log_camera_frustrum=True,
            radii=-0.95
        )
        print("[INFO] Logged RGB point clouds and camera frustums")
    
    # Visualize masks as togglable layers
    print(f"\n[INFO] ========== Visualizing Masks (Togglable Layers) ==========")
    
    masks_dict = data["masks"]
    all_stats = {}
    
    if args.use_global_ids:
        # Use global ID assignment for consistent IDs across cameras
        print(f"[INFO] Using GLOBAL ID mode (matching across cameras)")
        print(f"[INFO]   Distance threshold: {args.distance_threshold}m")
        print(f"[INFO]   Frame for matching: {args.frame_for_matching}")
        
        stats = visualize_masks_with_global_ids(
            masks_dict=masks_dict,
            depths=data["depths"],
            intrs=data["intrs"],
            extrs=data["extrs"],
            entity_base_path=args.entity_path,
            camera_ids=data["camera_ids"],
            rgbs=rgbs,
            radius=args.radius,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            fps=args.fps,
            max_frames=args.max_frames,
            distance_threshold=args.distance_threshold,
            frame_for_matching=args.frame_for_matching,
        )
        
        all_stats = stats["mask_stats"]
        print(f"\n[INFO] Global ID Mapping:")
        for mask_name, id_mapping in stats["global_id_mapping"].items():
            print(f"[INFO]   {mask_name}:")
            for cam_idx, mapping in id_mapping.items():
                print(f"[INFO]     Camera {cam_idx}: {mapping}")
    
    else:
        # Original per-camera visualization
        print(f"[INFO] Using PER-CAMERA mode (separate layers per camera)")
        
        for mask_name, mask_array in masks_dict.items():
            print(f"\n[INFO] --- Processing mask type: '{mask_name}' ---")
            
            # Each mask type gets its own entity path
            mask_entity_path = f"{args.entity_path}/{mask_name}"
            
            stats = visualize_masks_batch(
                masks=mask_array,
                depths=data["depths"],
                intrs=data["intrs"],
                extrs=data["extrs"],
                entity_base_path=mask_entity_path,
                camera_ids=data["camera_ids"],
                rgbs=rgbs,
                color=color,
                radius=args.radius,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
                fps=args.fps,
                max_frames=args.max_frames,
            )
            
            all_stats[mask_name] = stats
            print(f"[INFO] Statistics for '{mask_name}':")
            print(f"[INFO]   Total points: {stats['total_points']}")
            print(f"[INFO]   Cameras: {stats['num_cameras']}")
            print(f"[INFO]   Frames: {stats['num_frames']}")
            print(f"[INFO]   Points per camera: {stats['points_per_camera']}")
    
    print(f"\n[INFO] ========== Visualization Complete ==========")
    print(f"[INFO] Processed {len(masks_dict)} mask type(s): {list(masks_dict.keys())}")
    if args.use_global_ids:
        total_points_all = sum(s['total_points'] for s in all_stats.values())
    else:
        total_points_all = sum(s['total_points'] for s in all_stats.values())
    print(f"[INFO] Total points across all masks: {total_points_all}")
    
    # Save recording
    print(f"\n[INFO] Saving visualization to {args.output}...")
    rr.save(str(args.output))
    print(f"[INFO] Done! View with: rerun {args.output} --web-viewer")
    
    if not args.spawn:
        print(f"[INFO] Or spawn viewer now with --spawn flag")


if __name__ == "__main__":
    main()
