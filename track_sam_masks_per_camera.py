#!/usr/bin/env python3
"""
Track SAM Masks Using SpatialTrackerV2 Per Camera

This script takes SAM2 masks from an NPZ file and tracks all points within each mask
across time using SpatialTrackerV2. It operates per-camera, generating 2D tracks that
can later be lifted to 3D.

Workflow:
1. Load NPZ with sam2_masks [C, T, H, W] and camera data
2. For each camera separately:
   - Extract all pixel coordinates from masks in first frame
   - Use SpatialTrackerV2 to track those points through time
   - Save 2D tracks with visibility flags
3. Save tracks per camera to NPZ file

Output Format:
    - tracks_2d: Dict {camera_id: [N, T, 2]} - 2D pixel coordinates
    - visibility: Dict {camera_id: [N, T]} - boolean visibility flags
    - track_instance_ids: Dict {camera_id: [N]} - which instance each track belongs to
    - All original data from input NPZ

Usage:
    python track_sam_masks_per_camera.py \
        --npz third_party/HOISTFormer/sam2_tracking_output/task_0045_sam2.npz \
        --mask-key sam2_masks \
        --output sam_tracks_per_camera.npz \
        --track-mode offline \
        --max-points-per-mask 500
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import cv2
from tqdm import tqdm
import sys

# Add spatialtrackerv2 to path
sys.path.insert(0, str(Path(__file__).parent / "spatialtrackerv2"))

# SpatialTrackerV2 imports
from models.SpaTrackV2.models.predictor import Predictor
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image
from models.SpaTrackV2.models.vggt4track.utils.pose_enc import pose_encoding_to_extri_intri


def extract_mask_points(
    masks: Dict[str, np.ndarray],  # {instance_name: [T, H, W]}
    frame_idx: int = 0,
    max_points_per_mask: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract all pixel coordinates from masks at a given frame.
    
    Args:
        masks: Dictionary of instance masks {instance_name: [T, H, W]}
        frame_idx: Which frame to extract points from (default: 0)
        max_points_per_mask: Optional limit on points per mask (random sampling)
    
    Returns:
        points: [N, 2] array of (x, y) pixel coordinates
        instance_ids: [N] array mapping each point to instance index
    """
    all_points = []
    all_instance_ids = []
    
    for instance_idx, (instance_name, mask_array) in enumerate(masks.items()):
        # Get mask for this frame [H, W]
        mask = mask_array[frame_idx]
        
        if not mask.any():
            print(f"[WARN] Instance {instance_name} has empty mask at frame {frame_idx}")
            continue
        
        # Get all pixel coordinates where mask is True
        y_coords, x_coords = np.where(mask)
        points = np.stack([x_coords, y_coords], axis=1)  # [N, 2] - (x, y)
        
        # Subsample if too many points
        if max_points_per_mask is not None and len(points) > max_points_per_mask:
            indices = np.random.choice(len(points), max_points_per_mask, replace=False)
            points = points[indices]
        
        # Assign instance ID
        instance_ids = np.full(len(points), instance_idx, dtype=np.int32)
        
        all_points.append(points)
        all_instance_ids.append(instance_ids)
        
        print(f"[INFO] Instance {instance_name}: extracted {len(points)} points")
    
    if not all_points:
        raise ValueError(f"No points extracted from any mask at frame {frame_idx}")
    
    points = np.vstack(all_points)
    instance_ids = np.concatenate(all_instance_ids)
    
    return points, instance_ids


def track_points_spatialtrackerv2(
    video_tensor: torch.Tensor,  # [T, 3, H, W] - RGB video
    depth_tensor: Optional[torch.Tensor],  # [T, H, W] or None
    intrs: Optional[torch.Tensor],  # [T, 3, 3] or [3, 3] or None
    extrs: Optional[torch.Tensor],  # [T, 4, 4] or [4, 4] or None
    query_points: np.ndarray,  # [N, 2] - (x, y) at frame 0
    track_mode: str = "offline",
    fps: int = 1,
    track_num: int = 756,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Track query points using SpatialTrackerV2.
    

        video_tensor: RGB video [T, 3, H, W]
        depth_tensor: Optional depth maps [T, H, W]
        intrs: Optional intrinsics [T, 3, 3] or [3, 3]
        extrs: Optional extrinsics [T, 4, 4] or [4, 4]
        query_points: Starting points [N, 2] - (x, y) at frame 0
        track_mode: "offline" or "online"
        fps: Frame rate
        track_num: Max number of VO points for tracker
        device: "cuda" or "cpu"
    
    Returns:
        tracks_2d: [T, N, 2] - tracked 2D positions
        visibility: [T, N] - boolean visibility flags
    """
    print(f"[INFO] Loading SpatialTrackerV2 model ({track_mode} mode)...")
    
    # Load model
    if track_mode == "offline":
        model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline")
    else:
        model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Online")
    
    model.spatrack.track_num = track_num
    model.eval()
    model.to(device)
    
    # Prepare query format: [N, 3] with columns [frame_idx, x, y]
    N = len(query_points)
    query_xyt = np.zeros((N, 3), dtype=np.float32)
    query_xyt[:, 0] = 0  # All queries start at frame 0
    query_xyt[:, 1:] = query_points  # (x, y)
    
    print(f"[INFO] Tracking {N} points through {len(video_tensor)} frames...")
    print(f"[INFO] Note: SpatialTrackerV2 may add additional tracking points automatically")
    
    # Run tracking with full_point=True to avoid uncertainty filtering issues
    with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
        (
            c2w_traj, intrs_pred, point_map, conf_depth,
            track3d_pred, track2d_pred, vis_pred, conf_pred, video
        ) = model.forward(
            video_tensor,
            depth=depth_tensor,
            intrs=intrs,
            extrs=extrs,
            queries=query_xyt,
            fps=fps,
            full_point=True,  # Set to True to avoid uncertainty filtering issues
            iters_track=4,
            query_no_BA=True,
            fixed_cam=False,
            stage=1,
            unc_metric=None,
            support_frame=len(video_tensor) - 1,
            replace_ratio=0.2
        )
    
    # Extract tracks - note: may have more tracks than query points
    tracks_2d = track2d_pred.cpu().numpy()  # [T, N_total, 3] - (x, y, confidence)
    visibility = vis_pred.cpu().numpy()  # [T, N_total]
    
    # Keep only the tracks corresponding to our query points (first N tracks)
    tracks_2d = tracks_2d[:, :N, :]
    visibility = visibility[:, :N]
    
    # Keep only x, y coordinates (drop confidence)
    tracks_2d = tracks_2d[:, :, :2]  # [T, N, 2]
    
    print(f"[INFO] Tracking complete. Tracks shape: {tracks_2d.shape}")
    
    return tracks_2d, visibility


def track_sam_masks_per_camera(
    npz_path: Path,
    mask_key: str = "sam2_masks",
    track_mode: str = "offline",
    max_points_per_mask: Optional[int] = None,
    fps: int = 1,
    track_num: int = 756,
    device: str = "cuda",
    output_path: Optional[Path] = None,
):
    """
    Track SAM masks using SpatialTrackerV2 per camera.
    
    Args:
        npz_path: Path to NPZ file with masks and camera data
        mask_key: Key for masks in NPZ (default: 'sam2_masks')
        track_mode: "offline" or "online"
        max_points_per_mask: Optional limit on points per mask
        fps: Frame rate
        track_num: Max VO points for tracker
        device: "cuda" or "cpu"
        output_path: Optional output path
    """
    print(f"[INFO] ========== Loading NPZ File ==========")
    print(f"[INFO] Input: {npz_path}")
    
    # Load data
    data = np.load(npz_path, allow_pickle=True)
    
    # Check required keys
    required_keys = [mask_key, "depths", "intrs", "extrs", "rgbs", "camera_ids"]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Required key '{key}' not found in NPZ. Available: {list(data.keys())}")
    
    # Load arrays
    masks_dict = data[mask_key].item()  # Dict of masks
    depths = data["depths"]  # [C, T, H, W] or [C, T, 1, H, W]
    intrs = data["intrs"]  # [C, T, 3, 3] or [C, 3, 3]
    extrs = data["extrs"]  # [C, T, 3, 4] or [C, 3, 4]
    rgbs = data["rgbs"]  # [C, T, H, W, 3] or [C, T, 3, H, W]
    camera_ids = [str(cid) for cid in data["camera_ids"]]
    
    # Handle depth format
    if depths.ndim == 5 and depths.shape[2] == 1:
        depths = depths[:, :, 0]
    
    # Handle RGB format
    if rgbs.ndim == 5 and rgbs.shape[2] == 3:
        # [C, T, 3, H, W] -> [C, T, H, W, 3]
        rgbs = np.moveaxis(rgbs, 2, -1)
    
    C, T, H, W = depths.shape
    print(f"[INFO] Data dimensions: {C} cameras, {T} frames, {H}x{W} resolution")
    print(f"[INFO] Found {len(masks_dict)} instance(s): {list(masks_dict.keys())}")
    
    print(f"\n[INFO] ========== Tracking Masks Per Camera ==========")
    
    tracks_2d_all = {}
    visibility_all = {}
    track_instance_ids_all = {}
    
    # Process each camera separately
    for c, cam_id in enumerate(tqdm(camera_ids, desc="Processing cameras")):
        print(f"\n[INFO] Camera {cam_id} ({c+1}/{C})")
        
        # Extract masks for this camera
        masks_cam = {
            name: mask_array[c]  # [T, H, W]
            for name, mask_array in masks_dict.items()
        }
        
        # Extract query points from first frame
        try:
            query_points, instance_ids = extract_mask_points(
                masks_cam,
                frame_idx=0,
                max_points_per_mask=max_points_per_mask,
            )
        except ValueError as e:
            print(f"[WARN] {e}")
            print(f"[WARN] Skipping camera {cam_id}")
            continue
        
        # Prepare video tensor for this camera
        video_cam = rgbs[c]  # [T, H, W, 3]
        video_tensor = torch.from_numpy(video_cam).float()  # [T, H, W, 3]
        video_tensor = video_tensor.permute(0, 3, 1, 2)  # [T, 3, H, W]
        
        # Prepare depth/intrs/extrs for this camera
        depth_cam = depths[c] if depths is not None else None  # [T, H, W]
        depth_tensor = torch.from_numpy(depth_cam).float() if depth_cam is not None else None
        
        # Handle intrinsics/extrinsics format
        if intrs.ndim == 3 and intrs.shape[1] == 3:
            # [C, 3, 3] -> [3, 3]
            intrs_cam = intrs[c]
        else:
            intrs_cam = intrs[c]  # [T, 3, 3]
        intrs_tensor = torch.from_numpy(intrs_cam).float()
        
        if extrs.ndim == 3 and extrs.shape[1] in [3, 4]:
            # [C, 3, 4] -> [3, 4]
            extrs_cam = extrs[c]
        else:
            extrs_cam = extrs[c]  # [T, 3, 4]
        
        # Convert extrinsics to 4x4 if needed
        if extrs_cam.ndim == 2 and extrs_cam.shape == (3, 4):
            # Single [3, 4] -> [4, 4]
            extrs_4x4 = np.eye(4)
            extrs_4x4[:3] = extrs_cam
            extrs_tensor = torch.from_numpy(extrs_4x4).float()
        elif extrs_cam.ndim == 3 and extrs_cam.shape[1:] == (3, 4):
            # [T, 3, 4] -> [T, 4, 4]
            T_cam = extrs_cam.shape[0]
            extrs_4x4 = np.tile(np.eye(4)[None], (T_cam, 1, 1))
            extrs_4x4[:, :3] = extrs_cam
            extrs_tensor = torch.from_numpy(extrs_4x4).float()
        else:
            extrs_tensor = torch.from_numpy(extrs_cam).float()
        
        # Track points
        tracks_2d, visibility = track_points_spatialtrackerv2(
            video_tensor=video_tensor,
            depth_tensor=depth_tensor,
            intrs=intrs_tensor,
            extrs=extrs_tensor,
            query_points=query_points,
            track_mode=track_mode,
            fps=fps,
            track_num=track_num,
            device=device,
        )
        
        # Store results
        tracks_2d_all[cam_id] = tracks_2d  # [T, N, 2]
        visibility_all[cam_id] = visibility  # [T, N]
        track_instance_ids_all[cam_id] = instance_ids  # [N]
        
        print(f"[INFO] Camera {cam_id}: tracked {len(query_points)} points")
    
    print(f"\n[INFO] ========== Saving Results ==========")
    
    # Determine output path
    if output_path is None:
        output_path = npz_path.parent / f"{npz_path.stem}_tracks_per_camera.npz"
    
    # Prepare output
    payload = dict(data)
    payload["tracks_2d"] = tracks_2d_all
    payload["visibility"] = visibility_all
    payload["track_instance_ids"] = track_instance_ids_all
    payload["tracking_config"] = {
        "track_mode": track_mode,
        "max_points_per_mask": max_points_per_mask,
        "fps": fps,
        "track_num": track_num,
    }
    
    # Save
    np.savez_compressed(output_path, **payload)
    
    print(f"[INFO] Output: {output_path}")
    print(f"[INFO] Tracked {len(tracks_2d_all)} cameras")
    print(f"\n[INFO] ========== Done! ==========")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Track SAM masks using SpatialTrackerV2 per camera",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--npz",
        type=Path,
        required=True,
        help="Path to NPZ file with masks and camera data",
    )
    
    parser.add_argument(
        "--mask-key",
        type=str,
        default="sam2_masks",
        help="Key for masks in NPZ file (default: 'sam2_masks')",
    )
    
    parser.add_argument(
        "--track-mode",
        choices=["offline", "online"],
        default="offline",
        help="Tracking mode (default: offline)",
    )
    
    parser.add_argument(
        "--max-points-per-mask",
        type=int,
        default=None,
        help="Maximum points per mask (random sampling if exceeded). Default: no limit",
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        default=1,
        help="Frame rate (default: 1)",
    )
    
    parser.add_argument(
        "--track-num",
        type=int,
        default=756,
        help="Max VO points for tracker (default: 756)",
    )
    
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to use (default: cuda)",
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output NPZ path (default: input_path with _tracks_per_camera suffix)",
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.npz.exists():
        raise FileNotFoundError(f"Input file not found: {args.npz}")
    
    # Run tracking
    track_sam_masks_per_camera(
        npz_path=args.npz,
        mask_key=args.mask_key,
        track_mode=args.track_mode,
        max_points_per_mask=args.max_points_per_mask,
        fps=args.fps,
        track_num=args.track_num,
        device=args.device,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
