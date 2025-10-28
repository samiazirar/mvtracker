#!/usr/bin/env python3
"""
Mask Lifting Utilities for RH20T Data

This module provides functions for lifting 2D segmentation masks to 3D point clouds:
- lift_mask_to_3d: Lift a single 2D binary mask to 3D world coordinates using depth
- lift_mask_to_3d_batch: Batch version for multiple cameras/frames
- visualize_mask_3d: Visualize 2D masks as 3D point clouds in Rerun
- visualize_masks_batch: Batch visualization of multiple masks across cameras/frames

These functions are designed to work with the RH20T dataset format where:
- Masks are binary arrays [H, W] or batched [C, T, H, W]
- Depths are in meters [H, W] or batched [C, T, H, W]
- Intrinsics are 3x3 camera calibration matrices
- Extrinsics are [3, 4] or [4, 4] world-to-camera transforms
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import rerun as rr


def lift_mask_to_3d(
    mask: np.ndarray,
    depth: np.ndarray,
    intr: np.ndarray,
    extr: np.ndarray,
    min_depth: float = 0.0,
    max_depth: float = 10.0,
    rgb: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Lift a 2D binary mask to 3D world coordinates using depth map.
    
    This function takes a 2D segmentation mask and projects it into 3D space
    by using the corresponding depth values. Only pixels where the mask is True
    and depth is valid are converted to 3D points.
    
    Args:
        mask: Binary mask [H, W], True for pixels to lift
        depth: Depth map [H, W], depth values in meters
        intr: Intrinsic matrix [3, 3], camera calibration
        extr: Extrinsic matrix [3, 4] or [4, 4], world-to-camera transform
        min_depth: Minimum valid depth threshold in meters (default: 0.0)
        max_depth: Maximum valid depth threshold in meters (default: 10.0)
        rgb: Optional RGB image [H, W, 3] to extract colors for points
    
    Returns:
        points_3d: 3D points in world coordinates [N, 3], dtype float32
        colors: Optional RGB colors [N, 3], dtype uint8 (if rgb provided)
    
    Example:
        >>> mask = np.zeros((480, 640), dtype=bool)
        >>> mask[100:200, 150:250] = True  # Region of interest
        >>> depth = np.random.rand(480, 640) * 5.0  # Simulated depth
        >>> K = np.eye(3)
        >>> K[0, 0] = K[1, 1] = 500.0  # Focal length
        >>> K[0, 2], K[1, 2] = 320, 240  # Principal point
        >>> E = np.eye(4)[:3, :]  # Identity transform
        >>> points, colors = lift_mask_to_3d(mask, depth, K, E)
        >>> print(f"Lifted {len(points)} points to 3D")
    """
    # Validate inputs
    if mask.shape != depth.shape:
        raise ValueError(f"Mask shape {mask.shape} must match depth shape {depth.shape}")
    
    H, W = mask.shape
    
    # Find masked pixels
    ys, xs = np.nonzero(mask)
    
    if len(xs) == 0:
        empty_points = np.empty((0, 3), dtype=np.float32)
        empty_colors = np.empty((0, 3), dtype=np.uint8) if rgb is not None else None
        return empty_points, empty_colors
    
    # Get depth values at mask locations
    z = depth[ys, xs].astype(np.float32)
    
    # Filter by depth validity
    valid = (z > min_depth) & (z < max_depth)
    
    if not np.any(valid):
        empty_points = np.empty((0, 3), dtype=np.float32)
        empty_colors = np.empty((0, 3), dtype=np.uint8) if rgb is not None else None
        return empty_points, empty_colors
    
    # Apply validity mask
    xs = xs[valid]
    ys = ys[valid]
    z = z[valid]
    
    # Backproject to camera coordinates
    # pixel coordinates -> normalized image coordinates -> camera coordinates
    x_cam = (xs - intr[0, 2]) * z / intr[0, 0]
    y_cam = (ys - intr[1, 2]) * z / intr[1, 1]
    z_cam = z
    
    # Stack into [N, 3] array
    points_cam = np.stack([x_cam, y_cam, z_cam], axis=1).astype(np.float32)
    
    # Transform to world coordinates
    # Handle both [3, 4] and [4, 4] extrinsic matrices
    if extr.shape == (3, 4):
        R = extr[:3, :3]
        t = extr[:3, 3]
    elif extr.shape == (4, 4):
        R = extr[:3, :3]
        t = extr[:3, 3]
    else:
        raise ValueError(f"Extrinsic matrix must be [3, 4] or [4, 4], got {extr.shape}")
    
    # Apply rotation and translation: p_world = R @ p_cam + t
    points_world = (R @ points_cam.T).T + t
    
    # Extract colors if RGB image provided
    colors = None
    if rgb is not None:
        if rgb.shape[:2] != (H, W):
            raise ValueError(f"RGB shape {rgb.shape[:2]} must match mask shape {(H, W)}")
        colors = rgb[ys, xs].astype(np.uint8)
    
    return points_world.astype(np.float32), colors


def lift_mask_to_3d_batch(
    masks: np.ndarray,
    depths: np.ndarray,
    intrs: np.ndarray,
    extrs: np.ndarray,
    min_depth: float = 0.0,
    max_depth: float = 10.0,
    rgbs: Optional[np.ndarray] = None,
) -> Tuple[List[np.ndarray], List[Optional[np.ndarray]]]:
    """
    Batch version of lift_mask_to_3d for multiple cameras/frames.
    
    Args:
        masks: Binary masks [C, T, H, W] or [T, H, W] or [C, H, W]
        depths: Depth maps [C, T, H, W] or [T, H, W] or [C, H, W]
        intrs: Intrinsics [C, T, 3, 3] or [C, 3, 3] or [3, 3]
        extrs: Extrinsics [C, T, 3, 4] or [C, 3, 4] or [3, 4]
        min_depth: Minimum valid depth threshold
        max_depth: Maximum valid depth threshold
        rgbs: Optional RGB images [C, T, H, W, 3] or [T, H, W, 3] or [C, H, W, 3]
    
    Returns:
        all_points: List of 3D point arrays, one per mask
        all_colors: List of color arrays (or None if no RGB provided)
    
    Example:
        >>> # Multi-camera, multi-frame scenario
        >>> masks = np.random.rand(3, 50, 480, 640) > 0.5  # 3 cameras, 50 frames
        >>> depths = np.random.rand(3, 50, 480, 640) * 5.0
        >>> intrs = np.tile(np.eye(3)[None, None], (3, 50, 1, 1))
        >>> extrs = np.tile(np.eye(4)[:3, :][None, None], (3, 50, 1, 1))
        >>> points_list, colors_list = lift_mask_to_3d_batch(masks, depths, intrs, extrs)
        >>> print(f"Processed {len(points_list)} masks")
    """
    # Determine dimensionality
    mask_shape = masks.shape
    
    if len(mask_shape) == 2:
        # Single mask [H, W]
        C, T = 1, 1
        masks = masks[None, None]
        depths = depths[None, None]
    elif len(mask_shape) == 3:
        # Either [C, H, W] or [T, H, W]
        C, T = mask_shape[0], 1
        masks = masks[:, None]
        depths = depths[:, None]
    elif len(mask_shape) == 4:
        # [C, T, H, W]
        C, T = mask_shape[0], mask_shape[1]
    else:
        raise ValueError(f"Unsupported mask shape: {mask_shape}")
    
    # Ensure intrinsics have correct shape
    if intrs.ndim == 2:
        # Single intrinsic [3, 3] -> broadcast to [C, T, 3, 3]
        intrs = np.tile(intrs[None, None], (C, T, 1, 1))
    elif intrs.ndim == 3:
        # [C, 3, 3] -> broadcast to [C, T, 3, 3]
        intrs = np.tile(intrs[:, None], (1, T, 1, 1))
    
    # Ensure extrinsics have correct shape
    if extrs.ndim == 2:
        # Single extrinsic [3, 4] or [4, 4] -> broadcast
        extrs = np.tile(extrs[None, None], (C, T, 1, 1))
    elif extrs.ndim == 3:
        # [C, 3, 4] or [C, 4, 4] -> broadcast to [C, T, ...]
        extrs = np.tile(extrs[:, None], (1, T, 1, 1))
    
    # Handle RGB if provided
    if rgbs is not None:
        rgb_shape = rgbs.shape
        if len(rgb_shape) == 3:
            # [H, W, 3]
            rgbs = rgbs[None, None]
        elif len(rgb_shape) == 4:
            # [C, H, W, 3] or [T, H, W, 3]
            rgbs = rgbs[:, None] if len(rgb_shape[0]) == C else rgbs[None]
        # else assume [C, T, H, W, 3]
    
    # Process each mask
    all_points = []
    all_colors = []
    
    for c in range(C):
        for t in range(T):
            mask = masks[c, t]
            depth = depths[c, t]
            intr = intrs[c, t]
            extr = extrs[c, t]
            rgb = rgbs[c, t] if rgbs is not None else None
            
            points, colors = lift_mask_to_3d(
                mask, depth, intr, extr,
                min_depth=min_depth,
                max_depth=max_depth,
                rgb=rgb
            )
            
            all_points.append(points)
            all_colors.append(colors)
    
    return all_points, all_colors


def visualize_mask_3d(
    mask: np.ndarray,
    depth: np.ndarray,
    intr: np.ndarray,
    extr: np.ndarray,
    entity_path: str,
    rgb: Optional[np.ndarray] = None,
    color: Optional[np.ndarray] = None,
    radius: float = 0.005,
    min_depth: float = 0.0,
    max_depth: float = 10.0,
    time_seconds: Optional[float] = None,
) -> int:
    """
    Visualize a 2D mask as a 3D point cloud in Rerun.
    
    Lifts the mask to 3D using the depth map and logs it to Rerun for visualization.
    
    Args:
        mask: Binary mask [H, W], True for pixels to visualize
        depth: Depth map [H, W], depth values in meters
        intr: Intrinsic matrix [3, 3]
        extr: Extrinsic matrix [3, 4] or [4, 4], world-to-camera transform
        entity_path: Rerun entity path (e.g., "world/masks/hand")
        rgb: Optional RGB image [H, W, 3] to color the points
        color: Optional fixed color [3] (uint8) for all points (overrides RGB)
        radius: Point radius for visualization (default: 0.005)
        min_depth: Minimum valid depth threshold
        max_depth: Maximum valid depth threshold
        time_seconds: Optional time in seconds for temporal logging
    
    Returns:
        num_points: Number of 3D points visualized
    
    Example:
        >>> import rerun as rr
        >>> rr.init("mask_viz", spawn=True)
        >>> mask = np.zeros((480, 640), dtype=bool)
        >>> mask[100:200, 150:250] = True
        >>> depth = np.random.rand(480, 640) * 5.0
        >>> K = np.eye(3)
        >>> E = np.eye(4)[:3, :]
        >>> num_pts = visualize_mask_3d(mask, depth, K, E, "world/my_mask")
        >>> print(f"Visualized {num_pts} points")
    """
    # Lift mask to 3D
    points_3d, colors_from_rgb = lift_mask_to_3d(
        mask=mask,
        depth=depth,
        intr=intr,
        extr=extr,
        min_depth=min_depth,
        max_depth=max_depth,
        rgb=rgb,
    )
    
    if len(points_3d) == 0:
        return 0
    
    # Determine colors
    if color is not None:
        # Use fixed color for all points
        point_colors = np.tile(color, (len(points_3d), 1)).astype(np.uint8)
    elif colors_from_rgb is not None:
        # Use colors from RGB image
        point_colors = colors_from_rgb
    else:
        # Default to white
        point_colors = np.full((len(points_3d), 3), 255, dtype=np.uint8)
    
    # Set time if provided
    if time_seconds is not None:
        rr.set_time_seconds("frame", time_seconds)
    
    # Log to Rerun
    rr.log(
        entity_path,
        rr.Points3D(
            positions=points_3d,
            colors=point_colors,
            radii=radius,
        )
    )
    
    return len(points_3d)


def visualize_masks_batch(
    masks: np.ndarray,
    depths: np.ndarray,
    intrs: np.ndarray,
    extrs: np.ndarray,
    entity_base_path: str,
    camera_ids: Optional[List[str]] = None,
    rgbs: Optional[np.ndarray] = None,
    color: Optional[np.ndarray] = None,
    radius: float = 0.005,
    min_depth: float = 0.0,
    max_depth: float = 10.0,
    fps: float = 30.0,
    max_frames: Optional[int] = None,
) -> Dict[str, int]:
    """
    Batch visualization of masks across multiple cameras and frames.
    
    Args:
        masks: Binary masks [C, T, H, W]
        depths: Depth maps [C, T, H, W]
        intrs: Intrinsics [C, T, 3, 3] or [C, 3, 3]
        extrs: Extrinsics [C, T, 3, 4] or [C, 3, 4]
        entity_base_path: Base path for entities (e.g., "world/masks")
        camera_ids: Optional list of camera ID strings [C]
        rgbs: Optional RGB images [C, T, H, W, 3]
        color: Optional fixed color [3] (uint8) for all points
        radius: Point radius for visualization
        min_depth: Minimum valid depth threshold
        max_depth: Maximum valid depth threshold
        fps: Frame rate for temporal logging
        max_frames: Optional limit on number of frames to visualize
    
    Returns:
        stats: Dictionary with statistics
            - "total_points": Total number of points visualized
            - "num_cameras": Number of cameras processed
            - "num_frames": Number of frames processed
            - "points_per_camera": List of point counts per camera
    
    Example:
        >>> # Multi-camera hand mask visualization
        >>> masks = hand_masks  # [3, 50, 480, 640]
        >>> depths = depth_maps  # [3, 50, 480, 640]
        >>> intrs = intrinsics  # [3, 3, 3]
        >>> extrs = extrinsics  # [3, 50, 3, 4]
        >>> stats = visualize_masks_batch(
        ...     masks, depths, intrs, extrs,
        ...     "world/hand_masks",
        ...     camera_ids=["cam0", "cam1", "cam2"],
        ...     color=np.array([255, 0, 255], dtype=np.uint8),  # Magenta
        ... )
        >>> print(f"Visualized {stats['total_points']} hand points")
    """
    C, T, H, W = masks.shape
    
    if max_frames is not None and T > max_frames:
        T = max_frames
        masks = masks[:, :max_frames]
        depths = depths[:, :max_frames]
        if rgbs is not None:
            rgbs = rgbs[:, :max_frames]
    
    # Handle intrinsics
    if intrs.ndim == 3 and intrs.shape[1] == 3:
        # [C, 3, 3] - static intrinsics
        intrs_expanded = np.tile(intrs[:, None], (1, T, 1, 1))
    else:
        intrs_expanded = intrs
    
    # Handle extrinsics
    if extrs.ndim == 3 and extrs.shape[1] in [3, 4]:
        # [C, 3, 4] or [C, 4, 4] - static extrinsics
        extrs_expanded = np.tile(extrs[:, None], (1, T, 1, 1))
    else:
        extrs_expanded = extrs
    
    # Statistics
    total_points = 0
    points_per_camera = []
    
    print(f"[INFO] Visualizing masks for {C} cameras, {T} frames...")
    
    for c in range(C):
        cam_id = camera_ids[c] if camera_ids is not None else f"cam_{c:03d}"
        cam_points = 0
        
        for t in range(T):
            # Set temporal context
            time_seconds = t / fps
            
            # Get data for this frame
            mask = masks[c, t]
            depth = depths[c, t]
            intr = intrs_expanded[c, t]
            extr = extrs_expanded[c, t]
            rgb = rgbs[c, t] if rgbs is not None else None
            
            # Skip empty masks
            if not mask.any():
                continue
            
            # Create entity path
            entity_path = f"{entity_base_path}/{cam_id}"
            
            # Visualize
            num_points = visualize_mask_3d(
                mask=mask,
                depth=depth,
                intr=intr,
                extr=extr,
                entity_path=entity_path,
                rgb=rgb,
                color=color,
                radius=radius,
                min_depth=min_depth,
                max_depth=max_depth,
                time_seconds=time_seconds,
            )
            
            cam_points += num_points
            total_points += num_points
        
        points_per_camera.append(cam_points)
        print(f"[INFO]   Camera {cam_id}: {cam_points} points")
    
    print(f"[INFO] Total: {total_points} points visualized")
    
    return {
        "total_points": total_points,
        "num_cameras": C,
        "num_frames": T,
        "points_per_camera": points_per_camera,
    }
