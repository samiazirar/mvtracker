#!/usr/bin/env python3
"""
Mask Lifting Utilities for RH20T Data

This module provides functions for lifting 2D segmentation masks to 3D point clouds:
- lift_mask_to_3d: Lift a single 2D binary mask to 3D world coordinates using depth
- lift_mask_to_3d_batch: Batch version for multiple cameras/frames
- visualize_mask_3d: Visualize 2D masks as 3D point clouds in Rerun
- visualize_masks_batch: Batch visualization of multiple masks across cameras/frames
- assign_global_mask_ids: Assign consistent IDs across cameras by matching 3D positions
- visualize_masks_with_global_ids: Visualize masks with globally consistent IDs

These functions are designed to work with the RH20T dataset format where:
- Masks can be single arrays [C, T, H, W] or dictionaries of masks {name: [C, T, H, W]}
- Depths are in meters [H, W] or batched [C, T, H, W]
- Intrinsics are 3x3 camera calibration matrices
- Extrinsics are [3, 4] or [4, 4] world-to-camera transforms

When using dictionaries of masks (e.g., {'hand': mask1, 'object': mask2}), each mask
type gets its own entity path in Rerun for independent toggling in the viewer.

Global ID Assignment:
The assign_global_mask_ids function matches masks across cameras using 3D spatial proximity.
It computes centroids in world coordinates and matches them with a distance threshold.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import rerun as rr
from scipy.spatial.distance import cdist

from utils.camera_utils import scale_intrinsics_matrix, infer_calibration_resolution


def compute_mask_centroid_3d(
    mask: np.ndarray,
    depth: np.ndarray,
    intr: np.ndarray,
    extr: np.ndarray,
    min_depth: float = 0.0,
    max_depth: float = 10.0,
) -> Optional[np.ndarray]:
    """
    Compute the 3D centroid of a 2D mask in world coordinates.
    
    Args:
        mask: Binary mask [H, W]
        depth: Depth map [H, W]
        intr: Intrinsic matrix [3, 3]
        extr: Extrinsic matrix [3, 4] or [4, 4]
        min_depth: Minimum valid depth threshold
        max_depth: Maximum valid depth threshold
    
    Returns:
        centroid: 3D centroid [3] in world coordinates, or None if mask is empty
    """
    points_3d, _ = lift_mask_to_3d(mask, depth, intr, extr, min_depth, max_depth)
    
    if len(points_3d) == 0:
        return None
    
    return np.mean(points_3d, axis=0)


def assign_global_mask_ids(
    masks_dict: Dict[str, np.ndarray],
    depths: np.ndarray,
    intrs: np.ndarray,
    extrs: np.ndarray,
    distance_threshold: float = 0.15,
    min_depth: float = 0.0,
    max_depth: float = 10.0,
    frame_index: int = 0,
) -> Dict[str, Dict[int, int]]:
    """
    Assign global IDs to masks across cameras by matching 3D centroids.
    
    This function takes masks from multiple cameras (where each camera has its own
    local ID numbering) and assigns consistent global IDs by matching masks that
    represent the same object in 3D space.
    
    Args:
        masks_dict: Dictionary mapping mask names to mask arrays
                   Each mask array is [C, T, H, W] where C is cameras, T is frames
        depths: Depth maps [C, T, H, W]
        intrs: Intrinsics [C, T, 3, 3] or [C, 3, 3]
        extrs: Extrinsics [C, T, 3, 4] or [C, 3, 4]
        distance_threshold: Maximum distance (meters) to consider masks as the same object
        min_depth: Minimum valid depth threshold
        max_depth: Maximum valid depth threshold
        frame_index: Which frame to use for matching (default: 0)
    
    Returns:
        global_id_mapping: Dictionary mapping mask names to {local_id: global_id} dicts
                          Structure: {mask_name: {local_id: global_id}}
                          where local_id is the mask ID within each camera
    
    Example:
        >>> # Camera 0 has 3 masks (IDs 0, 1, 2), Camera 1 has 2 masks (IDs 0, 1)
        >>> # The function matches them based on 3D position
        >>> mapping = assign_global_mask_ids(masks_dict, depths, intrs, extrs)
        >>> # Result might be: {'hand': {0: 0, 1: 1, 2: 2}}  # cam0 IDs -> global IDs
        >>> #                   {'hand': {0: 0, 1: 2}}       # cam1 IDs -> global IDs
        >>> # This means cam1's ID 0 matches cam0's ID 0 (both get global ID 0)
        >>> # and cam1's ID 1 matches cam0's ID 2 (both get global ID 2)
    """
    if not masks_dict:
        return {}
    
    # Get dimensions
    first_mask_array = list(masks_dict.values())[0]
    C, T, H, W = first_mask_array.shape
    
    if frame_index >= T:
        frame_index = 0
    
    # Handle intrinsics
    if intrs.ndim == 3 and intrs.shape[1] == 3:
        intrs_expanded = np.tile(intrs[:, None], (1, T, 1, 1))
    else:
        intrs_expanded = intrs
    
    # Handle extrinsics
    if extrs.ndim == 3 and extrs.shape[1] in [3, 4]:
        extrs_expanded = np.tile(extrs[:, None], (1, T, 1, 1))
    else:
        extrs_expanded = extrs
    
    global_id_mapping = {}
    
    for mask_name, masks in masks_dict.items():
        print(f"[INFO] Assigning global IDs for mask type: '{mask_name}'")
        
        # Extract unique mask IDs per camera at the given frame
        # Assuming masks are stored with integer IDs as values
        camera_masks = []  # List of {local_id: mask_array} per camera
        camera_centroids = []  # List of {local_id: centroid_3d} per camera
        
        for c in range(C):
            mask_frame = masks[c, frame_index]
            unique_ids = np.unique(mask_frame)
            unique_ids = unique_ids[unique_ids > 0]  # Exclude background (0)
            
            cam_masks = {}
            cam_centroids = {}
            
            for local_id in unique_ids:
                # Create binary mask for this ID
                binary_mask = (mask_frame == local_id)
                
                # Compute 3D centroid
                centroid = compute_mask_centroid_3d(
                    binary_mask,
                    depths[c, frame_index],
                    intrs_expanded[c, frame_index],
                    extrs_expanded[c, frame_index],
                    min_depth,
                    max_depth,
                )
                
                if centroid is not None:
                    cam_masks[int(local_id)] = binary_mask
                    cam_centroids[int(local_id)] = centroid
            
            camera_masks.append(cam_masks)
            camera_centroids.append(cam_centroids)
            print(f"[INFO]   Camera {c}: Found {len(cam_centroids)} masks")
        
        # Now match masks across cameras using greedy matching
        # Start with camera 0 as reference
        next_global_id = 0
        local_to_global = [{} for _ in range(C)]  # One dict per camera
        
        if len(camera_centroids[0]) > 0:
            # Assign global IDs to camera 0 masks
            for local_id in sorted(camera_centroids[0].keys()):
                local_to_global[0][local_id] = next_global_id
                next_global_id += 1
        
        # Match subsequent cameras to existing global IDs
        for c in range(1, C):
            if len(camera_centroids[c]) == 0:
                continue
            
            # Get centroids for this camera
            local_ids_c = list(camera_centroids[c].keys())
            centroids_c = np.array([camera_centroids[c][lid] for lid in local_ids_c])
            
            # Get all existing global centroids
            existing_global_centroids = {}
            for prev_c in range(c):
                for local_id, global_id in local_to_global[prev_c].items():
                    if global_id not in existing_global_centroids:
                        existing_global_centroids[global_id] = camera_centroids[prev_c][local_id]
            
            if len(existing_global_centroids) > 0:
                global_ids_list = list(existing_global_centroids.keys())
                global_centroids = np.array([existing_global_centroids[gid] for gid in global_ids_list])
                
                # Compute pairwise distances
                distances = cdist(centroids_c, global_centroids, metric='euclidean')
                
                # Greedy matching: for each mask in current camera, find closest global ID
                matched = set()
                for i, local_id in enumerate(local_ids_c):
                    min_dist_idx = np.argmin(distances[i])
                    min_dist = distances[i, min_dist_idx]
                    
                    if min_dist < distance_threshold and min_dist_idx not in matched:
                        # Match to existing global ID
                        local_to_global[c][local_id] = global_ids_list[min_dist_idx]
                        matched.add(min_dist_idx)
                        print(f"[INFO]   Camera {c}, local ID {local_id} -> global ID {global_ids_list[min_dist_idx]} (dist: {min_dist:.3f}m)")
                    else:
                        # Create new global ID
                        local_to_global[c][local_id] = next_global_id
                        print(f"[INFO]   Camera {c}, local ID {local_id} -> NEW global ID {next_global_id} (min_dist: {min_dist:.3f}m)")
                        next_global_id += 1
            else:
                # No existing global IDs, assign new ones
                for local_id in local_ids_c:
                    local_to_global[c][local_id] = next_global_id
                    print(f"[INFO]   Camera {c}, local ID {local_id} -> NEW global ID {next_global_id}")
                    next_global_id += 1
        
        # Store the mapping for this mask type
        # Convert from list of dicts to dict of dicts indexed by camera
        global_id_mapping[mask_name] = {c: local_to_global[c] for c in range(C)}
        
        print(f"[INFO] Assigned {next_global_id} global IDs for '{mask_name}'")
    
    return global_id_mapping


def track_masks_temporal(
    masks: np.ndarray,
    depths: np.ndarray,
    intrs: np.ndarray,
    extrs: np.ndarray,
    temporal_distance_threshold: float = 0.10,
    min_depth: float = 0.0,
    max_depth: float = 10.0,
) -> Dict[int, List[Tuple[int, int]]]:
    """
    Track masks temporally within a single camera using centroid proximity.
    
    This function tracks masks frame-to-frame within one camera by matching
    centroids across consecutive frames. It handles masks appearing/disappearing
    and assigns temporal track IDs.
    
    Args:
        masks: Mask array for one camera [T, H, W] with integer IDs
        depths: Depth maps for one camera [T, H, W]
        intrs: Intrinsics for one camera [T, 3, 3] or [3, 3]
        extrs: Extrinsics for one camera [T, 3, 4] or [3, 4]
        temporal_distance_threshold: Max distance (meters) to link masks across frames
        min_depth: Minimum valid depth threshold
        max_depth: Maximum valid depth threshold
    
    Returns:
        tracks: Dictionary mapping track_id to list of (frame_idx, local_mask_id) tuples
                Example: {0: [(0, 1), (1, 1), (2, 2)], 1: [(0, 2), (1, 3)]}
                Track 0 appears in frames 0,1,2 with local IDs 1,1,2
    """
    T, H, W = masks.shape
    
    # Handle intrinsics
    if intrs.ndim == 2:
        intrs_expanded = np.tile(intrs[None], (T, 1, 1))
    else:
        intrs_expanded = intrs
    
    # Handle extrinsics
    if extrs.ndim == 2:
        extrs_expanded = np.tile(extrs[None], (T, 1, 1))
    else:
        extrs_expanded = extrs
    
    # Extract masks and centroids for each frame
    frame_data = []  # List of {local_id: centroid} per frame
    for t in range(T):
        mask_frame = masks[t]
        unique_ids = np.unique(mask_frame)
        unique_ids = unique_ids[unique_ids > 0]  # Exclude background
        
        centroids = {}
        for local_id in unique_ids:
            binary_mask = (mask_frame == local_id)
            centroid = compute_mask_centroid_3d(
                binary_mask,
                depths[t],
                intrs_expanded[t],
                extrs_expanded[t],
                min_depth,
                max_depth,
            )
            if centroid is not None:
                centroids[int(local_id)] = centroid
        
        frame_data.append(centroids)
    
    # Temporal tracking: link masks across frames
    tracks = {}  # {track_id: [(frame, local_id), ...]}
    next_track_id = 0
    active_tracks = {}  # {track_id: last_centroid}
    
    for t in range(T):
        current_centroids = frame_data[t]
        
        if not current_centroids:
            continue
        
        # Match current masks to active tracks
        current_local_ids = list(current_centroids.keys())
        current_positions = np.array([current_centroids[lid] for lid in current_local_ids])
        
        if active_tracks:
            # Compute distances to active tracks
            track_ids = list(active_tracks.keys())
            track_positions = np.array([active_tracks[tid] for tid in track_ids])
            
            distances = cdist(current_positions, track_positions, metric='euclidean')
            
            # Greedy matching: assign each mask to closest track if within threshold
            matched_current = set()
            matched_tracks = set()
            
            for i, local_id in enumerate(current_local_ids):
                if len(track_ids) == 0:
                    break
                    
                min_dist_idx = np.argmin(distances[i])
                min_dist = distances[i, min_dist_idx]
                track_id = track_ids[min_dist_idx]
                
                if min_dist < temporal_distance_threshold and min_dist_idx not in matched_tracks:
                    # Continue existing track
                    tracks[track_id].append((t, local_id))
                    active_tracks[track_id] = current_centroids[local_id]
                    matched_current.add(i)
                    matched_tracks.add(min_dist_idx)
            
            # Create new tracks for unmatched masks
            for i, local_id in enumerate(current_local_ids):
                if i not in matched_current:
                    tracks[next_track_id] = [(t, local_id)]
                    active_tracks[next_track_id] = current_centroids[local_id]
                    next_track_id += 1
        else:
            # First frame with masks: create new tracks for all
            for local_id in current_local_ids:
                tracks[next_track_id] = [(t, local_id)]
                active_tracks[next_track_id] = current_centroids[local_id]
                next_track_id += 1
    
    return tracks


def compute_track_representative_centroid(
    track: List[Tuple[int, int]],
    masks: np.ndarray,
    depths: np.ndarray,
    intrs: np.ndarray,
    extrs: np.ndarray,
    min_depth: float = 0.0,
    max_depth: float = 10.0,
    method: str = "median",
) -> Optional[np.ndarray]:
    """
    Compute a representative 3D centroid for a temporal track.
    
    Args:
        track: List of (frame_idx, local_mask_id) tuples
        masks: Mask array [T, H, W]
        depths: Depth maps [T, H, W]
        intrs: Intrinsics [T, 3, 3] or [3, 3]
        extrs: Extrinsics [T, 3, 4] or [3, 4]
        min_depth: Minimum valid depth threshold
        max_depth: Maximum valid depth threshold
        method: "median" or "mean" for aggregation
    
    Returns:
        representative_centroid: [3] array or None if track is empty
    """
    T, H, W = masks.shape
    
    # Handle intrinsics
    if intrs.ndim == 2:
        intrs_expanded = np.tile(intrs[None], (T, 1, 1))
    else:
        intrs_expanded = intrs
    
    # Handle extrinsics
    if extrs.ndim == 2:
        extrs_expanded = np.tile(extrs[None], (T, 1, 1))
    else:
        extrs_expanded = extrs
    
    # Compute centroid for each frame in track
    centroids = []
    for frame_idx, local_id in track:
        mask_frame = masks[frame_idx]
        binary_mask = (mask_frame == local_id)
        
        centroid = compute_mask_centroid_3d(
            binary_mask,
            depths[frame_idx],
            intrs_expanded[frame_idx],
            extrs_expanded[frame_idx],
            min_depth,
            max_depth,
        )
        
        if centroid is not None:
            centroids.append(centroid)
    
    if not centroids:
        return None
    
    centroids = np.array(centroids)
    
    if method == "median":
        return np.median(centroids, axis=0)
    else:  # mean
        return np.mean(centroids, axis=0)


def assign_global_ids_temporal(
    masks_dict: Dict[str, np.ndarray],
    depths: np.ndarray,
    intrs: np.ndarray,
    extrs: np.ndarray,
    distance_threshold: float = 0.15,
    temporal_distance_threshold: float = 0.10,
    min_depth: float = 0.0,
    max_depth: float = 10.0,
) -> Tuple[Dict[str, Dict[int, Dict[int, int]]], Dict[str, Dict[int, Dict[int, List[int]]]]]:
    """
    Assign global IDs by matching local mask IDs across cameras using average temporal distance.
    
    Key principle: Each local mask ID (e.g., camera A's id0) represents ONE object across all time.
    We compute the average distance between mask IDs across cameras over all frames where both exist.
    
    Args:
        masks_dict: Dictionary mapping mask names to mask arrays [C, T, H, W]
                   Masks contain integer IDs (0=background, 1,2,3,...=objects)
        depths: Depth maps [C, T, H, W]
        intrs: Intrinsics [C, T, 3, 3] or [C, 3, 3]
        extrs: Extrinsics [C, T, 3, 4] or [C, 3, 4]
        distance_threshold: Max average distance (meters) to match masks across cameras
        temporal_distance_threshold: Unused (kept for API compatibility)
        min_depth: Minimum valid depth threshold
        max_depth: Maximum valid depth threshold
    
    Returns:
        global_id_mapping: {mask_name: {camera: {local_id: global_id}}}
        frames_mapping: {mask_name: {camera: {local_id: [frame_indices]}}}
    """
    if not masks_dict:
        return {}, {}
    
    # Get dimensions
    first_mask_array = list(masks_dict.values())[0]
    C, T, H, W = first_mask_array.shape
    
    # Handle intrinsics
    if intrs.ndim == 3 and intrs.shape[1] == 3:
        intrs_per_camera = intrs  # [C, 3, 3]
    elif intrs.ndim == 4:
        intrs_per_camera = intrs[:, 0]  # Use first frame [C, 3, 3]
    else:
        intrs_per_camera = np.tile(intrs[None], (C, 1, 1))
    
    # Handle extrinsics
    if extrs.ndim == 3 and extrs.shape[1] in [3, 4]:
        extrs_per_camera = extrs  # [C, 3/4, 4]
    elif extrs.ndim == 4:
        extrs_per_camera = extrs[:, 0]  # Use first frame
    else:
        extrs_per_camera = np.tile(extrs[None], (C, 1, 1))
    
    global_id_mapping = {}
    frames_mapping = {}
    
    for mask_name, masks in masks_dict.items():
        print(f"\n[INFO] ========== Computing Temporal Averages for '{mask_name}' ==========")
        
        # Step 1: For each camera, extract all unique local IDs and their centroids over time
        camera_local_ids = {}  # {camera: [local_ids]}
        camera_centroids = {}  # {camera: {local_id: {frame: centroid}}}
        
        for c in range(C):
            print(f"[INFO] Processing camera {c}...")
            
            # Get intrinsics/extrinsics for this camera
            if intrs.ndim == 4:
                intr_cam = intrs[c]  # [T, 3, 3]
            else:
                intr_cam = intrs_per_camera[c]  # [3, 3]
            
            if extrs.ndim == 4:
                extr_cam = extrs[c]  # [T, 3, 4]
            else:
                extr_cam = extrs_per_camera[c]  # [3, 4]
            
            # Handle intrinsics/extrinsics expansion
            if intr_cam.ndim == 2:
                intrs_expanded = np.tile(intr_cam[None], (T, 1, 1))
            else:
                intrs_expanded = intr_cam
            
            if extr_cam.ndim == 2:
                extrs_expanded = np.tile(extr_cam[None], (T, 1, 1))
            else:
                extrs_expanded = extr_cam
            
            # Extract all unique local IDs across all frames
            all_local_ids = set()
            for t in range(T):
                unique_ids = np.unique(masks[c, t])
                unique_ids = unique_ids[unique_ids > 0]  # Exclude background
                all_local_ids.update(unique_ids.tolist())
            
            all_local_ids = sorted(list(all_local_ids))
            camera_local_ids[c] = all_local_ids
            
            # For each local ID, compute centroids at each frame where it exists
            centroids_per_id = {}
            for local_id in all_local_ids:
                frame_centroids = {}
                for t in range(T):
                    mask_frame = masks[c, t]
                    if local_id in mask_frame:  # Check if this ID exists in this frame
                        binary_mask = (mask_frame == local_id)
                        if binary_mask.any():
                            centroid = compute_mask_centroid_3d(
                                binary_mask,
                                depths[c, t],
                                intrs_expanded[t],
                                extrs_expanded[t],
                                min_depth,
                                max_depth,
                            )
                            if centroid is not None:
                                frame_centroids[t] = centroid
                
                centroids_per_id[local_id] = frame_centroids
                print(f"[INFO]   Camera {c}, local ID {local_id}: visible in {len(frame_centroids)} frames")
            
            camera_centroids[c] = centroids_per_id
        
        # Step 2: Compute average distances between all pairs of (camera_i, local_id_i) and (camera_j, local_id_j)
        print(f"\n[INFO] Computing pairwise average distances...")
        
        # Build list of all (camera, local_id) pairs
        all_masks = []
        for c in range(C):
            for local_id in camera_local_ids[c]:
                all_masks.append((c, local_id))
        
        # Compute average distance matrix
        n_masks = len(all_masks)
        avg_distances = np.full((n_masks, n_masks), np.inf)
        
        for i in range(n_masks):
            cam_i, id_i = all_masks[i]
            centroids_i = camera_centroids[cam_i][id_i]
            
            for j in range(i, n_masks):
                if i == j:
                    avg_distances[i, j] = 0.0
                    continue
                
                cam_j, id_j = all_masks[j]
                
                # Skip if same camera (can't match within camera)
                if cam_i == cam_j:
                    continue
                
                centroids_j = camera_centroids[cam_j][id_j]
                
                # Find common frames
                common_frames = set(centroids_i.keys()) & set(centroids_j.keys())
                
                if not common_frames:
                    continue  # No overlap, keep inf
                
                # Compute average distance over common frames
                distances = []
                for frame in common_frames:
                    dist = np.linalg.norm(centroids_i[frame] - centroids_j[frame])
                    distances.append(dist)
                
                avg_dist = np.mean(distances)
                avg_distances[i, j] = avg_dist
                avg_distances[j, i] = avg_dist  # Symmetric
        
        # Step 3: Assign global IDs using greedy matching
        print(f"\n[INFO] Assigning global IDs...")
        
        next_global_id = 0
        mask_to_global = {}  # {(camera, local_id): global_id}
        
        # Start with camera 0
        for local_id in camera_local_ids[0]:
            mask_to_global[(0, local_id)] = next_global_id
            print(f"[INFO]   Camera 0, local ID {local_id} -> global ID {next_global_id}")
            next_global_id += 1
        
        # Match subsequent cameras
        for c in range(1, C):
            for local_id in camera_local_ids[c]:
                # Find index in all_masks
                idx = all_masks.index((c, local_id))
                
                # Find best match among already assigned masks
                best_match = None
                best_dist = np.inf
                
                for (prev_c, prev_id), global_id in mask_to_global.items():
                    if prev_c >= c:
                        continue  # Only look at previously processed cameras
                    
                    prev_idx = all_masks.index((prev_c, prev_id))
                    dist = avg_distances[idx, prev_idx]
                    
                    if dist < best_dist:
                        best_dist = dist
                        best_match = global_id
                
                if best_match is not None and best_dist < distance_threshold:
                    # Match to existing global ID
                    mask_to_global[(c, local_id)] = best_match
                    print(f"[INFO]   Camera {c}, local ID {local_id} -> global ID {best_match} (avg dist: {best_dist:.3f}m)")
                else:
                    # Create new global ID
                    mask_to_global[(c, local_id)] = next_global_id
                    print(f"[INFO]   Camera {c}, local ID {local_id} -> NEW global ID {next_global_id} (min avg dist: {best_dist:.3f}m)")
                    next_global_id += 1
        
        # Convert to output format
        local_to_global_per_camera = {}
        frames_per_camera = {}
        
        for c in range(C):
            local_to_global_per_camera[c] = {}
            frames_per_camera[c] = {}
            
            for local_id in camera_local_ids[c]:
                global_id = mask_to_global[(c, local_id)]
                local_to_global_per_camera[c][local_id] = global_id
                
                # Store which frames this local_id appears in
                frames_with_id = list(camera_centroids[c][local_id].keys())
                frames_per_camera[c][local_id] = frames_with_id
        
        global_id_mapping[mask_name] = local_to_global_per_camera
        frames_mapping[mask_name] = frames_per_camera
        
        print(f"[INFO] Assigned {next_global_id} global IDs for '{mask_name}' (simplified approach)")
    
    return global_id_mapping, frames_mapping


def lift_mask_to_3d(
    mask: np.ndarray,
    depth: np.ndarray,
    intr: np.ndarray,
    extr: np.ndarray,
    min_depth: float = 0.0,
    max_depth: float = 10.0,
    rgb: Optional[np.ndarray] = None,
    mask_resolution: Optional[Tuple[int, int]] = None,  # (W, H) if mask is in different resolution than depth
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Lift a 2D binary mask to 3D world coordinates using depth map.
    
    This function takes a 2D segmentation mask and projects it into 3D space
    by using the corresponding depth values. Handles resolution mismatch between
    mask (typically from RGB tracking) and depth map by scaling coordinates.
    
    Args:
        mask: Binary mask [H, W], True for pixels to lift
        depth: Depth map [H, W], depth values in meters
        intr: Intrinsic matrix [3, 3], camera calibration (for DEPTH resolution)
        extr: Extrinsic matrix [3, 4] or [4, 4], world-to-camera transform
        min_depth: Minimum valid depth threshold in meters (default: 0.0)
        max_depth: Maximum valid depth threshold in meters (default: 10.0)
        rgb: Optional RGB image [H, W, 3] to extract colors for points
        mask_resolution: Optional (W, H) tuple if mask is in different resolution than depth
                        Used to scale mask pixel coordinates to depth space
    
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
    # Get dimensions
    H_mask, W_mask = mask.shape
    H_depth, W_depth = depth.shape
    
    # Handle resolution mismatch
    if mask_resolution is not None:
        # Mask is in different resolution (e.g., RGB) than depth
        W_mask_orig, H_mask_orig = mask_resolution
        if (H_mask, W_mask) != (H_mask_orig, W_mask_orig):
            raise ValueError(
                f"Mask shape {(H_mask, W_mask)} doesn't match provided mask_resolution {(H_mask_orig, W_mask_orig)}"
            )
        
        # Compute scale factors from mask space to depth space
        scale_x = W_depth / W_mask
        scale_y = H_depth / H_mask
    elif (H_mask, W_mask) != (H_depth, W_depth):
        # Auto-detect resolution mismatch
        scale_x = W_depth / W_mask
        scale_y = H_depth / H_mask
        print(
            f"[INFO] Auto-detected resolution mismatch: mask {W_mask}x{H_mask}, depth {W_depth}x{H_depth}. "
            f"Scaling by ({scale_x:.3f}, {scale_y:.3f})"
        )
    else:
        # No resolution mismatch
        scale_x = 1.0
        scale_y = 1.0
    
    # Find masked pixels in mask space
    ys_mask, xs_mask = np.nonzero(mask)
    
    if len(xs_mask) == 0:
        empty_points = np.empty((0, 3), dtype=np.float32)
        empty_colors = np.empty((0, 3), dtype=np.uint8) if rgb is not None else None
        return empty_points, empty_colors
    
    # Scale coordinates to depth space
    xs_depth = (xs_mask * scale_x).astype(int)
    ys_depth = (ys_mask * scale_y).astype(int)
    
    # Clip to valid depth range
    xs_depth = np.clip(xs_depth, 0, W_depth - 1)
    ys_depth = np.clip(ys_depth, 0, H_depth - 1)
    
    # Get depth values at scaled locations
    z = depth[ys_depth, xs_depth].astype(np.float32)
    
    # Filter by depth validity
    valid = (z > min_depth) & (z < max_depth)
    
    if not np.any(valid):
        empty_points = np.empty((0, 3), dtype=np.float32)
        empty_colors = np.empty((0, 3), dtype=np.uint8) if rgb is not None else None
        return empty_points, empty_colors
    
    # Apply validity mask
    xs_depth = xs_depth[valid]
    ys_depth = ys_depth[valid]
    xs_mask_valid = xs_mask[valid]  # Keep for RGB lookup
    ys_mask_valid = ys_mask[valid]
    z = z[valid]
    
    # Backproject to camera coordinates using DEPTH SPACE coordinates
    # (intrinsics are calibrated for depth resolution)
    x_cam = (xs_depth - intr[0, 2]) * z / intr[0, 0]
    y_cam = (ys_depth - intr[1, 2]) * z / intr[1, 1]
    z_cam = z
    
    # Stack into [N, 3] array
    points_cam = np.stack([x_cam, y_cam, z_cam], axis=1).astype(np.float32)
    
    # Transform to world coordinates.
    # Extrinsics are stored as world-to-camera, so invert the rigid transform.
    if extr.shape == (3, 4):
        R_wc = extr[:3, :3]
        t_wc = extr[:3, 3]
    elif extr.shape == (4, 4):
        R_wc = extr[:3, :3]
        t_wc = extr[:3, 3]
    else:
        raise ValueError(f"Extrinsic matrix must be [3, 4] or [4, 4], got {extr.shape}")
    
    R_cw = R_wc.T
    t_cw = -R_cw @ t_wc
    points_world = (R_cw @ points_cam.T).T + t_cw
    
    # Extract colors if RGB image provided (use mask-space coordinates)
    colors = None
    if rgb is not None:
        # RGB should match mask resolution, not depth resolution
        if rgb.shape[:2] != (H_mask, W_mask):
            raise ValueError(
                f"RGB shape {rgb.shape[:2]} must match mask shape {(H_mask, W_mask)}, not depth shape {(H_depth, W_depth)}"
            )
        colors = rgb[ys_mask_valid, xs_mask_valid].astype(np.uint8)
    
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
    print(f"[INFO] Each camera will be a separate toggleable layer in Rerun")
    
    for c in range(C):
        cam_id = camera_ids[c] if camera_ids is not None else f"cam_{c:03d}"
        cam_points = 0
        
        # Create entity path for this camera - each camera is a separate layer
        entity_path = f"{entity_base_path}/camera_{cam_id}"
        
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
            
            # Visualize at the camera-specific entity path
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
        print(f"[INFO]   Camera {cam_id}: {cam_points} points (toggleable at '{entity_path}')")
    
    print(f"[INFO] Total: {total_points} points visualized")
    
    return {
        "total_points": total_points,
        "num_cameras": C,
        "num_frames": T,
        "points_per_camera": points_per_camera,
    }


def visualize_masks_with_global_ids(
    masks_dict: Dict[str, np.ndarray],
    depths: np.ndarray,
    intrs: np.ndarray,
    extrs: np.ndarray,
    entity_base_path: str,
    camera_ids: Optional[List[str]] = None,
    rgbs: Optional[np.ndarray] = None,
    radius: float = 0.005,
    min_depth: float = 0.0,
    max_depth: float = 10.0,
    fps: float = 30.0,
    max_frames: Optional[int] = None,
    distance_threshold: float = 0.15,
    frame_for_matching: int = 0,
) -> Dict[str, any]:
    """
    Visualize masks with globally consistent IDs across cameras.
    
    This function assigns global IDs to masks across cameras (so the same object
    gets the same ID in all cameras) and visualizes them as separate toggleable
    layers per global ID.
    
    Args:
        masks_dict: Dictionary of mask arrays {mask_name: [C, T, H, W]}
                   Note: masks should contain integer IDs (not binary)
        depths: Depth maps [C, T, H, W]
        intrs: Intrinsics [C, T, 3, 3] or [C, 3, 3]
        extrs: Extrinsics [C, T, 3, 4] or [C, 3, 4]
        entity_base_path: Base path for entities (e.g., "world/masks")
        camera_ids: Optional list of camera ID strings [C]
        rgbs: Optional RGB images [C, T, H, W, 3]
        radius: Point radius for visualization
        min_depth: Minimum valid depth threshold
        max_depth: Maximum valid depth threshold
        fps: Frame rate for temporal logging
        max_frames: Optional limit on number of frames to visualize
        distance_threshold: Max distance (meters) to match masks across cameras
        frame_for_matching: Which frame to use for computing global ID assignments
    
    Returns:
        stats: Dictionary with statistics including global_id_mapping
    
    Example:
        >>> # Masks with integer IDs per camera
        >>> masks = {"hand": hand_masks}  # [3, 50, 480, 640] with values 0, 1, 2, ...
        >>> stats = visualize_masks_with_global_ids(
        ...     {"hand": masks},
        ...     depths, intrs, extrs,
        ...     "world/hands",
        ...     distance_threshold=0.15,  # 15cm threshold
        ... )
        >>> # Each global ID gets its own layer: world/hands/hand/global_id_0, etc.
    """
    # Assign global IDs
    print(f"[INFO] ========== Assigning Global IDs ==========")
    global_id_mapping = assign_global_mask_ids(
        masks_dict=masks_dict,
        depths=depths,
        intrs=intrs,
        extrs=extrs,
        distance_threshold=distance_threshold,
        min_depth=min_depth,
        max_depth=max_depth,
        frame_index=frame_for_matching,
    )
    
    # Get dimensions
    first_mask_array = list(masks_dict.values())[0]
    C, T, H, W = first_mask_array.shape
    
    if max_frames is not None and T > max_frames:
        T = max_frames
    
    # Handle intrinsics
    if intrs.ndim == 3 and intrs.shape[1] == 3:
        intrs_expanded = np.tile(intrs[:, None], (1, T, 1, 1))
    else:
        intrs_expanded = intrs
    
    # Handle extrinsics
    if extrs.ndim == 3 and extrs.shape[1] in [3, 4]:
        extrs_expanded = np.tile(extrs[:, None], (1, T, 1, 1))
    else:
        extrs_expanded = extrs
    
    # Determine colors for each global ID (cycling through a palette)
    color_palette = [
        np.array([255, 0, 0], dtype=np.uint8),    # Red
        np.array([0, 255, 0], dtype=np.uint8),    # Green
        np.array([0, 0, 255], dtype=np.uint8),    # Blue
        np.array([255, 255, 0], dtype=np.uint8),  # Yellow
        np.array([255, 0, 255], dtype=np.uint8),  # Magenta
        np.array([0, 255, 255], dtype=np.uint8),  # Cyan
        np.array([255, 128, 0], dtype=np.uint8),  # Orange
        np.array([128, 0, 255], dtype=np.uint8),  # Purple
    ]
    
    print(f"\n[INFO] ========== Visualizing with Global IDs ==========")
    
    all_stats = {}
    
    for mask_name, masks in masks_dict.items():
        print(f"\n[INFO] --- Processing mask type: '{mask_name}' ---")
        
        # Get the ID mapping for this mask type
        id_mapping_per_camera = global_id_mapping[mask_name]
        
        # Determine all global IDs
        all_global_ids = set()
        for cam_mapping in id_mapping_per_camera.values():
            all_global_ids.update(cam_mapping.values())
        all_global_ids = sorted(list(all_global_ids))
        
        print(f"[INFO] Global IDs for '{mask_name}': {all_global_ids}")
        
        # Visualize each global ID as a separate layer
        global_id_stats = {}
        
        for global_id in all_global_ids:
            entity_path = f"{entity_base_path}/{mask_name}/global_id_{global_id}"
            color = color_palette[global_id % len(color_palette)]
            
            total_points = 0
            
            # Process each camera and frame
            for c in range(C):
                cam_id = camera_ids[c] if camera_ids is not None else f"cam_{c:03d}"
                
                # Get local-to-global mapping for this camera
                local_to_global = id_mapping_per_camera[c]
                
                # Find which local ID(s) map to this global ID
                local_ids_for_global = [lid for lid, gid in local_to_global.items() if gid == global_id]
                
                if not local_ids_for_global:
                    continue  # This camera doesn't have this global ID
                
                for t in range(T):
                    time_seconds = t / fps
                    
                    mask_frame = masks[c, t]
                    depth = depths[c, t]
                    intr = intrs_expanded[c, t]
                    extr = extrs_expanded[c, t]
                    rgb_frame = rgbs[c, t] if rgbs is not None else None
                    
                    # Create binary mask for this global ID
                    # (combine all local IDs that map to this global ID)
                    binary_mask = np.zeros_like(mask_frame, dtype=bool)
                    for local_id in local_ids_for_global:
                        binary_mask |= (mask_frame == local_id)
                    
                    if not binary_mask.any():
                        continue
                    
                    # Visualize
                    num_points = visualize_mask_3d(
                        mask=binary_mask,
                        depth=depth,
                        intr=intr,
                        extr=extr,
                        entity_path=entity_path,
                        rgb=rgb_frame,
                        color=color,
                        radius=radius,
                        min_depth=min_depth,
                        max_depth=max_depth,
                        time_seconds=time_seconds,
                    )
                    
                    total_points += num_points
            
            global_id_stats[global_id] = total_points
            print(f"[INFO]   Global ID {global_id}: {total_points} points (color: RGB{tuple(color)}, path: '{entity_path}')")
        
        all_stats[mask_name] = {
            "global_id_stats": global_id_stats,
            "total_points": sum(global_id_stats.values()),
            "num_global_ids": len(all_global_ids),
        }
    
    print(f"\n[INFO] ========== Visualization Complete ==========")
    total_points_all = sum(s['total_points'] for s in all_stats.values())
    print(f"[INFO] Total points across all masks: {total_points_all}")
    
    return {
        "mask_stats": all_stats,
        "global_id_mapping": global_id_mapping,
        "total_points": total_points_all,
    }


def visualize_masks_with_temporal_global_ids(
    masks_dict: Dict[str, np.ndarray],
    depths: np.ndarray,
    intrs: np.ndarray,
    extrs: np.ndarray,
    entity_base_path: str,
    camera_ids: Optional[List[str]] = None,
    rgbs: Optional[np.ndarray] = None,
    radius: float = 0.005,
    min_depth: float = 0.0,
    max_depth: float = 10.0,
    fps: float = 30.0,
    max_frames: Optional[int] = None,
    distance_threshold: float = 0.15,
    temporal_distance_threshold: float = 0.10,
) -> Dict[str, any]:
    """
    Visualize masks with globally consistent IDs using average temporal distance.
    
    Key principle: Each local mask ID represents ONE object across all time.
    Matches objects by computing average 3D distance across all frames where both exist.
    
    Args:
        masks_dict: Dictionary of mask arrays {mask_name: [C, T, H, W]}
                   Note: masks should contain integer IDs (not binary)
        depths: Depth maps [C, T, H, W]
        intrs: Intrinsics [C, T, 3, 3] or [C, 3, 3]
        extrs: Extrinsics [C, T, 3, 4] or [C, 3, 4]
        entity_base_path: Base path for entities (e.g., "world/masks")
        camera_ids: Optional list of camera ID strings [C]
        rgbs: Optional RGB images [C, T, H, W, 3]
        radius: Point radius for visualization
        min_depth: Minimum valid depth threshold
        max_depth: Maximum valid depth threshold
        fps: Frame rate for temporal logging
        max_frames: Optional limit on number of frames to visualize
        distance_threshold: Max average distance (meters) to match masks across cameras
        temporal_distance_threshold: Unused (kept for API compatibility)
    
    Returns:
        stats: Dictionary with statistics including global_id_mapping and frames_mapping
    
    Example:
        >>> # Camera A has local IDs [0, 1], Camera B has local IDs [0, 1, 3]
        >>> # Each local ID persists across time (with possible temporal gaps)
        >>> masks = {"hand": hand_masks}  # [2, 50, 480, 640] with values 0, 1, 2, 3, ...
        >>> stats = visualize_masks_with_temporal_global_ids(
        ...     {"hand": masks},
        ...     depths, intrs, extrs,
        ...     "world/hands",
        ...     distance_threshold=0.15,  # 15cm average distance threshold
        ... )
        >>> # If cam_A_id0 and cam_B_id1 have avg distance < 0.15m, they get same global ID
    """
    # Assign global IDs using temporal tracking
    print(f"[INFO] ========== Temporal Tracking & Global ID Assignment ==========")
    global_id_mapping, track_mapping = assign_global_ids_temporal(
        masks_dict=masks_dict,
        depths=depths,
        intrs=intrs,
        extrs=extrs,
        distance_threshold=distance_threshold,
        temporal_distance_threshold=temporal_distance_threshold,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    
    # Get dimensions
    first_mask_array = list(masks_dict.values())[0]
    C, T, H, W = first_mask_array.shape
    
    if max_frames is not None and T > max_frames:
        T = max_frames
    
    # Handle intrinsics
    if intrs.ndim == 3 and intrs.shape[1] == 3:
        intrs_expanded = np.tile(intrs[:, None], (1, T, 1, 1))
    else:
        intrs_expanded = intrs
    
    # Handle extrinsics
    if extrs.ndim == 3 and extrs.shape[1] in [3, 4]:
        extrs_expanded = np.tile(extrs[:, None], (1, T, 1, 1))
    else:
        extrs_expanded = extrs
    
    # Color palette
    color_palette = [
        np.array([255, 0, 0], dtype=np.uint8),    # Red
        np.array([0, 255, 0], dtype=np.uint8),    # Green
        np.array([0, 0, 255], dtype=np.uint8),    # Blue
        np.array([255, 255, 0], dtype=np.uint8),  # Yellow
        np.array([255, 0, 255], dtype=np.uint8),  # Magenta
        np.array([0, 255, 255], dtype=np.uint8),  # Cyan
        np.array([255, 128, 0], dtype=np.uint8),  # Orange
        np.array([128, 0, 255], dtype=np.uint8),  # Purple
    ]
    
    print(f"\n[INFO] ========== Visualizing with Temporal Global IDs ==========")
    
    all_stats = {}
    
    for mask_name, masks in masks_dict.items():
        print(f"\n[INFO] --- Processing mask type: '{mask_name}' ---")
        
        # Get mappings for this mask type
        local_to_global_per_camera = global_id_mapping[mask_name]
        frames_per_camera = track_mapping[mask_name]
        
        # Determine all global IDs
        all_global_ids = set()
        for cam_mapping in local_to_global_per_camera.values():
            all_global_ids.update(cam_mapping.values())
        all_global_ids = sorted(list(all_global_ids))
        
        print(f"[INFO] Global IDs for '{mask_name}': {all_global_ids}")
        print(f"[INFO] Visualizing across {C} cameras, {T} frames...")
        
        # Visualize each global ID
        global_id_stats = {}
        
        for global_id in all_global_ids:
            entity_path = f"{entity_base_path}/{mask_name}/global_id_{global_id}"
            color = color_palette[global_id % len(color_palette)]
            
            total_points = 0
            
            # Process each camera and frame
            for c in range(C):
                cam_id = camera_ids[c] if camera_ids is not None else f"cam_{c:03d}"
                local_to_global = local_to_global_per_camera[c]
                
                # Find local IDs that map to this global ID
                local_ids_for_global = [lid for lid, gid in local_to_global.items() if gid == global_id]
                
                if not local_ids_for_global:
                    continue
                
                for t in range(T):
                    time_seconds = t / fps
                    
                    # Get frame data
                    mask_frame = masks[c, t]
                    depth = depths[c, t]
                    intr = intrs_expanded[c, t]
                    extr = extrs_expanded[c, t]
                    rgb_frame = rgbs[c, t] if rgbs is not None else None
                    
                    # Create binary mask for this global ID (combine all local IDs that map to it)
                    binary_mask = np.zeros_like(mask_frame, dtype=bool)
                    for local_id in local_ids_for_global:
                        binary_mask |= (mask_frame == local_id)
                    
                    if not binary_mask.any():
                        continue
                    
                    # Visualize
                    num_points = visualize_mask_3d(
                        mask=binary_mask,
                        depth=depth,
                        intr=intr,
                        extr=extr,
                        entity_path=entity_path,
                        rgb=rgb_frame,
                        color=color,
                        radius=radius,
                        min_depth=min_depth,
                        max_depth=max_depth,
                        time_seconds=time_seconds,
                    )
                    
                    total_points += num_points
            
            global_id_stats[global_id] = total_points
            print(f"[INFO]   Global ID {global_id}: {total_points} points (color: RGB{tuple(color)}, path: '{entity_path}')")
        
        all_stats[mask_name] = {
            "global_id_stats": global_id_stats,
            "total_points": sum(global_id_stats.values()),
            "num_global_ids": len(all_global_ids),
            "local_ids_per_camera": {c: len(local_to_global_per_camera[c]) for c in range(C)},
        }
    
    print(f"\n[INFO] ========== Visualization Complete ==========")
    total_points_all = sum(s['total_points'] for s in all_stats.values())
    print(f"[INFO] Total points across all masks: {total_points_all}")
    
    # Print summary
    for mask_name in all_stats:
        print(f"[INFO] '{mask_name}' summary:")
        for c, num_local_ids in all_stats[mask_name]['local_ids_per_camera'].items():
            print(f"[INFO]   Camera {c}: {num_local_ids} local mask IDs")
    
    return {
        "mask_stats": all_stats,
        "global_id_mapping": global_id_mapping,
        "frames_mapping": track_mapping,
        "total_points": total_points_all,
    }
