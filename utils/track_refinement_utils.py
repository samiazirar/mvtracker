#!/usr/bin/env python3
"""
Utilities for refining and filtering tracking results.

Functions for:
- Calculating track motion metrics
- Filtering static tracks based on motion threshold
- Aligning tracks to query points (removing invalid tracks)
- Computing track statistics
"""

from typing import Dict, Optional, Tuple
import numpy as np


def calculate_track_motion(
    tracks: np.ndarray,
    visibilities: Optional[np.ndarray] = None,
    method: str = "total_displacement",
) -> np.ndarray:
    """
    Calculate motion metric for each track.
    
    Args:
        tracks: Track trajectories [T, N, 3] in 3D world coordinates
        visibilities: Track visibility mask [T, N] (optional)
        method: Motion metric to compute:
            - "total_displacement": Sum of all frame-to-frame distances
            - "max_displacement": Maximum distance from starting point
            - "std_displacement": Standard deviation of positions
            - "endpoint_distance": Distance between first and last visible points
    
    Returns:
        motion: Motion metric per track [N]
    """
    T, N, D = tracks.shape
    assert D == 3, "Tracks must be 3D"
    
    if visibilities is None:
        visibilities = np.ones((T, N), dtype=bool)
    
    motion = np.zeros(N)
    
    for i in range(N):
        track = tracks[:, i, :]  # [T, 3]
        vis = visibilities[:, i]  # [T]
        
        # Skip if track has no visible points
        if not vis.any():
            motion[i] = 0.0
            continue
        
        visible_points = track[vis]  # [T_vis, 3]
        
        if method == "total_displacement":
            # Sum of frame-to-frame distances
            if len(visible_points) < 2:
                motion[i] = 0.0
            else:
                displacements = np.linalg.norm(
                    visible_points[1:] - visible_points[:-1], axis=-1
                )
                motion[i] = displacements.sum()
        
        elif method == "max_displacement":
            # Maximum distance from first visible point
            start_point = visible_points[0]
            distances = np.linalg.norm(visible_points - start_point, axis=-1)
            motion[i] = distances.max()
        
        elif method == "std_displacement":
            # Standard deviation of positions (captures overall movement)
            motion[i] = visible_points.std()
        
        elif method == "endpoint_distance":
            # Distance between first and last visible points
            if len(visible_points) < 2:
                motion[i] = 0.0
            else:
                motion[i] = np.linalg.norm(visible_points[-1] - visible_points[0])
        
        else:
            raise ValueError(f"Unknown motion method: {method}")
    
    return motion


def filter_static_tracks(
    tracks: np.ndarray,
    visibilities: Optional[np.ndarray] = None,
    motion_threshold: float = 0.01,
    motion_method: str = "total_displacement",
    query_points: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Filter out static tracks based on motion threshold.
    
    Args:
        tracks: Track trajectories [T, N, 3] in 3D world coordinates
        visibilities: Track visibility mask [T, N]
        motion_threshold: Minimum motion to keep a track (in world units)
        motion_method: Method to compute motion (see calculate_track_motion)
        query_points: Optional query points [N, 4] where cols=[t, x, y, z]
    
    Returns:
        filtered_tracks: Tracks with motion > threshold [T, N_filtered, 3]
        filtered_visibilities: Visibility for filtered tracks [T, N_filtered]
        filtered_query_points: Query points for filtered tracks [N_filtered, 4] (if provided)
        stats: Dictionary with filtering statistics
    """
    T, N, D = tracks.shape
    
    # Calculate motion for each track
    motion = calculate_track_motion(tracks, visibilities, method=motion_method)
    
    # Identify moving tracks
    moving_mask = motion > motion_threshold
    n_moving = moving_mask.sum()
    
    # Filter tracks
    filtered_tracks = tracks[:, moving_mask, :]
    
    if visibilities is not None:
        filtered_visibilities = visibilities[:, moving_mask]
    else:
        filtered_visibilities = np.ones((T, n_moving), dtype=bool)
    
    if query_points is not None:
        filtered_query_points = query_points[moving_mask]
    else:
        filtered_query_points = None
    
    # Compute statistics
    stats = {
        "n_original": N,
        "n_moving": n_moving,
        "n_static": N - n_moving,
        "fraction_moving": n_moving / N if N > 0 else 0.0,
        "motion_threshold": motion_threshold,
        "motion_method": motion_method,
        "motion_min": motion.min(),
        "motion_max": motion.max(),
        "motion_mean": motion.mean(),
        "motion_median": np.median(motion),
        "motion_moving_min": motion[moving_mask].min() if n_moving > 0 else 0.0,
        "motion_moving_mean": motion[moving_mask].mean() if n_moving > 0 else 0.0,
        "motion_static_max": motion[~moving_mask].max() if (N - n_moving) > 0 else 0.0,
    }
    
    return filtered_tracks, filtered_visibilities, filtered_query_points, stats


def align_tracks_to_points(
    tracks: np.ndarray,
    visibilities: np.ndarray,
    query_points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Remove tracks where no valid points exist (i.e., never visible).
    Also ensures query points are valid.
    
    Args:
        tracks: Track trajectories [T, N, 3]
        visibilities: Track visibility mask [T, N]
        query_points: Query points [N, 4] where cols=[t, x, y, z]
    
    Returns:
        aligned_tracks: Tracks with at least one visible point [T, N_valid, 3]
        aligned_visibilities: Visibility for valid tracks [T, N_valid]
        aligned_query_points: Query points for valid tracks [N_valid, 4]
        stats: Dictionary with alignment statistics
    """
    T, N, D = tracks.shape
    
    # Check which tracks have at least one visible point
    has_visible_points = visibilities.any(axis=0)  # [N]
    n_valid = has_visible_points.sum()
    
    # Filter to valid tracks only
    aligned_tracks = tracks[:, has_visible_points, :]
    aligned_visibilities = visibilities[:, has_visible_points]
    aligned_query_points = query_points[has_visible_points]
    
    # Compute statistics
    visibility_counts = visibilities.sum(axis=0)  # [N]
    stats = {
        "n_original": N,
        "n_valid": n_valid,
        "n_invalid": N - n_valid,
        "fraction_valid": n_valid / N if N > 0 else 0.0,
        "visibility_min": visibility_counts[has_visible_points].min() if n_valid > 0 else 0,
        "visibility_max": visibility_counts[has_visible_points].max() if n_valid > 0 else 0,
        "visibility_mean": visibility_counts[has_visible_points].mean() if n_valid > 0 else 0.0,
    }
    
    return aligned_tracks, aligned_visibilities, aligned_query_points, stats


def refine_tracks(
    tracks: np.ndarray,
    visibilities: Optional[np.ndarray] = None,
    query_points: Optional[np.ndarray] = None,
    motion_threshold: float = 0.01,
    motion_method: str = "total_displacement",
    remove_invalid: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict]:
    """
    Complete track refinement pipeline:
    1. Remove tracks with no visible points (if remove_invalid=True)
    2. Filter static tracks based on motion threshold
    
    Args:
        tracks: Track trajectories [T, N, 3]
        visibilities: Track visibility mask [T, N] (optional)
        query_points: Query points [N, 4] where cols=[t, x, y, z] (optional)
        motion_threshold: Minimum motion to keep a track
        motion_method: Method to compute motion
        remove_invalid: Whether to remove tracks with no visible points
        verbose: Print statistics
    
    Returns:
        refined_tracks: Refined tracks [T, N_refined, 3]
        refined_visibilities: Visibility for refined tracks [T, N_refined]
        refined_query_points: Query points for refined tracks [N_refined, 4] (if provided)
        stats: Dictionary with all refinement statistics
    """
    T, N, D = tracks.shape
    
    if visibilities is None:
        visibilities = np.ones((T, N), dtype=bool)
    
    all_stats = {"n_original": N}
    
    # Step 1: Remove invalid tracks (no visible points)
    if remove_invalid:
        tracks, visibilities, query_points, align_stats = align_tracks_to_points(
            tracks, visibilities, query_points
        )
        all_stats["alignment"] = align_stats
        
        if verbose:
            print(f"[INFO] Alignment: {align_stats['n_original']} -> {align_stats['n_valid']} tracks "
                  f"({align_stats['n_invalid']} removed with no visible points)")
    
    # Step 2: Filter static tracks
    tracks, visibilities, query_points, motion_stats = filter_static_tracks(
        tracks, visibilities, motion_threshold, motion_method, query_points
    )
    all_stats["motion_filtering"] = motion_stats
    
    if verbose:
        print(f"[INFO] Motion filtering: {motion_stats['n_original']} -> {motion_stats['n_moving']} tracks "
              f"({motion_stats['n_static']} static removed, threshold={motion_threshold:.4f})")
        print(f"[INFO]   Motion range: [{motion_stats['motion_min']:.4f}, {motion_stats['motion_max']:.4f}], "
              f"mean={motion_stats['motion_mean']:.4f}, median={motion_stats['motion_median']:.4f}")
    
    # Final statistics
    all_stats["n_refined"] = tracks.shape[1]
    all_stats["total_reduction_factor"] = N / tracks.shape[1] if tracks.shape[1] > 0 else float('inf')
    
    if verbose:
        print(f"[INFO] Total refinement: {N} -> {all_stats['n_refined']} tracks "
              f"({N - all_stats['n_refined']} removed, {all_stats['total_reduction_factor']:.2f}x reduction)")
    
    return tracks, visibilities, query_points, all_stats


def compute_track_statistics(
    tracks: np.ndarray,
    visibilities: Optional[np.ndarray] = None,
) -> Dict:
    """
    Compute comprehensive statistics about tracks.
    
    Args:
        tracks: Track trajectories [T, N, 3]
        visibilities: Track visibility mask [T, N]
    
    Returns:
        stats: Dictionary with track statistics
    """
    T, N, D = tracks.shape
    
    if visibilities is None:
        visibilities = np.ones((T, N), dtype=bool)
    
    # Basic info
    stats = {
        "n_frames": T,
        "n_tracks": N,
        "dimensionality": D,
    }
    
    # Visibility statistics
    visibility_counts = visibilities.sum(axis=0)  # Visible frames per track
    stats.update({
        "visibility_min": visibility_counts.min(),
        "visibility_max": visibility_counts.max(),
        "visibility_mean": visibility_counts.mean(),
        "visibility_median": np.median(visibility_counts),
    })
    
    # Motion statistics (using total displacement)
    motion = calculate_track_motion(tracks, visibilities, method="total_displacement")
    stats.update({
        "motion_total_disp_min": motion.min(),
        "motion_total_disp_max": motion.max(),
        "motion_total_disp_mean": motion.mean(),
        "motion_total_disp_median": np.median(motion),
    })
    
    # Spatial extent
    visible_tracks = tracks.copy()
    visible_tracks[~visibilities] = np.nan  # Mark invisible as NaN
    
    stats.update({
        "spatial_extent_x": np.nanmax(tracks[:, :, 0]) - np.nanmin(tracks[:, :, 0]),
        "spatial_extent_y": np.nanmax(tracks[:, :, 1]) - np.nanmin(tracks[:, :, 1]),
        "spatial_extent_z": np.nanmax(tracks[:, :, 2]) - np.nanmin(tracks[:, :, 2]),
        "center_x": np.nanmean(tracks[:, :, 0]),
        "center_y": np.nanmean(tracks[:, :, 1]),
        "center_z": np.nanmean(tracks[:, :, 2]),
    })
    
    return stats
