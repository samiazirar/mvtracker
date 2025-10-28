#!/usr/bin/env python3
"""
Helper script to identify potential grasp timestamps by analyzing finger distances.

This script analyzes hand keypoints from a hand-tracked NPZ file and suggests
frame indices where the fingers are likely grasping an object (close together).

Usage:
    python scripts/identify_grasp_timestamps.py \\
        --npz data/human_high_res_filtered/task_*_hand_tracked.npz \\
        --threshold 0.05 \\
        --min-duration 3
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


# MANO keypoint indices
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12


def load_hand_data(npz_path: Path) -> dict:
    """Load hand tracking data from NPZ file."""
    print(f"[INFO] Loading {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)
    return data


def compute_finger_distances(query_points: np.ndarray, num_frames: int) -> np.ndarray:
    """
    Compute the average distance between finger tips for each frame.
    
    Args:
        query_points: [N, 4] array where cols are [frame_idx, x, y, z]
        num_frames: Total number of frames
        
    Returns:
        distances: [T] array of average finger distances per frame
    """
    distances = np.zeros(num_frames, dtype=np.float32)
    
    for t in range(num_frames):
        # Get all query points for this frame
        frame_mask = query_points[:, 0] == t
        frame_pts = query_points[frame_mask, 1:4]  # [N, 3] world coords
        
        if len(frame_pts) < 2:
            distances[t] = np.nan  # No data
            continue
        
        # Compute pairwise distances between all finger keypoints
        dists = []
        for i in range(len(frame_pts)):
            for j in range(i + 1, len(frame_pts)):
                dist = np.linalg.norm(frame_pts[i] - frame_pts[j])
                dists.append(dist)
        
        if dists:
            distances[t] = np.mean(dists)  # Average distance
        else:
            distances[t] = np.nan
    
    return distances


def identify_grasp_periods(distances: np.ndarray, threshold: float, 
                           min_duration: int) -> List[Tuple[int, int]]:
    """
    Identify continuous periods where finger distance is below threshold.
    
    Args:
        distances: [T] array of finger distances
        threshold: Maximum distance to consider as grasping (in meters)
        min_duration: Minimum number of consecutive frames to consider
        
    Returns:
        List of (start_frame, end_frame) tuples
    """
    # Find frames where distance is below threshold
    is_grasping = distances < threshold
    
    # Find continuous periods
    periods = []
    start = None
    
    for t in range(len(is_grasping)):
        if is_grasping[t] and start is None:
            start = t
        elif not is_grasping[t] and start is not None:
            if t - start >= min_duration:
                periods.append((start, t - 1))
            start = None
    
    # Handle case where grasping continues to end
    if start is not None and len(is_grasping) - start >= min_duration:
        periods.append((start, len(is_grasping) - 1))
    
    return periods


def suggest_timestamps(periods: List[Tuple[int, int]], num_samples: int = 3) -> List[int]:
    """
    Suggest specific timestamps to use for each grasp period.
    
    Args:
        periods: List of (start, end) grasp periods
        num_samples: Number of samples to take from each period
        
    Returns:
        List of suggested frame indices
    """
    suggestions = []
    
    for start, end in periods:
        duration = end - start + 1
        
        if duration <= num_samples:
            # Use all frames if period is short
            suggestions.extend(range(start, end + 1))
        else:
            # Sample evenly across the period
            indices = np.linspace(start, end, num_samples, dtype=int)
            suggestions.extend(indices.tolist())
    
    return sorted(set(suggestions))  # Remove duplicates and sort


def plot_distances(distances: np.ndarray, threshold: float, 
                   periods: List[Tuple[int, int]], 
                   suggestions: List[int],
                   output_path: Path):
    """
    Create a visualization of finger distances and suggested grasp frames.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot distances
    frames = np.arange(len(distances))
    valid = ~np.isnan(distances)
    ax.plot(frames[valid], distances[valid], 'b-', linewidth=2, label='Finger Distance')
    
    # Plot threshold
    ax.axhline(threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold ({threshold}m)')
    
    # Highlight grasp periods
    for start, end in periods:
        ax.axvspan(start, end, alpha=0.3, color='green', label='Grasp Period' if start == periods[0][0] else '')
    
    # Mark suggested timestamps
    for t in suggestions:
        if t < len(distances) and not np.isnan(distances[t]):
            ax.plot(t, distances[t], 'ro', markersize=10, 
                   label='Suggested Timestamp' if t == suggestions[0] else '')
    
    ax.set_xlabel('Frame Index', fontsize=12)
    ax.set_ylabel('Average Finger Distance (m)', fontsize=12)
    ax.set_title('Hand Grasp Detection', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"[INFO] Saved visualization: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Identify grasp timestamps from hand tracking data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python scripts/identify_grasp_timestamps.py \\
        --npz data/human_high_res_filtered/task_*_hand_tracked.npz \\
        --threshold 0.05 \\
        --min-duration 3 \\
        --plot grasp_analysis.png
        """
    )
    
    parser.add_argument(
        "--npz",
        required=True,
        help="Path to *_hand_tracked.npz file"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Maximum finger distance to consider as grasping (meters, default: 0.05)"
    )
    parser.add_argument(
        "--min-duration",
        type=int,
        default=3,
        help="Minimum number of consecutive frames for grasp (default: 3)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of sample frames per grasp period (default: 3)"
    )
    parser.add_argument(
        "--plot",
        default=None,
        help="Save visualization plot to this path (optional)"
    )
    
    args = parser.parse_args()
    
    # Load data
    npz_path = Path(args.npz)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    
    data = load_hand_data(npz_path)
    query_points = data["query_points"]  # [N, 4]
    num_frames = data["rgbs"].shape[1]  # T
    
    print(f"[INFO] Analyzing {num_frames} frames with {len(query_points)} query points...")
    
    # Compute finger distances
    distances = compute_finger_distances(query_points, num_frames)
    
    # Find valid frames (where we have data)
    valid_frames = (~np.isnan(distances)).sum()
    print(f"[INFO] Valid frames with hand detections: {valid_frames}/{num_frames}")
    
    if valid_frames == 0:
        print("[ERROR] No valid hand detections found!")
        return
    
    # Compute statistics
    valid_dists = distances[~np.isnan(distances)]
    print(f"\n[INFO] Finger distance statistics:")
    print(f"  - Mean: {valid_dists.mean():.4f}m")
    print(f"  - Std: {valid_dists.std():.4f}m")
    print(f"  - Min: {valid_dists.min():.4f}m")
    print(f"  - Max: {valid_dists.max():.4f}m")
    
    # Identify grasp periods
    periods = identify_grasp_periods(distances, args.threshold, args.min_duration)
    
    if not periods:
        print(f"\n[WARNING] No grasp periods found with threshold {args.threshold}m")
        print(f"[HINT] Try increasing --threshold or decreasing --min-duration")
        
        # Suggest a threshold based on data
        suggested_threshold = np.percentile(valid_dists, 25)
        print(f"[HINT] Suggested threshold based on data: {suggested_threshold:.4f}m")
        return
    
    print(f"\n[INFO] Found {len(periods)} grasp period(s):")
    for i, (start, end) in enumerate(periods, 1):
        duration = end - start + 1
        avg_dist = np.nanmean(distances[start:end+1])
        print(f"  {i}. Frames {start}-{end} (duration: {duration}, avg dist: {avg_dist:.4f}m)")
    
    # Suggest specific timestamps
    suggestions = suggest_timestamps(periods, args.num_samples)
    
    print(f"\n[INFO] Suggested grasp timestamps for SAM prompting:")
    print(f"  {' '.join(map(str, suggestions))}")
    
    print(f"\n[COMMAND] Use these timestamps with:")
    print(f"  bash scripts/run_object_tracking_example.sh")
    print(f"  # Set: GRASP_TIMESTAMPS=\"{' '.join(map(str, suggestions))}\"")
    
    # Create visualization if requested
    if args.plot:
        plot_path = Path(args.plot)
        if not plot_path.is_absolute():
            plot_path = npz_path.parent / plot_path
        plot_distances(distances, args.threshold, periods, suggestions, plot_path)


if __name__ == "__main__":
    main()
