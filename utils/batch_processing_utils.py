#!/usr/bin/env python3
"""
Utilities for batch processing query points to avoid overloading trackers.

This module provides functions to:
- Split query points into batches based on max points per batch
- Process batches sequentially to avoid memory overload
- Combine batch tracking results
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np


def split_query_points_into_batches(
    query_points: np.ndarray,
    max_points_per_batch: int,
    strategy: str = "temporal",
) -> List[np.ndarray]:
    """
    Split query points into batches to avoid overloading tracker.
    
    Args:
        query_points: [N, 4] array with columns [frame_idx, x, y, z]
        max_points_per_batch: Maximum number of points per batch
        strategy: Splitting strategy:
            - "temporal": Split by frame ranges (preserves temporal coherence)
            - "random": Random splitting (simple but may break coherence)
            - "spatial": Split by spatial proximity (not implemented yet)
            
    Returns:
        List of query point batches, each [M, 4] where M <= max_points_per_batch
        
    Example:
        >>> query_points = np.array([[0, 1, 2, 3], [0, 4, 5, 6], [1, 7, 8, 9]])
        >>> batches = split_query_points_into_batches(query_points, max_points_per_batch=2)
        >>> len(batches)
        2
    """
    N = len(query_points)
    
    if N <= max_points_per_batch:
        print(f"[INFO] Query points ({N}) within limit ({max_points_per_batch}), no batching needed")
        return [query_points]
    
    num_batches = int(np.ceil(N / max_points_per_batch))
    print(f"[INFO] Splitting {N} query points into {num_batches} batches (max {max_points_per_batch} per batch)")
    
    if strategy == "temporal":
        return _split_temporal(query_points, max_points_per_batch)
    elif strategy == "random":
        return _split_random(query_points, max_points_per_batch)
    elif strategy == "spatial":
        raise NotImplementedError("Spatial splitting not yet implemented")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _split_temporal(query_points: np.ndarray, max_points_per_batch: int) -> List[np.ndarray]:
    """
    Split query points by temporal ranges to preserve coherence.
    
    This groups points by frames and tries to keep frame groups together.
    """
    # Sort by frame index
    sorted_indices = np.argsort(query_points[:, 0])
    sorted_points = query_points[sorted_indices]
    
    # Group by frame
    unique_frames = np.unique(sorted_points[:, 0])
    frame_groups = []
    
    for frame_idx in unique_frames:
        frame_mask = sorted_points[:, 0] == frame_idx
        frame_points = sorted_points[frame_mask]
        frame_groups.append((frame_idx, frame_points))
    
    # Distribute frame groups into batches
    batches = []
    current_batch = []
    current_size = 0
    
    for frame_idx, frame_points in frame_groups:
        frame_size = len(frame_points)
        
        # If single frame exceeds limit, split it
        if frame_size > max_points_per_batch:
            # Save current batch if not empty
            if current_batch:
                batches.append(np.vstack(current_batch))
                current_batch = []
                current_size = 0
            
            # Split large frame into sub-batches
            for i in range(0, frame_size, max_points_per_batch):
                sub_batch = frame_points[i:i + max_points_per_batch]
                batches.append(sub_batch)
            continue
        
        # Check if adding this frame would exceed limit
        if current_size + frame_size > max_points_per_batch:
            # Save current batch and start new one
            if current_batch:
                batches.append(np.vstack(current_batch))
            current_batch = [frame_points]
            current_size = frame_size
        else:
            # Add to current batch
            current_batch.append(frame_points)
            current_size += frame_size
    
    # Add remaining batch
    if current_batch:
        batches.append(np.vstack(current_batch))
    
    # Log batch info
    for i, batch in enumerate(batches):
        frame_range = f"[{batch[:, 0].min():.0f}, {batch[:, 0].max():.0f}]"
        print(f"[INFO]   Batch {i}: {len(batch)} points, frames {frame_range}")
    
    return batches


def _split_random(query_points: np.ndarray, max_points_per_batch: int) -> List[np.ndarray]:
    """
    Randomly split query points into batches.
    
    Simple but may break temporal coherence.
    """
    N = len(query_points)
    num_batches = int(np.ceil(N / max_points_per_batch))
    
    # Shuffle indices
    shuffled_indices = np.random.permutation(N)
    
    batches = []
    for i in range(num_batches):
        start_idx = i * max_points_per_batch
        end_idx = min((i + 1) * max_points_per_batch, N)
        batch_indices = shuffled_indices[start_idx:end_idx]
        batch = query_points[batch_indices]
        batches.append(batch)
        print(f"[INFO]   Batch {i}: {len(batch)} points")
    
    return batches


def create_batch_npzs(
    input_npz: Path,
    query_points_batches: List[np.ndarray],
    output_dir: Path,
    base_name: str,
) -> List[Path]:
    """
    Create separate NPZ files for each query point batch.
    
    Args:
        input_npz: Original NPZ file (to copy other data from)
        query_points_batches: List of query point arrays
        output_dir: Directory for batch NPZ files
        base_name: Base name for output files
        
    Returns:
        List of paths to created batch NPZ files
        
    Example:
        >>> batch_npzs = create_batch_npzs(
        ...     Path("scene_query.npz"),
        ...     [batch1, batch2],
        ...     Path("./batches/"),
        ...     "scene"
        ... )
    """
    print(f"\n[INFO] Creating {len(query_points_batches)} batch NPZ files")
    
    # Load original data
    data = np.load(input_npz, allow_pickle=True)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_npz_paths = []
    
    for i, batch_query_points in enumerate(query_points_batches):
        # Create batch NPZ with original data + batch query points
        payload = dict(data)
        payload["query_points"] = batch_query_points
        payload["batch_index"] = i
        payload["total_batches"] = len(query_points_batches)
        
        # Save
        output_path = output_dir / f"{base_name}_batch_{i}.npz"
        np.savez_compressed(output_path, **payload)
        
        batch_npz_paths.append(output_path)
        print(f"[INFO]   Created batch {i}: {output_path.name} ({len(batch_query_points)} points)")
    
    return batch_npz_paths


def get_batch_info_from_npz(npz_path: Path) -> Optional[Dict[str, int]]:
    """
    Extract batch information from NPZ file if it exists.
    
    Args:
        npz_path: Path to NPZ file
        
    Returns:
        Dict with batch_index and total_batches, or None if not a batch file
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
        
        if "batch_index" in data and "total_batches" in data:
            return {
                "batch_index": int(data["batch_index"]),
                "total_batches": int(data["total_batches"]),
            }
        return None
    except Exception as e:
        print(f"[WARN] Failed to read batch info from {npz_path}: {e}")
        return None


def verify_batch_npzs(batch_npz_paths: List[Path]) -> bool:
    """
    Verify that batch NPZ files are valid and complete.
    
    Args:
        batch_npz_paths: List of batch NPZ file paths
        
    Returns:
        True if all batches are valid, False otherwise
    """
    print(f"\n[INFO] Verifying {len(batch_npz_paths)} batch NPZ files")
    
    batch_infos = []
    for path in batch_npz_paths:
        if not path.exists():
            print(f"[ERROR] Batch file not found: {path}")
            return False
        
        info = get_batch_info_from_npz(path)
        if info is None:
            print(f"[ERROR] No batch info in {path.name}")
            return False
        
        batch_infos.append(info)
    
    # Check indices are sequential
    indices = [info["batch_index"] for info in batch_infos]
    expected_indices = list(range(len(batch_npz_paths)))
    
    if sorted(indices) != expected_indices:
        print(f"[ERROR] Batch indices not sequential: {indices}")
        return False
    
    # Check total_batches matches
    total_batches = batch_infos[0]["total_batches"]
    if not all(info["total_batches"] == total_batches for info in batch_infos):
        print(f"[ERROR] Inconsistent total_batches across files")
        return False
    
    if total_batches != len(batch_npz_paths):
        print(f"[ERROR] total_batches ({total_batches}) != actual files ({len(batch_npz_paths)})")
        return False
    
    print(f"[INFO] ✓ All {len(batch_npz_paths)} batch files are valid")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Split query points into batches")
    parser.add_argument("--npz", type=Path, required=True, help="Input NPZ with query_points")
    parser.add_argument("--max-points", type=int, required=True, help="Max points per batch")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--base-name", type=str, default="batch", help="Base name for output")
    parser.add_argument(
        "--strategy",
        type=str,
        default="temporal",
        choices=["temporal", "random"],
        help="Splitting strategy",
    )
    
    args = parser.parse_args()
    
    # Load query points
    data = np.load(args.npz, allow_pickle=True)
    
    if "query_points" not in data:
        raise KeyError("No 'query_points' in NPZ")
    
    query_points = data["query_points"]
    print(f"[INFO] Loaded {len(query_points)} query points from {args.npz.name}")
    
    # Split into batches
    batches = split_query_points_into_batches(
        query_points=query_points,
        max_points_per_batch=args.max_points,
        strategy=args.strategy,
    )
    
    # Create batch NPZ files
    batch_paths = create_batch_npzs(
        input_npz=args.npz,
        query_points_batches=batches,
        output_dir=args.output_dir,
        base_name=args.base_name,
    )
    
    # Verify
    verify_batch_npzs(batch_paths)
    
    print(f"\n[INFO] Done! Created {len(batch_paths)} batch files in {args.output_dir}")
