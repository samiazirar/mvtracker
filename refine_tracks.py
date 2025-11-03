#!/usr/bin/env python3
"""
Refine tracking results by filtering static tracks and aligning to query points.

This script:
1. Loads tracking results from NPZ files (or extracts from RRD)
2. Filters out static tracks based on motion threshold
3. Removes tracks with no visible points
4. Saves refined NPZ with only moving, valid tracks

Usage:
    python refine_tracks.py --input tracking_results.npz --output refined_tracks.npz
    
    # With custom motion threshold
    python refine_tracks.py --input tracks.npz --output refined.npz --motion-threshold 0.05
    
    # Process multiple NPZ files from per-mask tracking
    python refine_tracks.py --input-dir ./tracking_per_mask --output-dir ./refined_tracks
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import sys

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.track_refinement_utils import refine_tracks, compute_track_statistics


def load_tracking_results(npz_path: Path) -> Dict:
    """
    Load tracking results from NPZ file.
    
    Expected keys:
        - tracks_3d or traj_e: 3D trajectories [T, N, 3]
        - visibilities or vis_e: Visibility mask [T, N]
        - query_points (optional): Query points [N, 4]
        - rgbs (optional): RGB frames
        - intrinsics (optional): Camera intrinsics
        - extrinsics (optional): Camera extrinsics
        - camera_ids (optional): Camera identifiers
    
    Returns:
        data: Dictionary with tracking data
    """
    print(f"[INFO] Loading tracking results from: {npz_path}")
    
    data = np.load(npz_path, allow_pickle=True)
    
    # Try different key names
    if "tracks_3d" in data:
        tracks = data["tracks_3d"]
    elif "traj_e" in data:
        tracks = data["traj_e"]
    else:
        raise KeyError("No tracks found in NPZ. Expected 'tracks_3d' or 'traj_e'")
    
    if "visibilities" in data:
        visibilities = data["visibilities"]
    elif "vis_e" in data:
        visibilities = data["vis_e"]
    else:
        print("[WARN] No visibility mask found, assuming all tracks visible")
        visibilities = np.ones(tracks.shape[:-1], dtype=bool)
    
    query_points = data.get("query_points", None)
    
    # Load additional data if available
    result = {
        "tracks_3d": tracks,
        "visibilities": visibilities,
        "query_points": query_points,
    }
    
    # Copy other useful keys
    for key in ["rgbs", "camera_ids", "depths", "tracker", "temporal_stride", "spatial_downsample", "timestamps", "per_camera_timestamps"]:
        if key in data:
            result[key] = data[key]
    
    # Handle intrinsics (might be 'intrs' or 'intrinsics')
    if "intrinsics" in data:
        result["intrinsics"] = data["intrinsics"]
    elif "intrs" in data:
        result["intrs"] = data["intrs"]
    
    # Handle extrinsics (might be 'extrs' or 'extrinsics')
    if "extrinsics" in data:
        result["extrinsics"] = data["extrinsics"]
    elif "extrs" in data:
        result["extrs"] = data["extrs"]
    
    print(f"[INFO] Loaded tracks: {tracks.shape}, visibilities: {visibilities.shape}")
    if query_points is not None:
        print(f"[INFO] Query points: {query_points.shape}")
    
    return result


def save_refined_results(
    output_path: Path,
    tracks_3d: np.ndarray,
    visibilities: np.ndarray,
    query_points: Optional[np.ndarray],
    original_data: Dict,
    stats: Dict,
) -> Path:
    """
    Save refined tracking results to NPZ file.
    
    Args:
        output_path: Output NPZ file path
        tracks_3d: Refined 3D tracks [T, N_refined, 3]
        visibilities: Refined visibility mask [T, N_refined]
        query_points: Refined query points [N_refined, 4] (if available)
        original_data: Original data dict (for copying additional keys)
        stats: Refinement statistics
    
    Returns:
        output_path: Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data to save
    save_data = {
        "tracks_3d": tracks_3d,
        "visibilities": visibilities,
    }
    
    if query_points is not None:
        save_data["query_points"] = query_points
    
    # Copy additional keys from original data
    # Handle both 'intrs'/'extrs' and 'intrinsics'/'extrinsics' naming
    for key in ["rgbs", "camera_ids", "depths"]:
        if key in original_data:
            save_data[key] = original_data[key]
    
    # Handle intrinsics (might be 'intrs' or 'intrinsics')
    if "intrinsics" in original_data:
        save_data["intrinsics"] = original_data["intrinsics"]
    elif "intrs" in original_data:
        save_data["intrinsics"] = original_data["intrs"]
    
    # Handle extrinsics (might be 'extrs' or 'extrinsics')
    if "extrinsics" in original_data:
        save_data["extrinsics"] = original_data["extrinsics"]
    elif "extrs" in original_data:
        save_data["extrinsics"] = original_data["extrs"]
    
    # Copy optional metadata
    for key in ["tracker", "temporal_stride", "spatial_downsample", "timestamps", "per_camera_timestamps"]:
        if key in original_data:
            save_data[key] = original_data[key]
    
    # Save statistics as JSON string
    save_data["refinement_stats"] = json.dumps(stats, indent=2)
    
    # Save to NPZ
    np.savez_compressed(output_path, **save_data)
    
    print(f"[INFO] Saved refined results to: {output_path}")
    
    return output_path


def refine_single_npz(
    input_path: Path,
    output_path: Path,
    motion_threshold: float = 0.01,
    motion_method: str = "total_displacement",
    remove_invalid: bool = True,
    save_stats: bool = True,
    verbose: bool = True,
) -> Dict:
    """
    Refine a single NPZ file.
    
    Args:
        input_path: Input NPZ file
        output_path: Output NPZ file
        motion_threshold: Minimum motion to keep a track
        motion_method: Method to compute motion
        remove_invalid: Whether to remove tracks with no visible points
        save_stats: Whether to save statistics to JSON
        verbose: Print detailed information
    
    Returns:
        stats: Refinement statistics
    """
    if verbose:
        print(f"\n[INFO] ========================================")
        print(f"[INFO] Refining: {input_path.name}")
        print(f"[INFO] ========================================")
    
    # Load data
    data = load_tracking_results(input_path)
    
    # Compute original statistics
    if verbose:
        print(f"\n[INFO] Original track statistics:")
        orig_stats = compute_track_statistics(data["tracks_3d"], data["visibilities"])
        for key, val in orig_stats.items():
            print(f"[INFO]   {key}: {val}")
    
    # Refine tracks
    if verbose:
        print(f"\n[INFO] Refining tracks...")
    
    refined_tracks, refined_vis, refined_query, refine_stats = refine_tracks(
        tracks=data["tracks_3d"],
        visibilities=data["visibilities"],
        query_points=data["query_points"],
        motion_threshold=motion_threshold,
        motion_method=motion_method,
        remove_invalid=remove_invalid,
        verbose=verbose,
    )
    
    # Compute refined statistics
    if verbose:
        print(f"\n[INFO] Refined track statistics:")
        refined_stats = compute_track_statistics(refined_tracks, refined_vis)
        for key, val in refined_stats.items():
            print(f"[INFO]   {key}: {val}")
    
    # Save results
    save_refined_results(
        output_path=output_path,
        tracks_3d=refined_tracks,
        visibilities=refined_vis,
        query_points=refined_query,
        original_data=data,
        stats=refine_stats,
    )
    
    # Save statistics to JSON
    if save_stats:
        stats_path = output_path.parent / f"{output_path.stem}_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(refine_stats, f, indent=2)
        if verbose:
            print(f"[INFO] Saved statistics to: {stats_path}")
    
    return refine_stats


def refine_multiple_npz(
    input_dir: Path,
    output_dir: Path,
    pattern: str = "*.npz",
    motion_threshold: float = 0.01,
    motion_method: str = "total_displacement",
    remove_invalid: bool = True,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """
    Refine multiple NPZ files in a directory.
    
    Args:
        input_dir: Input directory with NPZ files
        output_dir: Output directory for refined NPZ files
        pattern: Glob pattern for NPZ files
        motion_threshold: Minimum motion to keep a track
        motion_method: Method to compute motion
        remove_invalid: Whether to remove tracks with no visible points
        verbose: Print detailed information
    
    Returns:
        all_stats: Dictionary mapping filename to refinement statistics
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all NPZ files
    npz_files = sorted(input_dir.glob(pattern))
    
    if not npz_files:
        raise FileNotFoundError(f"No NPZ files found in {input_dir} with pattern {pattern}")
    
    print(f"\n[INFO] ========================================")
    print(f"[INFO] Refining {len(npz_files)} NPZ files")
    print(f"[INFO] Input directory: {input_dir}")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] ========================================")
    
    # Process each file
    all_stats = {}
    
    for i, npz_file in enumerate(npz_files):
        print(f"\n[INFO] Processing {i+1}/{len(npz_files)}: {npz_file.name}")
        
        output_path = output_dir / f"{npz_file.stem}_refined.npz"
        
        try:
            stats = refine_single_npz(
                input_path=npz_file,
                output_path=output_path,
                motion_threshold=motion_threshold,
                motion_method=motion_method,
                remove_invalid=remove_invalid,
                save_stats=True,
                verbose=verbose,
            )
            all_stats[npz_file.name] = stats
        except Exception as e:
            print(f"[ERROR] Failed to process {npz_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save combined statistics
    combined_stats_path = output_dir / "combined_refinement_stats.json"
    with open(combined_stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    print(f"\n[INFO] ========================================")
    print(f"[INFO] Refined {len(all_stats)}/{len(npz_files)} files successfully")
    print(f"[INFO] Combined statistics saved to: {combined_stats_path}")
    print(f"[INFO] ========================================")
    
    return all_stats


def main():
    parser = argparse.ArgumentParser(
        description="Refine tracking results by filtering static tracks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Input/output
    parser.add_argument(
        "--input",
        type=Path,
        help="Input NPZ file with tracking results",
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        help="Output NPZ file for refined results",
    )
    
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Input directory with multiple NPZ files (alternative to --input)",
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for refined NPZ files (used with --input-dir)",
    )
    
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.npz",
        help="Glob pattern for NPZ files in input directory (default: *.npz)",
    )
    
    # Refinement parameters
    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=0.01,
        help="Minimum motion to keep a track (in world units, default: 0.01)",
    )
    
    parser.add_argument(
        "--motion-method",
        type=str,
        default="total_displacement",
        choices=["total_displacement", "max_displacement", "std_displacement", "endpoint_distance"],
        help="Method to compute track motion (default: total_displacement)",
    )
    
    parser.add_argument(
        "--keep-invalid",
        action="store_true",
        help="Keep tracks with no visible points (default: remove them)",
    )
    
    parser.add_argument(
        "--no-save-stats",
        action="store_true",
        help="Don't save statistics to JSON file",
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.input and args.input_dir:
        parser.error("Cannot specify both --input and --input-dir")
    
    if not args.input and not args.input_dir:
        parser.error("Must specify either --input or --input-dir")
    
    if args.input and not args.output:
        parser.error("--output is required when using --input")
    
    if args.input_dir and not args.output_dir:
        parser.error("--output-dir is required when using --input-dir")
    
    # Process single file or directory
    try:
        if args.input:
            refine_single_npz(
                input_path=args.input,
                output_path=args.output,
                motion_threshold=args.motion_threshold,
                motion_method=args.motion_method,
                remove_invalid=not args.keep_invalid,
                save_stats=not args.no_save_stats,
                verbose=not args.quiet,
            )
        else:
            refine_multiple_npz(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                pattern=args.pattern,
                motion_threshold=args.motion_threshold,
                motion_method=args.motion_method,
                remove_invalid=not args.keep_invalid,
                verbose=not args.quiet,
            )
        
        print(f"\n[INFO] Success! Refinement complete.")
        
    except Exception as e:
        print(f"\n[ERROR] Refinement failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
