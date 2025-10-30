#!/usr/bin/env python3
"""
Export tracking results to per-camera videos.

This script:
1. Loads refined tracking results from NPZ files
2. Projects 3D tracks to each camera view
3. Exports videos with track overlays for each camera

Usage:
    python export_tracks_to_videos.py --input refined_tracks.npz --output-dir ./videos
    
    # With custom trail length and FPS
    python export_tracks_to_videos.py --input tracks.npz --output-dir ./videos --trail-length 20 --fps 30
    
    # Process multiple NPZ files
    python export_tracks_to_videos.py --input-dir ./refined_tracks --output-dir ./videos
"""

import argparse
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import sys

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.video_export_utils import export_tracks_to_videos_per_camera


def load_data_for_video_export(npz_path: Path) -> Dict:
    """
    Load data needed for video export.
    
    Expected keys:
        - tracks_3d: 3D trajectories [T, N, 3]
        - visibilities: Visibility mask [T, N]
        - rgbs: RGB frames [V, T, 3, H, W]
        - intrinsics: Camera intrinsics [V, 3, 3]
        - extrinsics: Camera extrinsics [V, 4, 4]
        - query_points (optional): Query points [N, 4]
        - camera_ids (optional): Camera identifiers
    
    Returns:
        data: Dictionary with required data
    """
    print(f"[INFO] Loading data from: {npz_path}")
    
    data = np.load(npz_path, allow_pickle=True)
    
    # Check required keys
    required_keys = ["tracks_3d", "visibilities", "rgbs", "intrinsics", "extrinsics"]
    missing_keys = [k for k in required_keys if k not in data]
    
    if missing_keys:
        raise KeyError(f"Missing required keys: {missing_keys}")
    
    result = {
        "tracks_3d": data["tracks_3d"],
        "visibilities": data["visibilities"],
        "rgbs": data["rgbs"],
        "intrinsics": data["intrinsics"],
        "extrinsics": data["extrinsics"],
    }
    
    # Optional keys
    if "query_points" in data:
        result["query_points"] = data["query_points"]
    
    if "camera_ids" in data:
        # Handle object array
        cam_ids = data["camera_ids"]
        if isinstance(cam_ids, np.ndarray) and cam_ids.dtype == object:
            result["camera_ids"] = [str(c) for c in cam_ids]
        else:
            result["camera_ids"] = list(cam_ids)
    
    print(f"[INFO] Loaded:")
    print(f"[INFO]   Tracks: {result['tracks_3d'].shape}")
    print(f"[INFO]   Visibilities: {result['visibilities'].shape}")
    print(f"[INFO]   RGB frames: {result['rgbs'].shape}")
    print(f"[INFO]   Cameras: {result['intrinsics'].shape[0]}")
    if "query_points" in result:
        print(f"[INFO]   Query points: {result['query_points'].shape}")
    
    return result


def export_single_npz(
    input_path: Path,
    output_dir: Path,
    base_name: Optional[str] = None,
    fps: float = 30.0,
    trail_length: int = 10,
    verbose: bool = True,
) -> Dict[str, Path]:
    """
    Export videos for a single NPZ file.
    
    Args:
        input_path: Input NPZ file
        output_dir: Output directory for videos
        base_name: Base name for output files (defaults to input filename)
        fps: Frames per second
        trail_length: Number of previous frames to show in trail
        verbose: Print detailed information
    
    Returns:
        video_paths: Dictionary mapping camera_id to video path
    """
    if verbose:
        print(f"\n[INFO] ========================================")
        print(f"[INFO] Exporting videos: {input_path.name}")
        print(f"[INFO] ========================================")
    
    # Load data
    data = load_data_for_video_export(input_path)
    
    # Determine base name
    if base_name is None:
        base_name = input_path.stem
    
    # Export videos
    video_paths = export_tracks_to_videos_per_camera(
        output_dir=output_dir,
        base_name=base_name,
        rgbs=data["rgbs"],
        tracks_3d=data["tracks_3d"],
        visibilities=data["visibilities"],
        intrinsics=data["intrinsics"],
        extrinsics=data["extrinsics"],
        camera_ids=data.get("camera_ids"),
        query_points=data.get("query_points"),
        fps=fps,
        trail_length=trail_length,
        verbose=verbose,
    )
    
    return video_paths


def export_multiple_npz(
    input_dir: Path,
    output_dir: Path,
    pattern: str = "*.npz",
    fps: float = 30.0,
    trail_length: int = 10,
    verbose: bool = True,
) -> Dict[str, Dict[str, Path]]:
    """
    Export videos for multiple NPZ files in a directory.
    
    Args:
        input_dir: Input directory with NPZ files
        output_dir: Output directory for videos
        pattern: Glob pattern for NPZ files
        fps: Frames per second
        trail_length: Number of previous frames to show in trail
        verbose: Print detailed information
    
    Returns:
        all_video_paths: Dictionary mapping filename to {camera_id: video_path}
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all NPZ files
    npz_files = sorted(input_dir.glob(pattern))
    
    if not npz_files:
        raise FileNotFoundError(f"No NPZ files found in {input_dir} with pattern {pattern}")
    
    print(f"\n[INFO] ========================================")
    print(f"[INFO] Exporting videos for {len(npz_files)} NPZ files")
    print(f"[INFO] Input directory: {input_dir}")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] ========================================")
    
    # Process each file
    all_video_paths = {}
    
    for i, npz_file in enumerate(npz_files):
        print(f"\n[INFO] Processing {i+1}/{len(npz_files)}: {npz_file.name}")
        
        # Create subdirectory for this NPZ
        npz_output_dir = output_dir / npz_file.stem
        npz_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            video_paths = export_single_npz(
                input_path=npz_file,
                output_dir=npz_output_dir,
                base_name=npz_file.stem,
                fps=fps,
                trail_length=trail_length,
                verbose=verbose,
            )
            all_video_paths[npz_file.name] = video_paths
        except Exception as e:
            print(f"[ERROR] Failed to process {npz_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n[INFO] ========================================")
    print(f"[INFO] Exported videos for {len(all_video_paths)}/{len(npz_files)} files")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] ========================================")
    
    return all_video_paths


def main():
    parser = argparse.ArgumentParser(
        description="Export tracking results to per-camera videos",
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
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for videos",
    )
    
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Input directory with multiple NPZ files (alternative to --input)",
    )
    
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.npz",
        help="Glob pattern for NPZ files in input directory (default: *.npz)",
    )
    
    parser.add_argument(
        "--base-name",
        type=str,
        help="Base name for output files (default: input filename)",
    )
    
    # Video parameters
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frames per second (default: 30.0)",
    )
    
    parser.add_argument(
        "--trail-length",
        type=int,
        default=10,
        help="Number of previous frames to show in trail (0 = no trail, -1 = all, default: 10)",
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
    
    # Process single file or directory
    try:
        if args.input:
            video_paths = export_single_npz(
                input_path=args.input,
                output_dir=args.output_dir,
                base_name=args.base_name,
                fps=args.fps,
                trail_length=args.trail_length,
                verbose=not args.quiet,
            )
            
            print(f"\n[INFO] Exported {len(video_paths)} videos:")
            for cam_id, video_path in video_paths.items():
                print(f"[INFO]   {cam_id}: {video_path}")
        else:
            all_video_paths = export_multiple_npz(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                pattern=args.pattern,
                fps=args.fps,
                trail_length=args.trail_length,
                verbose=not args.quiet,
            )
            
            total_videos = sum(len(vp) for vp in all_video_paths.values())
            print(f"\n[INFO] Exported {total_videos} videos across {len(all_video_paths)} NPZ files")
        
        print(f"\n[INFO] Success! Video export complete.")
        
    except Exception as e:
        print(f"\n[ERROR] Video export failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
