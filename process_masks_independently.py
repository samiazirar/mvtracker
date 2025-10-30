#!/usr/bin/env python3
"""
Process and track masks independently, generating separate RRD files per mask.

This script orchestrates the per-mask tracking workflow:
1. Load NPZ file with multiple mask instances
2. Split into separate NPZ files (one per mask instance)
3. For each mask:
   - Create query points
   - Run tracking (using demo.py)
   - Generate individual .rrd file
4. Combine all .rrd files for joint visualization

Usage:
    python process_masks_independently.py \
        --npz path/to/masks.npz \
        --mask-key sam2_masks \
        --tracker mvtracker \
        --output-dir ./tracking_results \
        --base-name scene_tracking

Example:
    python process_masks_independently.py \
        --npz third_party/HOISTFormer/sam2_tracking_output/task_0034_processed_hand_tracked_hoist_sam2.npz \
        --mask-key sam2_masks \
        --tracker mvtracker \
        --output-dir ./tracking_per_mask \
        --frames-before 3 \
        --frames-after 10
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.mask_instance_utils import (
    get_mask_instance_ids,
    split_npz_by_instances,
    verify_instance_npz,
)
from utils.rerun_combine_utils import combine_tracking_results


def run_command(cmd: List[str], description: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command with logging."""
    print(f"\n[INFO] ========== {description} ==========")
    print(f"[CMD] {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=False, text=True, check=False)
    
    if check and result.returncode != 0:
        print(f"[ERROR] Command failed with exit code {result.returncode}")
        raise subprocess.CalledProcessError(result.returncode, cmd)
    
    return result


def create_query_points_for_instance(
    instance_npz: Path,
    mask_key: str,
    frames_before: int,
    frames_after: int,
    use_first_frame: bool,
) -> Path:
    """
    Run create_query_points_from_masks.py for a single instance.
    
    Returns path to output NPZ with query points.
    """
    output_npz = instance_npz.parent / f"{instance_npz.stem}_query.npz"
    
    cmd = [
        "python",
        "create_query_points_from_masks.py",
        "--npz", str(instance_npz),
        "--key", mask_key,
        "--frames-before", str(frames_before),
        "--frames-after", str(frames_after),
        "--output", str(output_npz),
    ]
    
    if use_first_frame:
        cmd.append("--use-first-frame")
    
    run_command(cmd, f"Creating query points for {instance_npz.stem}")
    
    if not output_npz.exists():
        raise FileNotFoundError(f"Query points NPZ not created: {output_npz}")
    
    return output_npz


def run_tracking_for_instance(
    query_npz: Path,
    tracker: str,
    output_rrd: Path,
    temporal_stride: int = 1,
    spatial_downsample: int = 1,
    depth_estimator: str = "gt",
    depth_cache_dir: Path = Path("./depth_cache"),
) -> Path:
    """
    Run demo.py tracking for a single instance.
    
    Returns path to output .rrd file.
    """
    cmd = [
        "python",
        "demo.py",
        "--temporal_stride", str(temporal_stride),
        "--spatial_downsample", str(spatial_downsample),
        "--depth_estimator", depth_estimator,
        "--depth_cache_dir", str(depth_cache_dir),
        "--rerun", "save",
        "--sample-path", str(query_npz),
        "--tracker", tracker,
        "--rrd", str(output_rrd),
    ]
    
    run_command(cmd, f"Running {tracker} tracking for {query_npz.stem}")
    
    if not output_rrd.exists():
        raise FileNotFoundError(f"Tracking RRD not created: {output_rrd}")
    
    return output_rrd


def process_masks_independently(
    input_npz: Path,
    mask_key: str,
    tracker: str,
    output_dir: Path,
    base_name: str,
    frames_before: int = 3,
    frames_after: int = 10,
    use_first_frame: bool = True,
    temporal_stride: int = 1,
    spatial_downsample: int = 1,
    depth_estimator: str = "gt",
    depth_cache_dir: Path = Path("./depth_cache"),
    skip_split: bool = False,
    instances_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Main workflow: process each mask instance independently.
    
    Args:
        input_npz: Path to NPZ with multiple mask instances
        mask_key: Key for masks in NPZ
        tracker: Tracker name (e.g., "mvtracker", "spatialtrackerv2")
        output_dir: Directory for all output files
        base_name: Base name for output files
        frames_before: Frames before contact for query points
        frames_after: Frames after contact for query points
        use_first_frame: Use first frame for all cameras
        temporal_stride: Temporal stride for tracking
        spatial_downsample: Spatial downsampling for tracking
        depth_estimator: Depth estimator to use
        depth_cache_dir: Directory for depth cache
        skip_split: Skip splitting if instance files already exist
        instances_dir: Directory with pre-split instance NPZ files
        
    Returns:
        Dictionary with result paths:
            - "instance_npzs": Dict[instance_id, npz_path]
            - "query_npzs": Dict[instance_id, query_npz_path]
            - "tracking_rrds": Dict[instance_id, rrd_path]
            - "combined_results": Dict with combined visualization paths
    """
    print(f"\n[INFO] ========================================")
    print(f"[INFO] Per-Mask Independent Tracking Pipeline")
    print(f"[INFO] ========================================")
    print(f"[INFO] Input NPZ: {input_npz}")
    print(f"[INFO] Mask key: {mask_key}")
    print(f"[INFO] Tracker: {tracker}")
    print(f"[INFO] Output directory: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Split NPZ by instances (or use existing)
    if skip_split and instances_dir:
        print(f"\n[INFO] Skipping split, using existing instances from {instances_dir}")
        instance_ids = get_mask_instance_ids(input_npz, mask_key=mask_key)
        instance_npzs = {
            inst_id: instances_dir / f"{input_npz.stem}_{inst_id}.npz"
            for inst_id in instance_ids
        }
    else:
        instances_dir = output_dir / "instances"
        instances_dir.mkdir(exist_ok=True)
        
        print(f"\n[INFO] Step 1: Splitting NPZ by mask instances")
        instance_npzs = split_npz_by_instances(
            input_npz_path=input_npz,
            output_dir=instances_dir,
            mask_key=mask_key,
            suffix="",
        )
    
    # Verify all instance files exist
    for inst_id, path in instance_npzs.items():
        if not path.exists():
            raise FileNotFoundError(f"Instance NPZ not found: {path}")
    
    # Step 2: Create query points for each instance
    print(f"\n[INFO] Step 2: Creating query points for each instance")
    query_npzs = {}
    
    for inst_id, instance_npz in instance_npzs.items():
        print(f"\n[INFO] --- Processing {inst_id} ---")
        
        query_npz = create_query_points_for_instance(
            instance_npz=instance_npz,
            mask_key=mask_key,
            frames_before=frames_before,
            frames_after=frames_after,
            use_first_frame=use_first_frame,
        )
        
        query_npzs[inst_id] = query_npz
    
    # Step 3: Run tracking for each instance
    print(f"\n[INFO] Step 3: Running tracking for each instance")
    tracking_rrds = {}
    
    for inst_id, query_npz in query_npzs.items():
        print(f"\n[INFO] --- Tracking {inst_id} ---")
        
        output_rrd = output_dir / f"{base_name}_{tracker}_{inst_id}.rrd"
        
        tracking_rrd = run_tracking_for_instance(
            query_npz=query_npz,
            tracker=tracker,
            output_rrd=output_rrd,
            temporal_stride=temporal_stride,
            spatial_downsample=spatial_downsample,
            depth_estimator=depth_estimator,
            depth_cache_dir=depth_cache_dir,
        )
        
        tracking_rrds[inst_id] = tracking_rrd
    
    # Step 4: Combine results for visualization
    print(f"\n[INFO] Step 4: Combining tracking results")
    
    instance_ids = sorted(tracking_rrds.keys())
    rrd_paths = [tracking_rrds[inst_id] for inst_id in instance_ids]
    
    combined_results = combine_tracking_results(
        rrd_paths=rrd_paths,
        instance_ids=instance_ids,
        output_dir=output_dir,
        output_name=f"{base_name}_{tracker}_combined",
    )
    
    # Summary
    print(f"\n[INFO] ========================================")
    print(f"[INFO] Pipeline Complete!")
    print(f"[INFO] ========================================")
    print(f"[INFO] Processed {len(instance_ids)} mask instances:")
    for inst_id in instance_ids:
        print(f"[INFO]   - {inst_id}")
    print(f"\n[INFO] Output files:")
    print(f"[INFO]   Instance NPZs: {instances_dir}/")
    print(f"[INFO]   Query NPZs: {instances_dir}/")
    print(f"[INFO]   Tracking RRDs: {output_dir}/")
    print(f"\n[INFO] View individual tracking:")
    for inst_id, rrd_path in tracking_rrds.items():
        print(f"[INFO]   rerun {rrd_path}")
    print(f"\n[INFO] View all tracking together:")
    print(f"[INFO]   bash {combined_results['viewing_script']}")
    
    return {
        "instance_npzs": instance_npzs,
        "query_npzs": query_npzs,
        "tracking_rrds": tracking_rrds,
        "combined_results": combined_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Process masks independently with per-mask tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--npz",
        type=Path,
        required=True,
        help="Input NPZ file with multiple mask instances",
    )
    
    parser.add_argument(
        "--mask-key",
        type=str,
        default="sam2_masks",
        help="Key for masks in NPZ (default: sam2_masks)",
    )
    
    parser.add_argument(
        "--tracker",
        type=str,
        default="mvtracker",
        choices=["mvtracker", "spatialtrackerv2", "cotracker3_offline"],
        help="Tracker to use (default: mvtracker)",
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for all results",
    )
    
    parser.add_argument(
        "--base-name",
        type=str,
        default="tracking",
        help="Base name for output files (default: tracking)",
    )
    
    parser.add_argument(
        "--frames-before",
        type=int,
        default=3,
        help="Frames before contact for query points (default: 3)",
    )
    
    parser.add_argument(
        "--frames-after",
        type=int,
        default=10,
        help="Frames after contact for query points (default: 10)",
    )
    
    parser.add_argument(
        "--use-first-frame",
        action="store_true",
        help="Use first frame for all cameras (recommended)",
    )
    
    parser.add_argument(
        "--temporal-stride",
        type=int,
        default=1,
        help="Temporal stride for tracking (default: 1)",
    )
    
    parser.add_argument(
        "--spatial-downsample",
        type=int,
        default=1,
        help="Spatial downsampling for tracking (default: 1)",
    )
    
    parser.add_argument(
        "--depth-estimator",
        type=str,
        default="gt",
        help="Depth estimator to use (default: gt)",
    )
    
    parser.add_argument(
        "--depth-cache-dir",
        type=Path,
        default=Path("./depth_cache"),
        help="Directory for depth cache (default: ./depth_cache)",
    )
    
    parser.add_argument(
        "--skip-split",
        action="store_true",
        help="Skip splitting if instance files already exist",
    )
    
    parser.add_argument(
        "--instances-dir",
        type=Path,
        default=None,
        help="Directory with pre-split instance NPZ files (requires --skip-split)",
    )
    
    args = parser.parse_args()
    
    # Validate
    if not args.npz.exists():
        parser.error(f"Input NPZ not found: {args.npz}")
    
    if args.skip_split and not args.instances_dir:
        parser.error("--skip-split requires --instances-dir")
    
    # Run pipeline
    try:
        results = process_masks_independently(
            input_npz=args.npz,
            mask_key=args.mask_key,
            tracker=args.tracker,
            output_dir=args.output_dir,
            base_name=args.base_name,
            frames_before=args.frames_before,
            frames_after=args.frames_after,
            use_first_frame=args.use_first_frame,
            temporal_stride=args.temporal_stride,
            spatial_downsample=args.spatial_downsample,
            depth_estimator=args.depth_estimator,
            depth_cache_dir=args.depth_cache_dir,
            skip_split=args.skip_split,
            instances_dir=args.instances_dir,
        )
        
        print(f"\n[INFO] Success! All masks processed.")
        
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
