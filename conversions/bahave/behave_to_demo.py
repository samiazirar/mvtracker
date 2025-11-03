#!/usr/bin/env python3
"""
One-step conversion: BEHAVE dataset → demo.py format

This script combines:
1. behave_to_npz.py (convert BEHAVE to intermediate format)
2. adapt_behave_for_demo.py (adapt to demo.py format)

Usage:
    python conversions/behave_to_demo.py --scene Date01_Sub01_backpack_back
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"Running: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"\n✗ Failed: {description}")
        return False
    
    print(f"\n✓ Success: {description}")
    return True


def behave_to_demo(
    scene: str,
    behave_root: str,
    output_dir: str,
    max_frames: int,
    downscale_factor: int,
    num_query_points: int,
    mask_type: str,
    camera_id: int,
    frame_id: int,
):
    """Convert BEHAVE scene to demo.py format in one step."""
    
    output_dir = Path(output_dir)
    intermediate_npz = output_dir / f"{scene}.npz"
    final_npz = output_dir / f"{scene}_demo.npz"
    
    print(f"\n{'='*70}")
    print(f"BEHAVE → demo.py Conversion Pipeline")
    print(f"{'='*70}")
    print(f"Scene: {scene}")
    print(f"Output: {final_npz}")
    print(f"{'='*70}\n")
    
    # Step 1: Convert BEHAVE to intermediate format
    step1_cmd = [
        sys.executable,
        "conversions/behave_to_npz.py",
        "--behave_root", behave_root,
        "--scene", scene,
        "--output_dir", str(output_dir),
        "--max_frames", str(max_frames),
        "--downscale_factor", str(downscale_factor),
        "--num_query_points", str(num_query_points),
        "--mask_type", mask_type,
    ]
    
    if not run_command(step1_cmd, "Step 1/2: Converting BEHAVE to .npz"):
        return False
    
    # Check if intermediate file was created
    if not intermediate_npz.exists():
        print(f"\n✗ Intermediate file not found: {intermediate_npz}")
        return False
    
    # Step 2: Adapt to demo.py format
    step2_cmd = [
        sys.executable,
        "conversions/adapt_behave_for_demo.py",
        str(intermediate_npz),
        "--camera_id", str(camera_id),
        "--frame_id", str(frame_id),
    ]
    
    if not run_command(step2_cmd, "Step 2/2: Adapting to demo.py format"):
        return False
    
    # Check if final file was created
    if not final_npz.exists():
        print(f"\n✗ Final file not found: {final_npz}")
        return False
    
    # Success!
    print(f"\n{'='*70}")
    print(f"✓ Conversion Complete!")
    print(f"{'='*70}")
    print(f"\nOutput file: {final_npz}")
    print(f"File size: {final_npz.stat().st_size / 1024 / 1024:.2f} MB")
    
    print(f"\n{'='*70}")
    print(f"Next Steps")
    print(f"{'='*70}")
    print(f"\n1. Run tracking with MVTracker:")
    print(f"   python demo.py \\")
    print(f"       --sample-path {final_npz} \\")
    print(f"       --tracker mvtracker \\")
    print(f"       --depth_estimator gt \\")
    print(f"       --rerun save \\")
    print(f"       --rrd {scene}_tracking.rrd")
    
    print(f"\n2. View results:")
    print(f"   rerun {scene}_tracking.rrd")
    
    print(f"\n3. Or try different trackers:")
    print(f"   --tracker cotracker3_offline")
    print(f"   --tracker spatialtrackerv2")
    print(f"   --tracker locotrack")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="One-step conversion: BEHAVE → demo.py format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required
    parser.add_argument(
        "--scene",
        type=str,
        required=True,
        help="Scene name (e.g., Date01_Sub01_backpack_back)"
    )
    
    # BEHAVE conversion options
    parser.add_argument(
        "--behave_root",
        type=str,
        default="/data/behave-dataset/behave_all",
        help="Path to BEHAVE dataset root"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./conversions/behave_converted",
        help="Output directory"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=100,
        help="Maximum number of frames to process"
    )
    parser.add_argument(
        "--downscale_factor",
        type=int,
        default=2,
        help="Image downscale factor (2 = half resolution)"
    )
    parser.add_argument(
        "--num_query_points",
        type=int,
        default=256,
        help="Number of query points to extract from masks"
    )
    parser.add_argument(
        "--mask_type",
        type=str,
        default="person",
        choices=["person", "hand"],
        help="Type of mask to use"
    )
    
    # Adaptation options
    parser.add_argument(
        "--camera_id",
        type=int,
        default=0,
        help="Camera to use for query point generation (0-3)"
    )
    parser.add_argument(
        "--frame_id",
        type=int,
        default=0,
        help="Frame to use for query point generation"
    )
    
    args = parser.parse_args()
    
    success = behave_to_demo(
        scene=args.scene,
        behave_root=args.behave_root,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        downscale_factor=args.downscale_factor,
        num_query_points=args.num_query_points,
        mask_type=args.mask_type,
        camera_id=args.camera_id,
        frame_id=args.frame_id,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
