#!/usr/bin/env python3
"""
Generate multiple contact flow visualizations from different DROID labs.
Creates RRD files for each shard in the training data.

Usage:
    python generate_all_visualizations.py --output_dir visualizations --num_samples_per_shard 10
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Generate all DROID visualizations")
    parser.add_argument(
        "--shards_dir",
        type=str,
        default="/data/droid/training_shards",
        help="Directory containing shard tar files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="visualizations",
        help="Output directory for RRD files",
    )
    parser.add_argument(
        "--num_samples_per_shard",
        type=int,
        default=10,
        help="Number of samples per shard",
    )
    parser.add_argument(
        "--max_shards",
        type=int,
        default=7,
        help="Maximum number of shards to process",
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Find all shards
    shards_dir = Path(args.shards_dir)
    shards = sorted(shards_dir.glob("shard_*.tar"))[:args.max_shards]
    
    print(f"[INFO] Found {len(shards)} shards in {args.shards_dir}")
    
    generated = []
    
    for shard_path in shards:
        shard_name = shard_path.stem
        print(f"\n{'='*60}")
        print(f"[INFO] Processing {shard_name}...")
        print(f"{'='*60}")
        
        # Run visualization script
        cmd = [
            sys.executable,
            "visualize_training_shards.py",
            "--shard_path", str(shard_path),
            "--num_samples", str(args.num_samples_per_shard),
            "--output_dir", args.output_dir,
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                rrd_path = os.path.join(args.output_dir, f"{shard_name}_contact_flow.rrd")
                generated.append(rrd_path)
                print(f"[SUCCESS] Generated {rrd_path}")
            else:
                print(f"[ERROR] Failed to process {shard_name}")
                print(result.stderr[:500] if result.stderr else "No error message")
        except subprocess.TimeoutExpired:
            print(f"[ERROR] Timeout processing {shard_name}")
        except Exception as e:
            print(f"[ERROR] {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("[SUMMARY] Generated visualizations:")
    print(f"{'='*60}")
    for rrd in generated:
        print(f"  - {rrd}")
    
    print(f"\n[INFO] Total: {len(generated)} RRD files generated")
    print(f"[INFO] View with: rerun <filename.rrd>")
    

if __name__ == "__main__":
    main()
