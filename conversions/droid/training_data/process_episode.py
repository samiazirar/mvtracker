"""Process a complete DROID episode for training data.

This is a convenience wrapper that runs both:
1. generate_tracks_and_metadata.py (CPU) - tracks, extrinsics, quality.json
2. extract_rgb_depth.py (GPU) - RGB and depth frames

Usage:
    python process_episode.py --episode_id "AUTOLab+84bd5053+2023-08-18-12h-01m-10s"
    python process_episode.py --episode_id "AUTOLab+84bd5053+2023-08-18-12h-01m-10s" --skip-frames
    python process_episode.py --episode_id "AUTOLab+84bd5053+2023-08-18-12h-01m-10s" --skip-tracks
"""

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Process a complete DROID episode for training data."
    )
    parser.add_argument(
        "--episode_id",
        required=True,
        help='Episode ID, e.g., "AUTOLab+84bd5053+2023-08-18-12h-01m-10s"',
    )
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
        help="Path to config YAML file.",
    )
    parser.add_argument(
        "--skip-tracks",
        action="store_true",
        help="Skip track and metadata generation",
    )
    parser.add_argument(
        "--skip-frames",
        action="store_true",
        help="Skip RGB/depth frame extraction",
    )
    parser.add_argument(
        "--cameras",
        nargs="*",
        help="Specific camera serials to process for frame extraction",
    )
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"=== Processing Episode: {args.episode_id} ===\n")
    
    # Step 1: Generate tracks and metadata
    if not args.skip_tracks:
        print("[Step 1/2] Generating tracks and metadata...")
        cmd = [
            sys.executable,
            os.path.join(script_dir, "generate_tracks_and_metadata.py"),
            "--episode_id", args.episode_id,
            "--config", args.config,
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print("[ERROR] Track generation failed!")
            sys.exit(1)
        print()
    else:
        print("[Step 1/2] Skipping track generation.\n")
    
    # Step 2: Extract RGB and depth frames
    if not args.skip_frames:
        print("[Step 2/2] Extracting RGB and depth frames...")
        cmd = [
            sys.executable,
            os.path.join(script_dir, "extract_rgb_depth.py"),
            "--episode_id", args.episode_id,
            "--config", args.config,
        ]
        if args.cameras:
            cmd.extend(["--cameras"] + args.cameras)
        
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print("[ERROR] Frame extraction failed!")
            sys.exit(1)
        print()
    else:
        print("[Step 2/2] Skipping frame extraction.\n")
    
    print("=== Processing Complete ===")


if __name__ == "__main__":
    main()
