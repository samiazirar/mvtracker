#!/usr/bin/env python3
"""Debug script to render visualizations for all processed episodes with metadata from huggingface."""

import os
import subprocess
import sys
import shutil
from pathlib import Path

# Episodes to process
EPISODES = [
    "PennPAL+06b0ffa5+2023-04-29-18h-30m-47s",
    "GuptaLab+553d1bd5+2023-04-30-15h-02m-19s",
    "REAL+de601749+2023-06-19-19h-24m-15s",
    "ILIAD+7ae1bcff+2023-05-31-18h-15m-27s",
    "RAIL+d027f2ae+2023-10-11-10h-46m-56s",
    "CLVR+13759f6e+2023-05-19-22h-14m-33s",
    "IRIS+7dfa2da3+2023-05-12-10h-51m-00s",
    "RPL+38ccaeb7+2023-04-26-18h-09m-40s",
    "AUTOLab+5d05c5aa+2023-07-07-10h-00m-27s",
    "IPRL+5085c3ce+2023-10-08-14h-28m-17s",
]

# Configuration
METADATA_REPO = "sazirarrwth99/droid_metadata_only"
GCS_BUCKET = "gs://gresearch/robotics/droid_raw/1.0.1"
OUTPUT_BASE = "./rendered_tracks"
RENDER_SCRIPT = "conversions/droid/training_data/render_tracks_from_mp4.py"

# HuggingFace cache for metadata
HF_CACHE = "./hf_render_cache/metadata"


def copy_quality_json(episode_id: str, output_dir: str):
    """Copy quality.json from HuggingFace cache to output directory."""
    import re
    from datetime import datetime
    
    # Parse episode ID to find the file
    parts = episode_id.split("+")
    if len(parts) != 3:
        print(f"  [WARN] Invalid episode ID format: {episode_id}")
        return
    
    lab, episode_hash, datetime_part = parts
    match = re.match(r"(\d{4}-\d{2}-\d{2})-(\d+)h-(\d+)m-(\d+)s", datetime_part)
    if not match:
        print(f"  [WARN] Invalid datetime format: {datetime_part}")
        return
    
    date = match.group(1)
    hour, minute, second = match.group(2), match.group(3), match.group(4)
    dt = datetime.strptime(f"{date} {hour}:{minute}:{second}", "%Y-%m-%d %H:%M:%S")
    timestamp_folder = dt.strftime("%a_%b_%e_%H:%M:%S_%Y").replace(" ", "_")
    
    # Try both success and failure paths
    for outcome in ["success", "failure"]:
        quality_path = os.path.join(
            HF_CACHE, lab, outcome, date, timestamp_folder, "quality.json"
        )
        
        if os.path.exists(quality_path):
            dest_path = os.path.join(output_dir, "quality.json")
            shutil.copy2(quality_path, dest_path)
            print(f"  [OK] Copied quality.json to {dest_path}")
            return
    
    print(f"  [WARN] quality.json not found for {episode_id}")


def render_episode(episode_id: str, index: int, total: int):
    """Render visualizations for a single episode."""
    print(f"\n{'='*80}")
    print(f"[{index}/{total}] Processing: {episode_id}")
    print(f"{'='*80}")
    
    output_dir = os.path.join(OUTPUT_BASE, episode_id)
    
    # Run render script
    cmd = [
        "python",
        RENDER_SCRIPT,
        "--episode_id", episode_id,
        "--metadata_repo", METADATA_REPO,
        "--gcs_bucket", GCS_BUCKET,
        "--mode", "all",
        "--output_dir", output_dir,
    ]
    
    print(f"[CMD] {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        
        print(f"\n[SUCCESS] Rendered {episode_id}")
        
        # Copy quality.json
        copy_quality_json(episode_id, output_dir)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Failed to render {episode_id}")
        print(f"  Exit code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n[INTERRUPTED] Stopped by user")
        sys.exit(1)


def main():
    print("="*80)
    print("DEBUG: Render All Episodes")
    print("="*80)
    print(f"Episodes to process: {len(EPISODES)}")
    print(f"Metadata repo: {METADATA_REPO}")
    print(f"Video source: {GCS_BUCKET}")
    print(f"Output directory: {OUTPUT_BASE}")
    print(f"Mode: all (all visualization types)")
    print("="*80)
    
    # Create output directory
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    
    # Process each episode
    success_count = 0
    failed_episodes = []
    
    for i, episode_id in enumerate(EPISODES, 1):
        success = render_episode(episode_id, i, len(EPISODES))
        
        if success:
            success_count += 1
        else:
            failed_episodes.append(episode_id)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total episodes: {len(EPISODES)}")
    print(f"Successful: {success_count}/{len(EPISODES)}")
    print(f"Failed: {len(failed_episodes)}/{len(EPISODES)}")
    
    if failed_episodes:
        print("\nFailed episodes:")
        for ep in failed_episodes:
            print(f"  - {ep}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
