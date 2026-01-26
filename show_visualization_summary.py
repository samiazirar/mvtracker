#!/usr/bin/env python3
"""
Print summary of all generated visualizations.
"""

import os
import sys
from pathlib import Path
from datetime import datetime


def format_size(size_bytes):
    """Format bytes to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}TB"


def main():
    workspace = Path("/workspace")
    viz_dir = workspace / "visualizations"
    viz_v2_dir = workspace / "visualizations_v2"
    viz_episodes_dir = workspace / "visualizations" / "episodes"
    video_dir = workspace / "output_videos"
    video_v2_dir = workspace / "output_videos_v2"
    test_data_dir = workspace / "test_data"
    
    print("=" * 70)
    print("  MVTracker Visualization Summary")
    print("=" * 70)
    print()
    
    # RRD Files (v1 and v2)
    rrd_files = list(viz_dir.glob("*.rrd")) + list(workspace.glob("*.rrd"))
    rrd_v2_files = list(viz_v2_dir.glob("*.rrd")) if viz_v2_dir.exists() else []
    total_rrd_size = 0
    
    print("📊 Rerun RRD Files (Interactive 3D Visualizations)")
    print("-" * 70)
    print(f"{'Filename':<45} {'Size':>10} {'Modified':>15}")
    print("-" * 70)
    
    for rrd in sorted(rrd_files):
        size = rrd.stat().st_size
        total_rrd_size += size
        mtime = datetime.fromtimestamp(rrd.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        print(f"{rrd.name:<45} {format_size(size):>10} {mtime:>15}")
    
    print("-" * 70)
    print(f"{'Total RRD files (v1):':<45} {len(rrd_files)} files, {format_size(total_rrd_size)}")
    print()
    
    # RRD Files v2
    if rrd_v2_files:
        total_rrd_v2_size = 0
        print("📊 Rerun RRD Files v2 (Training Shards v2)")
        print("-" * 70)
        print(f"{'Filename':<45} {'Size':>10} {'Modified':>15}")
        print("-" * 70)
        
        for rrd in sorted(rrd_v2_files):
            size = rrd.stat().st_size
            total_rrd_v2_size += size
            mtime = datetime.fromtimestamp(rrd.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            print(f"{rrd.name:<45} {format_size(size):>10} {mtime:>15}")
        
        print("-" * 70)
        print(f"{'Total RRD files (v2):':<45} {len(rrd_v2_files)} files, {format_size(total_rrd_v2_size)}")
        print()
        total_rrd_size += total_rrd_v2_size
        rrd_files = rrd_files + rrd_v2_files
    
    # RRD Files - Episode Tracks (normalized/unnormalized)
    episode_rrd_files = list(viz_episodes_dir.glob("*.rrd")) if viz_episodes_dir.exists() else []
    if episode_rrd_files:
        total_ep_size = 0
        print("📊 Episode Track Visualizations (Normalized + Unnormalized)")
        print("-" * 70)
        print(f"{'Filename':<45} {'Size':>10} {'Modified':>15}")
        print("-" * 70)
        
        for rrd in sorted(episode_rrd_files):
            size = rrd.stat().st_size
            total_ep_size += size
            mtime = datetime.fromtimestamp(rrd.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            print(f"{rrd.name:<45} {format_size(size):>10} {mtime:>15}")
        
        print("-" * 70)
        print(f"{'Total episode RRD files:':<45} {len(episode_rrd_files)} files, {format_size(total_ep_size)}")
        print()
        total_rrd_size += total_ep_size
        rrd_files = rrd_files + episode_rrd_files
    
    # Track NPZ files
    track_npz_files = list(test_data_dir.rglob("tracks.npz")) if test_data_dir.exists() else []
    if track_npz_files:
        total_npz_size = 0
        print("📁 Track Data Files (tracks.npz)")
        print("-" * 70)
        print(f"{'Relative Path':<50} {'Size':>10}")
        print("-" * 70)
        
        for npz in sorted(track_npz_files)[:10]:  # Show first 10
            size = npz.stat().st_size
            total_npz_size += size
            rel_path = str(npz.relative_to(test_data_dir))[:50]
            print(f"{rel_path:<50} {format_size(size):>10}")
        
        if len(track_npz_files) > 10:
            print(f"... and {len(track_npz_files) - 10} more files")
            for npz in track_npz_files[10:]:
                total_npz_size += npz.stat().st_size
        
        print("-" * 70)
        print(f"{'Total track NPZ files:':<45} {len(track_npz_files)} files, {format_size(total_npz_size)}")
        print()
    
    # Video Files
    video_files = list(video_dir.glob("*.mp4")) if video_dir.exists() else []
    total_video_size = 0
    
    print("🎬 Video Files (2D Track Overlays)")
    print("-" * 70)
    print(f"{'Filename':<45} {'Size':>10} {'Modified':>15}")
    print("-" * 70)
    
    for video in sorted(video_files):
        size = video.stat().st_size
        total_video_size += size
        mtime = datetime.fromtimestamp(video.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        print(f"{video.name:<45} {format_size(size):>10} {mtime:>15}")
    
    print("-" * 70)
    print(f"{'Total video files (v1):':<45} {len(video_files)} files, {format_size(total_video_size)}")
    print()
    
    # Video Files v2
    video_v2_files = list(video_v2_dir.glob("*.mp4")) if video_v2_dir.exists() else []
    if video_v2_files:
        total_video_v2_size = 0
        print("🎬 Video Files v2 (Training Shards v2)")
        print("-" * 70)
        print(f"{'Filename':<45} {'Size':>10} {'Modified':>15}")
        print("-" * 70)
        
        for video in sorted(video_v2_files):
            size = video.stat().st_size
            total_video_v2_size += size
            mtime = datetime.fromtimestamp(video.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            print(f"{video.name:<45} {format_size(size):>10} {mtime:>15}")
        
        print("-" * 70)
        print(f"{'Total video files (v2):':<45} {len(video_v2_files)} files, {format_size(total_video_v2_size)}")
        print()
        total_video_size += total_video_v2_size
        video_files = video_files + video_v2_files
    
    # Summary
    print("=" * 70)
    print("📋 SUMMARY")
    print("=" * 70)
    print(f"  • RRD visualizations: {len(rrd_files)} files ({format_size(total_rrd_size)})")
    print(f"  • Video outputs:      {len(video_files)} files ({format_size(total_video_size)})")
    print(f"  • Total disk usage:   {format_size(total_rrd_size + total_video_size)}")
    print()
    print("🔍 How to view:")
    print("  • RRD files: rerun visualizations/<filename>.rrd")
    print("  • Videos:    ffplay output_videos/<filename>.mp4")
    print("=" * 70)


if __name__ == "__main__":
    main()
