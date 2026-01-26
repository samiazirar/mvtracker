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
    video_dir = workspace / "output_videos"
    
    print("=" * 70)
    print("  MVTracker Visualization Summary")
    print("=" * 70)
    print()
    
    # RRD Files
    rrd_files = list(viz_dir.glob("*.rrd")) + list(workspace.glob("*.rrd"))
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
    print(f"{'Total RRD files:':<45} {len(rrd_files)} files, {format_size(total_rrd_size)}")
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
    print(f"{'Total video files:':<45} {len(video_files)} files, {format_size(total_video_size)}")
    print()
    
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
