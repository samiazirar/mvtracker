"""
Compare reprojected videos across different ICP methods using photometric error metrics.

This script automatically discovers all video folders and compares them against
a baseline (videos_no_icp). It supports comparing multiple ICP variants:
- videos_no_icp (baseline)
- videos_icp_z (Z-only ICP)
- videos_icp_xyz (full 3D ICP)
- videos_external_icp (external camera alignment)
- etc.

Metrics computed:
1. MSE (Mean Squared Error) - pixel-wise squared difference  
2. PSNR (Peak Signal-to-Noise Ratio) - log scale quality metric (higher = better)
3. SSIM (Structural Similarity Index) - perceptual similarity (higher = better)
4. MAE (Mean Absolute Error) - average pixel difference (lower = better)

Usage:
    python compare_photometric_error.py [--videos_dir PATH] [--baseline FOLDER]
"""

import cv2
import numpy as np
import os
import glob
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json


@dataclass
class FrameMetrics:
    """Metrics for a single frame."""
    mse: float
    psnr: float
    ssim: float
    mae: float


@dataclass 
class VideoMetrics:
    """Aggregated metrics for a video."""
    mean_mse: float
    mean_psnr: float
    mean_ssim: float
    mean_mae: float
    std_mse: float
    std_psnr: float
    std_ssim: float
    std_mae: float
    frame_count: int


def compute_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Mean Squared Error between two images."""
    return np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)


def compute_psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
    """Compute Peak Signal-to-Noise Ratio."""
    mse = compute_mse(img1, img2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10((max_val ** 2) / mse)


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute Structural Similarity Index (simplified version).
    For full SSIM, use skimage.metrics.structural_similarity.
    """
    # Convert to grayscale for SSIM
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray = img1
        img2_gray = img2
    
    img1_f = img1_gray.astype(np.float64)
    img2_f = img2_gray.astype(np.float64)
    
    # Constants for stability
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    # Mean
    mu1 = cv2.GaussianBlur(img1_f, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2_f, (11, 11), 1.5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Variance and covariance
    sigma1_sq = cv2.GaussianBlur(img1_f ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2_f ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1_f * img2_f, (11, 11), 1.5) - mu1_mu2
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return float(np.mean(ssim_map))


def compute_mae(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Mean Absolute Error between two images."""
    return np.mean(np.abs(img1.astype(np.float64) - img2.astype(np.float64)))


def compute_frame_metrics(frame1: np.ndarray, frame2: np.ndarray) -> FrameMetrics:
    """Compute all metrics for a frame pair."""
    return FrameMetrics(
        mse=compute_mse(frame1, frame2),
        psnr=compute_psnr(frame1, frame2),
        ssim=compute_ssim(frame1, frame2),
        mae=compute_mae(frame1, frame2)
    )


def compare_videos(video_path: str, gt_path: str) -> Tuple[VideoMetrics, List[FrameMetrics]]:
    """
    Compare a reprojected video against ground truth.
    
    Args:
        video_path: Path to reprojected video (ICP or no-ICP)
        gt_path: Path to ground truth video
        
    Returns:
        Tuple of (aggregated VideoMetrics, list of per-frame FrameMetrics)
    """
    cap_video = cv2.VideoCapture(video_path)
    cap_gt = cv2.VideoCapture(gt_path)
    
    if not cap_video.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    if not cap_gt.isOpened():
        raise ValueError(f"Cannot open ground truth: {gt_path}")
    
    frame_metrics_list = []
    
    while True:
        ret1, frame1 = cap_video.read()
        ret2, frame2 = cap_gt.read()
        
        if not ret1 or not ret2:
            break
        
        # Ensure same size
        if frame1.shape != frame2.shape:
            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
        
        metrics = compute_frame_metrics(frame1, frame2)
        frame_metrics_list.append(metrics)
    
    cap_video.release()
    cap_gt.release()
    
    if not frame_metrics_list:
        raise ValueError("No frames processed")
    
    # Aggregate metrics
    mse_vals = [m.mse for m in frame_metrics_list]
    psnr_vals = [m.psnr for m in frame_metrics_list if m.psnr != float('inf')]
    ssim_vals = [m.ssim for m in frame_metrics_list]
    mae_vals = [m.mae for m in frame_metrics_list]
    
    video_metrics = VideoMetrics(
        mean_mse=np.mean(mse_vals),
        mean_psnr=np.mean(psnr_vals) if psnr_vals else 0,
        mean_ssim=np.mean(ssim_vals),
        mean_mae=np.mean(mae_vals),
        std_mse=np.std(mse_vals),
        std_psnr=np.std(psnr_vals) if psnr_vals else 0,
        std_ssim=np.std(ssim_vals),
        std_mae=np.std(mae_vals),
        frame_count=len(frame_metrics_list)
    )
    
    return video_metrics, frame_metrics_list


def discover_video_folders(videos_dir: str) -> List[str]:
    """
    Discover all video folders in the videos directory.
    
    Returns:
        List of folder names (e.g., ['videos_no_icp', 'videos_icp_z', 'videos_icp_xyz'])
    """
    folders = []
    for item in os.listdir(videos_dir):
        item_path = os.path.join(videos_dir, item)
        if os.path.isdir(item_path) and item.startswith('videos'):
            # Check if it contains video files
            mp4_files = glob.glob(os.path.join(item_path, "*.mp4"))
            if mp4_files:
                folders.append(item)
    return sorted(folders)


def find_videos_in_folder(folder_path: str) -> Dict[str, str]:
    """
    Find all videos in a folder, keyed by camera serial.
    
    Returns:
        Dict mapping camera_serial to video path
    """
    videos = {}
    mp4_files = glob.glob(os.path.join(folder_path, "*.mp4"))
    
    for mp4_path in mp4_files:
        filename = os.path.basename(mp4_path)
        # Extract camera serial (e.g., "17368348_reprojection.mp4" -> "17368348")
        serial = filename.split('_')[0]
        if serial.isdigit():
            videos[serial] = mp4_path
    
    return videos


def find_matching_videos_across_folders(videos_dir: str, baseline_folder: str = "videos_no_icp") -> Dict[str, Dict[str, str]]:
    """
    Find matching video files across all video folders.
    
    Args:
        videos_dir: Root directory containing video folders
        baseline_folder: Folder to use as baseline for comparison
        
    Returns:
        Dict mapping camera_serial to {folder_name: video_path, ...}
    """
    folders = discover_video_folders(videos_dir)
    
    if not folders:
        print(f"[ERROR] No video folders found in {videos_dir}")
        return {}
    
    print(f"[INFO] Discovered {len(folders)} video folder(s): {folders}")
    
    # Collect all videos from all folders
    all_videos = {}  # folder -> {serial -> path}
    all_serials = set()
    
    for folder in folders:
        folder_path = os.path.join(videos_dir, folder)
        videos = find_videos_in_folder(folder_path)
        all_videos[folder] = videos
        all_serials.update(videos.keys())
    
    # Build matches - for each serial, collect paths from all folders
    matches = {}
    for serial in all_serials:
        matches[serial] = {}
        for folder in folders:
            if serial in all_videos[folder]:
                matches[serial][folder] = all_videos[folder][serial]
    
    return matches


def generate_comparison_report(results: Dict, videos_dir: str, baseline: str):
    """Generate a comparison report with visualizations."""
    output_dir = os.path.join(videos_dir, "comparison_report")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all methods
    all_methods = set()
    for serial_data in results.values():
        all_methods.update(serial_data.keys())
    methods = sorted(all_methods)
    
    # Summary table
    print("\n" + "=" * 100)
    print("PHOTOMETRIC COMPARISON REPORT")
    print("=" * 100)
    
    # Header
    header = f"{'Camera':<12} {'Method':<20} {'MSE':>10} {'PSNR':>10} {'SSIM':>10} {'MAE':>10}"
    print(f"\n{header}")
    print("-" * 100)
    
    # Per-camera results
    for serial in sorted(results.keys()):
        data = results[serial]
        for method in methods:
            if method in data:
                metrics = data[method]
                print(f"{serial:<12} {method:<20} {metrics.mean_mse:>10.2f} "
                      f"{metrics.mean_psnr:>10.2f} {metrics.mean_ssim:>10.4f} "
                      f"{metrics.mean_mae:>10.2f}")
        print()
    
    # Compute improvements relative to baseline
    if baseline in methods:
        print("\n" + "=" * 100)
        print(f"IMPROVEMENT vs BASELINE ({baseline})")
        print("  MSE/MAE: negative = better (lower error)")
        print("  PSNR/SSIM: positive = better (higher quality)")
        print("=" * 100)
        
        comparison_methods = [m for m in methods if m != baseline]
        
        for method in comparison_methods:
            print(f"\n--- {method} vs {baseline} ---")
            print(f"{'Camera':<12} {'ΔMSE':>12} {'ΔPSNR':>12} {'ΔSSIM':>12} {'ΔMAE':>12} {'Verdict':>12}")
            print("-" * 80)
            
            total_better = 0
            total_worse = 0
            total_same = 0
            
            for serial in sorted(results.keys()):
                data = results[serial]
                if baseline in data and method in data:
                    base_m = data[baseline]
                    comp_m = data[method]
                    
                    delta_mse = comp_m.mean_mse - base_m.mean_mse
                    delta_psnr = comp_m.mean_psnr - base_m.mean_psnr
                    delta_ssim = comp_m.mean_ssim - base_m.mean_ssim
                    delta_mae = comp_m.mean_mae - base_m.mean_mae
                    
                    # Verdict: count improvements
                    # Lower MSE/MAE = better, Higher PSNR/SSIM = better
                    improvements = 0
                    if delta_mse < -0.1: improvements += 1
                    if delta_psnr > 0.1: improvements += 1
                    if delta_ssim > 0.001: improvements += 1
                    if delta_mae < -0.1: improvements += 1
                    
                    if improvements >= 3:
                        verdict = "✓ BETTER"
                        total_better += 1
                    elif improvements <= 1:
                        verdict = "✗ WORSE"
                        total_worse += 1
                    else:
                        verdict = "~ MIXED"
                        total_same += 1
                    
                    # Arrows for direction
                    mse_arrow = "↓" if delta_mse < 0 else "↑"
                    psnr_arrow = "↑" if delta_psnr > 0 else "↓"
                    ssim_arrow = "↑" if delta_ssim > 0 else "↓"
                    mae_arrow = "↓" if delta_mae < 0 else "↑"
                    
                    print(f"{serial:<12} {delta_mse:>10.2f}{mse_arrow} "
                          f"{delta_psnr:>10.2f}{psnr_arrow} "
                          f"{delta_ssim:>10.4f}{ssim_arrow} "
                          f"{delta_mae:>10.2f}{mae_arrow} "
                          f"{verdict:>12}")
            
            print(f"\nSummary: {total_better} better, {total_worse} worse, {total_same} mixed")
    
    # Save JSON report
    json_report = {
        'baseline': baseline,
        'methods': methods,
        'cameras': {}
    }
    
    for serial, data in results.items():
        json_report['cameras'][serial] = {}
        for method, metrics in data.items():
            json_report['cameras'][serial][method] = {
                'mse': metrics.mean_mse,
                'psnr': metrics.mean_psnr,
                'ssim': metrics.mean_ssim,
                'mae': metrics.mean_mae,
                'std_mse': metrics.std_mse,
                'std_psnr': metrics.std_psnr,
                'std_ssim': metrics.std_ssim,
                'std_mae': metrics.std_mae,
                'frame_count': metrics.frame_count
            }
    
    json_path = os.path.join(output_dir, "photometric_comparison.json")
    with open(json_path, 'w') as f:
        json.dump(json_report, f, indent=2)
    print(f"\n[INFO] Report saved to: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare reprojected videos using photometric metrics")
    parser.add_argument("--videos_dir", type=str, default="point_clouds/videos",
                       help="Directory containing video folders")
    parser.add_argument("--baseline", type=str, default="videos_no_icp",
                       help="Baseline folder for comparison (default: videos_no_icp)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("PHOTOMETRIC ERROR COMPARISON (Auto-Discovery)")
    print("=" * 60)
    
    # Find all video folders and match videos
    print(f"\n[INFO] Scanning: {args.videos_dir}")
    matches = find_matching_videos_across_folders(args.videos_dir, args.baseline)
    
    if not matches:
        print("[ERROR] No videos found!")
        return
    
    print(f"[INFO] Found {len(matches)} camera(s)")
    
    # Get all folders
    all_folders = set()
    for serial_videos in matches.values():
        all_folders.update(serial_videos.keys())
    folders = sorted(all_folders)
    
    print(f"[INFO] Comparing folders: {folders}")
    
    # Compare videos from each folder against baseline
    results = {}
    
    for serial, paths in matches.items():
        print(f"\n[INFO] Processing camera: {serial}")
        results[serial] = {}
        
        # Get baseline video path
        baseline_path = paths.get(args.baseline)
        if not baseline_path:
            print(f"  [WARN] No baseline video for {serial}, skipping")
            continue
        
        for folder, video_path in paths.items():
            print(f"  -> Comparing {folder} vs {args.baseline}...")
            try:
                # Compare this video against baseline
                metrics, _ = compare_videos(video_path, baseline_path)
                results[serial][folder] = metrics
                print(f"    Frames: {metrics.frame_count}, MSE: {metrics.mean_mse:.2f}, SSIM: {metrics.mean_ssim:.4f}")
                    
            except Exception as e:
                print(f"    [ERROR] {e}")
    
    # Generate report
    generate_comparison_report(results, args.videos_dir, args.baseline)
    
    print("\n[SUCCESS] Comparison complete!")


if __name__ == "__main__":
    main()