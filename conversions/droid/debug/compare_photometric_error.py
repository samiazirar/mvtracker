"""
Compare reprojected videos (ICP vs No-ICP) against ground truth using photometric error metrics.

Metrics computed:
1. MSE (Mean Squared Error) - pixel-wise squared difference
2. PSNR (Peak Signal-to-Noise Ratio) - log scale quality metric
3. SSIM (Structural Similarity Index) - perceptual similarity
4. MAE (Mean Absolute Error) - average pixel difference

Usage:
    python compare_photometric_error.py [--videos_dir PATH] [--output_dir PATH]
"""

import cv2
import numpy as np
import os
import glob
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
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


def find_matching_videos(videos_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Find matching video files across ICP, no-ICP, and ground truth folders.
    
    Returns:
        Dict mapping camera_serial to {'icp': path, 'no_icp': path, 'gt': path}
    """
    icp_dir = os.path.join(videos_dir, "videos_icp")
    no_icp_dir = os.path.join(videos_dir, "videos_no_icp")
    gt_dir = os.path.join(videos_dir, "ground_truth")
    
    matches = {}
    
    # Find all ground truth videos
    gt_videos = glob.glob(os.path.join(gt_dir, "*_ground_truth.mp4"))
    
    for gt_path in gt_videos:
        filename = os.path.basename(gt_path)
        # Extract camera serial (e.g., "17368348_ground_truth.mp4" -> "17368348")
        serial = filename.replace("_ground_truth.mp4", "")
        
        icp_path = os.path.join(icp_dir, f"{serial}_reprojection.mp4")
        no_icp_path = os.path.join(no_icp_dir, f"{serial}_reprojection.mp4")
        
        if os.path.exists(icp_path) and os.path.exists(no_icp_path):
            matches[serial] = {
                'icp': icp_path,
                'no_icp': no_icp_path,
                'gt': gt_path
            }
    
    return matches


def generate_comparison_report(results: Dict, output_dir: str):
    """Generate a comparison report with visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Summary table
    print("\n" + "=" * 80)
    print("PHOTOMETRIC COMPARISON REPORT")
    print("=" * 80)
    print(f"\n{'Camera':<15} {'Type':<10} {'MSE':>12} {'PSNR':>12} {'SSIM':>12} {'MAE':>12}")
    print("-" * 80)
    
    summary = {}
    
    for serial, data in results.items():
        for method in ['no_icp', 'icp']:
            metrics = data[method]
            print(f"{serial:<15} {method:<10} {metrics.mean_mse:>12.2f} "
                  f"{metrics.mean_psnr:>12.2f} {metrics.mean_ssim:>12.4f} "
                  f"{metrics.mean_mae:>12.2f}")
        print()
        
        # Track improvements
        improvement = {
            'mse': data['no_icp'].mean_mse - data['icp'].mean_mse,
            'psnr': data['icp'].mean_psnr - data['no_icp'].mean_psnr,
            'ssim': data['icp'].mean_ssim - data['no_icp'].mean_ssim,
            'mae': data['no_icp'].mean_mae - data['icp'].mean_mae,
        }
        summary[serial] = improvement
    
    # Print improvement summary
    print("\n" + "=" * 80)
    print("ICP IMPROVEMENT SUMMARY (positive = ICP is better)")
    print("=" * 80)
    print(f"\n{'Camera':<15} {'ΔMSE':>12} {'ΔPSNR':>12} {'ΔSSIM':>12} {'ΔMAE':>12}")
    print("-" * 80)
    
    for serial, imp in summary.items():
        mse_better = "↓" if imp['mse'] > 0 else "↑"
        psnr_better = "↑" if imp['psnr'] > 0 else "↓"
        ssim_better = "↑" if imp['ssim'] > 0 else "↓"
        mae_better = "↓" if imp['mae'] > 0 else "↑"
        
        print(f"{serial:<15} {imp['mse']:>10.2f}{mse_better} "
              f"{imp['psnr']:>10.2f}{psnr_better} "
              f"{imp['ssim']:>10.4f}{ssim_better} "
              f"{imp['mae']:>10.2f}{mae_better}")
    
    # Save JSON report
    json_report = {
        'cameras': {},
        'summary': summary
    }
    
    for serial, data in results.items():
        json_report['cameras'][serial] = {
            'no_icp': {
                'mse': data['no_icp'].mean_mse,
                'psnr': data['no_icp'].mean_psnr,
                'ssim': data['no_icp'].mean_ssim,
                'mae': data['no_icp'].mean_mae,
            },
            'icp': {
                'mse': data['icp'].mean_mse,
                'psnr': data['icp'].mean_psnr,
                'ssim': data['icp'].mean_ssim,
                'mae': data['icp'].mean_mae,
            }
        }
    
    json_path = os.path.join(output_dir, "photometric_comparison.json")
    with open(json_path, 'w') as f:
        json.dump(json_report, f, indent=2)
    print(f"\n[INFO] Report saved to: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare reprojected videos against ground truth")
    parser.add_argument("--videos_dir", type=str, default="point_clouds/videos",
                       help="Directory containing video folders (videos_icp, videos_no_icp, ground_truth)")
    parser.add_argument("--output_dir", type=str, default="point_clouds/videos/comparison_report",
                       help="Directory to save comparison report")
    args = parser.parse_args()
    
    print("=" * 60)
    print("PHOTOMETRIC ERROR COMPARISON")
    print("=" * 60)
    
    # Find matching videos
    print(f"\n[INFO] Looking for videos in: {args.videos_dir}")
    matches = find_matching_videos(args.videos_dir)
    
    if not matches:
        print("[ERROR] No matching video sets found!")
        print("  Expected structure:")
        print("    videos_dir/")
        print("      videos_icp/<serial>_reprojection.mp4")
        print("      videos_no_icp/<serial>_reprojection.mp4")
        print("      ground_truth/<serial>_ground_truth.mp4")
        return
    
    print(f"[INFO] Found {len(matches)} camera(s) with complete video sets")
    
    # Compare each camera
    results = {}
    
    for serial, paths in matches.items():
        print(f"\n[INFO] Processing camera: {serial}")
        
        # Compare no-ICP vs ground truth
        print(f"  -> Comparing no-ICP...")
        no_icp_metrics, _ = compare_videos(paths['no_icp'], paths['gt'])
        
        # Compare ICP vs ground truth
        print(f"  -> Comparing ICP...")
        icp_metrics, _ = compare_videos(paths['icp'], paths['gt'])
        
        results[serial] = {
            'no_icp': no_icp_metrics,
            'icp': icp_metrics
        }
    
    # Generate report
    generate_comparison_report(results, args.output_dir)
    
    print("\n[SUCCESS] Comparison complete!")


if __name__ == "__main__":
    main()