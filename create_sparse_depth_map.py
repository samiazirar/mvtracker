import argparse
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import rerun as rr  # pip install rerun-sdk
import torch
from huggingface_hub import hf_hub_download
from mvtracker.utils.visualizer_rerun import log_pointclouds_to_rerun

"""
Demo script to create a Rerun point cloud video from a data sample.

This script loads RGB+D data, selects a specific segment of frames,
and logs the resulting point clouds to Rerun for visualization.

Example usage:
python create_sparse_depth_map.py --rerun save --max_frames 50 --frame_selection mid
"""

def main():
    p = argparse.ArgumentParser(description="Create a Rerun point cloud video from RGB-D data.")
    p.add_argument(
        "--rerun",
        choices=["save", "spawn", "stream"],
        default="save",
        help=(
            "Whether to save the recording to a file, spawn a new Rerun viewer, or stream to an existing one."
        ),
    )
    p.add_argument(
        "--lightweight",
        action="store_true",
        help="Use lightweight Rerun logging for reduced memory usage, recommended for web viewing.",
    )
    p.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum number of frames to process from the sequence. If not set, all frames are used.",
    )
    p.add_argument(
        "--frame_selection",
        choices=["first", "mid", "last"],
        default="first",
        help="Which part of the sequence to use if --max_frames is set: the first N, middle N, or last N frames.",
    )
    p.add_argument(
        "--clean_pointcloud",
        action="store_true",
        help="Run Open3D outlier removal before logging point clouds for a cleaner visualization."
    )
    p.add_argument(
        "--pc_clean_method",
        choices=["statistical", "radius"],
        default="statistical",
        help="Point-cloud cleaning algorithm to use when --clean_pointcloud is enabled."
    )
    p.add_argument(
        "--pc_clean_nb_neighbors",
        type=int,
        default=20,
        help="Number of neighbors for statistical outlier removal."
    )
    p.add_argument(
        "--pc_clean_std_ratio",
        type=float,
        default=2.0,
        help="Standard deviation ratio for statistical outlier removal."
    )
    p.add_argument(
        "--pc_clean_radius",
        type=float,
        default=0.05,
        help="Radius for radius-based outlier removal."
    )
    p.add_argument(
        "--pc_clean_min_points",
        type=int,
        default=5,
        help="Minimum number of neighbors within the radius for radius-based outlier removal."
    )
    p.add_argument(
        "--rrd",
        default="pointcloud_video.rrd",
        help="Path to save the .rrd file if --rerun save is used.",
    )
    args = p.parse_args()

    # --- Rerun Initialization ---
    rr.init("pointcloud_video_demo", recording_id="v0.1")
    if args.rerun == "stream":
        rr.connect_tcp()
    elif args.rerun == "spawn":
        rr.spawn()

    # --- Point Cloud Cleaning Configuration ---
    pc_clean_cfg: Optional[Dict[str, Any]] = None
    if args.clean_pointcloud:
        pc_clean_cfg = {
            "method": args.pc_clean_method,
            "nb_neighbors": args.pc_clean_nb_neighbors,
            "std_ratio": args.pc_clean_std_ratio,
            "radius": args.pc_clean_radius,
            "min_points": args.pc_clean_min_points,
        }
        print(f"Open3D point-cloud cleaning enabled (method={pc_clean_cfg['method']}).")


    # --- Load Data ---
    # Download demo sample from Hugging Face Hub if a local file isn't used
    # sample_path = hf_hub_download(
    #     repo_id="ethz-vlg/mvtracker",
    #     filename="data_sample.npz",
    #     token=os.getenv("HF_TOKEN"),
    #     repo_type="model",
    # )

    # Or, use a local path to your .npz file
    sample_path = "/data/npz_file/task_0065_user_0010_scene_0009_cfg_0004_pred.npz" # Example path
    
    if not os.path.exists(sample_path):
        print(f"Error: Data file not found at {sample_path}")
        print("Please update the 'sample_path' variable to point to your .npz file.")
        return

    print(f"Loading data from: {sample_path}")
    sample = np.load(sample_path, mmap_mode='r', allow_pickle=True)

    rgbs_full = torch.from_numpy(sample["rgbs"]).float()
    depths_full = torch.from_numpy(sample["depths"]).float()
    intrs_full = torch.from_numpy(sample["intrs"]).float()
    extrs_full = torch.from_numpy(sample["extrs"]).float()

    camera_ids: Optional[List[str]] = None
    if "camera_ids" in sample:
        try:
            camera_ids = [cid.decode('utf-8') for cid in sample["camera_ids"]]
        except (AttributeError, UnicodeDecodeError):
            camera_ids = [str(cid) for cid in sample["camera_ids"]]


    # --- Frame Selection Logic ---
    total_frames = rgbs_full.shape[1]
    
    if args.max_frames is not None and args.max_frames < total_frames:
        n_frames = args.max_frames
        if args.frame_selection == 'first':
            start_idx = 0
            end_idx = n_frames
        elif args.frame_selection == 'last':
            start_idx = total_frames - n_frames
            end_idx = total_frames
        elif args.frame_selection == 'mid':
            start_idx = (total_frames - n_frames) // 2
            end_idx = start_idx + n_frames
        else: # Default to first
            start_idx = 0
            end_idx = n_frames
            
        print(f"Selecting {n_frames} '{args.frame_selection}' frames (from index {start_idx} to {end_idx}).")
        
        rgbs = rgbs_full[:, start_idx:end_idx]
        depths = depths_full[:, start_idx:end_idx]
        intrs = intrs_full[:, start_idx:end_idx]
        extrs = extrs_full[:, start_idx:end_idx]
    else:
        print(f"Using all {total_frames} frames from the dataset.")
        rgbs = rgbs_full
        depths = depths_full
        intrs = intrs_full
        extrs = extrs_full

    print("Final data shapes (V, T, C, H, W):")
    print(f"  RGBs:   {rgbs.shape}")
    print(f"  Depths: {depths.shape}")


    # --- Log to Rerun ---
    print("Logging point clouds to Rerun...")
    log_pointclouds_to_rerun(
        dataset_name="demo_sequence",
        datapoint_idx=0,
        rgbs=rgbs[None],
        depths=depths[None],
        intrs=intrs[None],
        extrs=extrs[None],
        camera_ids=camera_ids,
        depths_conf=None,
        conf_thrs=[5.0],
        log_only_confident_pc=False,
        radii=-2.95,  # Controls point size
        fps=12,
        bbox_crop=None,
        sphere_radius_crop=12.0,
        sphere_center_crop=np.array([0, 0, 0]),
        log_rgb_image=False,
        log_depthmap_as_image_v1=False,
        log_depthmap_as_image_v2=False,
        log_camera_frustrum=True,
        log_rgb_pointcloud=True,
        pc_clean_cfg=pc_clean_cfg,
    )

    if args.rerun == "save":
        rr.save(args.rrd)
        print(f"✅ Saved Rerun recording to: {os.path.abspath(args.rrd)}")


if __name__ == "__main__":
    main()