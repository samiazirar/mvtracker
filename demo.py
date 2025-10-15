import argparse
import os
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rerun as rr  # pip install rerun-sdk==0.21.0
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from mvtracker.models.core.monocular_baselines import (
    CoTrackerOfflineWrapper,
    CoTrackerOnlineWrapper,
    MonocularToMultiViewAdapter,
)
from mvtracker.utils.visualizer_rerun import log_pointclouds_to_rerun, log_tracks_to_rerun

import sys
import types

# Create a shim package `numpy._core` that mirrors 1.x `numpy.core` modules so
# pickles produced with NumPy 2.x can still be loaded without upgrading numpy.
if "numpy._core" not in sys.modules:
    shim = types.ModuleType("numpy._core")
    shim.__dict__.update(np.core.__dict__)
    sys.modules["numpy._core"] = shim
    for _submodule in ("multiarray", "numerictypes", "umath"):
        full_name = f"numpy._core.{_submodule}"
        if full_name not in sys.modules and hasattr(np.core, _submodule):
            sys.modules[full_name] = getattr(np.core, _submodule)

"""Demo script for MVTracker with memory optimizations and optional VGGT depth estimation.
run with

python demo.py --batch_processing --optimize_performance --temporal_stride 1 --spatial_downsample 1 --depth_estimator gt --depth_cache_dir ./depth_cache --rerun save  --random_query_points
"""

def _prepare_uint8_rgbs(rgbs: torch.Tensor) -> torch.Tensor:
    """Convert float images to uint8 safely for depth estimation models."""
    if rgbs.dtype == torch.uint8:
        return rgbs
    if not torch.is_floating_point(rgbs):
        return rgbs.to(torch.uint8)
    rgbs_to_scale = rgbs
    max_val = rgbs_to_scale.max().item() if rgbs_to_scale.numel() else 0.0
    if max_val <= 1.000001:
        rgbs_to_scale = rgbs_to_scale * 255.0
    return rgbs_to_scale.clamp(0, 255).round().to(torch.uint8)

def maybe_estimate_depths_from_generic(
    rgbs: torch.Tensor,
    intrs: torch.Tensor,
    extrs: torch.Tensor,
    estimator: str,
    cache_root: Path,
    seq_name: str,
    skip_if_cached: bool,
    model_id: str = "facebook/VGGT-1B",
    depths_gt: Optional[torch.Tensor] = None,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Selectively replace GT depths/intrinsics/extrinsics using dataset utilities."""
    if estimator == "gt":
        return None

    cache_root = cache_root.expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)
    rgbs_uint8 = _prepare_uint8_rgbs(rgbs.cpu())

    if estimator == "duster":
        return _estimate_duster_depths(
            rgbs_uint8=rgbs_uint8,
            intrs=intrs,
            extrs=extrs,
            cache_root=cache_root,
            seq_name=seq_name,
            skip_if_cached=skip_if_cached,
        )

    if estimator == "vggt_raw":
        from mvtracker.datasets.generic_scene_dataset import _ensure_vggt_raw_cache_and_load

        depths_raw, confs_raw, intrs_raw, extrs_raw = _ensure_vggt_raw_cache_and_load(
            rgbs=rgbs_uint8,
            seq_name=seq_name,
            dataset_root=str(cache_root),
            skip_if_cached=skip_if_cached,
            model_id=model_id,
        )
        return depths_raw.float(), intrs_raw.float(), extrs_raw.float(), confs_raw.float()

    if estimator == "vggt_aligned":
        from mvtracker.datasets.generic_scene_dataset import _ensure_vggt_aligned_cache_and_load

        depths_aligned, confs_aligned, intrs_aligned, extrs_aligned = _ensure_vggt_aligned_cache_and_load(
            rgbs=rgbs_uint8,
            seq_name=seq_name,
            dataset_root=str(cache_root),
            extrs_gt=extrs.cpu().float(),
            skip_if_cached=skip_if_cached,
            model_id=model_id,
        )
        return depths_aligned.float(), intrs_aligned.float(), extrs_aligned.float(), confs_aligned.float()

    if estimator == "fusion":
        return _estimate_fused_depths(
            rgbs_uint8=rgbs_uint8,
            intrs=intrs,
            extrs=extrs,
            cache_root=cache_root,
            seq_name=seq_name,
            skip_if_cached=skip_if_cached,
            model_id=model_id,
            depths_gt=depths_gt,
        )

    raise ValueError(f"Unsupported depth estimator mode '{estimator}'.")


def _estimate_duster_depths(
    rgbs_uint8: torch.Tensor,
    intrs: torch.Tensor,
    extrs: torch.Tensor,
    cache_root: Path,
    seq_name: str,
    skip_if_cached: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    raise DeprecationWarning("DUSt3R is deprecated; use GT .")
    """Run (or load) DUSt3R depth estimates while keeping GT intrinsics/extrinsics."""
    from scripts.egoexo4d_preprocessing import main_estimate_duster_depth
            # "DUSt3R depth estimation requires the Duster repo in PYTHONPATH. "
            # "Follow scripts/estimate_depth_with_duster.py instructions to install it."

    V, T, _, H, W = rgbs_uint8.shape
    depth_root = cache_root / f"duster_depths__{seq_name}"
    depth_root.mkdir(parents=True, exist_ok=True)

    sentinel_frame = depth_root / f"3d_model__{T - 1:05d}__scene.npz"
    needs_run = not (skip_if_cached and sentinel_frame.exists())

    if needs_run:
        pkl_path = cache_root / f"{seq_name}.pkl"
        if not (skip_if_cached and pkl_path.exists()):
            scene_payload = {
                "ego_cam_name": None,
                "rgbs": {},
                "intrs": {},
                "extrs": {},
            }
            for v in range(V):
                cam_name = f"cam_{v:03d}"
                scene_payload["rgbs"][cam_name] = rgbs_uint8[v].numpy()
                intr_v = intrs[v]
                extr_v = extrs[v]
                if intr_v.ndim == 3:
                    intr0 = intr_v[0].cpu()
                    if not torch.allclose(intr_v, intr_v[0][None], atol=1e-4):
                        warnings.warn(
                            f"Intrinsics for {cam_name} vary across frames; using first frame for DUSt3R."
                        )
                else:
                    intr0 = intr_v.cpu()
                if extr_v.ndim == 3:
                    extr0 = extr_v[0].cpu()
                    if not torch.allclose(extr_v, extr_v[0][None], atol=1e-4):
                        warnings.warn(
                            f"Extrinsics for {cam_name} vary across frames; using first frame for DUSt3R."
                        )
                else:
                    extr0 = extr_v.cpu()
                scene_payload["intrs"][cam_name] = intr0.numpy()
                scene_payload["extrs"][cam_name] = extr0.numpy()
            with open(pkl_path, "wb") as f:
                pickle.dump(scene_payload, f)

        print(f"Running DUSt3R to estimate depths for sequence '{seq_name}' (may take a while)...")
        main_estimate_duster_depth(
            pkl_scene_file=str(pkl_path),
            depths_output_dir=str(depth_root),
            save_rerun_viz=False,
            skip_if_output_already_exists=skip_if_cached,
        )

    depth_slices = []
    conf_slices = []
    for t in range(T):
        scene_file = depth_root / f"3d_model__{t:05d}__scene.npz"
        if not scene_file.exists():
            raise FileNotFoundError(
                f"Expected DUSt3R output '{scene_file}' not found. "
                "Ensure DUSt3R finished successfully."
            )
        data = np.load(scene_file)
        d = torch.from_numpy(data["depths"]).float()  # [V, H', W']
        c = torch.from_numpy(data.get("confs", np.ones_like(data["depths"], dtype=np.float32))).float()
        d = F.interpolate(d[:, None], size=(H, W), mode="nearest")[:, 0]
        c = F.interpolate(c[:, None], size=(H, W), mode="nearest")[:, 0]
        depth_slices.append(d)
        conf_slices.append(c)

    depths = torch.stack(depth_slices, dim=1).unsqueeze(2)  # [V, T, 1, H, W]
    confs = torch.stack(conf_slices, dim=1).unsqueeze(2)
    return depths, intrs.float(), extrs.float(), confs


def _ensure_depth_channel(volume: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if volume is None:
        return None
    if volume.ndim == 4:
        return volume.unsqueeze(2)
    return volume


def _resize_like(template: torch.Tensor, volume: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    if volume.shape[-2:] == template.shape[-2:]:
        return volume
    v, t = volume.shape[:2]
    c = volume.shape[2]
    flat = volume.reshape(v * t, c, volume.shape[-2], volume.shape[-1])
    if mode == "nearest":
        resized = F.interpolate(flat, size=template.shape[-2:], mode="nearest")
    else:
        resized = F.interpolate(flat, size=template.shape[-2:], mode=mode, align_corners=False)
    return resized.reshape(v, t, c, template.shape[-2], template.shape[-1])


def _detect_static_prefix_frames(
    rgbs_uint8: torch.Tensor,
    diff_threshold: float = 0.5,
    max_frames: int = 10,
) -> List[int]:
    if rgbs_uint8.ndim != 5:
        return []
    _, t, _, _, _ = rgbs_uint8.shape
    if t == 0:
        return []
    if t == 1:
        return [0]
    diffs = (
        rgbs_uint8[:, 1:].float()
        .sub(rgbs_uint8[:, :-1].float())
        .abs()
        .mean(dim=(0, 2, 3, 4))
    )
    frames = [0]
    for idx, delta in enumerate(diffs):
        if delta.item() <= diff_threshold and len(frames) < max_frames:
            frames.append(idx + 1)
        else:
            break
    return frames


def _estimate_per_view_scale(
    pred_depths: torch.Tensor,
    gt_depths: torch.Tensor,
    frame_indices: List[int],
    eps: float = 1e-6,
) -> torch.Tensor:
    v, t = pred_depths.shape[:2]
    if not frame_indices:
        frame_indices = list(range(min(t, 3)))
    scales = []
    for vid in range(v):
        pred_slice = pred_depths[vid, frame_indices]
        gt_slice = gt_depths[vid, frame_indices]
        valid = (
            (pred_slice > eps)
            & (gt_slice > eps)
            & torch.isfinite(pred_slice)
            & torch.isfinite(gt_slice)
        )
        if valid.sum() < 16:
            scales.append(torch.tensor(1.0, device=pred_depths.device, dtype=pred_depths.dtype))
            continue
        ratios = (gt_slice[valid] / pred_slice[valid]).reshape(-1)
        if ratios.numel() == 0:
            scales.append(torch.tensor(1.0, device=pred_depths.device, dtype=pred_depths.dtype))
            continue
        scale = torch.median(ratios)
        if not torch.isfinite(scale):
            scale = torch.tensor(1.0, device=pred_depths.device, dtype=pred_depths.dtype)
        scales.append(scale)
    scale_tensor = torch.stack(scales).view(v, 1, 1, 1, 1)
    return scale_tensor


def _confidence_from_depth(depth: torch.Tensor) -> torch.Tensor:
    conf = torch.ones_like(depth)
    conf[~torch.isfinite(depth)] = 0
    conf[depth <= 0] = 0
    return conf


def _smooth_depth_with_weights(depth: torch.Tensor, weights: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    if kernel_size < 1:
        return depth
    padding = kernel_size // 2
    flat_depth = depth.reshape(-1, 1, depth.shape[-2], depth.shape[-1])
    flat_weights = weights.reshape(-1, 1, weights.shape[-2], weights.shape[-1])
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=depth.device, dtype=depth.dtype)
    weighted_depth = F.conv2d(flat_depth * flat_weights, kernel, padding=padding)
    summed_weights = F.conv2d(flat_weights, kernel, padding=padding).clamp_min(1e-6)
    smoothed = (weighted_depth / summed_weights).reshape_as(depth)
    return smoothed


def _estimate_fused_depths(
    rgbs_uint8: torch.Tensor,
    intrs: torch.Tensor,
    extrs: torch.Tensor,
    cache_root: Path,
    seq_name: str,
    skip_if_cached: bool,
    model_id: str,
    depths_gt: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if depths_gt is None:
        raise ValueError("'fusion' depth estimator requires ground-truth depths for calibration.")

    depths_gt = _ensure_depth_channel(depths_gt).float()

    depths_duster, _, _, confs_duster = _estimate_duster_depths(
        rgbs_uint8=rgbs_uint8,
        intrs=intrs,
        extrs=extrs,
        cache_root=cache_root,
        seq_name=seq_name,
        skip_if_cached=skip_if_cached,
    )

    from mvtracker.datasets.generic_scene_dataset import _ensure_vggt_aligned_cache_and_load

    depths_vggt, confs_vggt, _intrs_vggt, _extrs_vggt = _ensure_vggt_aligned_cache_and_load(
        rgbs=rgbs_uint8,
        seq_name=seq_name,
        dataset_root=str(cache_root),
        extrs_gt=extrs.cpu().float(),
        skip_if_cached=skip_if_cached,
        model_id=model_id,
    )

    depths_vggt = _ensure_depth_channel(depths_vggt).float()
    confs_vggt = _ensure_depth_channel(confs_vggt).float()
    depths_duster = _ensure_depth_channel(depths_duster).float()
    confs_duster = _ensure_depth_channel(confs_duster).float()

    depths_vggt = _resize_like(depths_gt, depths_vggt, mode="bilinear")
    depths_duster = _resize_like(depths_gt, depths_duster, mode="bilinear")
    confs_vggt = _resize_like(depths_gt, confs_vggt, mode="nearest")
    confs_duster = _resize_like(depths_gt, confs_duster, mode="nearest")

    static_frames = _detect_static_prefix_frames(rgbs_uint8)
    scale_duster = _estimate_per_view_scale(depths_duster, depths_gt, static_frames)
    scale_vggt = _estimate_per_view_scale(depths_vggt, depths_gt, static_frames)

    if static_frames:
        print(f"Fusion calibration: using {len(static_frames)} static frame(s) before motion.")
    else:
        print("Fusion calibration: no static prefix detected; falling back to first frames.")

    depths_duster = depths_duster * scale_duster
    depths_vggt = depths_vggt * scale_vggt

    conf_gt = _confidence_from_depth(depths_gt)
    conf_duster = confs_duster.clamp_min(0.0)
    conf_vggt = confs_vggt.clamp_min(0.0)

    conf_duster = conf_duster / (conf_duster.max().clamp_min(1e-6))
    conf_vggt = conf_vggt / (conf_vggt.max().clamp_min(1e-6))

    valid_gt = conf_gt > 0
    if valid_gt.any():
        median_depth = depths_gt[valid_gt].median().item()
        sigma = max(median_depth * 0.05, 0.02)
    else:
        sigma = 0.1
    sigma_tensor = torch.tensor(sigma, device=depths_gt.device, dtype=depths_gt.dtype)

    residual_duster = (depths_duster - depths_gt).abs()
    residual_vggt = (depths_vggt - depths_gt).abs()
    joint_residual = torch.min(residual_duster, residual_vggt)

    weight_gt = conf_gt * torch.exp(-joint_residual / (sigma_tensor * 1.5 + 1e-6))
    weight_duster = conf_duster * torch.exp(-residual_duster / (sigma_tensor + 1e-6))
    weight_vggt = conf_vggt * torch.exp(-residual_vggt / (sigma_tensor + 1e-6))

    outlier_thresh = sigma_tensor * 3.0
    weight_duster[residual_duster > outlier_thresh] *= 0.1
    weight_vggt[residual_vggt > outlier_thresh] *= 0.1
    weight_gt[~valid_gt] = 0

    weight_sum = (weight_gt + weight_duster + weight_vggt).clamp_min(1e-6)
    fused = (
        weight_gt * depths_gt
        + weight_duster * depths_duster
        + weight_vggt * depths_vggt
    ) / weight_sum

    fused = _smooth_depth_with_weights(fused, weight_sum)

    fused_conf = weight_sum.clamp(max=10.0)

    fused = fused.to(torch.float32)
    fused_conf = fused_conf.to(torch.float32)

    return fused, intrs.float(), extrs.float(), fused_conf


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--rerun",
        choices=["save", "spawn", "stream"],
        default="save",
        help=(
            "Whether to save recording to disk, spawn a new Rerun instance, or stream to an existing one. "
            "If 'spawn', make sure a rerun window can be spawned in your environment. "
            "If 'stream', make sure a rerun instance is running at port 9876. "
            "If 'save', the recording will be saved to a `.rrd` file that can be drag-and-dropped into "
            "a running rerun viewer, including the online viewer at https://app.rerun.io/version/0.21.0. "
            "For the online viewer, you want to create low memory-usage recordings with --lightweight."
        ),
    )
    p.add_argument(
        "--lightweight",
        action="store_true",
        help=(
            "Use lightweight rerun logging (less memory usage). This is recommended if you want to "
            "view the recording in the online Rerun viewer at https://app.rerun.io/version/0.21.0."
        ),
    )
    p.add_argument(
        "--random_query_points",
        action="store_true",
        help="Use random query points instead of demo ones.",
    )
    p.add_argument(
        "--tracker",
        choices=["mvtracker", "cotracker3_online", "cotracker3_offline"],
        default="mvtracker",
        help="Select which tracker to run. Defaults to MVTracker; CoTracker3 wrappers are available for comparison.",
    )
    p.add_argument(
        "--temporal_stride", 
        type=int, 
        default=1, 
        help="Temporal subsampling stride to reduce memory usage. Use every Nth frame."
    )
    p.add_argument(
        "--spatial_downsample", 
        type=int, 
        default=1, 
        help="Spatial downsampling factor (1=no downsampling, 2=half resolution)."
    )
    p.add_argument(
        "--batch_processing",
        action="store_true",
        help="Process data in batches to reduce peak memory usage."
    )
    p.add_argument(
        "--batch_size_views", 
        type=int, 
        default=8, 
        help="Number of views to process at once when batch processing is enabled."
    )
    p.add_argument(
        "--batch_size_frames", 
        type=int, 
        default=120, 
        help="Number of frames to process at once when batch processing is enabled."
    )
    p.add_argument(
        "--optimize_performance",
        action="store_true",
        help="Enable performance optimizations (larger batches, reduced I/O)."
    )
    p.add_argument(
        "--depth_estimator",
        choices=["gt", "duster", "vggt_raw", "vggt_aligned", "fusion"],
        default="gt",
        help="Optionally replace GT depths with DUSt3R or VGGT predictions or fuse all sources."
    )
    p.add_argument(
        "--depth_cache_dir",
        default="./depth_cache",
        help="Directory to store cached depth predictions."
    )
    p.add_argument(
        "--force_depth_recompute",
        action="store_true",
        help="Re-run depth estimation even when cached outputs exist."
    )
    p.add_argument(
        "--clean_pointcloud",
        action="store_true",
        help="Run Open3D outlier removal before logging point clouds."
    )
    p.add_argument(
        "--pc_clean_method",
        choices=["statistical", "radius"],
        default="statistical",
        help="Point-cloud cleaning mode when --clean_pointcloud is set."
    )
    p.add_argument(
        "--pc_clean_nb_neighbors",
        type=int,
        default=20,
        help="Neighbors for statistical outlier removal."
    )
    p.add_argument(
        "--pc_clean_std_ratio",
        type=float,
        default=2.0,
        help="Std-dev multiplier for statistical outlier removal."
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
        help="Minimum neighbors inside the radius filter."
    )
    p.add_argument(
        "--rrd",
        default="mvtracker_demo.rrd",
        help=(
            "Path to save a .rrd file if `--rerun save` is used. "
            "Note that rerun prefers recordings to have a .rrd suffix."
        ),
    )
    p.add_argument(
        "--include-robot",
        action="store_true",
        help="Overlay the robot URDF as a static mesh in the Rerun visualization."
    )
    p.add_argument(
        "--sample-path",
        type=str,
        default=None,
        help="Path to the task npz, if none will download the sample from HF mvtracker repo.",
    )

    args = p.parse_args()
    np.random.seed(72)
    torch.manual_seed(72)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    depth_cache_root = Path(args.depth_cache_dir)
    depth_skip_if_cached = not args.force_depth_recompute
    if args.depth_estimator != "gt":
        print(
            f"Depth estimator '{args.depth_estimator}' enabled; cache directory: "
            f"{depth_cache_root.expanduser()}"
        )
    pc_clean_cfg: Optional[Dict[str, Any]] = None
    if args.clean_pointcloud:
        pc_clean_cfg = {
            "method": args.pc_clean_method,
            "nb_neighbors": args.pc_clean_nb_neighbors,
            "std_ratio": args.pc_clean_std_ratio,
            "radius": args.pc_clean_radius,
            "min_points": args.pc_clean_min_points,
        }
        print(
            "Open3D point-cloud cleaning enabled "
            f"(method={pc_clean_cfg['method']})."
        )

    print(f"Using tracker: {args.tracker}")
    if args.tracker == "mvtracker":
        mvtracker = torch.hub.load(
            "ethz-vlg/mvtracker",
            "mvtracker",
            pretrained=True,
            device=device,
        )
        mvtracker.eval()
    else:
        if args.tracker == "cotracker3_online":
            base_tracker = CoTrackerOnlineWrapper(model_name="cotracker3_online")
        else:
            base_tracker = CoTrackerOfflineWrapper(model_name="cotracker3_offline")
        base_tracker = base_tracker.to(device)
        base_tracker.eval()
        mvtracker = MonocularToMultiViewAdapter(base_tracker).to(device)
        mvtracker.eval()

    
    if args.sample_path is not None:
        sample_path = args.sample_path
    elif sample_path is None:
        print("No sample path provided, downloading a small demo sample from HF...")
        raise NotImplementedError("Get the correct sample path")
        # sample_path = hf_hub_download(
        #     repo_id="ethz-vlg/mvtracker",
        #     filename="data/task_0001_user_0010_scene_0005_cfg_0004_demo.npz",
        #     repo_type="dataset",
        # )

    print("HUMANS NOT SUPPORTED YET") #TODO: find out why _human not work
    print("Loading large RH20T dataset - this may take a while...")
    print("Memory before loading:", torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else "N/A", "GB GPU")
    
    # Clean up the original demo data to free memory
    # del rgbs_original, depths_original, intrs_original, extrs_original, query_points_original, sample_original
    import gc
    gc.collect()
    
    # Load with memory mapping to avoid loading entire file into RAM at once
    sample = np.load(sample_path, mmap_mode='r',allow_pickle=True)  
    #only for now, remove all camera data from id "045322071843" for this copy the data
    camera_ids = sample["camera_ids"]
    if "045322071843" in camera_ids:
        print("Removing camera 045322071843 for this demo")
        raise NotImplementedError("Remove this exception after testing")
        mask = camera_ids != "045322071843"
        sample = {
            "rgbs": sample["rgbs"][mask],
            "depths": sample["depths"][mask],#CHANGED
            "intrs": sample["intrs"][mask],
            "extrs": sample["extrs"][mask],
            "camera_ids": sample["camera_ids"][mask],
            #use dummy
            "query_points": np.random.uniform(-1, 1, (10, 3)).astype(np.float32),
        }

        
    print(f"Dataset shapes - RGB: {sample['rgbs'].shape}, Depth: {sample['depths'].shape}")
    camera_ids: Optional[List[str]] = None
    if "camera_ids" in sample:
        raw_camera_ids = np.array(sample["camera_ids"], copy=False)
        flat_ids = raw_camera_ids.reshape(-1)
        decoded_ids: List[str] = []
        for cid in flat_ids:
            if isinstance(cid, (bytes, bytearray, np.bytes_)):
                decoded_ids.append(cid.decode("utf-8"))
            else:
                decoded_ids.append(str(cid))
        if decoded_ids and len(decoded_ids) != sample["rgbs"].shape[0]:
            warnings.warn(
                "Number of camera IDs does not match number of views; falling back to index-based naming."
            )
        else:
            camera_ids = decoded_ids if decoded_ids else None
    depth_seq_name_base = Path(sample_path).stem or "demo_sequence"
    #TODO: whats this
    # Load data in smaller chunks or subsample to fit memory
    temporal_stride = args.temporal_stride
    spatial_downsample = args.spatial_downsample
    
    if args.batch_processing:
        # For batch processing, we'll load data batch by batch instead of all at once
        print(f"Batch processing enabled: {args.batch_size_views} views, {args.batch_size_frames} frames per batch")
        
        # Performance optimization: increase batch sizes if requested
        if args.optimize_performance:
            # Try to fit more data per batch to reduce overhead
            effective_batch_frames = min(args.batch_size_frames * 2, 240)  # Cap at 240 frames
            effective_batch_views = args.batch_size_views
            print(f"Performance mode: Using larger batches - {effective_batch_views} views, {effective_batch_frames} frames")
        else:
            effective_batch_frames = args.batch_size_frames
            effective_batch_views = args.batch_size_views
            print("[Warning] Check ig the batch uses the last query poitns as inout, normally it is not to good to make batches as it then cannot track occluded. So if possible, redo. maybe alwways when gripper is wide open ") #TODO
        
        print("Loading data in batches to minimize memory usage...")
        
        num_views_total = sample['rgbs'].shape[0] 
        num_frames_total = sample['rgbs'].shape[1]
        
        pred_tracks_batched = []
        pred_vis_batched = []
        
        import time
        total_start_time = time.time()
        
        batch_count = 0
        for view_start in range(0, num_views_total, effective_batch_views):
            view_end = min(view_start + effective_batch_views, num_views_total)
            
            for frame_start in range(0, num_frames_total, effective_batch_frames * temporal_stride):
                frame_end = min(frame_start + effective_batch_frames * temporal_stride, num_frames_total)
                
                batch_start_time = time.time()
                batch_count += 1
                
                print(f"[Batch {batch_count}] Loading views {view_start}:{view_end}, frames {frame_start}:{frame_end}")
                
                # Load only this batch from memory-mapped file
                load_start = time.time()
                rgbs_batch = torch.from_numpy(sample["rgbs"][view_start:view_end, frame_start:frame_end:temporal_stride]).float()
                depths_batch = torch.from_numpy(sample["depths"][view_start:view_end, frame_start:frame_end:temporal_stride]).float()
                intrs_batch = torch.from_numpy(sample["intrs"][view_start:view_end, frame_start:frame_end:temporal_stride]).float()
                extrs_batch = torch.from_numpy(sample["extrs"][view_start:view_end, frame_start:frame_end:temporal_stride]).float()
                load_time = time.time() - load_start
                load_time = time.time() - load_start
                
                # Apply spatial downsampling if requested
                preprocess_start = time.time()
                if spatial_downsample > 1:
                    rgbs_batch = rgbs_batch[:, :, :, ::spatial_downsample, ::spatial_downsample]
                    depths_batch = depths_batch[:, :, :, ::spatial_downsample, ::spatial_downsample]
                    # Adjust intrinsics for downsampling
                    intrs_batch = intrs_batch.clone()
                    intrs_batch[:, :, 0, 0] /= spatial_downsample  # fx
                    intrs_batch[:, :, 1, 1] /= spatial_downsample  # fy
                    intrs_batch[:, :, 0, 2] /= spatial_downsample  # cx
                    intrs_batch[:, :, 1, 2] /= spatial_downsample  # cy

                preprocess_time = time.time() - preprocess_start

                if args.depth_estimator != "gt":
                    seq_name = (
                        f"{depth_seq_name_base}_v{view_start:03d}-{view_end:03d}_"
                        f"f{frame_start:05d}-{frame_end:05d}"
                    )
                    depth_estimate = maybe_estimate_depths_from_generic(
                        rgbs=rgbs_batch,
                        intrs=intrs_batch,
                        extrs=extrs_batch,
                        estimator=args.depth_estimator,
                        cache_root=depth_cache_root,
                        seq_name=seq_name,
                        skip_if_cached=depth_skip_if_cached,
                        depths_gt=depths_batch,
                    )
                    if depth_estimate is not None:
                        depths_batch, intrs_batch, extrs_batch, _ = depth_estimate

                batch_memory_gb = rgbs_batch.numel() * 4 / (1024**3) + depths_batch.numel() * 4 / (1024**3)
                print(f"  Batch shape: {rgbs_batch.shape}, Memory: {batch_memory_gb:.2f} GB, Load: {load_time:.2f}s")
                
                # Load query points once (they are small)
                if view_start == 0 and frame_start == 0:
                    if "query_points" in sample:
                        query_points = torch.from_numpy(sample["query_points"]).float()
                    
                    # Generate random query points if requested
                    if args.random_query_points:
                        from mvtracker.models.core.model_utils import init_pointcloud_from_rgbd
                        num_queries = 512
                        t0 = 0
                        xy_radius = 12.0
                        z_min, z_max = -1.0, 10.0
                        xyz, _ = init_pointcloud_from_rgbd(
                            fmaps=rgbs_batch[None],
                            depths=depths_batch[None],
                            intrs=intrs_batch[None],
                            extrs=extrs_batch[None],
                            stride=1,
                            level=0,
                        )
                        pts = xyz[t0]
                        assert pts.numel() > 0, "No valid depth points to sample queries from."

                        r2 = pts[:, 0] ** 2 + pts[:, 1] ** 2
                        mask = (r2 <= xy_radius ** 2) & (pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)
                        pool = pts[mask]
                        assert pool.shape[0] > 0, "Cylinder mask removed all points; increase radius or z-range."

                        idx = torch.randperm(pool.shape[0])[:num_queries]
                        pts = pool[idx]
                        ts = torch.full((pts.shape[0], 1), float(t0), device=pts.device)
                        query_points = torch.cat([ts, pts], dim=1).float()
                        print(f"  Sampled {pts.shape[0]} queries from depth at t={t0} within r<={xy_radius}, z∈[{z_min},{z_max}].")
                
                # Filter query points for this temporal batch
                if query_points.shape[0] > 0:
                    # Convert global frame indices to batch-local indices
                    batch_frame_start = frame_start // temporal_stride
                    batch_frame_end = (frame_end - 1) // temporal_stride + 1
                    
                    query_mask = (query_points[:, 0] >= batch_frame_start) & (query_points[:, 0] < batch_frame_end)
                    query_batch = query_points[query_mask].clone()
                    if query_batch.shape[0] > 0:
                        query_batch[:, 0] -= batch_frame_start  # Adjust timestamps to batch-local
                    else:
                        # No query points in this batch - use the original query points at t=0
                        query_batch = query_points.clone()
                        query_batch[:, 0] = 0  # Set all to first frame of batch
                else:
                    query_batch = query_points
                
                if query_batch.shape[0] == 0:
                    print(f"  No query points available, skipping batch...")
                    continue
                
                # Run prediction on batch
                inference_start = time.time()
                torch.set_float32_matmul_precision("high")
                amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
                
                if args.tracker == "mvtracker":
                    # MVTracker expects query_points_3d and supports mixed precision
                    with torch.no_grad(), torch.cuda.amp.autocast(enabled=device == "cuda", dtype=amp_dtype):
                        results_batch = mvtracker(
                            rgbs=rgbs_batch[None].to(device) / 255.0,
                            depths=depths_batch[None].to(device),
                            intrs=intrs_batch[None].to(device),
                            extrs=extrs_batch[None].to(device),
                            query_points_3d=query_batch[None].to(device),
                        )
                else:
                    # CoTracker wrappers expect query_points and may not support mixed precision
                    with torch.no_grad():
                        results_batch = mvtracker(
                            rgbs=rgbs_batch[None].to(device) / 255.0,
                            depths=depths_batch[None].to(device),
                            intrs=intrs_batch[None].to(device),
                            extrs=extrs_batch[None].to(device),
                            query_points=query_batch[None].to(device),
                        )
                inference_time = time.time() - inference_start
                
                # Collect results
                pred_tracks_batch = results_batch["traj_e"].cpu()
                pred_vis_batch = results_batch["vis_e"].cpu()
                
                pred_tracks_batched.append(pred_tracks_batch)
                pred_vis_batched.append(pred_vis_batch)
                
                # Clear GPU memory (less frequently if optimizing)
                if not args.optimize_performance or batch_count % 3 == 0:
                    torch.cuda.empty_cache()
                del rgbs_batch, depths_batch, intrs_batch, extrs_batch
                
                batch_total_time = time.time() - batch_start_time
                print(f"  Inference: {inference_time:.2f}s, Total batch: {batch_total_time:.2f}s")
                
        total_time = time.time() - total_start_time
        print(f"Total processing time: {total_time:.2f}s for {batch_count} batches ({total_time/batch_count:.2f}s per batch)")
                
        # Combine results from all batches
        if pred_tracks_batched:
            # For batch processing, we need to handle concatenation differently
            # Each batch produces [T_batch, N, 3] tensors
            # We need to combine them properly
            print(f"Combining results from {len(pred_tracks_batched)} batches...")
            
            # Find the batch with query points to get the right N dimension
            reference_batch_idx = 0
            for i, tracks in enumerate(pred_tracks_batched):
                if tracks.shape[1] > 1:  # More than 1 query point (not dummy)
                    reference_batch_idx = i
                    break
            
            reference_shape = pred_tracks_batched[reference_batch_idx].shape
            print(f"Using reference shape: {reference_shape}")
            
            # If we have varying temporal dimensions, we'll take predictions from the best batch
            # For a more complete implementation, you might want to interpolate between batches
            pred_tracks = pred_tracks_batched[reference_batch_idx]
            pred_vis = pred_vis_batched[reference_batch_idx]
            
            print(f"Using results from batch {reference_batch_idx}: tracks {pred_tracks.shape}, vis {pred_vis.shape}")
        else:
            print("No valid batches processed!")
            return
            
        # For visualization, we need the full data loaded once
        print("Loading subsampled data for visualization...")
        rgbs = torch.from_numpy(sample["rgbs"][:, ::temporal_stride]).float()
        depths = torch.from_numpy(sample["depths"][:, ::temporal_stride]).float()
        intrs = torch.from_numpy(sample["intrs"][:, ::temporal_stride]).float()
        extrs = torch.from_numpy(sample["extrs"][:, ::temporal_stride]).float()

        
        
        if spatial_downsample > 1:
            rgbs = rgbs[:, :, :, ::spatial_downsample, ::spatial_downsample]
            depths = depths[:, :, :, ::spatial_downsample, ::spatial_downsample]
            intrs = intrs.clone()
            intrs[:, :, 0, 0] /= spatial_downsample
            intrs[:, :, 1, 1] /= spatial_downsample
            intrs[:, :, 0, 2] /= spatial_downsample
            intrs[:, :, 1, 2] /= spatial_downsample
            
    else:
        # Original single-pass loading
        rgbs = torch.from_numpy(sample["rgbs"][:, ::temporal_stride]).float()
        depths = torch.from_numpy(sample["depths"][:, ::temporal_stride]).float()  # Convert mm to meters
        intrs = torch.from_numpy(sample["intrs"][:, ::temporal_stride]).float()
        extrs = torch.from_numpy(sample["extrs"][:, ::temporal_stride]).float()
        if "query_points" in sample:
            query_points = torch.from_numpy(sample["query_points"]).float()
        else:
            query_points = torch.zeros((0, 4), dtype=torch.float32)  # No query points
        
        # Apply spatial downsampling if requested
        if spatial_downsample > 1:
            print(f"Applying spatial downsampling by factor {spatial_downsample}")
            rgbs = rgbs[:, :, :, ::spatial_downsample, ::spatial_downsample]
            depths = depths[:, :, :, ::spatial_downsample, ::spatial_downsample]
            # Adjust intrinsics for downsampling
            intrs = intrs.clone()
            intrs[:, :, 0, 0] /= spatial_downsample  # fx
            intrs[:, :, 1, 1] /= spatial_downsample  # fy
            intrs[:, :, 0, 2] /= spatial_downsample  # cx
            intrs[:, :, 1, 2] /= spatial_downsample  # cy

        if args.depth_estimator != "gt":
            depth_estimate = maybe_estimate_depths_from_generic(
                rgbs=rgbs,
                intrs=intrs,
                extrs=extrs,
                estimator=args.depth_estimator,
                cache_root=depth_cache_root,
                seq_name=depth_seq_name_base,
                skip_if_cached=depth_skip_if_cached,
                depths_gt=depths,
            )
            if depth_estimate is not None:
                depths, intrs, extrs, _ = depth_estimate

        print(f"After temporal subsampling (stride={temporal_stride}):")
        print("Shapes: rgbs, depths, intrs, extrs, query_points:", rgbs.shape, depths.shape, intrs.shape, extrs.shape, query_points.shape)
        
        # Calculate memory usage
        rgb_memory_gb = rgbs.numel() * 4 / (1024**3)
        total_memory_gb = (rgbs.numel() + depths.numel()) * 4 / (1024**3)
        print(f"RGB tensor memory: {rgb_memory_gb:.2f} GB")
        print(f"Total tensor memory: {total_memory_gb:.2f} GB")
        
        # Optionally, sample random queries in a cylinder of radius 12, height [-1, +10] and replace the demo queries
        if args.random_query_points:
            from mvtracker.models.core.model_utils import init_pointcloud_from_rgbd
            num_queries = 512
            t0 = 0
            xy_radius = 12.0
            z_min, z_max = -1.0, 10.0
            xyz, _ = init_pointcloud_from_rgbd(
                fmaps=rgbs[None],  # [1,V,T,1,H,W], uint8 0–255
                depths=depths[None],  # [1,V,T,1,H,W]
                intrs=intrs[None],  # [1,V,T,3,3]
                extrs=extrs[None],  # [1,V,T,3,4]
                stride=1,
                level=0,
            )
            pts = xyz[t0]  # [V*H*W, 3] at t=0
            assert pts.numel() > 0, "No valid depth points to sample queries from."

            r2 = pts[:, 0] ** 2 + pts[:, 1] ** 2
            mask = (r2 <= xy_radius ** 2) & (pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)
            pool = pts[mask]
            assert pool.shape[0] > 0, "Cylinder mask removed all points; increase radius or z-range."

            idx = torch.randperm(pool.shape[0])[:num_queries]
            pts = pool[idx]
            ts = torch.full((pts.shape[0], 1), float(t0), device=pts.device)
            query_points = torch.cat([ts, pts], dim=1).float()  # (N,4): (t,x,y,z)
            print(f"Sampled {pts.shape[0]} queries from depth at t={t0} within r<={xy_radius}, z∈[{z_min},{z_max}].")
        
        # Original non-batch processing
        print("Processing all data at once...")
        torch.set_float32_matmul_precision("high")
        amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
        
        if args.tracker == "mvtracker":
            # MVTracker expects query_points_3d and supports mixed precision
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=device == "cuda", dtype=amp_dtype):
                results = mvtracker(
                    rgbs=rgbs[None].to(device) / 255.0,
                    depths=depths[None].to(device),
                    intrs=intrs[None].to(device),
                    extrs=extrs[None].to(device),
                    query_points_3d=query_points[None].to(device),
                )
        else:
            # CoTracker wrappers expect query_points and may not support mixed precision
            with torch.no_grad():
                results = mvtracker(
                    rgbs=rgbs[None].to(device) / 255.0,
                    depths=depths[None].to(device),
                    intrs=intrs[None].to(device),
                    extrs=extrs[None].to(device),
                    query_points=query_points[None].to(device),
                )
        pred_tracks = results["traj_e"].cpu()  # [T,N,3]
        pred_vis = results["vis_e"].cpu()  # [T,N]

    # Visualize results
    rr.init("3dpt", recording_id="v0.16")
    if args.rerun == "stream":
        rr.connect_tcp()
    elif args.rerun == "spawn":
        rr.spawn()
    log_pointclouds_to_rerun(
        dataset_name="demo",
        datapoint_idx=0,
        rgbs=rgbs[None],
        depths=depths[None],
        intrs=intrs[None],
        extrs=extrs[None],
        camera_ids=camera_ids,
        depths_conf=None,
        conf_thrs=[5.0],
        log_only_confident_pc=False,
        radii=-0.95,#make smaller for now SIZE
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

    if args.include_robot:
        raise NotImplementedError("Robot visualization from urdf not anymore supported.")

    #TODO: check if query points are correct!
    log_tracks_to_rerun(
        dataset_name="demo",
        datapoint_idx=0,
        predictor_name="MVTracker",
        gt_trajectories_3d_worldspace=None,
        gt_visibilities_any_view=None,
        query_points_3d=query_points[None],
        pred_trajectories=pred_tracks,
        pred_visibilities=pred_vis,
        per_track_results=None,
        radii_scale=1.0,
        fps=12,
        sphere_radius_crop=12.0,
        sphere_center_crop=np.array([0, 0, 0]),
        log_per_interval_results=False,
        max_tracks_to_log=100 if args.lightweight else None,
        track_batch_size=50,
        method_id=None,
        color_per_method_id=None,
        memory_lightweight_logging=args.lightweight,
    )
    if args.rerun == "save":
        rr.save(args.rrd)
        print(f"Saved Rerun recording to: {os.path.abspath(args.rrd)}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*DtypeTensor constructors are no longer.*", module="pointops.query")
    warnings.filterwarnings("ignore", message=".*Plan failed with a cudnnException.*", module="torch.nn.modules.conv")
    main()
