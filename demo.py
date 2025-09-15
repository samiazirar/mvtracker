import argparse
import os
import warnings

import numpy as np
import rerun as rr  # pip install rerun-sdk==0.21.0
import torch
from huggingface_hub import hf_hub_download

from mvtracker.utils.visualizer_rerun import log_pointclouds_to_rerun, log_tracks_to_rerun


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
        "--temporal_stride", 
        type=int, 
        default=4, 
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
        default=4, 
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
        "--rrd",
        default="mvtracker_demo.rrd",
        help=(
            "Path to save a .rrd file if `--rerun save` is used. "
            "Note that rerun prefers recordings to have a .rrd suffix."
        ),
    )
    args = p.parse_args()
    np.random.seed(72)
    torch.manual_seed(72)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load MVTracker predictor
    mvtracker = torch.hub.load("ethz-vlg/mvtracker", "mvtracker", pretrained=True, device=device)

    # Download demo sample from Hugging Face Hub
    sample_path = hf_hub_download(
        repo_id="ethz-vlg/mvtracker",
        filename="data_sample.npz",
        token=os.getenv("HF_TOKEN"),
        repo_type="model",
    )
    sample_original = np.load(sample_path)
    rgbs_original = torch.from_numpy(sample_original["rgbs"]).float() 
    depths_original = torch.from_numpy(sample_original["depths"]).float()  
    intrs_original = torch.from_numpy(sample_original["intrs"]).float()  
    extrs_original = torch.from_numpy(sample_original["extrs"]).float()  
    query_points_original = torch.from_numpy(sample_original["query_points"]).float()  
    print("Shapes: rgbs, depths, intrs, extrs, query_points:", rgbs_original.shape, depths_original.shape, intrs_original.shape, extrs_original.shape, query_points_original.shape)
    # Load the RH20T dataset with memory management
    sample_path = "/data/rh20t_api/data/RH20T/packed_npz/task_0001_user_0016_scene_0001_cfg_0003_wc_rotz180.npz"
    
    print("Loading large RH20T dataset - this may take a while...")
    print("Memory before loading:", torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else "N/A", "GB GPU")
    
    # Clean up the original demo data to free memory
    del rgbs_original, depths_original, intrs_original, extrs_original, query_points_original, sample_original
    import gc
    gc.collect()
    
    # Load with memory mapping to avoid loading entire file into RAM at once
    sample = np.load(sample_path, mmap_mode='r')
    print(f"Dataset shapes - RGB: {sample['rgbs'].shape}, Depth: {sample['depths'].shape}")
    
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
                
                batch_memory_gb = rgbs_batch.numel() * 4 / (1024**3) + depths_batch.numel() * 4 / (1024**3)
                print(f"  Batch shape: {rgbs_batch.shape}, Memory: {batch_memory_gb:.2f} GB, Load: {load_time:.2f}s")
                
                # Load query points once (they are small)
                if view_start == 0 and frame_start == 0:
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
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=device == "cuda", dtype=amp_dtype):
                    results_batch = mvtracker(
                        rgbs=rgbs_batch[None].to(device) / 255.0,
                        depths=depths_batch[None].to(device),
                        intrs=intrs_batch[None].to(device),
                        extrs=extrs_batch[None].to(device),
                        query_points_3d=query_batch[None].to(device),
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
        depths = torch.from_numpy(sample["depths"][:, ::temporal_stride]).float()
        intrs = torch.from_numpy(sample["intrs"][:, ::temporal_stride]).float()
        extrs = torch.from_numpy(sample["extrs"][:, ::temporal_stride]).float()
        query_points = torch.from_numpy(sample["query_points"]).float()
        
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
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=device == "cuda", dtype=amp_dtype):
            results = mvtracker(
                rgbs=rgbs[None].to(device) / 255.0,
                depths=depths[None].to(device),
                intrs=intrs[None].to(device),
                extrs=extrs[None].to(device),
                query_points_3d=query_points[None].to(device),
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
        depths_conf=None,
        conf_thrs=[5.0],
        log_only_confident_pc=False,
        radii=-2.45,
        fps=12,
        bbox_crop=None,
        sphere_radius_crop=12.0,
        sphere_center_crop=np.array([0, 0, 0]),
        log_rgb_image=False,
        log_depthmap_as_image_v1=False,
        log_depthmap_as_image_v2=False,
        log_camera_frustrum=True,
        log_rgb_pointcloud=True,
    )
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
