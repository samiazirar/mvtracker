"""
Inspect converted BEHAVE .npz files.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def inspect_npz(npz_path: Path):
    """Inspect a converted BEHAVE .npz file."""
    
    print(f"Loading: {npz_path}")
    data = np.load(npz_path)
    
    print("\n=== NPZ File Contents ===")
    for key in data.files:
        value = data[key]
        if isinstance(value, np.ndarray):
            print(f"{key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"{key}: {value}")
    
    # Extract key data
    rgbs = data['rgbs']
    masks = data['masks']
    depths = data['depths']
    intrinsics = data['intrinsics']
    extrinsics = data['extrinsics']
    query_points = data['query_points']
    
    num_cameras, num_frames, _, h, w = rgbs.shape
    
    print(f"\n=== Scene Info ===")
    print(f"Scene name: {data['scene_name']}")
    print(f"Mask type: {data['mask_type']}")
    print(f"Number of cameras: {num_cameras}")
    print(f"Number of frames: {num_frames}")
    print(f"Image resolution: {h}x{w}")
    print(f"Downscale factor: {data['downscale_factor']}")
    print(f"Number of query points per frame: {query_points.shape[2]}")
    
    print(f"\n=== Camera Parameters ===")
    for cam_id in range(num_cameras):
        print(f"\nCamera {cam_id}:")
        print(f"  Intrinsics:\n{intrinsics[cam_id]}")
        print(f"  Extrinsics (3x4):\n{extrinsics[cam_id]}")
    
    print(f"\n=== Data Statistics ===")
    print(f"RGB range: [{rgbs.min()}, {rgbs.max()}]")
    print(f"Depth range: [{depths.min():.3f}, {depths.max():.3f}] meters")
    print(f"Mask unique values: {np.unique(masks)}")
    print(f"Query points range: x=[{query_points[..., 0].min():.1f}, {query_points[..., 0].max():.1f}], "
          f"y=[{query_points[..., 1].min():.1f}, {query_points[..., 1].max():.1f}]")
    
    # Visualize a sample frame
    print("\n=== Visualizing Sample Frame ===")
    frame_idx = num_frames // 2
    
    fig, axes = plt.subplots(num_cameras, 3, figsize=(15, 4 * num_cameras))
    if num_cameras == 1:
        axes = axes[np.newaxis, :]
    
    for cam_id in range(num_cameras):
        # RGB
        rgb = rgbs[cam_id, frame_idx].transpose(1, 2, 0)
        axes[cam_id, 0].imshow(rgb)
        axes[cam_id, 0].set_title(f"Camera {cam_id} - RGB (Frame {frame_idx})")
        axes[cam_id, 0].axis('off')
        
        # Mask
        mask = masks[cam_id, frame_idx]
        axes[cam_id, 1].imshow(mask, cmap='gray')
        axes[cam_id, 1].set_title(f"Camera {cam_id} - Mask")
        axes[cam_id, 1].axis('off')
        
        # RGB with query points
        axes[cam_id, 2].imshow(rgb)
        qpts = query_points[frame_idx, cam_id]
        # Filter out zero points
        valid_mask = (qpts[:, 0] != 0) | (qpts[:, 1] != 0)
        if valid_mask.any():
            axes[cam_id, 2].scatter(qpts[valid_mask, 0], qpts[valid_mask, 1], 
                                   c='red', s=10, alpha=0.5)
        axes[cam_id, 2].set_title(f"Camera {cam_id} - Query Points")
        axes[cam_id, 2].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    output_path = npz_path.with_suffix('.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Inspect converted BEHAVE .npz files")
    parser.add_argument(
        "npz_path",
        type=str,
        help="Path to the .npz file"
    )
    
    args = parser.parse_args()
    npz_path = Path(args.npz_path)
    
    if not npz_path.exists():
        print(f"Error: File not found: {npz_path}")
        return
    
    inspect_npz(npz_path)


if __name__ == "__main__":
    main()
