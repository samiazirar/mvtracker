"""
Adapt BEHAVE .npz format to demo.py format.

The BEHAVE conversion script creates .npz files with:
- intrinsics/extrinsics per camera (broadcast to all frames)
- query_points in 2D (need to unproject to 3D)

This script converts to the format expected by demo.py:
- intrs: (C, T, 3, 3)
- extrs: (C, T, 3, 4)
- query_points: (N, 3) in 3D world coordinates
"""

import argparse
from pathlib import Path

import numpy as np


def unproject_2d_to_3d(
    points_2d: np.ndarray,
    depth_map: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
) -> np.ndarray:
    """
    Unproject 2D points to 3D world coordinates using depth.
    
    Args:
        points_2d: (N, 2) [x, y] in image coordinates
        depth_map: (H, W) depth in meters
        intrinsics: (3, 3) camera intrinsics
        extrinsics: (3, 4) camera-to-world [R|t]
    
    Returns:
        points_3d: (N, 3) in world coordinates
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    R = extrinsics[:, :3]
    t = extrinsics[:, 3]
    
    points_3d_world = []
    
    for x, y in points_2d:
        x_int, y_int = int(x), int(y)
        
        # Check bounds
        if not (0 <= y_int < depth_map.shape[0] and 0 <= x_int < depth_map.shape[1]):
            continue
        
        # Get depth
        z = depth_map[y_int, x_int]
        if z <= 0:
            continue
        
        # Unproject to camera coordinates
        X_cam = (x - cx) * z / fx
        Y_cam = (y - cy) * z / fy
        Z_cam = z
        
        # Transform to world coordinates
        point_cam = np.array([X_cam, Y_cam, Z_cam])
        point_world = R @ point_cam + t
        
        points_3d_world.append(point_world)
    
    return np.array(points_3d_world, dtype=np.float32)


def adapt_behave_for_demo(
    input_npz: Path,
    output_npz: Path,
    camera_id: int = 0,
    frame_id: int = 0,
    min_query_points: int = 10,
):
    """
    Adapt BEHAVE .npz format to demo.py format.
    
    Args:
        input_npz: Input BEHAVE .npz file
        output_npz: Output path for demo.py format
        camera_id: Camera to use for query point generation (0-3)
        frame_id: Frame to use for query point generation
        min_query_points: Minimum number of valid query points required
    """
    
    print(f"Loading: {input_npz}")
    data = np.load(input_npz)
    
    # Load data
    rgbs = data['rgbs']              # (C, T, 3, H, W)
    depths = data['depths']          # (C, T, H, W)
    intrinsics = data['intrinsics']  # (C, 3, 3)
    extrinsics = data['extrinsics']  # (C, 3, 4)
    query_points_2d = data['query_points']  # (T, C, N, 2)
    
    C, T = rgbs.shape[:2]
    N = query_points_2d.shape[2]
    
    print(f"\nInput data:")
    print(f"  Cameras: {C}")
    print(f"  Frames: {T}")
    print(f"  Resolution: {rgbs.shape[3]}×{rgbs.shape[4]}")
    print(f"  Query points per frame: {N}")
    
    # Validate camera and frame indices
    if camera_id >= C:
        print(f"Warning: camera_id {camera_id} >= num_cameras {C}, using camera 0")
        camera_id = 0
    if frame_id >= T:
        print(f"Warning: frame_id {frame_id} >= num_frames {T}, using frame 0")
        frame_id = 0
    
    # 1. Broadcast intrinsics to (C, T, 3, 3)
    print("\nBroadcasting intrinsics to (C, T, 3, 3)...")
    intrs = np.tile(intrinsics[:, None, :, :], (1, T, 1, 1))
    
    # 2. Broadcast extrinsics to (C, T, 3, 4)
    print("Broadcasting extrinsics to (C, T, 3, 4)...")
    extrs = np.tile(extrinsics[:, None, :, :], (1, T, 1, 1))
    
    # 3. Convert query points from 2D to 3D
    print(f"\nUnprojecting query points from camera {camera_id}, frame {frame_id}...")
    qpts_2d = query_points_2d[frame_id, camera_id]  # (N, 2)
    depth_map = depths[camera_id, frame_id]  # (H, W)
    K = intrinsics[camera_id]  # (3, 3)
    E = extrinsics[camera_id]  # (3, 4)
    
    # Filter out zero points
    valid_mask = (qpts_2d[:, 0] != 0) | (qpts_2d[:, 1] != 0)
    qpts_2d_valid = qpts_2d[valid_mask]
    
    print(f"  Valid 2D query points: {len(qpts_2d_valid)}/{N}")
    
    # Unproject to 3D
    query_points = unproject_2d_to_3d(qpts_2d_valid, depth_map, K, E)
    
    print(f"  Successfully unprojected: {len(query_points)}/{len(qpts_2d_valid)}")
    
    if len(query_points) < min_query_points:
        print(f"\nError: Only {len(query_points)} valid query points, need at least {min_query_points}")
        print("Try:")
        print("  - Different camera_id or frame_id")
        print("  - Increase --num_query_points during conversion")
        print("  - Use --random_query_points in demo.py instead")
        return False
    
    # Print query point statistics
    print(f"\n3D query points statistics:")
    print(f"  X range: [{query_points[:, 0].min():.3f}, {query_points[:, 0].max():.3f}]")
    print(f"  Y range: [{query_points[:, 1].min():.3f}, {query_points[:, 1].max():.3f}]")
    print(f"  Z range: [{query_points[:, 2].min():.3f}, {query_points[:, 2].max():.3f}]")
    
    print(f"\nFinal format:")
    print(f"  rgbs: {rgbs.shape}")
    print(f"  depths: {depths.shape}")
    print(f"  intrs: {intrs.shape}")
    print(f"  extrs: {extrs.shape}")
    print(f"  query_points: {query_points.shape}")
    
    # Save in demo.py format
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        output_npz,
        rgbs=rgbs,
        depths=depths,
        intrs=intrs,
        extrs=extrs,
        query_points=query_points,
    )
    
    print(f"\n✓ Saved to: {output_npz}")
    print(f"File size: {output_npz.stat().st_size / 1024 / 1024:.2f} MB")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Adapt BEHAVE .npz format to demo.py format"
    )
    parser.add_argument(
        "input_npz",
        type=str,
        help="Input BEHAVE .npz file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: add _demo suffix to input)"
    )
    parser.add_argument(
        "--camera_id",
        type=int,
        default=0,
        help="Camera to use for query point generation (0-3)"
    )
    parser.add_argument(
        "--frame_id",
        type=int,
        default=0,
        help="Frame to use for query point generation"
    )
    parser.add_argument(
        "--min_query_points",
        type=int,
        default=10,
        help="Minimum number of valid query points required"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_npz)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_demo.npz"
    
    success = adapt_behave_for_demo(
        input_npz=input_path,
        output_npz=output_path,
        camera_id=args.camera_id,
        frame_id=args.frame_id,
        min_query_points=args.min_query_points,
    )
    
    if success:
        print("\n" + "="*70)
        print("Ready to use with demo.py!")
        print("="*70)
        print(f"\nRun:")
        print(f"python demo.py --sample-path {output_path} --tracker mvtracker --depth_estimator gt --rerun save")
    else:
        print("\nConversion failed. See error messages above.")


if __name__ == "__main__":
    main()
