"""ICP Alignment for External Cameras Only.

This script performs ICP alignment between external cameras only,
excluding the wrist camera from the alignment process.

Use this when you want to refine the alignment between external cameras
without involving the wrist camera transforms.

Usage:
    python conversions/droid/icp_external_cams.py
"""

import numpy as np
import os
import glob
import h5py
import yaml
import cv2
from scipy.spatial.transform import Rotation as R

try:
    import pyzed.sl as sl
    import open3d as o3d
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False


def numpy_to_o3d_pointcloud(points, colors=None):
    """Convert numpy array to Open3D point cloud."""
    if points is None or len(points) == 0:
        return o3d.geometry.PointCloud()
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    
    if colors is not None:
        colors = colors.astype(np.float64)
        if colors.max() > 1.0:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd


def preprocess_cloud(pcd, voxel_size=0.01):
    """Preprocess point cloud for ICP."""
    if len(pcd.points) == 0:
        return pcd
    
    # Downsample
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    # Remove outliers
    if len(pcd.points) > 30:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    # Estimate normals
    if len(pcd.points) > 10:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size * 5, max_nn=30
            )
        )
    
    return pcd


def run_icp_alignment(source_pcd, target_pcd, max_correspondence_distance=0.05):
    """Run point-to-plane ICP alignment."""
    if len(source_pcd.points) == 0 or len(target_pcd.points) == 0:
        return np.eye(4), 0.0
    
    # Ensure target has normals
    if not target_pcd.has_normals():
        target_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
        )
    
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=50,
        relative_fitness=1e-6,
        relative_rmse=1e-6
    )
    
    result = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd,
        max_correspondence_distance,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria
    )
    
    return result.transformation, result.fitness


def transform_points(points, transform):
    """Transform 3D points using a 4x4 transformation matrix."""
    if len(points) == 0:
        return points
    
    ones = np.ones((points.shape[0], 1))
    points_homo = np.hstack([points, ones])
    points_transformed = (transform @ points_homo.T).T
    
    return points_transformed[:, :3]


def align_external_cameras(active_cams, config, reference_serial=None):
    """
    Align external cameras using ICP.
    
    This function excludes the wrist camera and only aligns external cameras
    to each other using point cloud ICP registration.
    
    Args:
        active_cams: Dictionary of active cameras
        config: Configuration dictionary
        reference_serial: Serial number of reference camera (first external if None)
        
    Returns:
        Dictionary of refined transforms for each external camera
    """
    # Filter to external cameras only
    external_cams = {
        serial: cam for serial, cam in active_cams.items() 
        if cam['type'] == 'external'
    }
    
    if len(external_cams) < 2:
        print("[ICP] Need at least 2 external cameras for alignment")
        return {}
    
    serials = list(external_cams.keys())
    
    # Select reference camera
    if reference_serial is None or reference_serial not in external_cams:
        reference_serial = serials[0]
    
    print(f"[ICP] Reference camera: {reference_serial}")
    print(f"[ICP] Aligning {len(serials) - 1} other external cameras")
    
    # Parameters
    voxel_size = config.get('icp_voxel_size', 0.01)
    max_corr_dist = config.get('icp_max_correspondence_distance', 0.05)
    max_depth = config.get('ext_max_depth', 1.5)
    min_depth = config.get('min_depth', 0.1)
    num_frames = config.get('icp_num_frames', 10)
    
    # Collect reference camera point cloud
    ref_cam = external_cams[reference_serial]
    ref_cam['zed'].set_svo_position(0)
    
    # Get multiple frames for robustness
    total_frames = ref_cam['zed'].get_svo_number_of_frames()
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    refined_transforms = {reference_serial: ref_cam['world_T_cam'].copy()}
    
    for serial in serials:
        if serial == reference_serial:
            continue
        
        source_cam = external_cams[serial]
        
        print(f"\n[ICP] Aligning camera {serial} to reference {reference_serial}")
        
        accumulated_transform = np.eye(4)
        total_fitness = 0.0
        valid_alignments = 0
        
        for frame_idx in frame_indices:
            # Get reference points
            ref_cam['zed'].set_svo_position(frame_idx)
            if ref_cam['zed'].grab(ref_cam['runtime']) != sl.ERROR_CODE.SUCCESS:
                continue
            
            mat = sl.Mat()
            ref_cam['zed'].retrieve_measure(mat, sl.MEASURE.XYZRGBA)
            ref_data = mat.get_data()
            
            ref_xyz = ref_data[:, :, :3].reshape(-1, 3)
            ref_rgb = ref_data[:, :, 3:4].view(np.uint8)[:, :, :3].reshape(-1, 3)
            
            # Filter
            valid = (np.isfinite(ref_xyz).all(axis=1) & 
                     (ref_xyz[:, 2] > min_depth) & 
                     (ref_xyz[:, 2] < max_depth))
            ref_xyz = ref_xyz[valid]
            ref_rgb = ref_rgb[valid]
            
            if len(ref_xyz) < 100:
                continue
            
            # Transform reference to world
            ref_world = transform_points(ref_xyz, ref_cam['world_T_cam'])
            
            # Get source points
            source_cam['zed'].set_svo_position(frame_idx)
            if source_cam['zed'].grab(source_cam['runtime']) != sl.ERROR_CODE.SUCCESS:
                continue
            
            mat = sl.Mat()
            source_cam['zed'].retrieve_measure(mat, sl.MEASURE.XYZRGBA)
            source_data = mat.get_data()
            
            source_xyz = source_data[:, :, :3].reshape(-1, 3)
            source_rgb = source_data[:, :, 3:4].view(np.uint8)[:, :, :3].reshape(-1, 3)
            
            # Filter
            valid = (np.isfinite(source_xyz).all(axis=1) & 
                     (source_xyz[:, 2] > min_depth) & 
                     (source_xyz[:, 2] < max_depth))
            source_xyz = source_xyz[valid]
            source_rgb = source_rgb[valid]
            
            if len(source_xyz) < 100:
                continue
            
            # Transform source to world (with current estimate)
            current_transform = source_cam['world_T_cam'] @ accumulated_transform
            source_world = transform_points(source_xyz, current_transform)
            
            # Create point clouds
            ref_pcd = numpy_to_o3d_pointcloud(ref_world, ref_rgb)
            source_pcd = numpy_to_o3d_pointcloud(source_world, source_rgb)
            
            # Preprocess
            ref_pcd = preprocess_cloud(ref_pcd, voxel_size)
            source_pcd = preprocess_cloud(source_pcd, voxel_size)
            
            # Run ICP
            delta_transform, fitness = run_icp_alignment(
                source_pcd, ref_pcd, max_corr_dist
            )
            
            if fitness > 0.3:  # Good alignment
                accumulated_transform = delta_transform @ accumulated_transform
                total_fitness += fitness
                valid_alignments += 1
        
        if valid_alignments > 0:
            avg_fitness = total_fitness / valid_alignments
            refined_transforms[serial] = source_cam['world_T_cam'] @ accumulated_transform
            print(f"[ICP] Camera {serial}: avg fitness = {avg_fitness:.4f}")
        else:
            refined_transforms[serial] = source_cam['world_T_cam'].copy()
            print(f"[ICP] Camera {serial}: no valid alignments, using original transform")
    
    return refined_transforms


def main():
    """Main function."""
    print("=" * 60)
    print("External Camera ICP Alignment")
    print("=" * 60)
    print("[INFO] This script aligns external cameras only")
    print("[INFO] Wrist camera is excluded from alignment")
    print("=" * 60)
    
    if not DEPENDENCIES_AVAILABLE:
        print("\n[ERROR] Required dependencies not available")
        print("[ERROR] Please install: pyzed, open3d")
        return
    
    config_path = 'conversions/droid/config.yaml'
    if not os.path.exists(config_path):
        print(f"\n[ERROR] Config file not found: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n[INFO] External camera ICP alignment module ready")
    print("[INFO] Use align_external_cameras(active_cams, config) to align cameras")
    print()
    
    print("[USAGE EXAMPLE]:")
    print("  from icp_external_cams import align_external_cameras")
    print("  ")
    print("  # Filter to external cameras only")
    print("  external_cams = {s: c for s, c in active_cams.items() if c['type'] == 'external'}")
    print("  ")
    print("  # Run alignment")
    print("  refined_transforms = align_external_cameras(external_cams, config)")
    print("  ")
    print("  # Apply refined transforms")
    print("  for serial, transform in refined_transforms.items():")
    print("      active_cams[serial]['world_T_cam'] = transform")
    print()


if __name__ == "__main__":
    main()