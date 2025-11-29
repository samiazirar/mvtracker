"""
Fuse Static Fields from Wrist Camera.

This script uses the wrist camera to build a high-quality "static" point cloud
of the scene by accumulating points from areas that are:
1. Far enough from the gripper (> 15cm) to avoid self-occlusion
2. Static (not changing much over time) - indicating they're not being manipulated

The wrist camera provides higher resolution depth in the manipulation area,
so we want to keep those observations for static surfaces while excluding
dynamic/manipulated regions.

Output:
- Fused point cloud combining external cameras + static wrist observations
- Video showing the accumulated static regions
"""

import numpy as np
import os
import glob
import h5py
import yaml
import cv2
from scipy.spatial.transform import Rotation as R
import pyzed.sl as sl
import open3d as o3d

from utils import (
    pose6_to_T,
    transform_points,
    compute_wrist_cam_offset,
    precompute_wrist_trajectory,
    external_cam_to_world,
    find_svo_for_camera,
    find_episode_data_by_date,
    get_zed_intrinsics,
    get_filtered_cloud,
    VideoRecorder,
    project_points_to_image,
    draw_points_on_image,
)


class StaticFieldAccumulator:
    """
    Accumulates static point cloud data from wrist camera observations.
    
    Points are kept if they:
    1. Are far enough from the camera (exclude gripper)
    2. Haven't changed significantly over time (static surfaces)
    """
    
    def __init__(self, voxel_size=0.005, min_observations=3, color_threshold=30):
        """
        Args:
            voxel_size: Size of voxels for accumulation (meters)
            min_observations: Minimum times a voxel must be seen to be kept
            color_threshold: Max color change (0-255) to consider static
        """
        self.voxel_size = voxel_size
        self.min_observations = min_observations
        self.color_threshold = color_threshold
        
        # Voxel grid storage: key = (vx, vy, vz), value = {points, colors, count, color_variance}
        self.voxel_data = {}
    
    def _get_voxel_key(self, point):
        """Convert a 3D point to voxel grid coordinates."""
        return (
            int(np.floor(point[0] / self.voxel_size)),
            int(np.floor(point[1] / self.voxel_size)),
            int(np.floor(point[2] / self.voxel_size))
        )
    
    def add_observation(self, points_world, colors):
        """
        Add a new observation of points in world coordinates.
        
        Args:
            points_world: Nx3 array of 3D points in world frame
            colors: Nx3 array of RGB colors (0-255)
        """
        if points_world is None or len(points_world) == 0:
            return
        
        for i in range(len(points_world)):
            point = points_world[i]
            color = colors[i].astype(np.float32)
            
            key = self._get_voxel_key(point)
            
            if key not in self.voxel_data:
                self.voxel_data[key] = {
                    'points': [point],
                    'colors': [color],
                    'count': 1,
                    'mean_color': color.copy(),
                    'color_variance': 0.0
                }
            else:
                voxel = self.voxel_data[key]
                voxel['points'].append(point)
                voxel['colors'].append(color)
                voxel['count'] += 1
                
                # Update running mean and variance of color
                n = voxel['count']
                old_mean = voxel['mean_color']
                new_mean = old_mean + (color - old_mean) / n
                
                # Variance update (Welford's algorithm)
                color_diff = np.linalg.norm(color - old_mean)
                voxel['color_variance'] += (color_diff ** 2) * (n - 1) / n
                voxel['mean_color'] = new_mean
    
    def get_static_cloud(self):
        """
        Extract the accumulated static point cloud.
        
        Returns:
            Tuple of (points, colors) for static voxels
        """
        static_points = []
        static_colors = []
        
        for key, voxel in self.voxel_data.items():
            # Check if voxel has enough observations
            if voxel['count'] < self.min_observations:
                continue
            
            # Check color variance (low variance = static)
            avg_variance = voxel['color_variance'] / voxel['count']
            if avg_variance > self.color_threshold ** 2:
                continue  # Too much color change, likely dynamic
            
            # Use mean position and color
            mean_point = np.mean(voxel['points'], axis=0)
            mean_color = voxel['mean_color']
            
            static_points.append(mean_point)
            static_colors.append(mean_color)
        
        if len(static_points) == 0:
            return np.empty((0, 3)), np.empty((0, 3))
        
        return np.array(static_points), np.array(static_colors)
    
    def get_stats(self):
        """Get statistics about the accumulator."""
        total_voxels = len(self.voxel_data)
        static_count = sum(1 for v in self.voxel_data.values() 
                         if v['count'] >= self.min_observations and 
                         v['color_variance'] / v['count'] <= self.color_threshold ** 2)
        return {
            'total_voxels': total_voxels,
            'static_voxels': static_count,
            'static_ratio': static_count / total_voxels if total_voxels > 0 else 0
        }


def fuse_point_clouds(static_cloud, external_clouds, voxel_size=0.005):
    """
    Fuse static wrist camera cloud with external camera clouds.
    
    The static cloud takes priority in overlapping regions since it's
    higher quality from the wrist camera.
    
    Args:
        static_cloud: Tuple of (points, colors) from wrist camera
        external_clouds: List of tuples (points, colors) from external cameras
        voxel_size: Voxel size for deduplication
        
    Returns:
        Tuple of (fused_points, fused_colors)
    """
    all_points = []
    all_colors = []
    
    # Add static cloud first (priority)
    if len(static_cloud[0]) > 0:
        all_points.append(static_cloud[0])
        all_colors.append(static_cloud[1])
    
    # Add external clouds
    for pts, cols in external_clouds:
        if pts is not None and len(pts) > 0:
            all_points.append(pts)
            all_colors.append(cols)
    
    if len(all_points) == 0:
        return np.empty((0, 3)), np.empty((0, 3))
    
    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)
    
    # Deduplicate using voxel grid (keeps first occurrence = static cloud priority)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    
    # Normalize colors
    colors_normalized = all_colors.astype(np.float64)
    if colors_normalized.max() > 1.0:
        colors_normalized = colors_normalized / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors_normalized)
    
    # Voxel downsample
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    fused_points = np.asarray(pcd_down.points)
    fused_colors = np.asarray(pcd_down.colors) * 255.0  # Back to 0-255
    
    return fused_points, fused_colors


def main():
    """Main function to run static field fusion."""
    
    # Load configuration
    config_path = 'conversions/droid/config.yaml'
    with open(config_path, 'r') as f:
        CONFIG = yaml.safe_load(f)
    
    print("=" * 60)
    print("Static Field Fusion from Wrist Camera")
    print("=" * 60)
    
    # Parameters
    min_depth_wrist = CONFIG.get('min_depth_wrist_icp', 0.15)  # Exclude gripper (15cm)
    max_depth_wrist = CONFIG.get('wrist_max_depth', 0.75)
    min_depth_ext = CONFIG.get('min_depth', 0.1)
    max_depth_ext = CONFIG.get('ext_max_depth', 1.5)
    
    print(f"[INFO] Wrist camera: excluding gripper (< {min_depth_wrist}m)")
    print(f"[INFO] Wrist camera depth range: {min_depth_wrist}m - {max_depth_wrist}m")
    
    # --- 1. Load Robot Data ---
    print("\n[INFO] Loading H5 Trajectory...")
    h5_file = h5py.File(CONFIG['h5_path'], 'r')
    cartesian_positions = h5_file['observation/robot_state/cartesian_position'][:]
    h5_file.close()
    num_frames = len(cartesian_positions)
    print(f"[INFO] Loaded {num_frames} frames")
    
    # --- 2. Calculate Wrist Transforms ---
    wrist_cam_transforms = []
    wrist_serial = None
    T_ee_cam = None
    
    metadata_path = CONFIG.get('metadata_path')
    if metadata_path is None:
        episode_dir = os.path.dirname(CONFIG['h5_path'])
        metadata_files = glob.glob(os.path.join(episode_dir, "metadata_*.json"))
        if metadata_files:
            metadata_path = metadata_files[0]
    
    if metadata_path and os.path.exists(metadata_path):
        import json
        with open(metadata_path, 'r') as f:
            meta = json.load(f)
        wrist_serial = str(meta.get("wrist_cam_serial", ""))
        wrist_pose_t0 = meta.get("wrist_cam_extrinsics")
        
        if wrist_pose_t0:
            T_ee_cam = compute_wrist_cam_offset(wrist_pose_t0, cartesian_positions[0])
            wrist_cam_transforms = precompute_wrist_trajectory(cartesian_positions, T_ee_cam)
            print(f"[INFO] Wrist camera serial: {wrist_serial}")
    
    # --- 3. Init Cameras ---
    cameras = {}
    
    # External Cameras
    ext_data = find_episode_data_by_date(CONFIG['h5_path'], CONFIG['extrinsics_json_path'])
    if ext_data:
        for cam_id, transform_list in ext_data.items():
            if not cam_id.isdigit():
                continue
            svo = find_svo_for_camera(CONFIG['recordings_dir'], cam_id)
            if svo:
                cameras[cam_id] = {
                    "type": "external",
                    "svo": svo,
                    "world_T_cam": external_cam_to_world(transform_list)
                }
    
    # Wrist Camera
    if wrist_serial and len(wrist_cam_transforms) > 0:
        svo = find_svo_for_camera(CONFIG['recordings_dir'], wrist_serial)
        if svo:
            cameras[wrist_serial] = {
                "type": "wrist",
                "svo": svo,
                "transforms": wrist_cam_transforms,
            }
    
    # Open ZED cameras
    active_cams = {}
    for serial, data in cameras.items():
        zed = sl.Camera()
        init = sl.InitParameters()
        init.set_from_svo_file(data['svo'])
        init.svo_real_time_mode = False
        init.coordinate_units = sl.UNIT.METER
        init.depth_mode = sl.DEPTH_MODE.NEURAL
        
        if zed.open(init) == sl.ERROR_CODE.SUCCESS:
            data['zed'] = zed
            data['runtime'] = sl.RuntimeParameters()
            data['K'], data['w'], data['h'] = get_zed_intrinsics(zed)
            active_cams[serial] = data
            print(f"[INFO] Opened camera {serial} ({data['type']})")
        else:
            print(f"[ERROR] Failed to open camera {serial}")
    
    # --- 4. Initialize Static Field Accumulator ---
    accumulator = StaticFieldAccumulator(
        voxel_size=0.005,  # 5mm voxels
        min_observations=3,  # Need to see 3 times to keep
        color_threshold=40  # Max color change to be considered static
    )
    
    # --- 5. Accumulate Static Fields from Wrist Camera ---
    print("\n[INFO] Accumulating static fields from wrist camera...")
    
    max_frames = min(num_frames, CONFIG.get('max_frames', 100))
    
    # Find wrist camera
    wrist_cam = None
    for serial, cam in active_cams.items():
        if cam['type'] == 'wrist':
            wrist_cam = cam
            wrist_cam['serial'] = serial
            break
    
    if wrist_cam is None:
        print("[ERROR] No wrist camera found!")
        return
    
    # Reset camera
    wrist_cam['zed'].set_svo_position(0)
    
    for i in range(max_frames):
        if i % 20 == 0:
            print(f"  -> Processing frame {i}/{max_frames}")
        
        wrist_cam['zed'].set_svo_position(i)
        if wrist_cam['zed'].grab(wrist_cam['runtime']) != sl.ERROR_CODE.SUCCESS:
            continue
        
        if i >= len(wrist_cam['transforms']):
            continue
        
        # Get points (excluding gripper)
        xyz_local, rgb = get_filtered_cloud(
            wrist_cam['zed'], wrist_cam['runtime'],
            max_depth=max_depth_wrist,
            min_depth=min_depth_wrist  # Exclude gripper
        )
        
        if xyz_local is None or len(xyz_local) == 0:
            continue
        
        # Transform to world
        T_wrist = wrist_cam['transforms'][i]
        xyz_world = transform_points(xyz_local, T_wrist)
        
        # Add to accumulator
        accumulator.add_observation(xyz_world, rgb)
    
    # Get stats
    stats = accumulator.get_stats()
    print(f"\n[INFO] Accumulator stats:")
    print(f"  - Total voxels observed: {stats['total_voxels']}")
    print(f"  - Static voxels kept: {stats['static_voxels']}")
    print(f"  - Static ratio: {stats['static_ratio']:.2%}")
    
    # --- 6. Get Static Cloud ---
    static_points, static_colors = accumulator.get_static_cloud()
    print(f"\n[INFO] Static cloud: {len(static_points)} points")
    
    # --- 7. Get External Camera Cloud (single frame for fusion) ---
    print("\n[INFO] Getting external camera point cloud...")
    external_clouds = []
    
    for serial, cam in active_cams.items():
        if cam['type'] != 'external':
            continue
        
        cam['zed'].set_svo_position(0)
        if cam['zed'].grab(cam['runtime']) != sl.ERROR_CODE.SUCCESS:
            continue
        
        xyz_local, rgb = get_filtered_cloud(
            cam['zed'], cam['runtime'],
            max_depth=max_depth_ext,
            min_depth=min_depth_ext
        )
        
        if xyz_local is None or len(xyz_local) == 0:
            continue
        
        # Transform to world
        xyz_world = transform_points(xyz_local, cam['world_T_cam'])
        external_clouds.append((xyz_world, rgb))
        print(f"  - {serial}: {len(xyz_world)} points")
    
    # --- 8. Fuse Point Clouds ---
    print("\n[INFO] Fusing point clouds...")
    fused_points, fused_colors = fuse_point_clouds(
        (static_points, static_colors),
        external_clouds,
        voxel_size=0.005
    )
    print(f"[INFO] Fused cloud: {len(fused_points)} points")
    
    # --- 9. Save Output ---
    output_dir = CONFIG.get('video_output_path', 'point_clouds/videos')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as PLY
    ply_path = os.path.join(output_dir, "fused_static_cloud.ply")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(fused_points)
    colors_norm = fused_colors / 255.0 if fused_colors.max() > 1.0 else fused_colors
    pcd.colors = o3d.utility.Vector3dVector(colors_norm)
    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"\n[INFO] Saved fused cloud to: {ply_path}")
    
    # Save static-only cloud
    ply_static_path = os.path.join(output_dir, "static_wrist_cloud.ply")
    pcd_static = o3d.geometry.PointCloud()
    pcd_static.points = o3d.utility.Vector3dVector(static_points)
    colors_static_norm = static_colors / 255.0 if static_colors.max() > 1.0 else static_colors
    pcd_static.colors = o3d.utility.Vector3dVector(colors_static_norm)
    o3d.io.write_point_cloud(ply_static_path, pcd_static)
    print(f"[INFO] Saved static cloud to: {ply_static_path}")
    
    # --- 10. Generate Visualization Video ---
    print("\n[INFO] Generating visualization video...")
    
    # Use first external camera for visualization
    viz_cam = None
    for serial, cam in active_cams.items():
        if cam['type'] == 'external':
            viz_cam = cam
            viz_cam['serial'] = serial
            break
    
    if viz_cam:
        video_path = os.path.join(output_dir, "fused_static_reprojection.mp4")
        recorder = VideoRecorder(output_dir, "fused_static", "reprojection", viz_cam['w'], viz_cam['h'])
        
        viz_cam['zed'].set_svo_position(0)
        
        for i in range(min(50, max_frames)):
            viz_cam['zed'].set_svo_position(i)
            if viz_cam['zed'].grab(viz_cam['runtime']) != sl.ERROR_CODE.SUCCESS:
                continue
            
            # Get image
            mat_img = sl.Mat()
            viz_cam['zed'].retrieve_image(mat_img, sl.VIEW.LEFT)
            img_bgra = mat_img.get_data()
            img_bgr = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)
            
            # Project fused cloud
            uv, cols = project_points_to_image(
                fused_points, viz_cam['K'], viz_cam['world_T_cam'],
                viz_cam['w'], viz_cam['h'], colors=fused_colors
            )
            
            img_out = draw_points_on_image(img_bgr, uv, colors=cols, point_size=1)
            recorder.write_frame(img_out)
        
        recorder.close()
    
    # --- 11. Cleanup ---
    for cam in active_cams.values():
        cam['zed'].close()
    
    print("\n" + "=" * 60)
    print("DONE - Static Field Fusion Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
