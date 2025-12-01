"""Utility modules for DROID point cloud generation."""

from .transforms import (
    # Basic utilities
    pose6_to_T, 
    rvec_tvec_to_matrix, 
    transform_points,
    decompose_transform,
    invert_transform,
    # DROID-specific pipelines
    compute_wrist_cam_offset,
    wrist_cam_to_world,
    wrist_points_to_world,
    external_cam_to_world,
    external_points_to_world,
    precompute_wrist_trajectory,
)
from .camera_utils import (
    find_svo_for_camera,
    find_episode_data_by_date,
    get_zed_intrinsics,
    get_filtered_cloud
)
from .gripper_visualizer import GripperVisualizer
from .object_detector import ObjectDetector, boxes_to_mask
from .video_utils import (
    VideoRecorder, 
    project_points_to_image,
    project_points_with_depth,
    draw_points_on_image,
    draw_points_on_image_fast,
    create_reprojection_video,
    draw_track_trails_on_image,
)
from .optimization import (
    # ICP optimization functions
    optimize_wrist_camera_icp,
    optimize_wrist_camera_icp_z_only,
    optimize_wrist_z_offset_icp,
    apply_z_offset_to_wrist_transforms,
    optimize_wrist_z_offset,
    optimize_wrist_z_offset_multi_frame,
    # Open3D utilities
    numpy_to_o3d_pointcloud,
    o3d_to_numpy,
    downsample_pointcloud,
    estimate_normals,
    # ICP registration
    run_icp_point_to_plane,
    run_icp_point_to_point,
    # Filtering
    filter_points_by_distance_from_camera,
    filter_wrist_cloud_for_icp,
    # Multi-frame optimization
    optimize_external_cameras_multi_frame,
    optimize_wrist_multi_frame,
)
from .tracking import (
    extract_contact_surface,
    sample_contact_points,
    compute_finger_transforms,
    ContactSurfaceTracker,
    MinimalGripperVisualizer,
)

__all__ = [
    # Basic transform utilities
    'pose6_to_T',
    'rvec_tvec_to_matrix',
    'transform_points',
    'decompose_transform',
    'invert_transform',
    # DROID-specific pipelines
    'compute_wrist_cam_offset',
    'wrist_cam_to_world',
    'wrist_points_to_world',
    'external_cam_to_world',
    'external_points_to_world',
    'precompute_wrist_trajectory',
    # Camera utilities
    'find_svo_for_camera',
    'find_episode_data_by_date',
    'get_zed_intrinsics',
    'get_filtered_cloud',
    # Visualization & Detection
    'GripperVisualizer',
    'ObjectDetector',
    'boxes_to_mask',
    # Video utilities
    'VideoRecorder',
    'project_points_to_image',
    'project_points_with_depth',
    'draw_points_on_image',
    'draw_points_on_image_fast',
    'create_reprojection_video',
    'draw_track_trails_on_image',
    # ICP Optimization
    'optimize_wrist_camera_icp',
    'optimize_wrist_camera_icp_z_only',
    'optimize_wrist_z_offset_icp',
    'apply_z_offset_to_wrist_transforms',
    'optimize_wrist_z_offset',
    'optimize_wrist_z_offset_multi_frame',
    # Open3D utilities
    'numpy_to_o3d_pointcloud',
    'o3d_to_numpy',
    'downsample_pointcloud',
    'estimate_normals',
    # ICP registration
    'run_icp_point_to_plane',
    'run_icp_point_to_point',
    # Filtering
    'filter_points_by_distance_from_camera',
    'filter_wrist_cloud_for_icp',
    # Multi-frame optimization
    'optimize_external_cameras_multi_frame',
    'optimize_wrist_multi_frame',
    # Tracking utilities
    'extract_contact_surface',
    'sample_contact_points',
    'compute_finger_transforms',
    'ContactSurfaceTracker',
    'MinimalGripperVisualizer',
]
