"""Utils package for DROID conversions.

This module exposes all utility functions for DROID point cloud generation.
"""

# Import from transforms
from .transforms import (
    pose6_to_T,
    rvec_tvec_to_matrix,
    transform_points,
    decompose_transform,
    invert_transform,
    compute_wrist_cam_offset,
    wrist_cam_to_world,
    wrist_points_to_world,
    external_cam_to_world,
    external_points_to_world,
    precompute_wrist_trajectory,
)

# Import from camera_utils
from .camera_utils import (
    find_svo_for_camera,
    find_episode_data_by_date,
    get_zed_intrinsics,
    get_filtered_cloud,
    reproject_point_cloud,
    reproject_all_to_one,
    backproject_2d_to_3d,
    combine_masks,
)

# Import from gripper_visualizer
from .gripper_visualizer import GripperVisualizer

# Import from object_detector (if available)
_object_detector_available = False
try:
    from .object_detector import ObjectDetector
    _object_detector_available = True
except ImportError:
    pass

__all__ = [
    # Transforms
    'pose6_to_T',
    'rvec_tvec_to_matrix',
    'transform_points',
    'decompose_transform',
    'invert_transform',
    'compute_wrist_cam_offset',
    'wrist_cam_to_world',
    'wrist_points_to_world',
    'external_cam_to_world',
    'external_points_to_world',
    'precompute_wrist_trajectory',
    # Camera utils
    'find_svo_for_camera',
    'find_episode_data_by_date',
    'get_zed_intrinsics',
    'get_filtered_cloud',
    'reproject_point_cloud',
    'reproject_all_to_one',
    'backproject_2d_to_3d',
    'combine_masks',
    # Gripper
    'GripperVisualizer',
]

# Add ObjectDetector to __all__ only if it was successfully imported
if _object_detector_available:
    __all__.append('ObjectDetector')
