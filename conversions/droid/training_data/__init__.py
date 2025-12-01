"""DROID Training Data Generation Package.

This package provides tools to generate training data from DROID episodes:

- generate_tracks_and_metadata.py: Generate gripper tracks and camera extrinsics (CPU)
- extract_rgb_depth.py: Extract RGB and depth frames as lossless PNGs (GPU)
- process_episode.py: Convenience wrapper to run both scripts

See README.md for usage instructions.
"""

from .generate_tracks_and_metadata import (
    parse_episode_id,
    find_episode_paths,
    load_cam2base_for_episode,
    generate_tracks,
    compute_extrinsics,
)

from .extract_rgb_depth import (
    find_all_svo_files,
    save_frame_png,
    process_camera,
)

__all__ = [
    'parse_episode_id',
    'find_episode_paths',
    'load_cam2base_for_episode',
    'generate_tracks',
    'compute_extrinsics',
    'find_all_svo_files',
    'save_frame_png',
    'process_camera',
]
