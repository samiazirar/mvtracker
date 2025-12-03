"""Extract RGB frames as PNG and depth frames as per-frame NPY from DROID SVO recordings.

This script processes DROID episodes and extracts frames to:
    {output_root}/{lab}/success/{date}/{timestamp}/
        └── recordings/
            ├── {camera_serial}/
            │   ├── rgb/
            │   │   ├── 000000.png
            │   │   ├── 000001.png
            │   │   └── ...
            │   └── depth/
            │       ├── 000000.npy  (float32 array, depth in meters)
            │       ├── 000001.npy
            │       └── ...
            └── ...

Compressed mode (--compressed):
    Skips RGB extraction entirely and stores depth as FFV1 lossless video:
    {output_root}/{lab}/success/{date}/{timestamp}/
        └── recordings/
            ├── {camera_serial}/
            │   └── depth.mkv       (FFV1 lossless, 16-bit depth in mm)
            │   └── depth_meta.json (scale factor and metadata for decoding)
            └── ...

NPZ mode (--npz):
    Stores RGB and depth as compressed numpy archives (all frames together):
    {output_root}/{lab}/success/{date}/{timestamp}/
        └── recordings/
            ├── {camera_serial}/
            │   ├── rgb.npz         (lossless compressed, uint8 RGB frames)
            │   ├── depth.npz       (lossless compressed, uint16 depth in mm)
            │   └── meta.json       (metadata for decoding)
            └── ...

Usage:
    python extract_rgb_depth.py --episode_id "AUTOLab+84bd5053+2023-08-18-12h-01m-10s"
    python extract_rgb_depth.py --episode_id "AUTOLab+84bd5053+2023-08-18-12h-01m-10s" --config custom_config.yaml
    python extract_rgb_depth.py --episode_id "..." --compressed  # Skip RGB, FFV1 depth
    python extract_rgb_depth.py --episode_id "..." --npz         # RGB + depth as .npz

Requirements:
    - GPU with CUDA support (for ZED depth computation)
    - pyzed SDK installed
"""

import argparse
import json
import os
import sys
import glob
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
import yaml

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ZED SDK import
try:
    import pyzed.sl as sl
except ImportError:
    print("[ERROR] pyzed SDK not installed. Please install the ZED SDK.")
    sys.exit(1)


def parse_episode_id(episode_id: str) -> dict:
    """Parse episode ID into components.
    
    Episode ID format: "{lab}+{hash}+{date}-{hour}h-{min}m-{sec}s"
    Example: "AUTOLab+84bd5053+2023-08-18-12h-01m-10s"
    
    Returns:
        dict with keys: lab, hash, date, time_str, timestamp_folder
    """
    parts = episode_id.split('+')
    if len(parts) != 3:
        raise ValueError(f"Invalid episode ID format: {episode_id}")
    
    lab = parts[0]
    episode_hash = parts[1]
    datetime_part = parts[2]
    
    # Parse datetime: "2023-08-18-12h-01m-10s"
    match = re.match(r'(\d{4}-\d{2}-\d{2})-(\d+)h-(\d+)m-(\d+)s', datetime_part)
    if not match:
        raise ValueError(f"Invalid datetime format in episode ID: {datetime_part}")
    
    date = match.group(1)
    hour = match.group(2)
    minute = match.group(3)
    second = match.group(4)
    time_str = f"{hour}:{minute}:{second}"
    
    # Reconstruct DROID timestamp folder format: "Fri_Aug_18_11:58:51_2023"
    # Note: Uses %e (space-padded day) then replaces space with underscore
    # e.g., for Oct 8 → "Sun_Oct__8_17:30:34_2023" (double underscore)
    from datetime import datetime
    dt = datetime.strptime(f"{date} {hour}:{minute}:{second}", "%Y-%m-%d %H:%M:%S")
    timestamp_folder = dt.strftime("%a_%b_%e_%H:%M:%S_%Y").replace(" ", "_")
    
    return {
        'lab': lab,
        'hash': episode_hash,
        'date': date,
        'time_str': time_str,
        'timestamp_folder': timestamp_folder,
        'full_id': episode_id,
    }


def find_episode_paths(droid_root: str, episode_info: dict, extra_roots: list = None) -> dict:
    """Find all paths for an episode.
    
    Searches in droid_root and any extra_roots (e.g., download directory).
    
    Returns:
        dict with keys: h5_path, recordings_dir, metadata_path, relative_path
    """
    lab = episode_info['lab']
    date = episode_info['date']
    timestamp_folder = episode_info['timestamp_folder']
    
    # Build list of roots to search
    roots_to_search = [droid_root]
    if extra_roots:
        roots_to_search.extend(extra_roots)
    
    # Search in all roots, both success and failure folders
    for root in roots_to_search:
        for outcome in ['success', 'failure']:
            base_path = os.path.join(root, lab, outcome, date, timestamp_folder)
            h5_path = os.path.join(base_path, 'trajectory.h5')
            
            if os.path.exists(h5_path):
                recordings_dir = os.path.join(base_path, 'recordings', 'SVO')
                
                # Find metadata file
                metadata_files = glob.glob(os.path.join(base_path, 'metadata_*.json'))
                metadata_path = metadata_files[0] if metadata_files else None
                
                # Relative path for output structure
                relative_path = os.path.join(lab, outcome, date, timestamp_folder)
                
                return {
                    'h5_path': h5_path,
                    'recordings_dir': recordings_dir,
                    'metadata_path': metadata_path,
                    'relative_path': relative_path,
                    'outcome': outcome,
                }
    
    raise FileNotFoundError(f"Episode not found: {episode_info['full_id']}")


def find_all_svo_files(recordings_dir: str) -> dict:
    """Find all SVO files in recordings directory.
    
    Returns:
        dict mapping camera serial to SVO path
    """
    svo_files = {}
    
    # Search for SVO and SVO2 files
    for pattern in ['*.svo', '*.svo2']:
        for svo_path in glob.glob(os.path.join(recordings_dir, pattern)):
            filename = os.path.basename(svo_path)
            # Extract serial number from filename
            # Format typically: "SN12345678.svo" or similar
            serial = filename.replace('.svo2', '').replace('.svo', '')
            # Clean up serial - remove common prefixes
            serial = serial.replace('SN', '').replace('sn', '')
            svo_files[serial] = svo_path
    
    return svo_files


def save_frame_png(
    rgb: np.ndarray,
    depth: np.ndarray,
    frame_idx: int,
    rgb_dir: str,
    depth_dir: str,
) -> Tuple[str, str]:
    """Save RGB as PNG and depth as raw NPY files."""
    filename = f"{frame_idx:06d}"
    
    # Save RGB (lossless/uncompressed PNG)
    rgb_path = os.path.join(rgb_dir, f"{filename}.png")
    # OpenCV expects BGR, convert from RGB
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(rgb_path, rgb_bgr)
    
    # Save depth as float32 NPY (meters)
    depth_path = os.path.join(depth_dir, f"{filename}.npy")
    depth_to_save = depth.copy()
    depth_to_save[~np.isfinite(depth_to_save)] = 0.0  # Keep invalid values consistent
    np.save(depth_path, depth_to_save)
    
    return rgb_path, depth_path


# =============================================================================
# COMPRESSED MODE: FFV1 depth video encoding
# =============================================================================
# Depth scale: meters -> millimeters (uint16)
# Range: 0-65535mm = 0-65.535m (sufficient for indoor robotics)
DEPTH_SCALE_MM = 1000.0  # meters to millimeters
DEPTH_MAX_MM = 65535     # uint16 max


class FFV1DepthVideoWriter:
    """Write depth frames to FFV1 lossless video in 16-bit format."""
    
    def __init__(self, output_path: str, width: int, height: int, fps: float = 30.0):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_count = 0
        self.writer = None
        self._init_writer()
    
    def _init_writer(self):
        """Initialize FFV1 video writer."""
        # FFV1 codec for lossless compression
        fourcc = cv2.VideoWriter_fourcc(*'FFV1')
        
        # Use grayscale 16-bit mode by writing as 2-channel 8-bit
        # OpenCV doesn't directly support 16-bit grayscale video,
        # so we split uint16 into two uint8 channels (low, high bytes)
        self.writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            self.fps,
            (self.width, self.height),
            isColor=True  # We'll use 3-channel BGR to store 16-bit data
        )
        
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open FFV1 video writer: {self.output_path}")
    
    def write_frame(self, depth_meters: np.ndarray):
        """Write a depth frame (float32 meters) to video.
        
        Converts to uint16 millimeters and encodes in BGR channels.
        Channel layout: B=low byte, G=high byte, R=0 (for future use/validation)
        """
        # Convert to millimeters and clip to uint16 range
        depth_mm = depth_meters * DEPTH_SCALE_MM
        depth_mm = np.nan_to_num(depth_mm, nan=0.0, posinf=0.0, neginf=0.0)
        depth_mm = np.clip(depth_mm, 0, DEPTH_MAX_MM)
        depth_uint16 = depth_mm.astype(np.uint16)
        
        # Split into low and high bytes
        low_byte = (depth_uint16 & 0xFF).astype(np.uint8)
        high_byte = (depth_uint16 >> 8).astype(np.uint8)
        
        # Pack into BGR image (B=low, G=high, R=0)
        bgr_frame = np.stack([low_byte, high_byte, np.zeros_like(low_byte)], axis=-1)
        
        self.writer.write(bgr_frame)
        self.frame_count += 1
    
    def close(self):
        """Close the video writer."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
    
    def get_metadata(self) -> dict:
        """Get metadata for decoding the depth video."""
        return {
            'format': 'ffv1_depth_z16',
            'encoding': 'bgr_split_uint16',
            'channel_layout': {'B': 'low_byte', 'G': 'high_byte', 'R': 'unused'},
            'depth_scale': DEPTH_SCALE_MM,
            'depth_unit': 'millimeters',
            'depth_max_mm': DEPTH_MAX_MM,
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'frame_count': self.frame_count,
            'decode_formula': 'depth_meters = ((G << 8) | B) / 1000.0',
        }


def decode_ffv1_depth_frame(bgr_frame: np.ndarray) -> np.ndarray:
    """Decode a single BGR frame back to depth in meters.
    
    Args:
        bgr_frame: BGR image from FFV1 video (H, W, 3) uint8
        
    Returns:
        Depth map in meters (H, W) float32
    """
    low_byte = bgr_frame[:, :, 0].astype(np.uint16)
    high_byte = bgr_frame[:, :, 1].astype(np.uint16)
    depth_uint16 = (high_byte << 8) | low_byte
    depth_meters = depth_uint16.astype(np.float32) / DEPTH_SCALE_MM
    return depth_meters


# =============================================================================
# NPZ MODE: Compressed numpy archives (all frames together)
# =============================================================================

class NPZFrameAccumulator:
    """Accumulate frames and save as compressed .npz archives."""
    
    def __init__(self, output_dir: str, width: int, height: int, expected_frames: int = 0):
        self.output_dir = output_dir
        self.width = width
        self.height = height
        self.expected_frames = expected_frames
        
        # Pre-allocate if we know the frame count, otherwise use lists
        if expected_frames > 0:
            self.rgb_frames = np.zeros((expected_frames, height, width, 3), dtype=np.uint8)
            self.depth_frames = np.zeros((expected_frames, height, width), dtype=np.uint16)
            self.use_preallocated = True
        else:
            self.rgb_frames_list = []
            self.depth_frames_list = []
            self.use_preallocated = False
        
        self.frame_count = 0
    
    def add_frame(self, rgb: np.ndarray, depth_meters: np.ndarray):
        """Add a frame pair (RGB uint8, depth float32 meters)."""
        # Convert depth to uint16 millimeters for storage
        depth_mm = depth_meters * DEPTH_SCALE_MM
        depth_mm = np.nan_to_num(depth_mm, nan=0.0, posinf=0.0, neginf=0.0)
        depth_mm = np.clip(depth_mm, 0, DEPTH_MAX_MM)
        depth_uint16 = depth_mm.astype(np.uint16)
        
        if self.use_preallocated:
            if self.frame_count < self.expected_frames:
                self.rgb_frames[self.frame_count] = rgb
                self.depth_frames[self.frame_count] = depth_uint16
        else:
            self.rgb_frames_list.append(rgb.copy())
            self.depth_frames_list.append(depth_uint16)
        
        self.frame_count += 1
    
    def add_depth_only(self, depth_meters: np.ndarray):
        """Add a depth-only frame (for modes that skip RGB)."""
        depth_mm = depth_meters * DEPTH_SCALE_MM
        depth_mm = np.nan_to_num(depth_mm, nan=0.0, posinf=0.0, neginf=0.0)
        depth_mm = np.clip(depth_mm, 0, DEPTH_MAX_MM)
        depth_uint16 = depth_mm.astype(np.uint16)
        
        if self.use_preallocated:
            if self.frame_count < self.expected_frames:
                self.depth_frames[self.frame_count] = depth_uint16
        else:
            self.depth_frames_list.append(depth_uint16)
        
        self.frame_count += 1
    
    def save(self, include_rgb: bool = True) -> dict:
        """Save accumulated frames to compressed .npz files.
        
        Returns:
            dict with file paths and metadata
        """
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Finalize arrays
        if self.use_preallocated:
            # Trim to actual frame count if fewer frames than expected
            depth_array = self.depth_frames[:self.frame_count]
            if include_rgb:
                rgb_array = self.rgb_frames[:self.frame_count]
        else:
            depth_array = np.stack(self.depth_frames_list, axis=0) if self.depth_frames_list else np.array([])
            if include_rgb:
                rgb_array = np.stack(self.rgb_frames_list, axis=0) if self.rgb_frames_list else np.array([])
        
        result = {
            'frame_count': self.frame_count,
            'width': self.width,
            'height': self.height,
        }
        
        # Save depth (always)
        depth_path = os.path.join(self.output_dir, 'depth.npz')
        np.savez_compressed(depth_path, depth=depth_array)
        result['depth_path'] = depth_path
        result['depth_shape'] = depth_array.shape
        result['depth_dtype'] = 'uint16'
        result['depth_unit'] = 'millimeters'
        result['depth_scale'] = DEPTH_SCALE_MM
        
        # Save RGB if included
        if include_rgb:
            rgb_path = os.path.join(self.output_dir, 'rgb.npz')
            np.savez_compressed(rgb_path, rgb=rgb_array)
            result['rgb_path'] = rgb_path
            result['rgb_shape'] = rgb_array.shape
            result['rgb_dtype'] = 'uint8'
        
        return result
    
    def get_metadata(self, include_rgb: bool = True) -> dict:
        """Get metadata for the saved archives."""
        meta = {
            'format': 'npz_compressed',
            'compression': 'zlib',
            'frame_count': self.frame_count,
            'width': self.width,
            'height': self.height,
            'depth': {
                'file': 'depth.npz',
                'array_key': 'depth',
                'dtype': 'uint16',
                'unit': 'millimeters',
                'scale': DEPTH_SCALE_MM,
                'decode_formula': 'depth_meters = depth_mm / 1000.0',
            },
        }
        if include_rgb:
            meta['rgb'] = {
                'file': 'rgb.npz',
                'array_key': 'rgb',
                'dtype': 'uint8',
                'channels': 'RGB',
            }
        return meta


def load_npz_frames(npz_dir: str) -> Tuple[Optional[np.ndarray], np.ndarray, dict]:
    """Load frames from NPZ archives.
    
    Args:
        npz_dir: Directory containing rgb.npz and/or depth.npz
        
    Returns:
        Tuple of (rgb_frames, depth_meters, metadata)
        rgb_frames may be None if only depth was saved
    """
    meta_path = os.path.join(npz_dir, 'meta.json')
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    # Load depth
    depth_path = os.path.join(npz_dir, 'depth.npz')
    with np.load(depth_path) as data:
        depth_mm = data['depth']
    depth_meters = depth_mm.astype(np.float32) / meta['depth']['scale']
    
    # Load RGB if available
    rgb_frames = None
    rgb_path = os.path.join(npz_dir, 'rgb.npz')
    if os.path.exists(rgb_path):
        with np.load(rgb_path) as data:
            rgb_frames = data['rgb']
    
    return rgb_frames, depth_meters, meta


def process_camera(
    svo_path: str,
    camera_serial: str,
    output_base_dir: str,
    max_frames: Optional[int] = None,
    compressed: bool = False,
    npz: bool = False,
    fps: float = 30.0,
) -> dict:
    """Process a single camera SVO file and extract frames.
    
    Args:
        svo_path: Path to SVO file
        camera_serial: Camera serial number for output folder
        output_base_dir: Base output directory
        max_frames: Maximum frames to process (None = all)
        compressed: If True, skip RGB and save depth as FFV1 video
        npz: If True, save RGB and depth as compressed .npz archives
        fps: Frame rate for video output (used in compressed mode)
    
    Returns:
        dict with processing statistics
    """
    if npz:
        mode_str = "NPZ (RGB + depth compressed archives)"
    elif compressed:
        mode_str = "COMPRESSED (FFV1 depth only)"
    else:
        mode_str = "STANDARD (RGB PNG + depth NPY)"
    print(f"  Processing camera: {camera_serial} [{mode_str}]")
    
    # Create output directories
    cam_dir = os.path.join(output_base_dir, 'recordings', camera_serial)
    os.makedirs(cam_dir, exist_ok=True)
    
    if not compressed and not npz:
        # Standard mode: create rgb and depth directories
        rgb_dir = os.path.join(cam_dir, 'rgb')
        depth_dir = os.path.join(cam_dir, 'depth')
        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
    
    # Initialize ZED camera
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo_path)
    init_params.svo_real_time_mode = False
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL  # Best quality depth
    
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"    [ERROR] Failed to open SVO: {err}")
        return {'serial': camera_serial, 'status': 'failed', 'error': str(err)}
    
    # Get camera info
    cam_info = zed.get_camera_information()
    width = cam_info.camera_configuration.resolution.width
    height = cam_info.camera_configuration.resolution.height
    total_frames = zed.get_svo_number_of_frames()
    
    if max_frames is not None:
        process_frames = min(max_frames, total_frames)
    else:
        process_frames = total_frames
    
    print(f"    Resolution: {width}x{height}")
    print(f"    Total frames: {total_frames}, processing: {process_frames}")
    
    # Prepare matrices
    runtime_params = sl.RuntimeParameters()
    mat_rgb = sl.Mat()
    mat_depth = sl.Mat()
    
    # Initialize writers based on mode
    depth_video_writer = None
    npz_accumulator = None
    
    if compressed:
        depth_video_path = os.path.join(cam_dir, 'depth.mkv')
        depth_video_writer = FFV1DepthVideoWriter(depth_video_path, width, height, fps)
        print(f"    Output: {depth_video_path}")
    elif npz:
        npz_accumulator = NPZFrameAccumulator(cam_dir, width, height, process_frames)
        print(f"    Output: {cam_dir}/rgb.npz, {cam_dir}/depth.npz")
    
    # Process frames
    frames_saved = 0
    for frame_idx in range(process_frames):
        if frame_idx % 100 == 0:
            print(f"    Frame {frame_idx}/{process_frames}")
        
        err = zed.grab(runtime_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"    [WARN] Failed to grab frame {frame_idx}: {err}")
            continue
        
        # Retrieve depth map (always needed)
        zed.retrieve_measure(mat_depth, sl.MEASURE.DEPTH)
        depth = mat_depth.get_data().copy()  # float32, meters
        
        if compressed:
            # Compressed mode: write depth to FFV1 video (skip RGB)
            depth_video_writer.write_frame(depth)
        elif npz:
            # NPZ mode: accumulate RGB and depth frames
            zed.retrieve_image(mat_rgb, sl.VIEW.LEFT)
            rgb_data = mat_rgb.get_data()
            rgb = cv2.cvtColor(rgb_data, cv2.COLOR_BGRA2RGB)
            npz_accumulator.add_frame(rgb, depth)
        else:
            # Standard mode: save RGB PNG and depth NPY
            zed.retrieve_image(mat_rgb, sl.VIEW.LEFT)
            rgb_data = mat_rgb.get_data()
            # ZED returns BGRA, convert to RGB
            rgb = cv2.cvtColor(rgb_data, cv2.COLOR_BGRA2RGB)
            
            save_frame_png(
                rgb, depth, frame_idx,
                rgb_dir, depth_dir,
            )
        
        frames_saved += 1
    
    # Finalize based on mode
    if compressed and depth_video_writer is not None:
        depth_video_writer.close()
        
        # Save depth video metadata
        depth_meta = depth_video_writer.get_metadata()
        depth_meta_path = os.path.join(cam_dir, 'depth_meta.json')
        with open(depth_meta_path, 'w') as f:
            json.dump(depth_meta, f, indent=2)
        print(f"    Saved depth metadata: {depth_meta_path}")
    
    elif npz and npz_accumulator is not None:
        print(f"    Saving compressed archives...")
        save_result = npz_accumulator.save(include_rgb=True)
        
        # Save metadata
        meta = npz_accumulator.get_metadata(include_rgb=True)
        meta_path = os.path.join(cam_dir, 'meta.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"    Saved: rgb.npz ({save_result['rgb_shape']}), depth.npz ({save_result['depth_shape']})")
    
    # Get intrinsics and save
    calib = cam_info.camera_configuration.calibration_parameters.left_cam
    intrinsics = {
        'fx': float(calib.fx),
        'fy': float(calib.fy),
        'cx': float(calib.cx),
        'cy': float(calib.cy),
        'width': width,
        'height': height,
        'k1': float(calib.disto[0]) if len(calib.disto) > 0 else 0.0,
        'k2': float(calib.disto[1]) if len(calib.disto) > 1 else 0.0,
        'p1': float(calib.disto[2]) if len(calib.disto) > 2 else 0.0,
        'p2': float(calib.disto[3]) if len(calib.disto) > 3 else 0.0,
        'k3': float(calib.disto[4]) if len(calib.disto) > 4 else 0.0,
    }
    
    intrinsics_path = os.path.join(cam_dir, 'intrinsics.json')
    with open(intrinsics_path, 'w') as f:
        json.dump(intrinsics, f, indent=2)
    
    zed.close()
    
    # Determine mode string for result
    if npz:
        mode = 'npz'
    elif compressed:
        mode = 'compressed'
    else:
        mode = 'standard'
    
    result = {
        'serial': camera_serial,
        'status': 'success',
        'frames_saved': frames_saved,
        'total_frames': total_frames,
        'width': width,
        'height': height,
        'mode': mode,
    }
    
    if compressed:
        result['depth_video'] = os.path.join(cam_dir, 'depth.mkv')
    elif npz:
        result['rgb_archive'] = os.path.join(cam_dir, 'rgb.npz')
        result['depth_archive'] = os.path.join(cam_dir, 'depth.npz')
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Extract RGB and depth frames from DROID SVO recordings."
    )
    parser.add_argument(
        "--episode_id",
        required=True,
        help='Episode ID, e.g., "AUTOLab+84bd5053+2023-08-18-12h-01m-10s"',
    )
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
        help="Path to config YAML file.",
    )
    parser.add_argument(
        "--cameras",
        nargs="*",
        help="Specific camera serials to process (default: all)",
    )
    parser.add_argument(
        "--compressed",
        action="store_true",
        help="Compressed mode: skip RGB, store depth as FFV1 lossless video (z16)",
    )
    parser.add_argument(
        "--npz",
        action="store_true",
        help="NPZ mode: store RGB and depth as lossless compressed .npz archives",
    )
    args = parser.parse_args()
    
    # Validate mutually exclusive modes
    if args.compressed and args.npz:
        print("[ERROR] Cannot use both --compressed and --npz. Choose one.")
        sys.exit(1)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.npz:
        mode_str = "NPZ (RGB + depth compressed archives)"
    elif args.compressed:
        mode_str = "COMPRESSED (depth FFV1 only)"
    else:
        mode_str = "STANDARD (RGB PNG + depth NPY)"
    print(f"=== Extracting Frames [{mode_str}] ===")
    print(f"Episode: {args.episode_id}")
    
    # Parse episode ID
    episode_info = parse_episode_id(args.episode_id)
    print(f"  Lab: {episode_info['lab']}")
    print(f"  Date: {episode_info['date']}")
    
    # Find episode paths (search both droid_root and download directory)
    extra_roots = [config.get('download_dir', './droid_downloads')]
    episode_paths = find_episode_paths(config['droid_root'], episode_info, extra_roots)
    print(f"  Found: {episode_paths['relative_path']}")
    print(f"  Recordings: {episode_paths['recordings_dir']}")
    
    # Find all SVO files
    svo_files = find_all_svo_files(episode_paths['recordings_dir'])
    print(f"  Found {len(svo_files)} cameras: {list(svo_files.keys())}")
    
    # Filter cameras if specified
    if args.cameras:
        svo_files = {k: v for k, v in svo_files.items() if k in args.cameras}
        print(f"  Filtering to: {list(svo_files.keys())}")
    
    if not svo_files:
        print("[ERROR] No SVO files found!")
        sys.exit(1)
    
    # Create output directory
    output_dir = os.path.join(config['output_root'], episode_paths['relative_path'])
    os.makedirs(output_dir, exist_ok=True)
    print(f"  Output: {output_dir}")
    
    # Process each camera
    results = []
    fps = config.get('fps', 30.0)
    for serial, svo_path in svo_files.items():
        result = process_camera(
            svo_path=svo_path,
            camera_serial=serial,
            output_base_dir=output_dir,
            max_frames=config.get('max_frames'),
            compressed=args.compressed,
            npz=args.npz,
            fps=fps,
        )
        results.append(result)
    
    # Determine mode for metadata
    if args.npz:
        extraction_mode = 'npz'
    elif args.compressed:
        extraction_mode = 'compressed'
    else:
        extraction_mode = 'standard'
    
    # Save extraction metadata
    metadata = {
        'episode_id': args.episode_id,
        'source_path': episode_paths['relative_path'],
        'mode': extraction_mode,
        'compressed': args.compressed,
        'npz': args.npz,
        'cameras': results,
    }
    
    metadata_path = os.path.join(output_dir, 'recordings', 'extraction_metadata.json')
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Summary
    print(f"\n=== Extraction Summary [{mode_str}] ===")
    for result in results:
        if result['status'] == 'success':
            mode_info = f" [{result.get('mode', 'standard')}]"
            print(f"  {result['serial']}: {result['frames_saved']} frames saved{mode_info}")
        else:
            print(f"  {result['serial']}: FAILED - {result.get('error', 'unknown')}")
    
    print(f"\nOutput: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
