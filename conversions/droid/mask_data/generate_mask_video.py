#!/usr/bin/env python3
"""
Generate sample videos with 2D masks for gripper and robot arm.

This script:
1. Loads a DROID episode (H5 trajectory + SVO camera files)
2. Computes forward kinematics for each frame
3. Renders 2D masks for gripper and robot arm
4. Creates overlay videos showing masks on camera frames

Usage:
    python generate_mask_video.py --config config.yaml
    python generate_mask_video.py --h5_path /path/to/trajectory.h5 --output_dir ./output

Output:
    - {camera}_gripper_mask.mp4: Video with gripper mask overlay
    - {camera}_robot_mask.mp4: Video with robot arm mask overlay  
    - {camera}_combined_mask.mp4: Video with both masks
"""

import argparse
import numpy as np
import os
import sys
import glob
import h5py
import json
import yaml
import cv2
from typing import Dict, Optional, Tuple
from scipy.spatial.transform import Rotation as R

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mask_utils import (
    GripperMaskRenderer,
    RobotArmMaskRenderer,
    CombinedRobotMaskRenderer,
    pose6_to_T,
    panda_forward_kinematics,
)

# Try importing ZED SDK for SVO reading
try:
    import pyzed.sl as sl
    HAS_ZED = True
except ImportError:
    print("[WARN] ZED SDK not available. Using MP4 fallback if available.")
    HAS_ZED = False


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    "h5_path": None,
    "recordings_dir": None,
    "extrinsics_json_path": "/data/cam2base_extrinsic_superset.json",
    "metadata_path": None,
    "output_dir": "./mask_output",
    "max_frames": 500,
    "width": 640,
    "height": 360,
    "fps": 30.0,
    
    # Mask rendering
    "gripper_mask_color": [0, 255, 0],      # Green
    "robot_mask_color": [255, 0, 0],        # Blue (BGR)
    "combined_mask_alpha": 0.4,
    "min_depth": 0.01,
    
    # Robot base transform (if known)
    "robot_base_pose": None,  # [x, y, z, roll, pitch, yaw]
}


# =============================================================================
# CAMERA UTILITIES
# =============================================================================

def find_episode_data_by_date(h5_path: str, extrinsics_json_path: str) -> Optional[Dict]:
    """Find camera extrinsics for an episode based on date matching."""
    if not os.path.exists(extrinsics_json_path):
        print(f"[WARN] Extrinsics file not found: {extrinsics_json_path}")
        return None
    
    # Extract date from h5 path
    # Format: .../YYYY-MM-DD/...
    import re
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', h5_path)
    if not date_match:
        print(f"[WARN] Could not extract date from path: {h5_path}")
        return None
    
    date_str = date_match.group(1)
    
    with open(extrinsics_json_path, 'r') as f:
        extrinsics_data = json.load(f)
    
    # Find matching entry
    for entry in extrinsics_data:
        if date_str in entry.get('date', ''):
            return entry
    
    print(f"[WARN] No extrinsics found for date: {date_str}")
    return None


def get_camera_intrinsics_from_svo(svo_path: str) -> Tuple[np.ndarray, int, int]:
    """Extract camera intrinsics from SVO file."""
    if not HAS_ZED:
        raise RuntimeError("ZED SDK required for SVO intrinsics extraction")
    
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo_path)
    init_params.svo_real_time_mode = False
    
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open SVO: {status}")
    
    # Get calibration
    calib = zed.get_camera_information().camera_configuration.calibration_parameters
    left_cam = calib.left_cam
    
    fx = left_cam.fx
    fy = left_cam.fy
    cx = left_cam.cx
    cy = left_cam.cy
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    
    width = zed.get_camera_information().camera_configuration.resolution.width
    height = zed.get_camera_information().camera_configuration.resolution.height
    
    zed.close()
    
    return K, width, height


def find_svo_for_camera(recordings_dir: str, cam_serial: str) -> Optional[str]:
    """Find SVO file for a specific camera serial."""
    patterns = [f"*{cam_serial}*.svo", f"*{cam_serial}*.svo2"]
    for pat in patterns:
        matches = glob.glob(os.path.join(recordings_dir, pat))
        if matches:
            return matches[0]
    return None


def find_mp4_for_camera(recordings_dir: str, cam_serial: str) -> Optional[str]:
    """Find MP4 file for a specific camera serial."""
    # Check for recordings_dir/MP4/{serial}_left.mp4
    mp4_dir = os.path.join(os.path.dirname(recordings_dir), "MP4")
    if os.path.exists(mp4_dir):
        mp4_path = os.path.join(mp4_dir, f"{cam_serial}_left.mp4")
        if os.path.exists(mp4_path):
            return mp4_path
    return None


class SVOReader:
    """Read frames from SVO file."""
    
    def __init__(self, svo_path: str):
        self.svo_path = svo_path
        self.zed = sl.Camera()
        
        init_params = sl.InitParameters()
        init_params.set_from_svo_file(svo_path)
        init_params.svo_real_time_mode = False
        init_params.depth_mode = sl.DEPTH_MODE.NONE
        
        status = self.zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open SVO: {status}")
        
        # Get calibration
        calib = self.zed.get_camera_information().camera_configuration.calibration_parameters
        left_cam = calib.left_cam
        
        self.K = np.array([
            [left_cam.fx, 0, left_cam.cx],
            [0, left_cam.fy, left_cam.cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        self.width = self.zed.get_camera_information().camera_configuration.resolution.width
        self.height = self.zed.get_camera_information().camera_configuration.resolution.height
        self.num_frames = self.zed.get_svo_number_of_frames()
        
        self.image = sl.Mat()
        self.current_frame = 0
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read next frame."""
        if self.current_frame >= self.num_frames:
            return False, None
        
        status = self.zed.grab()
        if status != sl.ERROR_CODE.SUCCESS:
            return False, None
        
        self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
        frame = self.image.get_data()[:, :, :3].copy()  # BGR
        
        self.current_frame += 1
        return True, frame
    
    def release(self):
        self.zed.close()


class MP4Reader:
    """Read frames from MP4 file."""
    
    def __init__(self, mp4_path: str, K: np.ndarray = None):
        self.cap = cv2.VideoCapture(mp4_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open MP4: {mp4_path}")
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Use provided intrinsics or estimate
        if K is not None:
            self.K = K
        else:
            # Estimate intrinsics (typical ZED values)
            fx = self.width * 0.7
            fy = fx
            cx = self.width / 2
            cy = self.height / 2
            self.K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float64)
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        return self.cap.read()
    
    def release(self):
        self.cap.release()


# =============================================================================
# MASK OVERLAY UTILITIES
# =============================================================================

def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float = 0.4
) -> np.ndarray:
    """Overlay a binary mask on an image with specified color and alpha."""
    result = image.copy()
    mask_bool = mask > 0
    
    # Create colored overlay
    overlay = np.zeros_like(image)
    overlay[mask_bool] = color
    
    # Blend
    result[mask_bool] = cv2.addWeighted(
        image[mask_bool], 1 - alpha,
        overlay[mask_bool], alpha,
        0
    )
    
    # Draw contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, color, 2)
    
    return result


def overlay_dual_masks(
    image: np.ndarray,
    gripper_mask: np.ndarray,
    robot_mask: np.ndarray,
    gripper_color: Tuple[int, int, int] = (0, 255, 0),
    robot_color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.4
) -> np.ndarray:
    """Overlay both gripper and robot masks with different colors."""
    result = overlay_mask(image, robot_mask, robot_color, alpha)
    result = overlay_mask(result, gripper_mask, gripper_color, alpha)
    return result


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_episode(
    h5_path: str,
    recordings_dir: str,
    extrinsics_json_path: str,
    output_dir: str,
    metadata_path: Optional[str] = None,
    max_frames: int = 500,
    config: Dict = None
):
    """Process a DROID episode and generate mask videos."""
    
    if config is None:
        config = DEFAULT_CONFIG.copy()
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("DROID Mask Video Generator")
    print("=" * 60)
    print(f"H5 Path: {h5_path}")
    print(f"Recordings: {recordings_dir}")
    print(f"Output: {output_dir}")
    print()
    
    # Load H5 data
    print("[1/5] Loading trajectory data...")
    h5_file = h5py.File(h5_path, 'r')
    
    # Get robot state
    cartesian_positions = h5_file['observation/robot_state/cartesian_position'][:]
    gripper_positions = h5_file['observation/robot_state/gripper_position'][:]
    
    # Check for joint positions (needed for full robot arm)
    has_joint_data = 'observation/robot_state/joint_positions' in h5_file
    if has_joint_data:
        joint_positions = h5_file['observation/robot_state/joint_positions'][:]
        print(f"  Found joint positions: {joint_positions.shape}")
    else:
        joint_positions = None
        print("  No joint positions found - using gripper only")
    
    num_frames = min(len(cartesian_positions), max_frames)
    h5_file.close()
    
    print(f"  Frames: {num_frames}")
    print(f"  Cartesian shape: {cartesian_positions.shape}")
    print(f"  Gripper shape: {gripper_positions.shape}")
    
    # Load metadata for wrist camera
    print("\n[2/5] Loading camera data...")
    
    # Auto-discover metadata
    if metadata_path is None:
        episode_dir = os.path.dirname(h5_path)
        metadata_files = glob.glob(os.path.join(episode_dir, "metadata_*.json"))
        if metadata_files:
            metadata_path = metadata_files[0]
    
    wrist_serial = None
    wrist_cam_offset = None
    
    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        wrist_serial = str(metadata.get("wrist_cam_serial", ""))
        wrist_cam_extrinsics = metadata.get("wrist_cam_extrinsics")
        
        if wrist_cam_extrinsics:
            # Compute constant offset from EE to wrist camera
            T_base_cam0 = pose6_to_T(wrist_cam_extrinsics)
            T_base_ee0 = pose6_to_T(cartesian_positions[0])
            
            # Apply 90-degree Z rotation fix (DROID convention)
            R_fix = R.from_euler('z', 90, degrees=True).as_matrix()
            T_base_ee0[:3, :3] = T_base_ee0[:3, :3] @ R_fix
            
            wrist_cam_offset = np.linalg.inv(T_base_ee0) @ T_base_cam0
            print(f"  Wrist camera: {wrist_serial}")
    
    # Find external cameras from extrinsics
    ext_data = find_episode_data_by_date(h5_path, extrinsics_json_path)
    
    cameras = {}
    
    if ext_data:
        for cam_info in ext_data.get('cameras', []):
            serial = str(cam_info.get('serial', ''))
            if not serial:
                continue
            
            # Skip wrist camera (we handle it separately)
            if serial == wrist_serial:
                continue
            
            # Parse extrinsics
            extr = cam_info.get('extrinsics', {})
            rvec = np.array(extr.get('rvec', [0, 0, 0]))
            tvec = np.array(extr.get('tvec', [0, 0, 0]))
            
            # Build world_T_cam
            R_mat = R.from_rotvec(rvec).as_matrix()
            T_world_cam = np.eye(4)
            T_world_cam[:3, :3] = R_mat
            T_world_cam[:3, 3] = tvec
            
            cameras[serial] = {
                'type': 'external',
                'world_T_cam': T_world_cam,
            }
    
    # Add wrist camera if we have offset
    if wrist_serial and wrist_cam_offset is not None:
        cameras[wrist_serial] = {
            'type': 'wrist',
            'ee_T_cam': wrist_cam_offset,
        }
    
    print(f"  Found {len(cameras)} cameras")
    for serial in cameras:
        print(f"    - {serial} ({cameras[serial]['type']})")
    
    if not cameras:
        print("[ERROR] No cameras found!")
        return
    
    # Initialize mask renderers
    print("\n[3/5] Initializing mask renderers...")
    gripper_renderer = GripperMaskRenderer()
    
    if has_joint_data:
        arm_renderer = RobotArmMaskRenderer()
        print("  Gripper renderer: OK")
        print("  Robot arm renderer: OK")
    else:
        arm_renderer = None
        print("  Gripper renderer: OK")
        print("  Robot arm renderer: Skipped (no joint data)")
    
    # Process each camera
    print("\n[4/5] Processing cameras...")
    
    for serial, cam_info in cameras.items():
        print(f"\n  Processing camera: {serial}")
        
        # Find video source
        if HAS_ZED:
            svo_path = find_svo_for_camera(recordings_dir, serial)
            if svo_path:
                print(f"    SVO: {os.path.basename(svo_path)}")
                reader = SVOReader(svo_path)
            else:
                mp4_path = find_mp4_for_camera(recordings_dir, serial)
                if mp4_path:
                    print(f"    MP4: {os.path.basename(mp4_path)}")
                    reader = MP4Reader(mp4_path)
                else:
                    print(f"    [SKIP] No video source found")
                    continue
        else:
            mp4_path = find_mp4_for_camera(recordings_dir, serial)
            if mp4_path:
                print(f"    MP4: {os.path.basename(mp4_path)}")
                reader = MP4Reader(mp4_path)
            else:
                print(f"    [SKIP] No video source found (ZED SDK not available)")
                continue
        
        K = reader.K
        width = reader.width
        height = reader.height
        
        # Resize for output
        out_width = config.get('width', 640)
        out_height = config.get('height', 360)
        
        # Adjust intrinsics for resize
        scale_x = out_width / width
        scale_y = out_height / height
        K_scaled = K.copy()
        K_scaled[0, :] *= scale_x
        K_scaled[1, :] *= scale_y
        
        # Initialize video writers
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = config.get('fps', 30.0)
        
        writers = {}
        writers['gripper'] = cv2.VideoWriter(
            os.path.join(output_dir, f"{serial}_gripper_mask.mp4"),
            fourcc, fps, (out_width, out_height)
        )
        
        if arm_renderer:
            writers['robot'] = cv2.VideoWriter(
                os.path.join(output_dir, f"{serial}_robot_mask.mp4"),
                fourcc, fps, (out_width, out_height)
            )
            writers['combined'] = cv2.VideoWriter(
                os.path.join(output_dir, f"{serial}_combined_mask.mp4"),
                fourcc, fps, (out_width, out_height)
            )
        
        # Process frames
        frame_idx = 0
        
        while frame_idx < num_frames:
            ret, frame = reader.read()
            if not ret:
                break
            
            # Resize frame
            frame = cv2.resize(frame, (out_width, out_height))
            
            # Get robot state for this frame
            cart_pos = cartesian_positions[frame_idx]
            grip_pos = gripper_positions[frame_idx]
            
            # Compute end-effector transform
            T_base_ee = pose6_to_T(cart_pos)
            
            # Apply rotation fix
            R_fix = R.from_euler('z', 90, degrees=True).as_matrix()
            T_base_ee[:3, :3] = T_base_ee[:3, :3] @ R_fix
            
            # Compute camera extrinsics
            if cam_info['type'] == 'wrist':
                # Wrist camera follows end-effector
                T_world_cam = T_base_ee @ cam_info['ee_T_cam']
            else:
                T_world_cam = cam_info['world_T_cam']
            
            # Camera to world transform
            T_cam_world = np.linalg.inv(T_world_cam)
            
            # Render gripper mask
            gripper_mask = gripper_renderer.render_mask(
                T_base_ee, grip_pos, K_scaled, T_cam_world,
                out_width, out_height, config.get('min_depth', 0.01)
            )
            
            # Overlay on frame
            gripper_color = tuple(config.get('gripper_mask_color', [0, 255, 0]))
            alpha = config.get('combined_mask_alpha', 0.4)
            
            gripper_overlay = overlay_mask(frame, gripper_mask, gripper_color, alpha)
            writers['gripper'].write(gripper_overlay)
            
            # Robot arm mask (if available)
            if arm_renderer and has_joint_data:
                joint_angles = joint_positions[frame_idx][:7]
                
                robot_mask = arm_renderer.render_mask(
                    joint_angles, K_scaled, T_cam_world,
                    out_width, out_height, np.eye(4),
                    config.get('min_depth', 0.01), exclude_hand=True
                )
                
                robot_color = tuple(config.get('robot_mask_color', [255, 0, 0]))
                
                robot_overlay = overlay_mask(frame, robot_mask, robot_color, alpha)
                writers['robot'].write(robot_overlay)
                
                combined_overlay = overlay_dual_masks(
                    frame, gripper_mask, robot_mask,
                    gripper_color, robot_color, alpha
                )
                writers['combined'].write(combined_overlay)
            
            frame_idx += 1
            
            if frame_idx % 50 == 0:
                print(f"    Frame {frame_idx}/{num_frames}")
        
        # Cleanup
        reader.release()
        for w in writers.values():
            w.release()
        
        print(f"    Saved {frame_idx} frames")
    
    print("\n[5/5] Complete!")
    print(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate mask videos for DROID episodes"
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to YAML config file'
    )
    parser.add_argument(
        '--h5_path', type=str, default=None,
        help='Path to trajectory.h5 file'
    )
    parser.add_argument(
        '--recordings_dir', type=str, default=None,
        help='Path to recordings/SVO directory'
    )
    parser.add_argument(
        '--extrinsics', type=str, 
        default='/data/cam2base_extrinsic_superset.json',
        help='Path to camera extrinsics JSON'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./mask_output',
        help='Output directory for videos'
    )
    parser.add_argument(
        '--max_frames', type=int, default=500,
        help='Maximum frames to process'
    )
    
    args = parser.parse_args()
    
    config = DEFAULT_CONFIG.copy()
    
    # Load config file if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config.update(yaml.safe_load(f))
    
    # Override with command line args
    if args.h5_path:
        config['h5_path'] = args.h5_path
    if args.recordings_dir:
        config['recordings_dir'] = args.recordings_dir
    if args.extrinsics:
        config['extrinsics_json_path'] = args.extrinsics
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.max_frames:
        config['max_frames'] = args.max_frames
    
    # Validate required paths
    if not config.get('h5_path'):
        print("[ERROR] h5_path is required. Use --h5_path or --config")
        sys.exit(1)
    
    h5_path = config['h5_path']
    
    # Auto-discover recordings directory
    recordings_dir = config.get('recordings_dir')
    if not recordings_dir:
        episode_dir = os.path.dirname(h5_path)
        recordings_dir = os.path.join(episode_dir, 'recordings', 'SVO')
        if not os.path.exists(recordings_dir):
            recordings_dir = os.path.join(episode_dir, 'recordings')
    
    process_episode(
        h5_path=h5_path,
        recordings_dir=recordings_dir,
        extrinsics_json_path=config.get('extrinsics_json_path', ''),
        output_dir=config.get('output_dir', './mask_output'),
        metadata_path=config.get('metadata_path'),
        max_frames=config.get('max_frames', 500),
        config=config
    )


if __name__ == '__main__':
    main()
