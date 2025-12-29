#!/usr/bin/env python3
"""
Demo script: Generate sample mask video with synthetic robot motion.

This script creates a demo video showing:
- Gripper mask (green) with articulated fingers
- Robot arm mask (blue) with all 7 links
- Combined overlay on a synthetic background

Usage:
    python demo_mask_rendering.py --output_dir ./demo_output
    python demo_mask_rendering.py --gripper_only  # Only render gripper
    python demo_mask_rendering.py --robot_only    # Only render robot arm
"""

import argparse
import numpy as np
import cv2
import os
import sys
from typing import Tuple

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mask_utils import (
    GripperMaskRenderer,
    RobotArmMaskRenderer,
    CombinedRobotMaskRenderer,
    pose6_to_T,
    panda_forward_kinematics,
)


def create_checkerboard_background(width: int, height: int, square_size: int = 40) -> np.ndarray:
    """Create a checkerboard pattern for background."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    for y in range(0, height, square_size):
        for x in range(0, width, square_size):
            if ((x // square_size) + (y // square_size)) % 2 == 0:
                img[y:y+square_size, x:x+square_size] = [60, 60, 60]
            else:
                img[y:y+square_size, x:x+square_size] = [40, 40, 40]
    
    return img


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float = 0.5
) -> np.ndarray:
    """Overlay a binary mask on an image."""
    result = image.copy()
    mask_bool = mask > 0
    
    if not np.any(mask_bool):
        return result
    
    # Colored overlay
    overlay = np.zeros_like(image)
    overlay[mask_bool] = color
    
    # Blend
    result[mask_bool] = cv2.addWeighted(
        image[mask_bool].astype(np.float32), 1 - alpha,
        overlay[mask_bool].astype(np.float32), alpha,
        0
    ).astype(np.uint8)
    
    # Draw contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, color, 2)
    
    return result


def generate_camera_params(width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate camera intrinsics and extrinsics for a reasonable view."""
    # Intrinsics (typical camera)
    fx = width * 1.2
    fy = fx
    cx = width / 2
    cy = height / 2
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Camera looking at origin from front-right-above
    # Position camera at (1.2, 0.8, 0.8) looking at (0.3, 0, 0.3)
    cam_pos = np.array([1.2, 0.8, 0.8])
    target = np.array([0.3, 0, 0.3])
    up = np.array([0, 0, 1])
    
    # Build rotation matrix (camera Z points at target)
    z_axis = target - cam_pos
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    x_axis = np.cross(z_axis, up)
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    # World to camera rotation
    R = np.stack([x_axis, y_axis, z_axis], axis=0)
    
    # World to camera transform
    T_cam_world = np.eye(4)
    T_cam_world[:3, :3] = R
    T_cam_world[:3, 3] = -R @ cam_pos
    
    return K, T_cam_world


def generate_demo_trajectory(num_frames: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a demo trajectory for the robot.
    
    Returns:
        joint_angles: (num_frames, 7) array
        gripper_pos: (num_frames,) array
    """
    t = np.linspace(0, 2 * np.pi, num_frames)
    
    # Base joint angles (neutral pose)
    base_joints = np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.8, 0.7])
    
    # Add sinusoidal motion to some joints
    joint_angles = np.zeros((num_frames, 7))
    
    for i in range(num_frames):
        joints = base_joints.copy()
        joints[0] += 0.3 * np.sin(t[i])           # Base rotation
        joints[1] += 0.2 * np.sin(t[i] * 0.5)     # Shoulder
        joints[3] += 0.3 * np.sin(t[i] * 0.7)     # Elbow
        joints[5] += 0.2 * np.sin(t[i] * 1.2)     # Wrist pitch
        joints[6] += 0.3 * np.sin(t[i] * 0.8)     # Wrist roll
        joint_angles[i] = joints
    
    # Gripper: open -> close -> open
    gripper_pos = 0.5 + 0.5 * np.sin(t * 2)
    gripper_pos = np.clip(gripper_pos, 0, 1)
    
    return joint_angles, gripper_pos


def main():
    parser = argparse.ArgumentParser(description="Demo mask rendering with synthetic data")
    parser.add_argument('--output_dir', type=str, default='./demo_output',
                        help='Output directory')
    parser.add_argument('--width', type=int, default=640, help='Video width')
    parser.add_argument('--height', type=int, default=480, help='Video height')
    parser.add_argument('--num_frames', type=int, default=120, help='Number of frames')
    parser.add_argument('--fps', type=float, default=30.0, help='FPS')
    parser.add_argument('--gripper_only', action='store_true', help='Only render gripper')
    parser.add_argument('--robot_only', action='store_true', help='Only render robot arm')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    width = args.width
    height = args.height
    num_frames = args.num_frames
    
    print("=" * 60)
    print("DROID Mask Rendering Demo")
    print("=" * 60)
    print(f"Output: {args.output_dir}")
    print(f"Resolution: {width}x{height}")
    print(f"Frames: {num_frames}")
    print()
    
    # Initialize renderers
    print("[1/4] Loading meshes...")
    
    render_gripper = not args.robot_only
    render_robot = not args.gripper_only
    
    gripper_renderer = None
    arm_renderer = None
    
    if render_gripper:
        try:
            gripper_renderer = GripperMaskRenderer()
            print("  Gripper meshes: OK")
        except Exception as e:
            print(f"  [WARN] Gripper meshes failed: {e}")
            render_gripper = False
    
    if render_robot:
        try:
            arm_renderer = RobotArmMaskRenderer()
            print("  Robot arm meshes: OK")
        except Exception as e:
            print(f"  [WARN] Robot arm meshes failed: {e}")
            render_robot = False
    
    if not render_gripper and not render_robot:
        print("[ERROR] No renderers available!")
        return
    
    # Generate camera parameters
    print("\n[2/4] Setting up camera...")
    K, T_cam_world = generate_camera_params(width, height)
    print(f"  Intrinsics: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")
    
    # Generate trajectory
    print("\n[3/4] Generating trajectory...")
    joint_angles, gripper_pos = generate_demo_trajectory(num_frames)
    print(f"  Joint range: [{joint_angles.min():.2f}, {joint_angles.max():.2f}] rad")
    print(f"  Gripper range: [{gripper_pos.min():.2f}, {gripper_pos.max():.2f}]")
    
    # Initialize video writers
    print("\n[4/4] Rendering frames...")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = args.fps
    
    writers = {}
    
    if render_gripper:
        writers['gripper'] = cv2.VideoWriter(
            os.path.join(args.output_dir, 'demo_gripper_mask.mp4'),
            fourcc, fps, (width, height)
        )
    
    if render_robot:
        writers['robot'] = cv2.VideoWriter(
            os.path.join(args.output_dir, 'demo_robot_mask.mp4'),
            fourcc, fps, (width, height)
        )
    
    if render_gripper and render_robot:
        writers['combined'] = cv2.VideoWriter(
            os.path.join(args.output_dir, 'demo_combined_mask.mp4'),
            fourcc, fps, (width, height)
        )
    
    # Render frames
    for frame_idx in range(num_frames):
        # Create background
        bg = create_checkerboard_background(width, height)
        
        # Get robot state
        joints = joint_angles[frame_idx]
        grip = gripper_pos[frame_idx]
        
        # Compute forward kinematics
        link_transforms = panda_forward_kinematics(joints)
        T_world_ee = link_transforms.get("hand", np.eye(4))
        
        gripper_mask = None
        robot_mask = None
        
        # Render gripper mask
        if render_gripper and gripper_renderer:
            gripper_mask = gripper_renderer.render_mask(
                T_world_ee, grip, K, T_cam_world, width, height
            )
            
            overlay = overlay_mask(bg, gripper_mask, (0, 255, 0), 0.5)
            
            # Add text
            cv2.putText(overlay, f"Gripper: {grip:.2f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(overlay, f"Frame: {frame_idx+1}/{num_frames}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            writers['gripper'].write(overlay)
        
        # Render robot arm mask
        if render_robot and arm_renderer:
            robot_mask = arm_renderer.render_mask(
                joints, K, T_cam_world, width, height,
                exclude_hand=True
            )
            
            overlay = overlay_mask(bg, robot_mask, (255, 100, 0), 0.5)
            
            # Add text
            cv2.putText(overlay, f"Joint 1: {joints[0]:.2f} rad", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(overlay, f"Frame: {frame_idx+1}/{num_frames}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            writers['robot'].write(overlay)
        
        # Combined
        if 'combined' in writers and gripper_mask is not None and robot_mask is not None:
            overlay = bg.copy()
            overlay = overlay_mask(overlay, robot_mask, (255, 100, 0), 0.5)
            overlay = overlay_mask(overlay, gripper_mask, (0, 255, 0), 0.5)
            
            cv2.putText(overlay, f"Gripper + Robot Arm", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(overlay, f"Frame: {frame_idx+1}/{num_frames}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            writers['combined'].write(overlay)
        
        if (frame_idx + 1) % 30 == 0:
            print(f"  Frame {frame_idx + 1}/{num_frames}")
    
    # Cleanup
    for name, writer in writers.items():
        writer.release()
        print(f"  Saved: demo_{name}_mask.mp4")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print(f"Videos saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
