#!/usr/bin/env python3
"""
Simple script to filter SAM masks for grasped objects between finger positions.

This script:
1. Takes a list of timestamps where gripping occurs
2. Loads the hand-tracked NPZ file 
3. Identifies points between the finger tips (thumb and index)
4. Uses SAM2.1 to segment the grasped object
5. Tracks it across all frames
6. Saves a new NPZ with object masks and creates videos

Author: Generated for MVTracker project
"""

import argparse
import os
from pathlib import Path
import numpy as np
import torch
import cv2
from tqdm import tqdm
import sys
import tempfile

# Add SAM2 to path
sys.path.insert(0, str(Path(__file__).parent.parent / "third_party" / "sam2"))

from sam2.build_sam import build_sam2_video_predictor


# ============================================================================
# CONSTANTS
# ============================================================================

# MANO hand keypoint indices (21 keypoints total)
# 0: wrist
# 1-4: thumb (1=base, 4=tip)
# 5-8: index finger (5=base, 8=tip)
# 9-12: middle finger
# 13-16: ring finger  
# 17-20: pinky finger
THUMB_TIP_IDX = 4
INDEX_TIP_IDX = 8
MIDDLE_TIP_IDX = 12


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_hand_tracked_npz(npz_path):
    """
    Load the hand-tracked NPZ file.
    
    Args:
        npz_path: Path to *_hand_tracked.npz file
        
    Returns:
        Dictionary with all NPZ data
    """
    print(f"[INFO] Loading hand-tracked data from: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    
    print(f"[INFO] Data contains:")
    print(f"  - RGB images: {data['rgbs'].shape}")  # [C, T, 3, H, W]
    print(f"  - Depths: {data['depths'].shape}")     # [C, T, 1, H, W]
    print(f"  - Hand masks: {data['sam_hand_masks'].shape}")  # [C, T, H, W]
    print(f"  - Query points: {data['query_points'].shape}")  # [N, 4] where cols=[t,x,y,z]
    
    return data


def find_gripper_center_in_image(query_points, frame_idx, intr, extr, img_shape):
    """
    Find the 2D image location of the gripper center at a given frame.
    The gripper center is computed as the average of finger tip query points.
    
    Args:
        query_points: [N, 4] array where cols are [frame_idx, x, y, z] in world coords
        frame_idx: Frame index to query
        intr: Camera intrinsic matrix [3, 3]
        extr: Camera extrinsic matrix [3, 4] (world-to-camera)
        img_shape: (H, W) image dimensions
        
    Returns:
        (center_x, center_y) in image coordinates, or None if no points found
    """
    # Filter query points for this frame
    frame_mask = query_points[:, 0] == frame_idx
    frame_points = query_points[frame_mask, 1:4]  # [N, 3] world coordinates
    
    if len(frame_points) == 0:
        return None
    
    # Compute center in world coordinates (average of all finger keypoints)
    world_center = np.mean(frame_points, axis=0)  # [3,]
    
    # Transform world point to camera coordinates
    # cam = R @ world + t
    R = extr[:3, :3]
    t = extr[:3, 3]
    cam_center = R @ world_center + t  # [3,]
    
    # Check if point is in front of camera
    if cam_center[2] <= 0:
        return None
    
    # Project to image coordinates
    img_point = intr @ cam_center  # [3,]
    img_point = img_point[:2] / img_point[2]  # [2,] normalize by depth
    
    x, y = int(round(img_point[0])), int(round(img_point[1]))
    
    # Check if point is within image bounds
    H, W = img_shape
    if 0 <= x < W and 0 <= y < H:
        return (x, y)
    
    return None


def get_gripper_prompt_points(query_points, frame_idx, intr, extr, img_shape, 
                               num_points=5):
    """
    Get multiple prompt points around the gripper for SAM.
    Returns points representing the finger tips and gripper center.
    
    Args:
        query_points: [N, 4] array of query points
        frame_idx: Frame index
        intr: Camera intrinsic [3, 3]
        extr: Camera extrinsic [3, 4]
        img_shape: (H, W)
        num_points: Number of prompt points to return
        
    Returns:
        np.array of shape [M, 2] with image coordinates, or None
    """
    # Filter query points for this frame
    frame_mask = query_points[:, 0] == frame_idx
    frame_points_3d = query_points[frame_mask, 1:4]  # [N, 3] world coordinates
    
    if len(frame_points_3d) == 0:
        return None
    
    # Sample points evenly from the available finger keypoints
    if len(frame_points_3d) > num_points:
        # Sample uniformly
        indices = np.linspace(0, len(frame_points_3d) - 1, num_points, dtype=int)
        sampled_points_3d = frame_points_3d[indices]
    else:
        sampled_points_3d = frame_points_3d
    
    # Project all points to image space
    R = extr[:3, :3]
    t = extr[:3, 3]
    
    prompt_points_2d = []
    H, W = img_shape
    
    for world_pt in sampled_points_3d:
        # Transform to camera space
        cam_pt = R @ world_pt + t
        
        if cam_pt[2] <= 0:  # Behind camera
            continue
        
        # Project to image
        img_pt = intr @ cam_pt
        img_pt = img_pt[:2] / img_pt[2]
        
        x, y = int(round(img_pt[0])), int(round(img_pt[1]))
        
        if 0 <= x < W and 0 <= y < H:
            prompt_points_2d.append([x, y])
    
    if len(prompt_points_2d) == 0:
        return None
    
    return np.array(prompt_points_2d, dtype=np.float32)


def segment_object_with_sam2(video_frames, gripper_frame_idx, prompt_points, 
                              sam_predictor, score_threshold=0.5):
    """
    Use SAM2.1 video predictor to segment and track an object.
    
    Args:
        video_frames: List of [H, W, 3] RGB frames (uint8)
        gripper_frame_idx: Frame index where gripping occurs (for prompt)
        prompt_points: [N, 2] array of prompt point coordinates at gripper_frame
        sam_predictor: SAM2 video predictor model
        score_threshold: Confidence threshold for masks
        
    Returns:
        masks: List of [H, W] binary masks for each frame
        scores: List of confidence scores for each frame
    """
    print(f"[INFO] Segmenting object using SAM2.1 video predictor...")
    print(f"  - Video length: {len(video_frames)} frames")
    print(f"  - Prompt frame: {gripper_frame_idx}")
    print(f"  - Prompt points: {len(prompt_points)}")
    
    # SAM2 video predictor expects a path to frames; dump RGBs to a temp directory.
    video_segments = {}
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        for idx, frame in enumerate(video_frames):
            if frame.dtype != np.uint8:
                frame_to_save = (frame * 255).clip(0, 255).astype(np.uint8)
            else:
                frame_to_save = frame
            if frame_to_save.ndim != 3 or frame_to_save.shape[2] != 3:
                raise ValueError(
                    f"Expected RGB frame with shape [H, W, 3], got {frame_to_save.shape}"
                )
            # SAM expects JPEGs with consecutive numbering.
            frame_bgr = cv2.cvtColor(frame_to_save, cv2.COLOR_RGB2BGR)
            frame_path = tmp_dir_path / f"{idx:05d}.jpg"
            if not cv2.imwrite(str(frame_path), frame_bgr):
                raise RuntimeError(f"Failed to write temporary frame {frame_path}")

        with torch.inference_mode():
            inference_state = sam_predictor.init_state(
                video_path=str(tmp_dir_path),
            )
            
            # Add positive point prompts at the gripper frame
            # Point labels: 1 = foreground point, 0 = background point
            _, out_obj_ids, out_mask_logits = sam_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=gripper_frame_idx,
                obj_id=0,  # First object
                points=prompt_points,
                labels=np.ones(len(prompt_points), dtype=np.int32),  # All foreground
            )
            
            # Propagate masks to all frames
            for out_frame_idx, out_obj_ids, out_mask_logits in sam_predictor.propagate_in_video(
                inference_state
            ):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
    
    # Extract masks and scores for each frame
    masks = []
    scores = []
    
    for frame_idx in range(len(video_frames)):
        if frame_idx in video_segments and 0 in video_segments[frame_idx]:
            mask = video_segments[frame_idx][0][0]  # [H, W] bool
            masks.append(mask.astype(np.uint8))
            # Use mask coverage as a simple score
            score = mask.mean()
            scores.append(score)
        else:
            # No mask for this frame
            H, W = video_frames[0].shape[:2]
            masks.append(np.zeros((H, W), dtype=np.uint8))
            scores.append(0.0)
    
    print(f"[INFO] Segmentation complete. Mask coverage: {np.mean(scores):.3f}")
    
    return masks, scores


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def process_grasped_object(npz_path, grasp_timestamps, output_dir, 
                           sam_config, sam_checkpoint, device):
    """
    Main processing function.
    
    Args:
        npz_path: Path to *_hand_tracked.npz file
        grasp_timestamps: List of frame indices where grasping occurs
        output_dir: Directory to save outputs
        sam_config: SAM2 config identifier (Hydra config path)
        sam_checkpoint: SAM2 checkpoint path
        device: 'cuda' or 'cpu'
    """
    # Load data
    data = load_hand_tracked_npz(npz_path)
    
    rgbs = data["rgbs"]              # [C, T, 3, H, W]
    depths = data["depths"]          # [C, T, 1, H, W]
    intrs = data["intrs"]            # [C, T, 3, 3] or [C, 3, 3]
    extrs = data["extrs"]            # [C, T, 3, 4] or [C, 3, 4]
    query_points = data["query_points"]  # [N, 4] where cols=[t,x,y,z]
    camera_ids = data.get("camera_ids", None)
    
    C, T, _, H, W = rgbs.shape
    print(f"\n[INFO] Processing {C} cameras, {T} frames, resolution {H}x{W}")
    print(f"[INFO] Grasp timestamps: {grasp_timestamps}")
    
    # Choose a reference grasp frame (use the middle one if multiple)
    grasp_frame_idx = grasp_timestamps[len(grasp_timestamps) // 2]
    print(f"[INFO] Using grasp frame {grasp_frame_idx} as reference")
    
    # Load SAM2 video predictor
    print(f"[INFO] Loading SAM2.1 video predictor from: {sam_checkpoint}")
    sam_predictor = build_sam2_video_predictor(sam_config, sam_checkpoint, device=device)
    
    # Storage for object masks (one per camera)
    object_masks_all_cams = np.zeros((C, T, H, W), dtype=np.uint8)
    object_scores_all_cams = np.zeros((C, T, H, W), dtype=np.float32)
    
    # Storage for videos
    video_buffers = {}
    
    # Process each camera
    for cam_idx in tqdm(range(C), desc="Processing cameras"):
        cam_id = str(camera_ids[cam_idx]) if camera_ids is not None else f"cam_{cam_idx:03d}"
        print(f"\n[INFO] Processing camera {cam_id}...")
        
        # Get intrinsics/extrinsics for this camera
        intr = intrs[cam_idx]
        extr = extrs[cam_idx]
        
        # Handle static vs time-varying intrinsics/extrinsics
        if intr.ndim == 2:  # Static [3, 3]
            intr_at_grasp = intr
        else:  # Time-varying [T, 3, 3]
            intr_at_grasp = intr[grasp_frame_idx]
        
        if extr.ndim == 2:  # Static [3, 4]
            extr_at_grasp = extr
        else:  # Time-varying [T, 3, 4]
            extr_at_grasp = extr[grasp_frame_idx]
        
        # Get prompt points for SAM at the grasp frame
        prompt_points = get_gripper_prompt_points(
            query_points, 
            grasp_frame_idx, 
            intr_at_grasp, 
            extr_at_grasp, 
            img_shape=(H, W),
            num_points=5
        )
        
        if prompt_points is None:
            print(f"[WARNING] No valid prompt points for camera {cam_id}, skipping")
            continue
        
        print(f"[INFO] Found {len(prompt_points)} prompt points at grasp frame")
        
        # Prepare video frames (convert from [T, 3, H, W] to list of [H, W, 3] RGB)
        video_frames = []
        for t in range(T):
            frame = np.moveaxis(rgbs[cam_idx, t], 0, 2)  # [3,H,W] -> [H,W,3]
            # Ensure uint8
            if frame.dtype != np.uint8:
                frame = (frame * 255).clip(0, 255).astype(np.uint8)
            video_frames.append(frame)
        
        # Segment object with SAM2
        object_masks, object_scores = segment_object_with_sam2(
            video_frames,
            grasp_frame_idx,
            prompt_points,
            sam_predictor,
            score_threshold=0.5
        )
        
        # Store masks
        for t in range(T):
            object_masks_all_cams[cam_idx, t] = object_masks[t]
            object_scores_all_cams[cam_idx, t] = object_masks[t].astype(np.float32)
        
        # Create overlay video
        overlay_frames = []
        for t in range(T):
            frame = video_frames[t].copy()
            mask = object_masks[t].astype(bool)
            
            if mask.any():
                # Overlay mask in cyan
                overlay = frame.astype(np.float32)
                overlay_color = np.array([0, 255, 255], dtype=np.float32)  # Cyan in RGB
                alpha = 0.5
                overlay[mask] = overlay[mask] * (1.0 - alpha) + overlay_color * alpha
                frame = overlay.clip(0, 255).astype(np.uint8)
            
            # Draw prompt points on grasp frame
            if t == grasp_frame_idx:
                for pt in prompt_points:
                    x, y = int(pt[0]), int(pt[1])
                    cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)  # Red dots
                    cv2.circle(frame, (x, y), 7, (255, 255, 255), 2)  # White outline
            
            overlay_frames.append(frame)
        
        video_buffers[cam_id] = overlay_frames
    
    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save NPZ with object masks
    payload = dict(data)
    payload["sam_object_masks"] = object_masks_all_cams
    payload["sam_object_scores"] = object_scores_all_cams
    payload["grasp_timestamps"] = np.array(grasp_timestamps)
    
    out_npz_path = output_dir / f"{Path(npz_path).stem}_with_object_masks.npz"
    np.savez_compressed(out_npz_path, **payload)
    print(f"\n[INFO] Saved NPZ with object masks: {out_npz_path}")
    
    # Save videos
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 12.0
    
    for cam_id, frames in video_buffers.items():
        if not frames:
            continue
        
        height, width = frames[0].shape[:2]
        video_path = output_dir / f"{Path(npz_path).stem}_cam{cam_id}_object_tracked.mp4"
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        
        for frame in frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
        
        writer.release()
        print(f"[INFO] Saved video: {video_path}")
    
    print("\n[INFO] Processing complete!")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Filter SAM masks for grasped objects between finger positions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python filter_grasped_object_with_sam.py \\
    --npz data/human_high_res_filtered/task_0045_user_0020_scene_0004_cfg_0006_human_processed_hand_tracked.npz \\
    --grasp-timestamps 10 15 20 \\
    --output-dir data/human_high_res_filtered/object_tracking \\
    --device cuda
        """
    )
    
    parser.add_argument(
        "--npz",
        required=True,
        help="Path to *_hand_tracked.npz file"
    )
    parser.add_argument(
        "--grasp-timestamps",
        nargs="+",
        type=int,
        required=True,
        help="Frame indices where grasping occurs (e.g., 10 15 20)"
    )
    parser.add_argument(
        "--output-dir",
        default="object_tracking_output",
        help="Directory to save outputs (default: object_tracking_output)"
    )
    parser.add_argument(
        "--sam-config",
        default="third_party/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
        help="SAM2 config name or YAML file under third_party/sam2/sam2"
    )
    parser.add_argument(
        "--sam-checkpoint",
        default="third_party/sam2/checkpoints/sam2.1_hiera_large.pt",
        help="SAM2 checkpoint path"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda or cpu)"
    )
    
    args = parser.parse_args()
    
    # Convert paths to absolute
    npz_path = Path(args.npz).resolve()
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    
    # Make SAM paths relative to workspace root if they're relative
    workspace_root = Path(__file__).parent.parent
    
    sam_module_root = workspace_root / "third_party" / "sam2" / "sam2"

    sam_config_arg = Path(args.sam_config)
    if sam_config_arg.suffix.lower() in {".yaml", ".yml"}:
        if not sam_config_arg.is_absolute():
            sam_config_path = (workspace_root / sam_config_arg).resolve()
        else:
            sam_config_path = sam_config_arg.resolve()

        if not sam_config_path.exists():
            raise FileNotFoundError(f"SAM config not found: {sam_config_path}")

        try:
            sam_config_rel = sam_config_path.relative_to(sam_module_root)
        except ValueError:
            raise ValueError(
                f"SAM config must live inside {sam_module_root}, got {sam_config_path}"
            )

        sam_config = str(sam_config_rel).replace(os.sep, "/")
    else:
        sam_config = args.sam_config
    
    sam_checkpoint = Path(args.sam_checkpoint)
    if not sam_checkpoint.is_absolute():
        sam_checkpoint = workspace_root / sam_checkpoint
    
    if not sam_checkpoint.exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {sam_checkpoint}")
    
    process_grasped_object(
        npz_path=npz_path,
        grasp_timestamps=args.grasp_timestamps,
        output_dir=args.output_dir,
        sam_config=sam_config,
        sam_checkpoint=str(sam_checkpoint),
        device=args.device
    )


if __name__ == "__main__":
    main()
