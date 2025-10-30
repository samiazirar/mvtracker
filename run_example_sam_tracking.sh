#!/bin/bash
# QUICK START - SAM Mask Tracking
# Edit the NPZ_FILE line and run: bash run_example_sam_tracking.sh

# YOUR FILE HERE
NPZ_FILE=third_party/HOISTFormer/sam2_tracking_output/task_0045_user_0020_scene_0004_cfg_0006_human_processed_hand_tracked_hoist_sam2.npz

# Track all mask points per camera (2D) -> Lift to 3D -> Visualize in Rerun
python track_sam_masks_per_camera.py --npz "$NPZ_FILE" --mask-key sam2_masks --track-mode offline --max-points-per-mask 200 --fps 1 && \
python lift_and_visualize_tracks.py --npz "${NPZ_FILE%.npz}_tracks_per_camera.npz" --fps 10 --max-frames 50

# Done! The .rrd file is saved next to the input NPZ
echo ""
echo "Done! View with:"
echo "  rerun ${NPZ_FILE%.npz}_tracks_per_camera_tracks_3d.rrd"

