#!/bin/bash
# Quick and dirty SAM mask tracking pipeline - edit and run!

# EDIT THIS - your SAM2 output file
NPZ_FILE=third_party/HOISTFormer/sam2_tracking_output/task_0045_user_0020_scene_0004_cfg_0006_human_processed_hand_tracked_hoist_sam2.npz

# Step 1: Track mask points per camera
python track_sam_masks_per_camera.py \
    --npz "$NPZ_FILE" \
    --mask-key sam2_masks \
    --track-mode offline \
    --max-points-per-mask 500 \
    --fps 1

# Step 2: Lift to 3D and visualize  
python lift_and_visualize_tracks.py \
    --npz "${NPZ_FILE%.npz}_tracks_per_camera.npz" \
    --fps 10 \
    --spawn

echo "Done! Check the .rrd file"
