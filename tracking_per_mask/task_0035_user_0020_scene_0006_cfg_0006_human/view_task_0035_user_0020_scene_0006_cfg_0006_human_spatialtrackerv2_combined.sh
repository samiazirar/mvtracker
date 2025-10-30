#!/usr/bin/env bash
# Auto-generated script to view multiple tracking results together
# Created: rerun_combine_utils.py

echo "Launching Rerun viewer with 2 recordings..."
echo ""

# View all RRD files together
rerun \
  tracking_per_mask/task_0035_user_0020_scene_0006_cfg_0006_human/task_0035_user_0020_scene_0006_cfg_0006_human_spatialtrackerv2_instance_0.rrd \
  tracking_per_mask/task_0035_user_0020_scene_0006_cfg_0006_human/task_0035_user_0020_scene_0006_cfg_0006_human_spatialtrackerv2_instance_1.rrd

echo ""
echo "Viewer closed."
