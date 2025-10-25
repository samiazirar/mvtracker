#!/bin/bash
set -e




echo "Trying depth completed"
DIR="/workspace/data/human_high_res_filtered_mg_depth_completion/task_0045_user_0020_scene_0004_cfg_0006_human_mg_depth_completion"

python demo.py --temporal_stride 1 --spatial_downsample 1 --depth_estimator gt --depth_cache_dir ./depth_cache --rerun save --sample-path "$DIR/task_0045_user_0020_scene_0004_cfg_0006_human_mg_completed_camera_id_037522061512.npz" --tracker cotracker3_offline --random_query_points --rrd "$DIR/task_0045_user_0020_scene_0004_cfg_0006_human_mg_completed_camera_id_037522061512.rrd"

python demo.py --temporal_stride 1 --spatial_downsample 1 --depth_estimator gt --depth_cache_dir ./depth_cache --rerun save --sample-path "$DIR/task_0045_user_0020_scene_0004_cfg_0006_human_mg_completed_camera_id_104122060811.npz" --tracker cotracker3_offline --random_query_points --rrd "$DIR/task_0045_user_0020_scene_0004_cfg_0006_human_mg_completed_camera_id_104122060811.rrd"

python demo.py --temporal_stride 1 --spatial_downsample 1 --depth_estimator gt --depth_cache_dir ./depth_cache --rerun save --sample-path "$DIR/task_0045_user_0020_scene_0004_cfg_0006_human_mg_completed_camera_id_104122061018.npz" --tracker cotracker3_offline --random_query_points --rrd "$DIR/task_0045_user_0020_scene_0004_cfg_0006_human_mg_completed_camera_id_104122061018.rrd"

python demo.py --temporal_stride 1 --spatial_downsample 1 --depth_estimator gt --depth_cache_dir ./depth_cache --rerun save --sample-path "$DIR/task_0045_user_0020_scene_0004_cfg_0006_human_mg_completed_camera_id_104122061330.npz" --tracker cotracker3_offline --random_query_points --rrd "$DIR/task_0045_user_0020_scene_0004_cfg_0006_human_mg_completed_camera_id_104122061330.rrd"

python demo.py --temporal_stride 1 --spatial_downsample 1 --depth_estimator gt --depth_cache_dir ./depth_cache --rerun save --sample-path "$DIR/task_0045_user_0020_scene_0004_cfg_0006_human_mg_completed_camera_id_104122061602.npz" --tracker cotracker3_offline --random_query_points --rrd "$DIR/task_0045_user_0020_scene_0004_cfg_0006_human_mg_completed_camera_id_104122061602.rrd"

python demo.py --temporal_stride 1 --spatial_downsample 1 --depth_estimator gt --depth_cache_dir ./depth_cache --rerun save --sample-path "$DIR/task_0045_user_0020_scene_0004_cfg_0006_human_mg_completed_camera_id_104122063633.npz" --tracker cotracker3_offline --random_query_points --rrd "$DIR/task_0045_user_0020_scene_0004_cfg_0006_human_mg_completed_camera_id_104122063633.rrd"

python demo.py --temporal_stride 1 --spatial_downsample 1 --depth_estimator gt --depth_cache_dir ./depth_cache --rerun save --sample-path "$DIR/task_0045_user_0020_scene_0004_cfg_0006_human_mg_completed_camera_id_104122064161.npz" --tracker cotracker3_offline --random_query_points --rrd "$DIR/task_0045_user_0020_scene_0004_cfg_0006_human_mg_completed_camera_id_104122064161.rrd"

python demo.py --temporal_stride 1 --spatial_downsample 1 --depth_estimator gt --depth_cache_dir ./depth_cache --rerun save --sample-path "$DIR/task_0045_user_0020_scene_0004_cfg_0006_human_mg_completed_camera_id_f0271510.npz" --tracker cotracker3_offline --random_query_points --rrd "$DIR/task_0045_user_0020_scene_0004_cfg_0006_human_mg_completed_camera_id_f0271510.rrd"

cp -r $DIR/* /data/rh20t_api/test_data_generated/
