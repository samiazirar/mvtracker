

# python create_query_points_from_masks.py --npz third_party/HOISTFormer/hoist_output/task_0045_user_0020_scene_0004_cfg_0006_human_processed_hand_tracked_hoist.npz --frames-before 3 --frames-after 1 --output third_party/HOISTFormer/hoist_output/task_0045_user_0020_scene_0004_cfg_0006_human_processed_hand_tracked_hoist_query.npz


TASK_FOLDER="task_0001_user_0015_scene_0002_cfg_0"
# TASK_FOLDER="task_0092_user_0010_scene_0004_cfg_0003_human"
DEPTH_FOLDER="/data/rh20t_api/data/low_res_data/RH20T_cfg6/$TASK_FOLDER"
# RGB_FOLDER="/data/rh20t_api/data/RH20T/RH20T_cfg6/$TASK_FOLDER" # Does not exist rn
RGB_FOLDER="/data/rh20t_api/hf_download/RH20T/RH20T_cfg6/$TASK_FOLDER"

SAMPLE_PATH_HAND_TRACKED="third_party/HOISTFormer/sam2_tracking_output/${TASK_FOLDER}_processed_hand_tracked_hoist_sam2_query.npz"

python create_query_points_from_masks.py --npz third_party/HOISTFormer/sam2_tracking_output/task_0045_user_0020_scene_0004_cfg_0006_human_processed_hand_tracked_hoist_sam2.npz --frames-before 3 --frames-after 15 --output $SAMPLE_PATH_HAND_TRACKED  --key sam2_masks --use-first-frame


python demo.py --temporal_stride 1 --spatial_downsample 1 --depth_estimator gt --depth_cache_dir ./depth_cache --rerun save --sample-path "$SAMPLE_PATH_HAND_TRACKED" --tracker spatialtrackerv2

# SAMPLE_PATH_HAND_TRACKED="third_party/HOISTFormer/hoist_output/
# python demo.py --temporal_stride 1 --spatial_downsample 1 --depth_estimator vggt_raw --depth_cache_dir ./depth_cache --rerun save --sample-path "$SAMPLE_PATH_HAND_TRACKED" --tracker spatialtrackerv2