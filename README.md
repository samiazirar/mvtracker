## Quick start (Docker)
- Launch a CUDA + ZED-ready container:  
  `docker run --gpus all --rm -it -v $PWD:/workspace -v /abs/path/to/droid_data:/data/droid stereolabs/zed:4.1-gl-devel-cuda12.1-ubuntu22.04 bash`
- Inside the container: `cd /workspace && pip install -r requirements.txt`
- If you use a different base image, make sure the ZED SDK and CUDA are available for `pyzed`.

## Run on the DROID dataset
- Point the config at your data by editing `conversions/droid/config.yaml` (`h5_path`, `extrinsics_json_path`, `recordings_dir`, optional `metadata_path`). Defaults assume the dataset is mounted at `/data/droid`.
- Generate the fused point cloud: `python conversions/droid/generate_pointcloud_from_droid_refactored.py`
- (Optional) Run the version with object masks: `python conversions/droid/generate_pointcloud_from_droid_with_object_tracking_refactored.py`
- Outputs are written as `.rrd` files under `point_clouds/`; view them with `rerun`.



### Debug 

- Nvidia driver suddenly not detected:
  Failed to initialize NVML: Driver/library version mismatch
  NVML library version: 570.195

  - This can be if the driver is updated on the host while the container is running. Restart the host machine without stopping the container.



### Test full

root@d3400090bd88:/workspace# python conversions/droid/training_data/render_tracks_from_mp4.py     --episode_id "AUTOLab+84bd5053+2023-08-18-12h-01m-10s"     --metadata_repo "sazirarrwth99/trajectories_droid"     --gcs_bucket "gs://gresearch/robotics/droid_raw/1.0.1"