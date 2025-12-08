## Quick start (Docker)
- Launch a CUDA + ZED-ready container:  
  `docker run --gpus all --rm -it -v $PWD:/workspace -v /abs/path/to/droid_data:/data/droid stereolabs/zed:4.1-gl-devel-cuda12.1-ubuntu22.04 bash`
- Inside the container: `cd /workspace && pip install -r requirements.txt`
- If you use a different base image, make sure the ZED SDK and CUDA are available for `pyzed`.

## Run on the DROID dataset

### Training Data Processing Pipelines

Two pipelines available for processing DROID episodes for training:

**1. Metadata-Only Pipeline (CPU-only, no GPU needed)**
```bash
cd conversions/droid/training_data
./run_pipeline_cluster_huggingface_metadata_only_no_depth.sh 100  # Process 100 episodes
```
- Extracts intrinsics only (no depth maps)
- Generates 3D + 2D tracks with normalized flow
- Fast processing (16+ CPU workers)
- Output: tracks.npz, extrinsics.npz, quality.json, intrinsics.json
- Uploads to: `sazirarrwth99/droid_metadata_only`

**2. Full Pipeline with Compressed Depth (GPU required)**
```bash
cd conversions/droid/training_data
./run_pipeline_cluster_huggingface_compressed_lossy.sh 100  # Process 100 episodes
```
- Extracts depth maps as FFV1 lossless video
- Generates 3D + 2D tracks with normalized flow
- Multi-GPU parallel processing
- Output: depth.mkv + tracks.npz + extrinsics.npz + quality.json + intrinsics.json
- Uploads to: `sazirarrwth99/lossy_comr_traject` AND `sazirarrwth99/droid_metadata_only`

See `conversions/droid/training_data/PIPELINE_SUMMARY.md` for detailed documentation.

### Point Cloud Visualization

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