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




## How to use eroot

# how_to_use_enroot
how to use the enroot of the Lamarr gpu cluster

### Dev container
#### Mount the NFS in a dev Container
  "mounts": [
    "source=/home/nfs/datasets/internal,target=/data,type=bind,consistency=cached,readonly=false"
  ],



### Enroot 

#### Run a container
- create a docker and push to gitlab
- use script in howto with name as argument
- all data in data/enroot/..
- exec into with listing the container using the **--fancy** argument then enroot exec <PID> bash

### copy files
- lmgpu to node:
  - (from node)
  - scp azirar@lmgpu-login.informatik.uni-bonn.de:/home/azirar/how_to_use/run_container/create_enroot.sh  

- node to lmgpu:
  -  (from node) 
  - scp  -r /workspace/output azirar@lmgpu-login.informatik.uni-bonn.de:/home/azirar/assets/outputs


- cedar to lmgpu
  - (from cedar)
  - scp ./assets/own_blocks.png azirar@lmgpu-login.informatik.uni-bonn.de:/home/azirar/
 
- lmgpu to cedar
  - (from cedar)
  - scp -r  azirar@lmgpu-login.informatik.uni-bonn.de:/home/azirar/assets/outputs/* ./assets/outputs/


### multiple "sessions"
tmux
if not then 
ctrz+z then with 'bg' to keep in background. Check with 'jobs'. 


### Forward a port
- node to lmgpu
  - (from node)
  - ssh -N -f -R 8888:localhost:8188 lmgpu-login
- lmgpu to cedar
  - (from cedar)
  -  ssh -L 8888:localhost:8888 azirar@lmgpu
-  cedar to mac
  - (from macbook)
  -  ssh -L 8888:localhost:8888 azirar@cedar

  -  

### Shorten a Video 
ffmpeg -i video.mp4 -t 2 -c copy video_01.mp4 -> first 2 seconds

### Video processing
- with vllm
  - check the folder openai_video_analysis
  - *important* you have to ad  --max-num-batched-tokens [NUM larger than max_model_len] 

