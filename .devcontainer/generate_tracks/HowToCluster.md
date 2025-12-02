<!-- nvidia/cuda:12.2.2-runtime-ubuntu22.04 -->


enroot import -o ~/depth_tracks_generator.sqsh docker://stereolabs/zed:5.0-gl-devel-cuda11.8-ubuntu22.04

export NVIDIA_DRIVER_CAPABILITIES=all
enroot create --force --name depth_tracks_generator ~/depth_tracks_generator.sqsh
enroot start -r -w depth_tracks_generator

# Inside the container, run:
apt update
apt install -y \
  build-essential cmake git wget pkg-config \
  libboost-all-dev \
  libopencv-dev \
    libeigen3-dev \

apt install git git-lfs -y
git lfs install

# clone the repository
git clone https://github.com/samiazirar/mvtracker.git
cd mvtracker
bash .devcontainer/generate_tracks/post-create.sh


enroot export --output ~/depth_tracks_generator_final.sqsh depth_tracks_generator



finally run with


enroot create --force --name depth_tracks_generator_final ~/depth_tracks_generator_final.sqsh
enroot start -r -w depth_tracks_generator_final






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



## Generate Training data on enroot
THIS DOES NOT WORK RN

### 1\. The Commands (Run these every time)

**Step A: Workstation (Cedar)**
Forward your local SSH port to the login node.

```bash
ssh -R 10022:localhost:22 azirar@lmgpu-login
```

**Step B: HPC Node (Compute Node)**
Bring the port to the node, mount the folder, and start the container.

```bash
# 1. Bring port 10022 from login node to here (background)
ssh -f -N -L 10022:localhost:10022 azirar@lmgpu-login

# 2. Create host mount point & mount your workstation folder
mkdir -p /tmp/my_local_data
sshfs -p 10022 azirar@localhost:/home/nfs/datasets/internal/Droid_tracks_depth_processed /tmp/my_local_data -o reconnect

# 3. Start Enroot (bind mount /tmp/my_local_data to /mnt/nfs inside container)
enroot start -r -w -m /tmp/my_local_data:/mnt/nfs 
```

-----

### 2\. The Script Change (One-time edit)

**File:** `conversions/droid/training_data/run_pipeline_cluster_debug.sh`

**Change Line 33:**

```bash
# OLD:
PERMANENT_STORAGE_DIR="./droid_processed"

# NEW:
PERMANENT_STORAGE_DIR="/mnt/nfs/droid_processed"
```

\*\*

-----

### 3\. The Documentation (Add to README)

**Add this section to `HowToCluster.md`:**

### Data Saving Workflow (Reverse Tunnel)

To save generated training data directly to local NFS storage (bypassing cluster quota):

1.  **On Workstation:** Run `ssh -R 10022:localhost:22 lmgpu-login` to expose local disk.
2.  **On Node:** Run `ssh -f -N -L 10022:localhost:10022 lmgpu-login` to fetch the port.
3.  **Mount:** `sshfs -p 10022 azirar@localhost:/local/data /tmp/my_local_data`.
4.  **Container:** Start with `-m /tmp/my_local_data:/mnt/nfs`.
5.  **Script:** Ensure `PERMANENT_STORAGE_DIR` points to `/mnt/nfs`.