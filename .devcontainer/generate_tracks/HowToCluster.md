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