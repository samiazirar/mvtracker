enroot import -o ~/depth_tracks_generator.sqsh docker://nvidia/cuda:12.2.2-runtime-ubuntu22.04

enroot create --name depth_tracks_generator ~/depth_tracks_generator.sqsh
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