<!-- nvidia/cuda:12.2.2-runtime-ubuntu22.04 -->


enroot import -o /data/cosmos_predict2.sqsh docker://registry.gitlab.uni-bonn.de:5050/rpl/public_registry/cosmos-predict2:latest

export NVIDIA_DRIVER_CAPABILITIES=all
enroot create --force --name cosmos_inference /data/cosmos_predict2.sqsh
enroot start -r -w cosmos_inference

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


enroot export --output ~/cosmos_inference_final.sqsh cosmos_inference



finally run with


enroot create --force --name cosmos_inference_final ~/cosmos_inference_final.sqsh
enroot start -r -w cosmos_inference_final