#!/bin/bash

apt-get update && apt-get install -y libavcodec-dev libavdevice-dev libavformat-dev libavfilter-dev \
    libswscale-dev libswresample-dev libavutil-dev
pip install av

git clone --recursive https://github.com/ethz-vlg/duster.git /workspace/duster

pip install -r /workspace/duster/requirements.txt  # or follow their instructions

cd /workspace/duster/
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -P checkpoints
md5sum checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth

cd /workspace/

export PYTHONPATH=/workspace/duster:$PYTHONPATH
