#!/bin/bash

apt-get update && apt-get install -y libavcodec-dev libavdevice-dev libavformat-dev libavfilter-dev \
    libswscale-dev libswresample-dev libavutil-dev
pip install av

git clone --recursive https://github.com/ethz-vlg/duster.git /workspace/duster

pip install -r /workspace/duster/requirements.txt  # or follow their instructions

cd /workspace/duster/
export PYTHONPATH=/workspace/duster:$PYTHONPATH
