# Use CUDA 12.1 devel image with Ubuntu 22.04
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install basic utilities and Python 3.10
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    git curl wget build-essential ca-certificates \
    libgl1-mesa-glx libxkbcommon-x11-0 x11-apps ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Make "python" command refer to Python 3
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Upgrade pip and install PyTorch with CUDA 12.1
RUN python -m pip install --upgrade pip && \
    pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# Copy requirement files
COPY requirements.txt requirements.full.txt /tmp/

# Install full repository requirements
# RUN pip install -r /tmp/requirements.full.txt && rm /tmp/requirements.txt /tmp/requirements.full.txt
 
# # Speeds up attention
# RUN pip install --upgrade --no-build-isolation flash-attn==2.5.8 
# # Speeds up kNN
# RUN pip install "git+https://github.com/ethz-vlg/pointcept.git@2082918#subdirectory=libs/pointops"  
WORKDIR /workspace

EXPOSE 7860

CMD ["bash"]
