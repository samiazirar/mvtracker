ARG BASE_IMAGE=nvcr.io/nvidia/cuda:12.2.2-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    TORCH_CUDA_ARCH_LIST="80"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    git curl wget ca-certificates build-essential pkg-config openssh-client \
    ffmpeg \
    libgl1 libglu1-mesa libglib2.0-0 libusb-1.0-0 \
    libxext6 libxrender1 libsm6 libx11-6 \
    file zstd unzip \
 && rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/bin/python3 /usr/bin/python

RUN python -m pip install --upgrade pip setuptools wheel

RUN git config --system url."ssh://git@github.com/".insteadOf https://github.com/

RUN mkdir -p /etc/ssh && ssh-keyscan -t rsa,ecdsa,ed25519 github.com >> /etc/ssh/ssh_known_hosts 2>/dev/null || true

RUN git config --global --add safe.directory /workspace

WORKDIR /workspace
CMD ["bash"]
