# =============================================================================
# LATEST Dockerfile - Minimal base, pulls dependencies at runtime
# Tagged as: latest
# Uses mounted workspace with host's third_party
# =============================================================================

# CUDA 12.8 devel image for sm_100 (Blackwell) support
ARG BASE_IMAGE=nvcr.io/nvidia/cuda:12.8.1-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# Ensure any CUDA extensions compile for Blackwell
ENV TORCH_CUDA_ARCH_LIST="80;86;89;90;100"

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    git curl wget ca-certificates build-essential \
    libgl1-mesa-glx libxkbcommon-x11-0 x11-apps ffmpeg \
    ninja-build cmake pkg-config openssh-client \
 && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

# Fresh pip
RUN python -m pip install --upgrade pip

# Huggingface hub optimization
RUN pip install -U "huggingface_hub[hf_xet]"

# Git/SSH config - prefer SSH for GitHub
RUN git config --system url."ssh://git@github.com/".insteadOf https://github.com/

# Pre-populate GitHub host key
RUN mkdir -p /etc/ssh && ssh-keyscan -t rsa,ecdsa,ed25519 github.com >> /etc/ssh/ssh_known_hosts 2>/dev/null || true

# Copy requirements for reference (actual install happens in post-create)
COPY requirements.txt /tmp/

# Git safe directory
RUN git config --global --add safe.directory /workspace

WORKDIR /workspace
EXPOSE 7860
CMD ["bash"]
