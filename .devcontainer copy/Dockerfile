# CUDA >=12.4 is required to build for sm_100 (Blackwell)
# FROM nvidia/cuda:12.6.2-devel-ubuntu22.04
# ↑ OLD base image kept for reference (commented)

# NEW: move to CUDA 12.8 devel image to match cu128 wheels/toolkit cleanly
# FROM nvidia/cuda:12.8.0-devel-ubuntu22.04
# only duing outage

# Prefer NGC (nvcr.io) during Docker Hub outages; override via build arg if needed
ARG BASE_IMAGE=nvcr.io/nvidia/cuda:12.8.1-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# Ensure any CUDA extensions compile for Blackwell
# ENV TORCH_CUDA_ARCH_LIST=100           # OLD (too narrow) — kept commented
# NEW: safe superset (A100/Ada/H100/Blackwell)
ENV TORCH_CUDA_ARCH_LIST="80;86;89;90;100"

# Make python point to python3
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    git curl wget ca-certificates build-essential \
    libgl1-mesa-glx libxkbcommon-x11-0 x11-apps ffmpeg \
    ninja-build cmake pkg-config openssh-client \
 && rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Fresh pip
RUN python -m pip install --upgrade pip

# === PyTorch with cu124/cu126 lines (OLD) kept but commented ===
# # === PyTorch with cu124 (works on driver 570.172 with your Blackwell) ===
# # Pin to latest 2.5.x line
# RUN install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# ^^^ OLD: this was wrong ('install' not 'pip install') and mismatched. Kept commented.

# Helpful perf + your earlier warning
RUN pip install -U "huggingface_hub[hf_xet]"

# Optional: common GPU libs — build from source to pick up sm_100
# (Skip if you don't use them; uncomment if you do.)
# RUN pip install --no-cache-dir --no-build-isolation --no-binary=:all: triton
# RUN pip install --no-cache-dir --no-build-isolation --no-binary=:all: flash-attn
# RUN pip install --no-cache-dir --no-build-isolation --no-binary=:all: xformers
# Prefer SSH for GitHub remotes to leverage mounted host SSH keys
RUN git config --system url."ssh://git@github.com/".insteadOf https://github.com/

# Pre-populate GitHub host key to avoid prompts in fresh containers
RUN mkdir -p /etc/ssh && ssh-keyscan -t rsa,ecdsa,ed25519 github.com >> /etc/ssh/ssh_known_hosts 2>/dev/null || true


# Copy requirements (if you have them) and install
COPY requirements.txt requirements.full.txt /tmp/
# RUN pip install -r /tmp/requirements.full.txt && rm /tmp/requirements.txt /tmp/requirements.full.txt

# --- Git/SSH quality-of-life (kept minimal; relies on agent forwarding or mounted ~/.ssh) ---
# (No secrets written here; we just ensure Git won't complain about ownership)
RUN git config --global --add safe.directory /workspace

WORKDIR /workspace
EXPOSE 7860
CMD ["bash"]