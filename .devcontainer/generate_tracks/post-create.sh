#!/bin/bash
set -euo pipefail

echo "[post-create] Minimal setup for DROID training data generation"

# Persist useful CUDA env (A100 = sm80)
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
export NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-compute,utility,video}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-80}

persist_env() {
  local key="$1" val="$2"
  if ! grep -q "^${key}=" /etc/environment 2>/dev/null; then
    echo "${key}=${val}" | tee -a /etc/environment >/dev/null
  fi
}

persist_env CUDA_HOME "$CUDA_HOME"
persist_env NVIDIA_VISIBLE_DEVICES "$NVIDIA_VISIBLE_DEVICES"
persist_env NVIDIA_DRIVER_CAPABILITIES "$NVIDIA_DRIVER_CAPABILITIES"
persist_env TORCH_CUDA_ARCH_LIST "$TORCH_CUDA_ARCH_LIST"

echo "[post-create] Installing system packages..."
apt-get update
apt-get install -y --no-install-recommends \
  python3 python3-pip python3-venv python3-dev \
  git curl wget ca-certificates build-essential pkg-config openssh-client \
  ffmpeg \
  libgl1 libglu1-mesa libglib2.0-0 libusb-1.0-0 \
  libxext6 libxrender1 libsm6 libx11-6 \
  libturbojpeg0 udev \
  libnvidia-encode-535 libnvidia-decode-535 \
  file zstd unzip \
  && rm -rf /var/lib/apt/lists/*

ln -sf /usr/bin/python3 /usr/bin/python

# Base Python tooling
python -m pip install --upgrade pip
pip install --upgrade setuptools wheel

echo "[post-create] Installing gsutil and GCS clients for episode downloads"
pip install --no-cache-dir gsutil google-cloud-storage gcsfs

echo "[post-create] Installing Python deps for track generation + depth extraction"
pip install --no-cache-dir \
  numpy scipy h5py PyYAML opencv-python-headless \
  pytransform3d trimesh rerun-sdk tqdm

# Optional ZED SDK install (needed for extract_rgb_depth.py)
INSTALL_ZED_SDK="${INSTALL_ZED_SDK:-1}"
ZED_INSTALLER_URL="${ZED_INSTALLER_URL:-https://download.stereolabs.com/zedsdk/5.1/cu12/ubuntu22}"
ZED_INSTALLER="${ZED_INSTALLER:-/tmp/ZED_SDK.run}"

if [ "$INSTALL_ZED_SDK" = "1" ]; then
  if [ ! -d "/usr/local/zed" ]; then
    echo "[post-create] Installing ZED SDK (set INSTALL_ZED_SDK=0 to skip)..."
    mkdir -p /etc/udev/rules.d || true
    wget -O "$ZED_INSTALLER" "$ZED_INSTALLER_URL"
    chmod +x "$ZED_INSTALLER"
    set +e
    "$ZED_INSTALLER" -- silent skip_ai_module
    INSTALL_EXIT=$?
    set -e
    if [ $INSTALL_EXIT -ne 0 ]; then
      echo "[WARN] ZED installer exited with code $INSTALL_EXIT"
    fi
    rm -f "$ZED_INSTALLER"
    rm -rf /var/lib/apt/lists/*
  else
    echo "[post-create] ZED SDK already present, skipping installer."
  fi

  if python - <<'PY' >/dev/null 2>&1; then
import pyzed.sl as sl
print(sl.__version__)
PY
    echo "[post-create] pyzed already available."
  else
    echo "[post-create] Installing pyzed wheel from ZED SDK..."
    PYZED_WHEEL=$(find /usr/local/zed -name "pyzed-*.whl" | head -n 1 || true)
    if [ -n "$PYZED_WHEEL" ]; then
      pip install --no-cache-dir "$PYZED_WHEEL"
    else
      echo "[WARN] Could not locate pyzed wheel under /usr/local/zed."
    fi
  fi
else
  echo "[post-create] INSTALL_ZED_SDK=0, skipping ZED install."
fi

# echo "[post-create] Configuring git defaults..."
# git config --global --add safe.directory /workspace
# git config --system url."ssh://git@github.com/".insteadOf https://github.com/
# mkdir -p /etc/ssh && ssh-keyscan -t rsa,ecdsa,ed25519 github.com >> /etc/ssh/ssh_known_hosts 2>/dev/null || true
# git config --global core.sshCommand 'ssh -o StrictHostKeyChecking=accept-new' || true
# git config --global credential.helper store || true

echo "[post-create] Fixing libturbojpeg link for ZED..."
if [ -f "/usr/lib/x86_64-linux-gnu/libturbojpeg.so.0" ]; then
    ln -sf /usr/lib/x86_64-linux-gnu/libturbojpeg.so.0 /usr/lib/libturbojpeg.so.0
fi

echo "[post-create] Done."