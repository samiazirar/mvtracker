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
  cmake git-lfs \
  libboost-all-dev libopencv-dev libeigen3-dev \
  ffmpeg \
  libgl1 libglu1-mesa libglib2.0-0 libusb-1.0-0 \
  libxext6 libxrender1 libsm6 libx11-6 \
  libturbojpeg libturbojpeg0-dev udev \
  file zstd unzip \
  && rm -rf /var/lib/apt/lists/*

# Initialize Git LFS
git lfs install

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

echo "[post-create] Ensuring libturbojpeg compatibility..."
# Force link creation if the dev package didn't do it perfectly for pyzed
if [ -f "/usr/lib/x86_64-linux-gnu/libturbojpeg.so.0" ]; then
    ln -sf /usr/lib/x86_64-linux-gnu/libturbojpeg.so.0 /usr/lib/libturbojpeg.so.0
fi
ldconfig


# [Install libcuda for enroot containers]
# --- RUN INSIDE CONTAINER (root@...) ---
set -e

echo "=== 1. Setup & Detection ==="
# Remove any broken/32-bit links from previous attempts
rm -f /usr/lib/x86_64-linux-gnu/libnvidia-encode.so*
rm -f /usr/lib/x86_64-linux-gnu/libnvcuvid.so*

# Detect driver version from host kernel
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
echo "   -> Detected Driver: $DRIVER_VERSION"

echo "=== 2. Downloading NVIDIA Installer ==="
# Download the exact driver match (contains the missing files)
wget -q "https://us.download.nvidia.com/tesla/${DRIVER_VERSION}/NVIDIA-Linux-x86_64-${DRIVER_VERSION}.run" -O nvidia_driver.run
chmod +x nvidia_driver.run

echo "=== 3. Extracting Files ==="
./nvidia_driver.run --extract-only --target extract_dir > /dev/null

echo "=== 4. Installing 64-BIT Libraries ==="

# Function to find and install ONLY the 64-bit version
install_64bit_lib() {
    local lib_name="$1"
    local search_name="${lib_name}.${DRIVER_VERSION}"
    local found=0
    
    # Find all files with this name
    local candidates=$(find extract_dir -name "$search_name")
    
    for f in $candidates; do
        # Check if file is 64-bit (x86-64) using 'file' command
        if file -L "$f" | grep -q "x86-64"; then
            echo "   -> Installing 64-bit: $lib_name"
            cp "$f" /usr/lib/x86_64-linux-gnu/
            
            # Create necessary symlinks
            ln -sf "/usr/lib/x86_64-linux-gnu/$search_name" "/usr/lib/x86_64-linux-gnu/${lib_name}.1"
            ln -sf "/usr/lib/x86_64-linux-gnu/$search_name" "/usr/lib/x86_64-linux-gnu/${lib_name}"
            found=1
            break
        fi
    done
    
    if [ "$found" -eq 0 ]; then
        echo "   [ERROR] Could not find 64-bit version of $lib_name"
        exit 1
    fi
}

# Install both missing Video Codec libraries
install_64bit_lib "libnvidia-encode.so"
install_64bit_lib "libnvcuvid.so"

echo "=== 5. Finalizing ==="
ldconfig
rm -rf nvidia_driver.run extract_dir

echo "=== VERIFICATION ==="
python3 -c "import pyzed.sl; print('SUCCESS: ZED SDK Imported!')"


echo "[Download] Robotiq Gripper if missing..."

if [ ! -d "/data/third_party/robotiq_arg85_description" ]; then
    echo "loading the model for the gripper..."
    cd /data/
    git clone https://github.com/a-price/robotiq_arg85_description.git 
fi

apt update && apt install -y rsync

echo "[post-create] Done."