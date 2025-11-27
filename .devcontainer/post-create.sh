#!/bin/bash
set -e

echo "Starting post-create setup..."

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip
echo "Upgrading setuptools, wheel, ninja..."
pip install --upgrade setuptools wheel ninja

# Install PyTorch with CUDA 12.8 support
echo "Installing PyTorch with CUDA 12.8..."
pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.7.1 \
    torchvision==0.22.* \
    torchaudio==2.7.*

# Install project requirements (allow failures)
echo "Installing project requirements..."
pip install -r requirements.txt || true

# Set CUDA architecture list for building extensions
# echo "Setting CUDA architecture list..."
# export TORCH_CUDA_ARCH_LIST='80;86;89;90;100'

# # Uninstall existing CUDA packages to rebuild with correct arch
# echo "Cleaning existing CUDA packages..."
# pip uninstall -y flash-attn xformers pointops || true

# # Reinstall flash-attn with correct CUDA architectures
# echo "Installing flash-attn..."
# pip install -v --no-build-isolation --no-cache-dir --force-reinstall flash-attn

# # Reinstall pointops with correct CUDA architectures
# echo "Installing pointops..."
# pip install -v --no-build-isolation --no-cache-dir --force-reinstall \
#     'git+https://github.com/ethz-vlg/pointcept.git@2082918#subdirectory=libs/pointops'
echo "Update safetensors"
pip install --upgrade safetensors
#clone if not exist
# pip install -r rh20t_api/requirements.txt

# Install SAM2 in editable mode (clone if missing)
SAM2_DIR="/workspace/third_party/sam2"

if [ ! -d "$SAM2_DIR" ]; then
    mkdir -p /workspace/third_party
    echo "Cloning SAM2 repository..."
    git clone https://github.com/facebookresearch/sam2.git "$SAM2_DIR"
    cd checkpoints && \
    ./download_ckpts.sh && \
    cd ..
else
    echo "SAM2 repository already exists, skipping clone."
fi


FLOWACTION_DIR="/workspace/third_party/3DFlowAction"

if [ ! -d "$FLOWACTION_DIR" ]; then
    mkdir -p /workspace/third_party
    echo "Cloning 3D-FlowAction repository..."
    git clone https://github.com/Hoyyyaard/3DFlowAction.git "$FLOWACTION_DIR"
else
    echo "3D-FlowAction repository already exists, skipping clone."
fi


echo "Installing SAM2..."
pip install -e "$SAM2_DIR"

SAM3_DIR="/workspace/third_party/sam3"
if [ -d "$SAM3_DIR" ]; then
    echo "Installing SAM3..."
    pip install -e "$SAM3_DIR"
else
    echo "SAM3 directory not found at $SAM3_DIR, skipping install."
fi

# Configure git
echo "Configuring git..."
git config --global --add safe.directory /workspace
git config --global core.sshCommand 'ssh -o StrictHostKeyChecking=accept-new'
git config --global credential.helper store || true

echo "Post-create setup completed successfully!"
unset TORCH_CUDA_ARCH_LIST
#clone if does not exist

if [ ! -d "/workspace/third_party/pytorch3d" ]; then
    echo "Cloning PyTorch3D repository..."
    cd /workspace/third_party
    git clone https://github.com/facebookresearch/pytorch3d.git
fi
# PyTorch3D's setup.py imports torch, so disable build isolation so torch from the main env is available
cd /workspace/third_party/pytorch3d
pip install --no-build-isolation -e .
cd /workspace/
# echo "if errors invoke unset TORCH_CUDA_ARCH_LIST"

apt-get update
apt-get install -y build-essential gcc-11 g++-11 ninja-build

export TORCH_CUDA_ARCH_LIST="12.0"      # dotted, not 120
export MAX_JOBS=$(nproc)
pip install -v --no-build-isolation \
  "git+https://github.com/ethz-vlg/pointcept.git@2082918#subdirectory=libs/pointops"

pip install trimesh

if [ ! -d "/workspace/third_party/spatialtrackerv2" ]; then
    cd /workspace/third_party
    echo "Cloning SpaTrackerV2 repository..."
    git clone https://github.com/henry123-boy/SpaTrackerV2.git spatialtrackerv2
    cd spatialtrackerv2
    git checkout 1673230
    git submodule update --init --recursive
    cd ..
    #rname the depth_edge with depth_map_edge, led to issues 
    # sed -i -E 's/(^|[^_a-zA-Z0-9])depth_edge([^_a-zA-Z0-9]|$)/\1depth_map_edge\2/g' ./models/SpaTrackV2/models/SpaTrack.py
fi
pip install pycolmap==3.11.1
# pip install git+https://github.com/EasternJournalist/utils3d.git#egg=utils3d
#then no depth_edge chaaneg needed? re
pip install git+https://github.com/EasternJournalist/utils3d.git@d790d33#egg=utils3d
pip install pyceres==2.4
pip install jaxtyping
pip install decord
# Update the threshold for weighted_procrustes_torch from 1e-3 to 5e-3
cd /workspace/third_party/spatialtrackerv2
sed -i 's/(torch.det(R) - 1).abs().max() < 1e-3/(torch.det(R) - 1).abs().max() < 5e-3/' ./models/SpaTrackV2/models/tracker3D/spatrack_modules/utils.py

# Verify the change: this should print a line with 5e-3
cat ./models/SpaTrackV2/models/tracker3D/spatrack_modules/utils.py | grep "(torch.det(R) - 1).abs().max()"
cd /workspace/

echo "Installing HaMeR Hand tracking..."
HAMER_DIR="/workspace/third_party/hamer"
if [ ! -d "$HAMER_DIR" ]; then
    mkdir -p /workspace/third_party
    echo "Cloning HaMER repository..."
    cd /workspace/third_party
    git clone --recursive https://github.com/geopavlakos/hamer.git
    cd $HAMER_DIR
    bash fetch_demo_data.sh
    python3.10 -m venv .hamer
    cd ../..
else
    echo "HaMER repository already exists, skipping clone."
fi

# pip install --upgrade --no-cache-dir --force-reinstall xtcocotools
#pip install --upgrade --no-cache-dir --force-reinstall scikit-image

cd $HAMER_DIR
#create a virtual environment and install hamer if not already done
if [ ! -d ".hamer" ]; then
    echo "Creating HaMeR virtual environment..."
    python -m venv .hamer
else
    echo "HaMeR virtual environment already exists, skipping creation."
fi
source .hamer/bin/activate
unset TORCH_CUDA_ARCH_LIST
export TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"
echo "Setting up HaMeR environment..."
pip install torch torchvision torchaudio
pip install --no-build-isolation -e .
pip install --no-build-isolation -e .[all]
pip install 'detectron2@git+https://github.com/facebookresearch/detectron2'
pip install -v -e third-party/ViTPose

#here stuff neded to run w/o a display
echo "Installing headless OpenGL (OSMesa + Xvfb)..."
apt-get update
apt-get install -y --no-install-recommends \
    libosmesa6 libosmesa6-dev mesa-common-dev libgl-dev \
    libegl1 libgles2 mesa-utils mesa-utils-bin \
    xvfb xserver-common libunwind8 libfontenc1 libxfont2 xauth x11-xkb-utils
rm -rf /var/lib/apt/lists/*
export PYOPENGL_PLATFORM=osmesa

# pip install mmpose mmengine mmdet
# pip install "mmcv>=2.0.0rc4,<=3.0.0"
echo "installing Sam2 inside HaMeR..."
pip install -e ../sam2 
#change --config-settings editable_mode=compat if annoys just skipping if already installed
echo "deactivating HaMeR virtual environment..."
deactivate
cd ../..

echo "Finished with HaMeR setup."



echo "Download HOIST Former and install..."
HOIST_DIR="/workspace/third_party/HOISTFormer"
if [ ! -d "$HOIST_DIR" ]; then
    mkdir -p /workspace/third_party
    echo "Cloning HOISTFormer repository..."
    git clone https://github.com/samiazirar/hoist_former "$HOIST_DIR"
fi 




echo "Setting up HOISTFormer environment..."

cd $HOIST_DIR
# ===== System deps (Ubuntu/Debian) =====
apt-get update
apt-get install -y --no-install-recommends \
  build-essential cmake ninja-build git python3-venv python3-dev \
  libgl1 libglib2.0-0

pip install scipy opencv-python tqdm pyzed

#TODO: Check if we should keep it lul
echo "Patching CUDA extension for PyTorch 2.7 compatibility..."
# Fix deprecated .type() calls in CUDA extension
CUDA_FILE="Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu"
if [ -f "$CUDA_FILE" ]; then
    sed -i 's/AT_DISPATCH_FLOATING_TYPES(value\.type()/AT_DISPATCH_FLOATING_TYPES(value.scalar_type()/g' "$CUDA_FILE"
    echo "Patched $CUDA_FILE"
fi

echo "Running Installation script for HOISTFormer..."
# ===== Create & activate venv (use `source` to activate) =====
bash install_make_env.sh
echo "HOISTFormer setup completed."

#remove this later on 
echo "Running human example script..."

INSTALL_ZED_SDK=${INSTALL_ZED_SDK:-1}
ZED_INSTALLER="ZED_SDK_Ubuntu22_cu124_v5.1.0_linux_x86_64.zstd.run"
ZED_INSTALLER_URL_DEFAULT="https://download.stereolabs.com/zedsdk/5.1/cu12/ubuntu22?_gl=1*m21mnx*_gcl_au*MTAxNjgyOTA5MS4xNzYzMDUxNzg5"
ZED_INSTALLER_URL="${ZED_INSTALLER_URL:-$ZED_INSTALLER_URL_DEFAULT}"
if [ "$INSTALL_ZED_SDK" = "1" ] && [ ! -d "/usr/local/zed" ]; then
    echo "Installing ZED SDK v5.1.0 (CUDA 12.4)..."
    apt-get update
    apt-get install -y curl file zstd wget
    pushd /tmp >/dev/null
    if [ -n "${ZED_INSTALLER_SOURCE:-}" ] && [ -f "${ZED_INSTALLER_SOURCE}" ]; then
        echo "Copying ZED SDK installer from ${ZED_INSTALLER_SOURCE}..."
        cp "${ZED_INSTALLER_SOURCE}" "${ZED_INSTALLER}"
    elif [ ! -f "${ZED_INSTALLER}" ]; then
        echo "Downloading ZED SDK installer via wget..."
        wget -O "${ZED_INSTALLER}" "${ZED_INSTALLER_URL}"
    fi
    if ! file "${ZED_INSTALLER}" | grep -qi "shell script"; then
        echo "Cached ZED installer invalid, re-downloading via wget..."
        rm -f "${ZED_INSTALLER}"
        wget -O "${ZED_INSTALLER}" "${ZED_INSTALLER_URL}"
    fi
    if ! file "${ZED_INSTALLER}" | grep -qi "shell script"; then
        cat <<'EOF' >&2
ZED SDK installer download failed.
The ZED 5.x downloads now require an authenticated URL from stereolabs.com.
Please download ZED_SDK_Ubuntu22_cu124_v5.1.0_linux_x86_64.zstd.run manually,
place it in /tmp (or set ZED_INSTALLER_URL to a valid signed link), and rerun the post-create step.
EOF
        echo "Skipping ZED SDK installation for now."
        popd >/dev/null
    else
        chmod +x "${ZED_INSTALLER}"
        # Run the installer in silent mode and skip the AI module download (it is fetched lazily).
        ./"${ZED_INSTALLER}" -- silent skip_ai_module
        rm -f "${ZED_INSTALLER}"
        popd >/dev/null
    fi
else
    if [ "$INSTALL_ZED_SDK" != "1" ]; then
        echo "INSTALL_ZED_SDK=0, skipping ZED SDK installation."
    else
        echo "ZED SDK already installed, skipping download."
    fi
fi

# Ensure NVIDIA video codec runtime libraries (libnvcuvid, libnvidia-encode) are present
NVIDIA_VIDEO_LIB_VERSION=${NVIDIA_VIDEO_LIB_VERSION:-570}
NVIDIA_VIDEO_LIB_DIR="/usr/local/nvidia-video-libs/${NVIDIA_VIDEO_LIB_VERSION}"
if command -v nvidia-smi >/dev/null 2>&1; then
    if ldconfig -p | grep -q libnvcuvid.so; then
        echo "NVIDIA video codec libs already available."
    else
        echo "Fetching NVIDIA video codec libs (version ${NVIDIA_VIDEO_LIB_VERSION})..."
        apt-get update
        TMPDIR=$(mktemp -d /tmp/nvidia-codec.XXXXXX)
        pushd "$TMPDIR" >/dev/null
        DOWNLOAD_OK=1
        for pkg in "libnvidia-decode-${NVIDIA_VIDEO_LIB_VERSION}" "libnvidia-encode-${NVIDIA_VIDEO_LIB_VERSION}"; do
            if ! apt-get download "$pkg"; then
                echo "Failed to download ${pkg}. Set NVIDIA_VIDEO_LIB_VERSION to a valid driver version and retry."
                DOWNLOAD_OK=0
                break
            fi
        done
        if [ "$DOWNLOAD_OK" = "1" ]; then
            mkdir -p extracted
            for deb in ./*.deb; do
                dpkg-deb -x "$deb" extracted
            done
            mkdir -p "$NVIDIA_VIDEO_LIB_DIR"
            cp -an extracted/usr/lib/x86_64-linux-gnu/libnvcuvid.so* "$NVIDIA_VIDEO_LIB_DIR" || true
            cp -an extracted/usr/lib/x86_64-linux-gnu/libnvidia-encode.so* "$NVIDIA_VIDEO_LIB_DIR" || true
            echo "$NVIDIA_VIDEO_LIB_DIR" > /etc/ld.so.conf.d/zed-nvidia-codec.conf
            ldconfig
            echo "Installed NVIDIA video codec libs into ${NVIDIA_VIDEO_LIB_DIR}."
        else
            echo "Skipping manual NVIDIA codec install due to download failure."
        fi
        popd >/dev/null
        rm -rf "$TMPDIR"
    fi
else
    echo "nvidia-smi not detected; skipping NVIDIA video codec libs."
fi

pip install cupy-cuda12x

# Install Grounding DINO
echo "Installing Grounding DINO..."
unset TORCH_CUDA_ARCH_LIST
unset CUDA_HOME

GROUNDINGDINO_DIR="/workspace/third_party/groundingdino-cu128"
if [ ! -d "$GROUNDINGDINO_DIR" ]; then
    mkdir -p /workspace/third_party
    echo "Cloning Grounding DINO repository..."
    git clone https://github.com/ghostcipher1/groundingdino-cu128.git "$GROUNDINGDINO_DIR"
    
    cd "$GROUNDINGDINO_DIR"
    
    # Fix setup.py paths for proper compilation
    echo "Patching setup.py for build..."
    sed -i 's|sources = \[os.path.join(extensions_dir, s) for s in sources\]|sources = [os.path.relpath(s, this_dir) for s in sources]|g' setup.py
    sed -i 's|include_dirs = \[extensions_dir\]|include_dirs = [os.path.relpath(extensions_dir, start=os.path.dirname(__file__))]|g' setup.py
    
    # Install Grounding DINO
    pip install -e . --no-build-isolation
    
    cd /workspace
else
    echo "Grounding DINO already exists, skipping clone."
fi

echo "Grounding DINO installation completed."

pip install yourdfpy --no-build-isolation

echo "Almost ready just installing some stuff for convinience"
apt install -y rsync

if [ ! -d "/workspace/third_party/robotiq_arg85_description" ]; then
    echo "loading the model for the gripper..."
    cd /workspace/third_party
    git clone https://github.com/a-price/robotiq_arg85_description.git 
fi

if [ ! -d "/workspace/third_party/CtRNet-X" ]; then
    echo "Cloning CtRNet-X repository..."
    cd /workspace/third_party
    git clone https://github.com/darthandvader/CtRNet-X.git
fi

if [ ! -d "/workspace/third_party/robosuite" ]; then
    echo "Cloning robosuite repository..."
    cd /workspace/third_party
    git clone https://github.com/ARISE-Initiative/robosuite.git
fi

if [ ! -d "/workspace/third_party/droid_policy_learning" ]; then
    echo "Cloning droid_policy_learning repository..."
    cd /workspace/third_party
    git clone https://github.com/droid-dataset/droid_policy_learning.git
fi

echo "installing ffmpeg and libx264-dev for video encoding"

apt update
apt install ffmpeg libx264-dev -y
echo "Setup complete!"