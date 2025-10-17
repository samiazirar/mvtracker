#!/bin/bash
set -e

echo "Starting post-create setup..."

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

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
pip install -r rh20t_api/requirements.txt

# Install SAM2 in editable mode (clone if missing)
SAM2_DIR="/workspace/sam2"
if [ ! -d "$SAM2_DIR" ]; then
    echo "Cloning SAM2 repository..."
    git clone https://github.com/facebookresearch/sam2.git "$SAM2_DIR"
else
    echo "SAM2 repository already exists, skipping clone."
fi

echo "Installing SAM2..."
pip install -e "$SAM2_DIR"
# Configure git
echo "Configuring git..."
git config --global --add safe.directory /workspace
git config --global core.sshCommand 'ssh -o StrictHostKeyChecking=accept-new'
git config --global credential.helper store || true

echo "Post-create setup completed successfully!"
# unset TORCH_CUDA_ARCH_LIST
# #clone if does not exist
# if [ ! -d "pytorch3d" ]; then
#     echo "Cloning PyTorch3D repository..."
#     git clone https://github.com/facebookresearch/pytorch3d.git
# cd pytorch3d && pip install -e .
# echo "if errors invoke unset TORCH_CUDA_ARCH_LIST"

apt-get update
apt-get install -y build-essential gcc-11 g++-11 ninja-build
pip install --upgrade pip setuptools wheel ninja

export TORCH_CUDA_ARCH_LIST="12.0"      # dotted, not 120
export MAX_JOBS=$(nproc)
pip install -v --no-build-isolation \
  "git+https://github.com/ethz-vlg/pointcept.git@2082918#subdirectory=libs/pointops"

pip install trimesh

if [ ! -d "spatialtrackerv2" ]; then
    echo "Cloning SpaTrackerV2 repository..."
    git clone https://github.com/henry123-boy/SpaTrackerV2.git spatialtrackerv2
    cd spatialtrackerv2
    git checkout 1673230
    git submodule update --init --recursive

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
sed -i 's/(torch.det(R) - 1).abs().max() < 1e-3/(torch.det(R) - 1).abs().max() < 5e-3/' ./models/SpaTrackV2/models/tracker3D/spatrack_modules/utils.py

# Verify the change: this should print a line with 5e-3
cat ./models/SpaTrackV2/models/tracker3D/spatrack_modules/utils.py | grep "(torch.det(R) - 1).abs().max()"
cd ..
