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


# Configure git
echo "Configuring git..."
git config --global --add safe.directory /workspace
git config --global core.sshCommand 'ssh -o StrictHostKeyChecking=accept-new'
git config --global credential.helper store || true

echo "Post-create setup completed successfully!"
unset TORCH_CUDA_ARCH_LIST
#clone if does not exist
