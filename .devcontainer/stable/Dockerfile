# =============================================================================
# STABLE Dockerfile - Everything pre-installed, no git pulls needed at runtime
# Tagged as: YYYY-MM-stable
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
    gcc-11 g++-11 \
    libosmesa6 libosmesa6-dev mesa-common-dev libgl-dev \
    libegl1 libgles2 mesa-utils mesa-utils-bin \
    xvfb xserver-common libunwind8 libfontenc1 libxfont2 xauth x11-xkb-utils \
    libgl1 libglib2.0-0 \
    rsync file zstd libx264-dev \
 && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

# Fresh pip
RUN python -m pip install --upgrade pip setuptools wheel ninja

# =============================================================================
# PyTorch with CUDA 12.8 support
# =============================================================================
RUN pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.7.1 \
    torchvision==0.22.* \
    torchaudio==2.7.*

# Huggingface hub optimization
RUN pip install -U "huggingface_hub[hf_xet]"

# Git/SSH config
RUN git config --system url."ssh://git@github.com/".insteadOf https://github.com/
RUN mkdir -p /etc/ssh && ssh-keyscan -t rsa,ecdsa,ed25519 github.com >> /etc/ssh/ssh_known_hosts 2>/dev/null || true

WORKDIR /workspace

# =============================================================================
# Copy requirements and install Python packages
# =============================================================================
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt || true && rm /tmp/requirements.txt

RUN pip install --upgrade safetensors trimesh pycolmap==3.11.1 pyceres==2.4 jaxtyping decord cupy-cuda12x yourdfpy scipy opencv-python tqdm pyzed

# Install pointops
RUN pip install -v --no-build-isolation \
    "git+https://github.com/ethz-vlg/pointcept.git@2082918#subdirectory=libs/pointops"

# Install utils3d
RUN pip install git+https://github.com/EasternJournalist/utils3d.git@d790d33#egg=utils3d

# =============================================================================
# Clone and install all third_party repositories (BAKED IN)
# =============================================================================
RUN mkdir -p /workspace/third_party_stable

# --- SAM2 ---
RUN git clone https://github.com/facebookresearch/sam2.git /workspace/third_party_stable/sam2 \
    && pip install -e /workspace/third_party_stable/sam2

# --- 3DFlowAction ---
RUN git clone https://github.com/Hoyyyaard/3DFlowAction.git /workspace/third_party_stable/3DFlowAction

# --- PyTorch3D ---
RUN git clone https://github.com/facebookresearch/pytorch3d.git /workspace/third_party_stable/pytorch3d \
    && cd /workspace/third_party_stable/pytorch3d \
    && pip install --no-build-isolation -e .

# --- SpaTrackerV2 ---
RUN git clone https://github.com/henry123-boy/SpaTrackerV2.git /workspace/third_party_stable/spatialtrackerv2 \
    && cd /workspace/third_party_stable/spatialtrackerv2 \
    && git checkout 1673230 \
    && git submodule update --init --recursive \
    && sed -i 's/(torch.det(R) - 1).abs().max() < 1e-3/(torch.det(R) - 1).abs().max() < 5e-3/' ./models/SpaTrackV2/models/tracker3D/spatrack_modules/utils.py

# --- HaMeR (separate venv) ---
RUN git clone --recursive https://github.com/geopavlakos/hamer.git /workspace/third_party_stable/hamer \
    && cd /workspace/third_party_stable/hamer \
    && bash fetch_demo_data.sh || true

RUN cd /workspace/third_party_stable/hamer \
    && python -m venv .hamer \
    && . .hamer/bin/activate \
    && pip install torch torchvision torchaudio \
    && pip install --no-build-isolation -e . \
    && pip install --no-build-isolation -e .[all] || true \
    && pip install 'detectron2@git+https://github.com/facebookresearch/detectron2' || true \
    && pip install -v -e third-party/ViTPose || true \
    && pip install -e /workspace/third_party_stable/sam2 \
    && deactivate

# --- HOISTFormer ---
RUN git clone https://github.com/samiazirar/hoist_former /workspace/third_party_stable/HOISTFormer

# Patch HOISTFormer CUDA extension
RUN cd /workspace/third_party_stable/HOISTFormer \
    && CUDA_FILE="Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu" \
    && if [ -f "$CUDA_FILE" ]; then \
         sed -i 's/AT_DISPATCH_FLOATING_TYPES(value\.type()/AT_DISPATCH_FLOATING_TYPES(value.scalar_type()/g' "$CUDA_FILE"; \
       fi \
    && bash install_make_env.sh || true

# --- Grounding DINO ---
RUN git clone https://github.com/ghostcipher1/groundingdino-cu128.git /workspace/third_party_stable/groundingdino-cu128 \
    && cd /workspace/third_party_stable/groundingdino-cu128 \
    && sed -i 's|sources = \[os.path.join(extensions_dir, s) for s in sources\]|sources = [os.path.relpath(s, this_dir) for s in sources]|g' setup.py \
    && sed -i 's|include_dirs = \[extensions_dir\]|include_dirs = [os.path.relpath(extensions_dir, start=os.path.dirname(__file__))]|g' setup.py \
    && pip install -e . --no-build-isolation

# --- Other repos (no special install) ---
RUN git clone https://github.com/a-price/robotiq_arg85_description.git /workspace/third_party_stable/robotiq_arg85_description
RUN git clone https://github.com/darthandvader/CtRNet-X.git /workspace/third_party_stable/CtRNet-X
RUN git clone https://github.com/ARISE-Initiative/robosuite.git /workspace/third_party_stable/robosuite
RUN git clone https://github.com/droid-dataset/droid_policy_learning.git /workspace/third_party_stable/droid_policy_learning

# =============================================================================
# Copy project source code into the image
# =============================================================================
COPY mvtracker/ /workspace/mvtracker/
COPY configs/ /workspace/configs/
COPY conversions/ /workspace/conversions/
COPY explainers/ /workspace/explainers/
COPY post_process/ /workspace/post_process/
COPY preprocess_data/ /workspace/preprocess_data/
COPY *.py /workspace/
COPY *.sh /workspace/
COPY *.md /workspace/
COPY requirements.txt /workspace/

# Git safe directory
RUN git config --global --add safe.directory /workspace

WORKDIR /workspace
EXPOSE 7860
CMD ["bash"]
