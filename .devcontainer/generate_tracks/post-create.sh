#!/bin/bash
set -euo pipefail

echo "[post-create] Minimal setup for DROID training data generation"

# Base Python tooling
python -m pip install --upgrade pip
pip install --upgrade setuptools wheel

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
    apt-get update
    apt-get install -y --no-install-recommends curl wget file zstd libusb-1.0-0
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
  then
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

echo "[post-create] Configuring git defaults..."
git config --global --add safe.directory /workspace
git config --global core.sshCommand 'ssh -o StrictHostKeyChecking=accept-new' || true
git config --global credential.helper store || true

echo "[post-create] Done."
