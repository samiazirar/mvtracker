#!/bin/bash
set -e

echo "=============================================="
echo "  MVTracker STABLE Build - Post-Create Setup"
echo "=============================================="
echo ""
echo "  Build Type: STABLE"
echo "  - All third_party dependencies are PRE-INSTALLED in Docker image"
echo "  - Located at: /workspace/third_party_stable/"
echo "  - No git cloning required"
echo "  - All Python packages pre-installed"
echo ""
echo "  To use the stable third_party:"
echo "    - Symlink: ln -sf /workspace/third_party_stable /workspace/third_party"
echo "    - Or import directly from third_party_stable"
echo ""
echo "=============================================="

# Create symlink to stable third_party if workspace third_party doesn't exist
if [ ! -d "/workspace/third_party" ] && [ ! -L "/workspace/third_party" ]; then
    echo "Creating symlink: /workspace/third_party -> /workspace/third_party_stable"
    ln -sf /workspace/third_party_stable /workspace/third_party
elif [ -L "/workspace/third_party" ]; then
    echo "Symlink /workspace/third_party already exists, pointing to: $(readlink /workspace/third_party)"
else
    echo "WARNING: /workspace/third_party exists as a directory (from mounted workspace)"
    echo "         The stable version is at /workspace/third_party_stable/"
    echo "         To use stable: rm -rf /workspace/third_party && ln -sf /workspace/third_party_stable /workspace/third_party"
fi

# Configure git
echo ""
echo "Configuring git..."
git config --global --add safe.directory /workspace
git config --global core.sshCommand 'ssh -o StrictHostKeyChecking=accept-new'
git config --global credential.helper store || true

echo ""
echo "=============================================="
echo "  STABLE setup complete!"
echo "  All dependencies are ready to use."
echo "=============================================="
