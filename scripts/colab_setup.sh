#!/bin/bash
# ============================================================================
# CCFinetune - Google Colab Setup Script
# ============================================================================
# Usage: Run this script in a Colab cell with:
#   !curl -sSL https://raw.githubusercontent.com/YOUR_USERNAME/CCFinetune/main/scripts/colab_setup.sh | bash
#
# Or copy-paste into a Colab cell:
#   !bash /content/CCFinetune/scripts/colab_setup.sh
# ============================================================================

set -e  # Exit on error

echo "=============================================="
echo "CCFinetune - Colab Setup"
echo "=============================================="

# ----------------------------------------------------------------------------
# 1. Check GPU
# ----------------------------------------------------------------------------
echo ""
echo "[1/6] Checking GPU..."

if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Please enable GPU runtime:"
    echo "       Runtime > Change runtime type > Hardware accelerator > GPU"
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "GPU check passed!"

# ----------------------------------------------------------------------------
# 2. Install Unsloth (Colab optimized)
# Reference: https://unsloth.ai/docs/get-started/install-and-update/google-colab
# ----------------------------------------------------------------------------
echo ""
echo "[2/6] Installing Unsloth (Colab optimized)..."

pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl triton -q
pip install --no-deps cut_cross_entropy unsloth_zoo -q
pip install sentencepiece protobuf datasets huggingface_hub hf_transfer -q
pip install unsloth -q

echo "Unsloth installed!"

# ----------------------------------------------------------------------------
# 3. Install uv package manager
# ----------------------------------------------------------------------------
echo ""
echo "[3/6] Installing uv package manager..."

curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH
export PATH="$HOME/.local/bin:$PATH"

uv --version
echo "uv installed!"

# ----------------------------------------------------------------------------
# 4. Clone/Update repository
# ----------------------------------------------------------------------------
echo ""
echo "[4/6] Setting up repository..."


# ----------------------------------------------------------------------------
# 5. Install project dependencies
# ----------------------------------------------------------------------------
echo ""
echo "[5/6] Installing project dependencies with uv..."

# Sync dependencies (skip unsloth since we installed it separately)
uv sync --no-install-project

echo "Dependencies installed!"

# ----------------------------------------------------------------------------
# 6. Setup environment
# ----------------------------------------------------------------------------
echo ""
echo "[6/6] Setting up environment..."

# Create outputs directory
mkdir -p outputs

# Enable faster HuggingFace downloads
export HF_HUB_ENABLE_HF_TRANSFER=1


echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "To activate the environment in new cells, run:"
echo "  source /content/activate_ccfinetune.sh"
echo ""
echo "Project directory: $PROJECT_DIR"
echo "=============================================="
