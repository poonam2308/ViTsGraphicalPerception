#!/usr/bin/env bash

# setup_conda.sh
set -e

ENV_NAME="vitsgp"

echo "Setting up Conda environment '$ENV_NAME' with PyTorch + CUDA..."

# 1. Check conda
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Please install Miniconda/Anaconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# 2. Make sure 'conda activate' works inside scripts
eval "$(conda shell.bash hook)"

# 3. Create environment (only if it doesn't exist yet)
if conda env list | grep -q "^\s*$ENV_NAME\s"; then
    echo "Conda environment '$ENV_NAME' already exists. Skipping creation."
else
    echo "Creating conda environment '$ENV_NAME'..."
    conda create -y -n "$ENV_NAME" python=3.12
fi

# 4. Activate env
echo "Activating conda environment '$ENV_NAME'..."
conda activate "$ENV_NAME"

# 5. Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# 6. Install PyTorch (CUDA 12.4) via pip inside the conda env
echo "Installing PyTorch with CUDA 12.4 (pip)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 7. Install other dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# 8 (optional): Default to offline W&B if no API key
if python - <<'PY'
import os, pathlib
netrc = pathlib.Path.home()/".netrc"
data = netrc.read_text() if netrc.exists() else ""
exit(0 if ("api.wandb.ai" in data or os.environ.get("WANDB_API_KEY")) else 1)
PY
then
  echo "W&B online mode available."
else
  export WANDB_MODE=offline
  echo "W&B API key not found. Setting WANDB_MODE=offline."
fi

# 9. Check CUDA availability
echo "Verifying CUDA setup..."
python - <<'EOF'
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. Check your NVIDIA drivers.")
EOF

echo "Conda setup complete."
echo "To activate this environment later, run:"
echo "  conda activate" "$ENV_NAME"
