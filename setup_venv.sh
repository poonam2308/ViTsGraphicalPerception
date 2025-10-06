#!/usr/bin/env bash

# Exit on first error
set -e

echo "Setting up Python virtual environment with PyTorch + CUDA..."

# 1. Check Python
if ! command -v python3 &> /dev/null; then
    echo "Python3 not found. Please install Python 3.8+ first."
    exit 1
fi

# 2. Create venv
echo "Creating virtual environment 'venv'..."
python3 -m venv venv

# 3. Activate venv
echo "Activating virtual environment..."
source venv/bin/activate

# 4. Upgrade pip & tools
echo "Upgrading pip..."
pip install --upgrade pip

# 5. Install PyTorch (CUDA 12.1)
echo "Installing PyTorch with CUDA 12.1..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# another: CUDA 11.8
#pip install torch==2.3.0+cu118 torchvision --extra-index-url https://download.pytorch.org/whl/cu118

# 6. Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt


# 7. Check CUDA availability
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

echo "Setup complete."
echo "To activate your virtual environment later, run:"
echo "  source venv/bin/activate"



