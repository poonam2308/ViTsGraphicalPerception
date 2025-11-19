#!/usr/bin/env bash
set -euo pipefail

### CONFIG ###########################################################
REPO_URL="https://github.com/poonam2308/ViTsGraphicalPerception.git"
REPO_DIR="$HOME/ViTsGraphicalPerception"
CONDA_DIR="$HOME/miniconda3"
ENV_NAME="vitsgp"
#####################################################################

echo "=== Step 0: Install basic system packages (git, wget, etc.) ==="
if command -v apt >/dev/null 2>&1; then
    sudo apt update
    sudo apt install -y wget git
else
    echo "No apt found. Please install git & wget manually and re-run."
fi

echo
echo "=== Step 1: Ensure Conda (Miniconda) is installed ==="
if ! command -v conda >/dev/null 2>&1; then
    echo "Conda not found. Installing Miniconda into $CONDA_DIR ..."
    mkdir -p /tmp/miniconda-install
    cd /tmp/miniconda-install

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh

    bash miniconda.sh -b -p "$CONDA_DIR"

    # Load conda into this shell
    source "$CONDA_DIR/etc/profile.d/conda.sh"

    # Optional: initialize for future interactive shells
    conda init bash || true
else
    echo "Conda already installed."
    # Try to load it into this script
    if [ -f "$CONDA_DIR/etc/profile.d/conda.sh" ]; then
        source "$CONDA_DIR/etc/profile.d/conda.sh"
    else
        # Fallback: use whatever conda is on PATH
        eval "$(conda shell.bash hook)"
    fi
fi

# Make sure 'conda' works with 'conda activate'
eval "$(conda shell.bash hook)"

echo
echo "=== Step 2: Clone or update the repository ==="
if [ -d "$REPO_DIR/.git" ]; then
    echo "Repo already exists at $REPO_DIR, pulling latest changes..."
    cd "$REPO_DIR"
    git pull --rebase
else
    echo "Cloning repository into $REPO_DIR ..."
    git clone "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
fi

echo
echo "=== Step 3: Run project conda setup script (creates env, installs deps) ==="
chmod +x setup_conda.sh
bash setup_conda.sh   # This will create/prepare env '$ENV_NAME'

echo
echo "=== Step 4: Activate env and prepare folders / Jupyter ==="
conda activate "$ENV_NAME"

# Just in case, ensure jupyter + nbconvert are present
pip install --upgrade pip
pip install jupyter nbconvert

# Create required experiment folders
mkdir -p src/Experiments/chkpt
mkdir -p src/Experiments/trainingplots

echo
echo "=== Step 5: Run replication script (figures, etc.) ==="
chmod +x replicate.sh
bash replicate.sh

echo
echo "=== ALL DONE  ==="
echo "Conda env: $ENV_NAME"
echo "Repo dir : $REPO_DIR"
echo
echo "Next time, you can just do:"
echo "  cd \"$REPO_DIR\""
echo "  conda activate $ENV_NAME"
echo "  bash replicate.sh    # (or run individual notebooks / training scripts)"
