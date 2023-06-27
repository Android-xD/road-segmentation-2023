#!/usr/bin/env bash
set -e
source startup.sh
mkdir -p /cluster/scratch/$USER/CIL
echo "Creating virtual environment"
python -m venv /cluster/scratch/$USER/CIL
echo "Activating virtual environment"

source /cluster/scratch/$USER/CIL/bin/activate
/cluster/scratch/$USER/CIL/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
/cluster/scratch/$USER/CIL/bin/pip install -r requirements.txt
