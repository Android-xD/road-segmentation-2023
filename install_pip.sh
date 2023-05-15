#!/usr/bin/env bash
set -e
source startup.sh
mkdir -p /cluster/scratch/horatan/CIL
echo "Creating virtual environment"
python -m venv /cluster/scratch/horatan/CIL
echo "Activating virtual environment"

source /cluster/scratch/horatan/CIL/bin/activate
/cluster/scratch/horatan/CIL/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
/cluster/scratch/horatan/CIL/bin/pip install -r requirements.txt
