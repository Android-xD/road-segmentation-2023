#!/usr/bin/env bash
set -e

mkdir -p /cluster/scratch/horatan/CIL
echo "Creating virtual environment"
python3.8 -m venv /cluster/scratch/horatan/CIL
echo "Activating virtual environment"

source /cluster/scratch/horatan/CIL/bin/activate

/cluster/scratch/horatan/CIL/bin/pip install -r requirements.txt
