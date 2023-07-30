#!/bin/bash
#SBATCH -n 16
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=4096
#SBATCH --tmp=4096 # per node!!
#SBATCH --job-name=cil_train
#SBATCH --output=cil_train.out # specify a file to direct output stream
#SBATCH --error=paperX.err
#SBATCH --open-mode=truncate # to overrides out and err files, you can also use

source startup.sh
python train.py --lr 0.0001 --data './data_google/training' --model 'fpn' --epochs 17 --full True --augmentations True
python train.py --lr 0.0001 --data './data/training' --model 'fpn' --epochs 30 --full True --augmentations True --load_model './out/model_best.pth.tar'
python train.py --lr 0.00001 --data './data/training' --model 'fpn' --epochs 30 --full True --augmentations True --load_model './out/model_best.pth.tar'
python submit.py --model 'fpn' --n_samples 50 --threshold 0.35
python mask_to_submission.py