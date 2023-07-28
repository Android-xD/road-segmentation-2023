#!/bin/bash
#SBATCH -n 4
#SBATCH --time=10:00:00
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=4000
#SBATCH --tmp=4000 # per node!!
#SBATCH --job-name=cil_train
#SBATCH --output=cil_train.out # specify a file to direct output stream
#SBATCH --error=paperX.err
#SBATCH --open-mode=truncate # to overrides out and err files, you can also use

source startup.sh
python train.py --data ./data_google/training --model fpn --epochs 30 --full True --augmentations True
python train.py --data ./data/training --model fpn --epochs 30 --full True --augmentations True --load_model ./out/model_best.pth.tar
python train.py --data ./data_google/training --model fpn --epochs 10 --full True --augmentations False --load_model ./out/model_best.pth.tar
python train.py --data ./data/training --model fpn --epochs 10 --full True --augmentations False --load_model ./out/model_best.pth.tar
python submit.py --model fpn --n_samples 100 --threshold 0.35
python mask_to_submission.py

python train
