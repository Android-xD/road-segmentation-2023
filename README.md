# Road Segmentation

## Setup on local Compuer
```shell
conda env create -f env.yml
conda activate CIL
```

## Setup on Euler
```shell
source install_pip.sh
```
After having everything installed once, you can just load the environment at the next login:
```shell
# load required modules
source startup.sh
```

## Testing
Start an interactive GPU session to check if the code runs.
```shell
srun --gpus=1 -n 16 --mem-per-cpu=4096 --pty bash
```
See if GPU is on:
```shell
python -c "import torch; print(torch.cuda.is_available())"
```
Start an interative CPU session
```shell
srun --cpus-per-task=1 -n 16 --mem-per-cpu=4096 --pty bash
```

## Dataset Structure
```shell
cd road-segmentation-2023
wget https://polybox.ethz.ch/index.php/s/bn3hG0mcpfsiMbk/download
unzip download
rm download
```

```
data
├── test
│   ├── images
│   │   ├── satimage_144.png
│   │   ├── satimage_145.png
│   │   ├── ...
├── training
│   ├── images
│   │   ├── satimage_0.png
│   │   ├── satimage_1.png
│   │   ├── ...
│   ├── groundtruth
│   │   ├── satimage_0.png
│   │   ├── satimage_1.png
│   │   ├── ...
│   ├── groundtruth_rich
│   │   ├── satimage_0.png
│   │   ├── satimage_1.png
│   │   ├── ...
```