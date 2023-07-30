# Road Segmentation

## Setup on local Computer
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

## Google Dataset Structure
```shell
cd road-segmentation-2023
wget https://www.polybox.ethz.ch/index.php/s/MMOxAqAsaW81uwC/download 
unzip download
rm download
```

```
data_google
├── images
│   ├── satimage_0.png
│   ├── satimage_1.png
│   ├── ...
├── groundtruth
│   ├── satimage_0.png
│   ├── satimage_1.png
│   ├── ...
```

## Reproducing our best performing model
For the very last run we changed the parameters of the transform to keep the scale. So in order to get the exact same result, you will need to change the settings a bit. 

Change the scale in `./utils/transforms.py` like so:
```python
class GeometricTransform:
    def __init__(self):
        self.min_scale = 0.999
        self.max_scale = 1.
```
Change the threshold in `mask_to_submision.py`
```python
foreground_threshold = 0.35 # percentage of pixels of val 255 required to assign a foreground label to a patch
```
To reproduce our best performing model, you need to run the install script on euler and then submit the batch script `job_full.sh`, which runs the training on the full set of images. Make sure to load the data first, as described above.
```shell
# sets up the environment with its dependencies
source install_pip.sh
# runs the whole training
sbatch job_full.sh
# the output masks are written to ./out/predictions/
# the submission file is written to ./submission.csv
```