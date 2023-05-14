# Road Segmentation

## Setup
```shell
conda env create -f env.yml
conda activate CIL
```

## Dataset Structure
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