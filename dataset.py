import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image

import utils.transforms as transforms
import utils.visualize as vis
from utils.make_rich_labels import dir_from_sdf


def split(dataset_size, split_proportions=[0.8,0.2]):
    """
    Splits a dataset into given proportions.

    Args:
        dataset_size: number of samples in the dataset
        split_proportions: list of proportions for each split

    returns: tuple of indices for each split
    """
    assert sum(split_proportions) == 1, "The sum of split_proportions should be 1."
    indices = torch.randperm(dataset_size)

    splits = []
    start_idx = 0
    for proportion in split_proportions:
        split_size = int(proportion * dataset_size)
        end_idx = start_idx + split_size
        split_indices = indices[start_idx:end_idx]
        splits.append(split_indices)
        start_idx = end_idx

    return tuple(splits)


class CustomImageDataset(Dataset):
    def __init__(self, data_dir, train=True, rich=True, geo_aug=True, color_aug=True):
        """
        Custom PyTorch Dataset for image data with optional augmentation and rich labels.

        Args:
            data_dir (str): The directory path where the satellite image and map data is stored.
            train (bool): If True, there are labels (masks) for the images (ie. training set)
            rich (bool): If True, loads additional rich ground truth information.
            geo_aug (bool): If True, enables geometric augmentation of the images.
            color_aug (bool): If True, enables color augmentation of the images.
        """
        self.train = train
        self.rich = rich
        self.geo_aug = geo_aug
        self.color_aug = color_aug

        # get image and label paths
        assert os.path.exists(os.path.join(data_dir, "images")), "The data directory does not exist."
        self.img_list = glob.glob(os.path.join(data_dir, "images", "*"))
        if self.train:
            assert os.path.exists(os.path.join(data_dir, "groundtruth")), "The groundtruth directory does not exist."
            self.mask_list_gt = glob.glob(os.path.join(data_dir, "groundtruth", "*"))
            if self.rich:
                assert os.path.exists(os.path.join(data_dir, "groundtruth_sdf")), "The groundtruth_sdf directory does not exist. Make sure to run make_rich_labels.py first."
                assert os.path.exists(os.path.join(data_dir, "groundtruth_width")), "The groundtruth_width directory does not exist. Make sure to run make_rich_labels.py first."
                self.mask_list_sdf = glob.glob(os.path.join(data_dir, "groundtruth_sdf", "*"))
                self.mask_list_width = glob.glob(os.path.join(data_dir, "groundtruth_width", "*"))

        # initialize geometric and color transforms
        self.affineTransform = transforms.GeometricTransform()
        self.color_transform = T.Compose(
            [
                T.ColorJitter(0.1, 0.2, 0.1),
                transforms.AddPerlinNoise(8, 0.02, 0.1, 10),
            ]
        )

    def __len__(self):
        """
        Returns the number of images in the dataset.
        """
        return len(self.img_list)

    def __getitem__(self, idx):
        """
        Returns a tuple of (image, mask) where mask is the ground truth plus optional rich labels.
        """
        image = read_image(self.img_list[idx], ImageReadMode.RGB)
        mask = np.array([])

        # apply augmentations
        if self.geo_aug:
            self.affineTransform.sample_params()
            image = self.affineTransform(image)
        if self.color_aug:
            image = self.color_transform(image)

        image = image.to(torch.uint8)

        # load labels
        if self.train:
            # load ground truth label
            mask_gt = read_image(self.mask_list_gt[idx])
            mask_gt = self.affineTransform(mask_gt)
            mask_gt[mask_gt > 0.5] = 1. # binarize
            mask = mask_gt.to(torch.float32)

            # load rich labels
            if self.rich:
                # load sdf label
                mask_sdf = read_image(self.mask_list_sdf[idx])
                mask_sdf = self.affineTransform(mask_sdf)

                # compute direction label
                mask_dir = dir_from_sdf(mask_sdf)

                mask_sdf *= self.affineTransform.scale
                mask_sdf = torch.clip(mask_sdf/64, 0, 1)

                # load width label
                mask_width = read_image(self.mask_list_width[idx])
                mask_width = self.affineTransform(mask_width)
                mask_width *= self.affineTransform.scale
                mask_width = mask_width/70 # be in [0,1]

                mask = torch.cat([mask_gt, mask_sdf, mask_width, mask_dir], 0).to(torch.float32)
                
        return image, mask


if __name__ == "__main__":
    """
    Test the dataset class.
    This script will generate example plots of the augmentations and rich labels and store them in the figures directory.
    """
    # create figures directory
    store_figures = r"./figures"
    os.makedirs(store_figures, exist_ok=True)

    # create datasets
    dataset_clean = CustomImageDataset(r"./data/training", rich=False, geo_aug=False, color_aug=False)
    dataset_color = CustomImageDataset(r"./data/training", rich=False, geo_aug=False, color_aug=True)
    dataset_geo = CustomImageDataset(r"./data/training", rich=False, geo_aug=True, color_aug=False)
    dataset_rich = CustomImageDataset(r"./data/training", rich=True, geo_aug=False, color_aug=False)

    # augmentation plots
    for i in range(10):
        img, label = dataset_clean[i]
        color, _ = dataset_color[i]
        geo, label_aug = dataset_geo[i]

        img_list = [img, color, geo]
        img_list = [np.transpose(im.squeeze(), (1, 2, 0)) for im in img_list]
        img_list = [img_list[0], label[0], img_list[1], img_list[2], label_aug[0]]
        vis.plot_images(img_list, titles=["Original Image", "Original Mask", "Color Augmentation", "Geometric Augmentation", "Augmented Mask"])
        plt.savefig(f"{store_figures}/augmentations_{i}.png")

    # rich label plots
    for i in range(10):
        img, label = dataset_rich[i]
        img = np.transpose(img.squeeze(), (1, 2, 0))
        label[3, label[0] != 1] = 0

        img_list = [img] + [label[j] for j in range(label.shape[0])]
        vis.plot_images(img_list, titles=["Original Image", "Original Mask", "Distance Function", "Road Width", "Road Direction"])
        plt.savefig(f"{store_figures}/rich_labels_{i}.png")