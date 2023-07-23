import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Subset
import transforms
import glob
from make_rich_labels import dir_from_sdf



def split(dataset_size, split_proportions=[0.8,0.2]):

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
        train: specifies if there are labels or not
        """
        self.train = train
        self.rich = rich
        self.geo_aug = geo_aug
        self.color_aug = color_aug

        self.img_list = glob.glob(os.path.join(data_dir, "images", "*"))
        if self.train:
            self.mask_list_gt = glob.glob(os.path.join(data_dir, "groundtruth", "*"))
            if self.rich:
                self.mask_list_sdf = glob.glob(os.path.join(data_dir, "groundtruth_sdf", "*"))
                self.mask_list_width = glob.glob(os.path.join(data_dir, "groundtruth_width", "*"))

        self.affineTransform = transforms.GeometricTransform()
        self.color_transform = T.Compose([
            T.ColorJitter(0.1, 0.2, 0.1),
            transforms.AddPerlinNoise(8, 0.02, 0.1, 10),
        ]
        )

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = read_image(self.img_list[idx], ImageReadMode.RGB)

        if self.geo_aug:
            self.affineTransform.sample_params()
        else:
            self.affineTransform.zero_params()

        image = self.affineTransform(image)
        image = image.to(torch.uint8)
        if self.color_aug:
            image = self.color_transform(image)

        if self.train:
            mask_gt = read_image(self.mask_list_gt[idx])
            mask_gt = self.affineTransform(mask_gt)
            mask_gt[mask_gt > 0.5] = 1.
            if self.rich:
                mask_sdf = read_image(self.mask_list_sdf[idx])
                mask_width = read_image(self.mask_list_width[idx])
                mask_sdf = self.affineTransform(mask_sdf)
                mask_width = self.affineTransform(mask_width)
                mask_width *= self.affineTransform.scale

                # compute direction
                mask_dir = dir_from_sdf(mask_sdf)

                mask_sdf *= self.affineTransform.scale
                # range (0,255) -> (0, 64) -rescale-> (0,1)
                mask_sdf = torch.clip(mask_sdf/64, 0, 1)

                mask_width = mask_width/70 # be in [0,1]

                mask = torch.cat([mask_gt, mask_sdf, mask_width, mask_dir], 0).to(torch.float32)
            else:
                mask = mask_gt.to(torch.float32)
        else:
            mask = self.img_list[idx]
        return image, mask

if __name__ == "__main__":
    import visualize as vis
    import numpy as np
    store_figures = r"./figures"
    os.makedirs(store_figures, exist_ok=True)

    ## augmentation plots
    dataset_clean = CustomImageDataset(r"./data/training", rich=False, geo_aug=False, color_aug=False)
    dataset_color = CustomImageDataset(r"./data/training", rich=False, geo_aug=False, color_aug=True)
    dataset_geo = CustomImageDataset(r"./data/training", rich=False, geo_aug=True, color_aug=False)

    for i in range(10):
        img, label = dataset_clean[i]
        color, _ = dataset_color[i]
        geo, _ = dataset_geo[i]

        img_list = [img, color, geo]
        img_list = [np.transpose(im.squeeze(), (1, 2, 0)) for im in img_list]
        vis.plot_images(img_list, titles=["Original Image", "Color Augmentation", "Geometric Augmentation"], hpad=0.5)
        plt.show()#plt.savefig(f"{store_figures}/augmentations_{i}.jpg")

    ## Rich label plots
    dataset_rich = CustomImageDataset(r"./data/training", rich=True, geo_aug=False, color_aug=False)

    for i in range(10):
        img, label = dataset_rich[i]
        img = np.transpose(img.squeeze(), (1, 2, 0))
        label[3, label[0] != 1] = 0

        img_list = [img] + [label[j] for j in range(label.shape[0])]
        vis.plot_images(img_list, titles=["Original Image", "Original Mask", "Distance Function", "Road Width", "Road Direction"], hpad=0.5)
        plt.savefig(f"{store_figures}/rich_labels_{i}.png")
