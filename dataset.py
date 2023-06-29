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



def test_train_split(dataset, train_split=0.8):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(train_split * dataset_size)
    train_indices, val_indices = indices[:split], indices[split:]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)

class CustomImageDataset(Dataset):
    def __init__(self, data_dir, train=True, test=False):
        """
        train: specifies if there are labels or not
        """
        self.train = train
        self.test = test
        self.img_list = glob.glob(os.path.join(data_dir, "images", "*"))
        if self.train:
            self.mask_list_gt = glob.glob(os.path.join(data_dir, "groundtruth", "*"))
            self.mask_list_sdf = glob.glob(os.path.join(data_dir, "groundtruth_sdf", "*"))
            self.mask_list_width = glob.glob(os.path.join(data_dir, "groundtruth_width", "*"))

        self.affineTransform = transforms.GeometricTransform()
        self.color_transform = T.Compose([
            T.ColorJitter(0.1, 0.5, 0.1),
            transforms.AddPerlinNoise()
        ]
        )

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = read_image(self.img_list[idx], ImageReadMode.RGB)
        if self.train:
            mask_gt = read_image(self.mask_list_gt[idx])
            mask_sdf = read_image(self.mask_list_sdf[idx])
            mask_width = read_image(self.mask_list_width[idx])
            
            if not self.test:
                self.affineTransform.sample_params()
            image = self.affineTransform(image)
            mask_gt = self.affineTransform(mask_gt)
            mask_sdf = self.affineTransform(mask_sdf)
            mask_width = self.affineTransform(mask_width)
            mask_width *= self.affineTransform.scale

            mask_gt[mask_gt > 0] = 1.

            mask_sdf *= self.affineTransform.scale
            # range (0,255) -> (0, 64) -rescale-> (0,1)
            mask_sdf = torch.clip(mask_sdf/64, 0, 1)

            mask_width = mask_width/70.*6 # To match the range of relu6

            # crop = T.CenterCrop(300)
            # image = crop(image)
            # mask = crop(mask)

            # image = self.affineTransform.backward(image)
            # mask = self.affineTransform.backward(mask)
            mask = torch.cat([mask_gt, mask_sdf, mask_width],0)
            image = image.to(torch.uint8)
            if self.color_transform and not self.test:
                image = self.color_transform(image)
        else:
            mask = self.img_list[idx]
        return image, mask

if __name__ == "__main__":
    dataset = CustomImageDataset(r"./data/training")
    print(len(dataset))
    for i in range(10):
        img, label = dataset[i]
        print(img.shape)
        print(label.shape)
        import visualize as vis
        for j in range(3):
            vis.show_img_mask(img, label[j])
