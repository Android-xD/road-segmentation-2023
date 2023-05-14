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
    def __init__(self, data_dir):

        self.img_list = glob.glob(os.path.join(data_dir, "images", "*"))
        self.mask_list = glob.glob(os.path.join(data_dir, "groundtruth", "*"))

        self.affineTransform = transforms.GeometricTransform()
        self.color_transform = T.Compose([
            T.ColorJitter(0.1, 0.5, 0.1)
        ]
        )

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = read_image(self.img_list[idx], ImageReadMode.RGB)
        mask = read_image(self.mask_list[idx])
        self.affineTransform.sample_params()
        image = self.affineTransform(image)
        mask = self.affineTransform(mask)
        mask[mask > 0] = 1
        # crop = T.CenterCrop(300)
        # image = crop(image)
        # mask = crop(mask)

        # image = self.affineTransform.backward(image)
        # mask = self.affineTransform.backward(mask)
        image = torch.tensor(image, dtype=torch.uint8)
        if self.color_transform:
            image = self.color_transform(image)

        return image, torch.tensor(mask, dtype=torch.long)

if __name__ == "__main__":
    dataset = CustomImageDataset(r"./data/training")
    print(len(dataset))
    for i in range(10):
        img, label = dataset[i]
        print(img.shape)
        print(label.shape)
        import visualize as vis
        vis.show_img_mask(img, label)