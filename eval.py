import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Subset

import utils.visualize as vis
from dataset import CustomImageDataset, split
from models.deeplabv3 import createDeepLabv3
from utils.visualize import plot_images

torch.manual_seed(0)


def eval(model, dataset:CustomImageDataset, num_samples=10):
    """
    Evaluate the model on a subset of the dataset.

    Args:
        model (torch.nn.Module): The trained model to be evaluated.
        dataset (CustomImageDataset): The dataset on which the model will be evaluated.
        num_samples (int, optional): Number of samples to visualize. Defaults to 10.
    """
    for i in range(int(len(dataset)*0.9), len(dataset)):
        combined = torch.zeros(dataset[0][1].size()[-3:])
        for j in range(num_samples):
            img, target = dataset[i]

            output = model(model.preprocess(img.unsqueeze(0)))
            output = dataset.affineTransform.backward(output)
            #combined[output>0.01] = combined[output>0.01]*0.5 + output*0.5
            img = dataset.affineTransform.backward(img)
            vis.show_img_mask(img.detach(), output.detach())

def nms(img, dir=None ):
    """
    Non-Maximum Suppression (NMS) on the input image.

    Args:
        img (np.ndarray): Input image to perform NMS on.
        dir (np.ndarray, optional): Direction map for NMS. Defaults to None.

    Returns:
        np.ndarray: NMS result.
    """
    if dir is None:
        img = cv2.GaussianBlur(img,ksize=(15, 15),sigmaX=5)
        dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        dir = (-np.arctan2(dy, dx)) % np.pi / np.pi
        gradient_magnitude = (dx ** 2 + dy ** 2) ** 0.5
    #else:
        #dir =cv2.GaussianBlur(dir,ksize=(15,15),sigmaX=5)

    angle_quantization = 4  # Number of discrete directions

    quantized_angle = np.round(dir*angle_quantization)

    suppressed = np.zeros_like(img)
    height, width = img.shape
    r = 4
    ri = np.array([i_ for i_ in range(-r, r + 1) if i_ != 0])
    for i in range(0, height):
        for j in range(0, width):
            if img[i, j] <= 0.1:
                suppressed[i, j] = 0
            else:
                direction = quantized_angle[i, j]
                if direction == 0:
                    idx_i, idx_j = i, j + ri
                elif direction == 1:
                    idx_i, idx_j = i + ri, j + ri
                elif direction == 2:
                    idx_i, idx_j = i + ri, j
                else:
                    idx_i, idx_j = i + ri, j - ri
                idx_i, idx_j = np.clip(idx_i, 0, height-1), np.clip(idx_j, 0, width-1)
                neighbors = img[idx_i, idx_j]

                if img[i, j] <= np.max(neighbors):
                    suppressed[i, j] = 0
                else:
                    suppressed[i, j] = 1

    return suppressed

def prop(img,dir):
    """
    Perform propagation on the input image using the given direction map.

    Args:
        img (np.ndarray): Input image to perform propagation on.
        dir (np.ndarray): Direction map for propagation.

    Returns:
        np.ndarray: Result of propagation.
    """
    if dir is None:
        img = cv2.GaussianBlur(img,ksize=(15,15),sigmaX=5)
        dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        dir = (-np.arctan2(dy, dx)) % np.pi / np.pi
        gradient_magnitude = (dx ** 2 + dy ** 2) ** 0.5
    #else:
        #dir =cv2.GaussianBlur(dir,ksize=(15,15),sigmaX=5)

    angle_quantization = 4  # Number of discrete directions

    quantized_angle = np.round(dir*angle_quantization)

    suppressed = np.zeros_like(img)
    height, width = img.shape
    r = 4
    p = np.zeros((5,*img.shape))
    for it in range(5):
        ri = np.array([i_ for i_ in range(-r, r + 1) if i_ != 0])
        for i in range(0, height):
            for j in range(0, width):
                if img[i, j] == 0:
                    suppressed[i, j] = 0
                else:
                    direction = (quantized_angle[i, j] + 2 )% 4
                    if direction == 0:
                        idx_i, idx_j = i, j + ri
                    elif direction == 1:
                        idx_i, idx_j = i + ri, j + ri
                    elif direction == 2:
                        idx_i, idx_j = i + ri, j
                    else:
                        idx_i, idx_j = i + ri, j - ri
                    idx_i, idx_j = np.clip(idx_i, 0, height-1), np.clip(idx_j, 0, width-1)
                    neighbors = img[idx_i, idx_j]

                    suppressed[i, j] = np.mean(neighbors)
        img = suppressed
        p[it]=suppressed

    return suppressed


if __name__ == '__main__':
    test_set = r"./data/test/images"
    training_set = r"./data/training"

    # Check if GPU is available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Create dataset and dataloaders for training and validation
    dataset= CustomImageDataset(training_set, train=True, geo_aug=False, color_aug=False)
    train_indices, val_indices = split(len(dataset), [0.8, 0.2])
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    # Load the pre-trained DeepLabV3 model and its state dictionary
    model, preprocess, _ = createDeepLabv3(1, 400)
    state_dict = torch.load("out/model_best.pth.tar", map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)

    for i, (input, target) in enumerate(val_loader):

        # Move input and target tensors to the device (CPU or GPU)
        input = input.to(device)
        #input = input.squeeze()
        target = target.to(device)
        output = model(preprocess(input))['out']
        # normalize the output
        pred = output[:, :1]
        # sdf = output[:, 1:2]
        # width = output[:, 2:3]/70
        # pred = F.sigmoid(pred)
        # sdf = F.sigmoid(sdf)
        # width = F.relu6(width)
        # dir = output[:,3:4] % 1
        # tile = F.sigmoid(output[:,4:5])
        # dir[tile < 0.5] = 0
        # width[tile < 0.5] = 0

        for j in range(target.shape[0]):
            img = np.transpose(input.cpu().detach()[j] / 255., (1, 2, 0))
            out = [pred] #, sdf, width, dir, tile]
            out = [o[j, 0].cpu().detach().numpy() for o in out]
            images = [img]+out
            names = ["Input Image", "Prediction"]
            plot_images(images, names, hpad=0.5)
            #plt.show()
            plt.savefig(f"./figures/out_{i}_{j}.png")


