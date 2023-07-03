import torch
import torch.nn as nn
from unet import UNet

from torchvision.models.segmentation.deeplabv3 import DeepLabHead

class CycleCNN(nn.Module):
    def __init__(self, in_channels=5):
        super(CycleCNN, self).__init__()
        self.unet = UNet(8, 5)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.unet = self.unet.to(device)


    def forward(self, x):
        return x[:,3:] + self.unet(x)

