import torch
import torch.nn as nn

from torchvision.models.segmentation.deeplabv3 import DeepLabHead

class CycleCNN(nn.Module):
    def __init__(self, in_channels=5):
        super(CycleCNN, self).__init__()
        self.dlhead1 = DeepLabHead(in_channels+3, in_channels)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self = self.to(device)

    def forward(self, x):
        x = x[:, 3:] + self.dlhead1(x)
        return x
