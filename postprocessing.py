import torch
import torch.nn as nn
from seg_net_lite import get_seg_net

from torchvision.models.segmentation.deeplabv3 import DeepLabHead

class CycleCNN(nn.Module):
    def __init__(self, in_channels=5):
        super(CycleCNN, self).__init__()
        self.segnet = get_seg_net()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.segnet = self.segnet.to(device)

    def forward(self, x):
        x = x[:, 3:] + self.segnet(x)
        return x
