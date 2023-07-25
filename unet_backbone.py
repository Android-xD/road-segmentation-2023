import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet152, ResNet152_Weights
import numpy as np


class ResnetWithHead(nn.Module):
    def __init__(self, pretrained=False, freeze=False):
        super().__init__()

        self.freeze = freeze
        if freeze:
            assert pretrained

        # ResNet-18, optionally pretrained in
        weights = ResNet50_Weights.DEFAULT
        self.resnet = torchvision.models.resnet50(weights=weights)
        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # Override the final layers: we want to extract the full 7x7 final feature map
        self.resnet.avgpool = nn.Identity()
        self.resnet.fc = nn.Identity()

        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)  # 7 -> 14
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)  # 14 -> 28
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)  # 28 -> 56
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)  # 56 -> 112
        self.conv5 = nn.Conv2d(32, 16, kernel_size=3, padding=1)  # 112 -> 224
        self.conv_out = nn.Conv2d(16, 1, kernel_size=3, padding=1)  # 112 -> 224
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.freeze:
            self.resnet.eval()
            with torch.no_grad():
                x = self.resnet(x)
        else:
            x = self.resnet(x)
        x = x.view(x.shape[0], 512, 7, 7)
        x = self.up(self.relu(self.conv1(x)))
        x = self.up(self.relu(self.conv2(x)))
        x = self.up(self.relu(self.conv3(x)))
        x = self.up(self.relu(self.conv4(x)))
        x = self.up(self.relu(self.conv5(x)))
        x = self.conv_out(x).squeeze(1)
        return x


class Block(nn.Module):
    # a repeating structure composed of two convolutional layers with batch normalization and ReLU activations
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(out_ch),
                                   nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1),
                                   nn.ReLU())

    def forward(self, x):
        return self.block(x)


class ResnetUNet(nn.Module):
    def __init__(self, pretrained=False, freeze=False):
        super().__init__()

        self.freeze = freeze
        if freeze:
            assert pretrained

        # ResNet-18, optionally pretrained in ImageNet
        weights = ResNet152_Weights.DEFAULT
        self.resnet = torchvision.models.resnet152(weights=weights)
        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # Override the final layers: we want to extract the full 7x7 final feature map
        self.resnet.avgpool = nn.Identity()
        self.resnet.fc = nn.Identity()

        self.conv1 = nn.Conv2d(2048, 1024, kernel_size=3, padding=1)  # 7 -> 14
        self.batch1 = nn.BatchNorm2d(1024)
        self.conv2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.batch2 = nn.BatchNorm2d(512)  # 14 -> 28
        self.conv3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)  # 28 -> 56
        self.batch3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, padding=1)  # 56 -> 112
        self.batch4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, padding=1)  # 112 -> 224
        self.batch5 = nn.BatchNorm2d(64)
        self.conv_out = nn.Conv2d(64, 2, kernel_size=3, padding=1)  # 112 -> 224
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.freeze:
            self.resnet.eval()

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x0 = self.resnet.relu(x)
        x = self.resnet.maxpool(x0)

        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x = self.resnet.layer3(x2)
        # x = self.resnet.layer4(x3)

        # x = self.up(self.relu(self.conv1(x))) + x3
        # x = self.batch1(x)
        x = self.up(self.relu(self.conv2(x))) + x2
        # x = self.batch2(x)
        x = self.up(self.relu(self.conv3(x))) + x1
        # x = self.batch3(x)
        x = self.up(self.relu(self.conv4(x)))
        # x = self.batch4(x)
        x = self.up(self.relu(self.conv5(x)))
        x = self.conv_out(x).squeeze(1)
        return x


class ResnetUNet2(nn.Module):
    def __init__(self, pretrained=False, freeze=False):
        super().__init__()

        self.freeze = freeze
        if freeze:
            assert pretrained

        # ResNet-18, optionally pretrained in ImageNet
        weights = ResNet152_Weights.DEFAULT
        self.resnet = torchvision.models.resnet152(weights=weights)
        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # Override the final layers: we want to extract the full 7x7 final feature map
        self.resnet.avgpool = nn.Identity()
        self.resnet.fc = nn.Identity()

        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout2d(p=0.5)
        self.upconv6 = nn.ConvTranspose2d(2048, 1024, 2, 2)
        self.upconv1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.upconv2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.upconv4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec_block6 = Block(2048, 1024)
        self.dec_block1 = Block(1024, 512)
        self.dec_block2 = Block(512, 256)
        self.dec_block3 = Block(256, 128)
        self.dec_block4 = Block(128, 64)
        self.dec_block7 = Block(64, 64)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.head = nn.Sequential(nn.Conv2d(64, 2, 1))

    def forward(self, x):
        if self.freeze:
            self.resnet.eval()

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x0 = self.resnet.relu(x)
        x = self.resnet.maxpool(x0)

        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x = self.resnet.layer3(x2)
        # x = self.resnet.layer4(x3)

        # x = self.upconv6(x)
        # x = torch.cat([x, x3], dim=1)
        # x = self.dec_block6(x)
        x = self.upconv1(x)
        # x = self.drop(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec_block1(x)
        x = self.upconv2(x)
        # x = self.drop(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec_block2(x)
        x = self.upconv3(x)
        # x = self.drop(x)
        x = torch.cat([x, x], dim=1)
        x = self.dec_block3(x)
        x = self.upconv4(x)
        # x = self.drop(x)
        x = torch.cat([x, x], dim=1)
        x = self.dec_block4(x)

        return {"out": self.head(x)}


class ResnetUNet3(nn.Module):
    def __init__(self, pretrained=False, freeze=False):
        super().__init__()

        self.freeze = freeze
        if freeze:
            assert pretrained

        # ResNet-18, optionally pretrained in ImageNet
        weights = ResNet152_Weights.DEFAULT
        self.resnet = torchvision.models.resnet152(weights=weights)
        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # Override the final layers: we want to extract the full 7x7 final feature map
        self.resnet.avgpool = nn.Identity()
        self.resnet.fc = nn.Identity()

        self.pool = nn.MaxPool2d(2)

        self.upconv6 = nn.ConvTranspose2d(2048, 1024, 2, 2)
        self.upconv1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.upconv2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.upconv4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec_block6 = Block(2048, 1024)
        self.dec_block61 = Block(1024, 1024)
        self.dec_block1 = Block(1024, 512)
        self.dec_block11 = Block(512, 512)
        self.dec_block2 = Block(512, 256)
        self.dec_block21 = Block(256, 256)
        self.dec_block3 = Block(256, 128)
        self.dec_block31 = Block(128, 128)
        self.dec_block4 = Block(128, 64)
        self.dec_block41 = Block(64, 64)
        self.dec_block7 = Block(64, 64)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.head = nn.Sequential(nn.Conv2d(64, 2, 1), nn.Sigmoid())

    def forward(self, x):
        if self.freeze:
            self.resnet.eval()

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x0 = self.resnet.relu(x)
        x = self.resnet.maxpool(x0)

        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        # x = self.resnet.layer4(x3)

        # x = self.upconv6(x)
        # x = torch.cat([x, x3], dim=1)
        # x = self.dec_block6(x)
        xa = self.upconv1(x3)
        x = torch.cat([xa, x2], dim=1)
        x = self.dec_block1(x) + xa
        xb = self.upconv2(x)
        xa = self.upconv2(xa)
        x = torch.cat([xb, x1], dim=1)
        x = self.dec_block2(x) + xb + xa
        xc = self.upconv3(xb)
        xa = self.upconv3(xa)
        xb = self.upconv3(xb)
        x = torch.cat([xc, xc], dim=1)
        x = self.dec_block3(x) + xc + xb + xa
        xd = self.upconv4(x)
        xa = self.upconv4(xa)
        xb = self.upconv4(xb)
        xc = self.upconv4(xc)
        x = torch.cat([xd, xd], dim=1)
        x = self.dec_block4(x) + xd + xc + xb + xa

        return self.head(x)


def get_Unet(outputchannels=1, input_size=512):
    """ Basic Unet Architecture with resnet backbone.

    Args:
        outputchannels (int, optional): The number of output channels
        in your dataset masks. Defaults to 1.

    Returns:
        model: Returns the Unet model with preprocess method.
    """
    # weights = DeepLabV3_ResNet50_Weights.DEFAULT
    # model = models.segmentation.deeplabv3_resnet50(weights=weights, progress=True)
    model = ResnetUNet2(pretrained=True, freeze=False)

    # Set the model in training mode
    model.train()

    # preprocess.resize_size = [input_size]

    # Move the model to the GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if torch.cuda.is_available():
        pre = lambda x: x.type('torch.cuda.FloatTensor')
    else:
        pre = lambda x: x.type('torch.FloatTensor')

    return model, pre


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from torchvision import transforms as T
    import cv2

    # Check if GPU is available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    img_path = r"./data/training/images/satimage_0.png"
    label_path = r"./data/training/groundtruth/satimage_0.png"

    image = cv2.imread(img_path)
    label = cv2.imread(label_path, 0)

    image = torch.tensor(image, dtype=float)
    image = image.permute((2, 0, 1))

    image = image.unsqueeze(0)  # ahape -> [1,3,h,w]

    print(image.shape)
    size = 384
    crop = T.CenterCrop(size)

    label = torch.tensor(label, dtype=float)
    label = label.unsqueeze(0).unsqueeze(0)  # shape -> [1,1,h,w]

    label = crop(label)
    image = crop(image)
    print(image.shape)
    model = ResnetUNet2()
    model.eval()
    model.to(device)
    image = image.to(device)
    batch = image.type(torch.FloatTensor).to(device)
    print(type(batch[0]))
    output = model(batch)  # ["out"] #.detach()

    print(output.shape)
    prediction = output.squeeze(0).squeeze(0)