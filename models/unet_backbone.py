import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet152_Weights


class Block(nn.Module):
    """ 
    Source: project_3_final.ipynb provided in class
    A repeating structure composed of two convolutional layers 
    with batch normalization and ReLU activations.
    """
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
    """
    ResnetUNet with skip connections.
    """
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

        x = self.upconv1(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec_block1(x)
        x = self.upconv2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec_block2(x)
        x = self.upconv3(x)
        x = torch.cat([x, x], dim=1)
        x = self.dec_block3(x)
        x = self.upconv4(x)
        x = torch.cat([x, x], dim=1)
        x = self.dec_block4(x)

        return self.head(x)


def get_Unet(outputchannels=1, input_size=512):
    """ 
    Basic Unet Architecture with resnet backbone.

    Args:
        outputchannels (int, optional): The number of output channels
        in your dataset masks. Defaults to 1.

    Returns:
        model: Returns the Unet model with preprocess method.
    """
    model = ResnetUNet(pretrained=True, freeze=False)

    # Set the model in training mode
    model.train()

    # Move the model to the GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define the preprocessing function
    if torch.cuda.is_available():
        pre = lambda x: x.type('torch.cuda.FloatTensor')
    else:
        pre = lambda x: x.type('torch.FloatTensor')

    # return model, preprocessing function, and postprocessing function
    return model, pre, lambda x:x


if __name__ == "__main__":
    """
    Test the model.
    """
    import cv2
    import numpy as np
    from torchvision import transforms as T

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
    model = ResnetUNet()
    model.eval()
    model.to(device)
    image = image.to(device)
    batch = image.type(torch.FloatTensor).to(device)
    print(type(batch[0]))
    output = model(batch)  # ["out"] #.detach()

    print(output.shape)
    prediction = output.squeeze(0).squeeze(0)