import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision


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


class Unet(nn.Module):
    def __init__(self, output_channels=2, chs=(3, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.enc_block1 = Block(3, 64)
        self.enc_block2 = Block(64, 128)
        self.enc_block3 = Block(128, 256)
        self.enc_block4 = Block(256, 512)
        self.enc_block5 = Block(512, 1024)
        self.enc_block6 = Block(1024, 2048)

        self.pool = nn.MaxPool2d(2)
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
        self.head = nn.Sequential(nn.Conv2d(64, output_channels,1), nn.Sigmoid())

    def forward(self, x):
        # enc_features = []
        enc1 = self.enc_block1(x)
        x = self.pool(enc1)
        enc2 = self.enc_block2(x)
        x = self.pool(enc2)
        enc3 = self.enc_block3(x)
        x = self.pool(enc3)
        enc4 = self.enc_block4(x)
        x = self.pool(enc4)
        enc5 = self.enc_block5(x)

        x = self.upconv1(enc5)
        x = torch.cat([x, enc4], dim=1)
        x = self.dec_block1(x)
        x = self.upconv2(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec_block2(x)
        x = self.upconv3(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec_block3(x)
        x = self.upconv4(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec_block4(x)
        return {"out": self.head(x)}


def get_Unet(outputchannels=1, input_size=512):
    """ Basic Unet Architecture.

    Args:
        outputchannels (int, optional): The number of output channels
        in your dataset masks. Defaults to 1.

    Returns:
        model: Returns the Unet model with preprocess method.
    """
    model = Unet(outputchannels, chs=(3, 64, 128, 256, 512, 1024))

    # Set the model in training mode
    model.train()

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
    size = 400
    crop = T.CenterCrop(size)

    label = torch.tensor(label, dtype=float)
    label = label.unsqueeze(0).unsqueeze(0)  # shape -> [1,1,h,w]

    label = crop(label)
    image = crop(image)
    print(image.shape)
    model = Unet()
    model.train()
    model.to(device)
    image = image.to(device)
    batch = image.type(torch.FloatTensor).to(device)
    print(type(batch[0]))
    output = model(batch)  # ["out"] #.detach()

    print(output.shape)
    prediction = output.squeeze(0).squeeze(0)
