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
        
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1) # 7 -> 14
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1) # 14 -> 28
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1) # 28 -> 56
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1) # 56 -> 112
        self.conv5 = nn.Conv2d(32, 16, kernel_size=3, padding=1) # 112 -> 224
        self.conv_out = nn.Conv2d(16, 1, kernel_size=3, padding=1) # 112 -> 224
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
        self.resnet = torchvision.models.resnet152( weights=weights )
        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        # Override the final layers: we want to extract the full 7x7 final feature map
        self.resnet.avgpool = nn.Identity()
        self.resnet.fc = nn.Identity()
        
        self.conv1 = nn.Conv2d(2048, 1024, kernel_size=3, padding=1) # 7 -> 14
        self.batch1 = nn.BatchNorm2d(1024)
        self.conv2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.batch2 = nn.BatchNorm2d(512) # 14 -> 28
        self.conv3 = nn.Conv2d(512, 256, kernel_size=3, padding=1) # 28 -> 56
        self.batch3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, padding=1) # 56 -> 112
        self.batch4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, padding=1) # 112 -> 224
        self.batch5 = nn.BatchNorm2d(64)
        self.conv_out = nn.Conv2d(64, 2, kernel_size=3, padding=1) # 112 -> 224
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
        #x = self.resnet.layer4(x3)
        
        #x = self.up(self.relu(self.conv1(x))) + x3
        #x = self.batch1(x)
        x = self.up(self.relu(self.conv2(x))) + x2
        #x = self.batch2(x)
        x = self.up(self.relu(self.conv3(x))) + x1
        #x = self.batch3(x)
        x = self.up(self.relu(self.conv4(x))) 
        #x = self.batch4(x)
        x = self.up(self.relu(self.conv5(x))) 
        x = self.conv_out(x).squeeze(1)
        return x
    
def get_Unet(outputchannels=1, input_size=512):
    """DeepLabv3 class with custom head.

    Args:
        outputchannels (int, optional): The number of output channels
        in your dataset masks. Defaults to 1.

    Returns:
        model: Returns the DeepLabv3 model with the ResNet101 backbone.
    """
    #weights = DeepLabV3_ResNet50_Weights.DEFAULT
    #model = models.segmentation.deeplabv3_resnet50(weights=weights, progress=True)
    model =ResnetUNet(pretrained=True, freeze=False)
    
    #model.classifier = DeepLabHead(2048, outputchannels)
    # Set the model in training mode
    #model.train()

    #preprocess.resize_size = [input_size]

    # Move the model to the GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return model

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
    model= ResnetUNet()
    model.eval()
    model.to(device)
    image = image.to(device)
    batch = image.type(torch.FloatTensor).to(device)
    print(type(batch[0]))
    output = model(batch)#["out"] #.detach()

    print(output.shape)
    prediction = output.squeeze(0).squeeze(0)      