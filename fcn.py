from torchvision import models
from torchvision.models.segmentation import fcn_resnet101, FCN_ResNet101_Weights
from torchvision.models.segmentation.fcn import FCNHead
import torch

def createFCN(outputchannels=1, input_size=512):
    """DeepLabv3 class with custom head.

    Args:
        outputchannels (int, optional): The number of output channels
        in your dataset masks. Defaults to 1.

    Returns:
        model: Returns the DeepLabv3 model with the ResNet101 backbone.
    """
    weights = FCN_ResNet101_Weights.DEFAULT
    model = models.segmentation.fcn_resnet101(weights=weights, progress=True)
    model.classifier = FCNHead(2048, outputchannels)
    # Set the model in training mode
    model.train()

    preprocess = weights.transforms(antialias=True)
    preprocess.resize_size = [input_size]

    # Move the model to the GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return model, preprocess