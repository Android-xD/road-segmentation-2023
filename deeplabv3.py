"""Model Pretrained on segmentation.

https://github.com/msminhas93/DeepLabv3FineTuning/blob/master/model.py

DeepLabv3 Model download and change the head for your prediction.
"""
from torchvision import models
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, LRASPP_MobileNet_V3_Large_Weights, DeepLabV3_ResNet101_Weights, DeepLabV3_MobileNet_V3_Large_Weights

from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import torch

def createDeepLabv3(outputchannels=1, input_size=512):
    """DeepLabv3 class with custom head.

    Args:
        outputchannels (int, optional): The number of output channels
        in your dataset masks. Defaults to 1.

    Returns:
        model: Returns the DeepLabv3 model with the ResNet101 backbone.
    """

    #weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    #model = models.segmentation.deeplabv3_mobilenet_v3_large(weights=weights, progress=True)
    #model.classifier = torch.nn.Sequential(DeepLabHead(960, outputchannels))

    #weights = DeepLabV3_ResNet101_Weights.DEFAULT
    #model = models.segmentation.deeplabv3_resnet101(weights=weights, progress=True)
    #model.classifier = torch.nn.Sequential(DeepLabHead(2048, outputchannels))


    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = models.segmentation.deeplabv3_resnet50(weights=weights, progress=True)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout2d(p=0.5),
        DeepLabHead(2048, 256),
        torch.nn.Dropout2d(p=0.5),
        DeepLabHead(256, outputchannels)
    )

    # Set the model in training mode
    model.train()

    preprocess = weights.transforms(antialias=True)
    preprocess.resize_size = [input_size]

    # Move the model to the GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return model, preprocess


def load_model(model_state_file, outputchannels=1, input_size=512):
    """Load file from path."""
    createDeepLabv3(outputchannels=outputchannels, input_size=input_size)

    state_dict = torch.load(model_state_file, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict["state_dict"])
    return model, preprocess


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
    model, preprocess = createDeepLabv3(1, size)
    model.eval()
    image = image.to(device)
    batch = preprocess(image)
    print(batch.shape)
    output = model(batch)["out"].detach()

    print(output.shape)
    prediction = output.squeeze(0).squeeze(0)  # shape [1,1,h,w ] -> [h,w]
    plt.imshow(prediction)
    plt.show()
