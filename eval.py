import cv2
import os
import matplotlib.pyplot as plt
from dataset import CustomImageDataset,test_train_split
import numpy as np
import torch
import visualize as vis
import torchvision.transforms as T
import seg_net_lite
from deeplabv3 import createDeepLabv3,load_model
from sklearn.metrics import f1_score, accuracy_score
import torch.nn.functional as F

torch.manual_seed(0)


def eval(model, dataset:CustomImageDataset, num_samples=10):
    for i in range(int(len(dataset)*0.9), len(dataset)):
        combined = torch.zeros(dataset[0][1].size()[-3:])
        for j in range(num_samples):
            img, target = dataset[i]

            output = model(model.preprocess(img.unsqueeze(0)))
            output = dataset.affineTransform.backward(output)
            #combined[output>0.01] = combined[output>0.01]*0.5 + output*0.5
            img = dataset.affineTransform.backward(img)
            vis.show_img_mask(img.detach(), output.detach())



if __name__ == '__main__':
    dataset_path = r"./data/test_set_images"
    training_set = r"./data/training"

    # Check if GPU is available
    use_cuda = torch.cuda.is_available()

    # Define the device to be used for computation
    device = torch.device("cuda" if use_cuda else "cpu")

    dataset = CustomImageDataset(training_set)

    train_dataset, val_dataset = test_train_split(dataset, 0.8)

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

    model, preprocess = createDeepLabv3(2, 512)
    state_dict = torch.load("out/model_best.pth.tar", map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)

    import torch.nn as nn
    import torchvision.transforms as T
    pad = nn.ReflectionPad2d(56)
    crop = T.CenterCrop(400)

    for i, (input, target) in enumerate(val_loader):
        # Move input and target tensors to the device (CPU or GPU)
        input = input.to(device)
        #print(input.shape, input.dtype)
        #input = input.squeeze()
        input = pad(input.squeeze().to(float)).to(torch.uint8)
        #print(input.shape, input.dtype)
        target = target.to(device)
        output = model(preprocess(input))['out']

        output = crop(output)
        input = crop(input)
        # normalize the output
        output = F.softmax(output)
        #for j in range(target.shape[0]):
            #vis.output_target_heat(input.detach()[j] / 255, output.detach()[j, 1], 0.3, target[j])
            #plt.imshow(output[j, 1].detach().cpu().numpy())
            #plt.show()
        print(torch.count_nonzero(target == (output[:, 1:2] > 0.5))/target.numel())

"""
# with pad
tensor(0.9248)
tensor(0.9220)
tensor(0.9046)
tensor(0.9250)
tensor(0.9346)
tensor(0.9063)
tensor(0.9157)

#  no pad
tensor(0.9274)
tensor(0.9220)
tensor(0.9076)
tensor(0.9255)
tensor(0.9352)
tensor(0.9055)
tensor(0.9172)
"""
