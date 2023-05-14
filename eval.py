import cv2
import os
import matplotlib.pyplot as plt
from dataset import CustomImageDataset,test_train_split
import numpy as np
import torch
import visualize as vis
import torchvision.transforms as T
import seg_net_lite
from sklearn.metrics import f1_score, accuracy_score



def eval(model, dataset:CustomImageDataset, num_samples=10):
    for i in range(int(len(dataset)*0.9), len(dataset)):
        combined = torch.zeros(dataset[0][1].size()[-3:])
        for j in range(num_samples):
            img, target = dataset[i]

            output = model(img.unsqueeze(0))
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


    model = seg_net_lite.get_seg_net()
    model_state_file = os.path.join('out', 'model_best.pth.tar')
    print('=> loading model from {}'.format(model_state_file))
    state_dict = torch.load(model_state_file, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    eval(model, dataset, 100)