import cv2
import os
import matplotlib.pyplot as plt
from dataset import CustomImageDataset,test_train_split
import numpy as np
import torch
import visualize as vis
import torchvision.transforms as T
from deeplabv3 import createDeepLabv3,load_model
from sklearn.metrics import f1_score, accuracy_score
import torch.nn.functional as F
from mask_to_submission import mask_to_submission_strings
from utils import aggregate_tile, f1_score
from visualize import plot_images
from tqdm import tqdm

torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
# Define the device to be used for computation
device = torch.device("cuda" if use_cuda else "cpu")

def resample(query, path, index, n_samples=20):
    with torch.no_grad():
        dataset = CustomImageDataset(path,train=False, test=True)
        dataset_adapt = CustomImageDataset(path,train=False, test=False)
        dataset_adapt.affineTransform.hfilp_prob=0
        dataset_adapt.affineTransform.vfilp_prob=0
    
    
        input, target = dataset[index]
        input = input.unsqueeze(0).to(device)
        output = query(input)
    
        #target = target.unsqueeze(0)
        #target = target.to(device)
    
    
        acc = output
        count = torch.ones_like(output)
        for _ in tqdm(range(n_samples)):
            input_sample, _ = dataset_adapt[index]
            # Move input and target tensors to the device (CPU or GPU)
            input_sample = input_sample.unsqueeze(0).to(device)
            output_sample = query(input_sample)
            # rotate direction
            output_sample[:, 3:4] = (output_sample[:, 3:4] - dataset_adapt.affineTransform.angle/180) % 1
            # scale with
            output_sample[:, 2:3] = output_sample[:, 2:3]/dataset_adapt.affineTransform.scale
    
            output_sample = dataset_adapt.affineTransform.backward(output_sample)
            mask = dataset_adapt.affineTransform.backward(torch.ones_like(output))
            count += mask
            acc += output_sample
        output_adapt = acc / torch.maximum(count, torch.ones_like(count))
        return output_adapt        
#view_output(input, output_adapt, target)

def view_output(input,output,target):
    # normalize the output
    pred = output[:, :1]
    sdf = output[:, 1:2]
    width = output[:, 2:3]
    pred = F.sigmoid(pred)
    sdf = F.sigmoid(sdf)
    width = F.relu6(width)
    dir = output[:, 3:4]
    tile = F.sigmoid(output[:, :1])
    # dir[tile < 0.5] = 0
    # width[tile < 0.5] = 0

    tiled = tile > 0.2
    gt = target[:,:1]> 0.5

    j = 0
    img = np.transpose(input.cpu().detach()[j] / 255., (1, 2, 0))
    #vis.output_target_heat(input.detach()[j] / 255, output.detach()[j, 4], 0.3, target[j, :1].detach())
    #plt.show()
    out = [pred, sdf, width, dir, tile, tiled, gt]
    out = [o[j, 0].cpu().detach().numpy() for o in out]
    images = [img] + out
    names = ["img", "probability", "signed distance", "width", "direction", "patch prediction", "final prediction",
             "gt prediction"]
    plot_images(images, names)
    plt.savefig(f"./figures/out_resampled_{i}.jpg")


if __name__ == '__main__':
    test_set = r"./data/test/images"
    training_set = r"./data_/training"

    # Check if GPU is available
    use_cuda = torch.cuda.is_available()

    # Define the device to be used for computation
    device = torch.device("cuda" if use_cuda else "cpu")


    model, preprocess = createDeepLabv3(5, 400)
    state_dict = torch.load("out/model_best_google30.pth.tar", map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()

    # Check the current allocated memory on the default GPU (device 0)
    allocated_memory = torch.cuda.memory_allocated()
    print("Currently allocated GPU memory:", allocated_memory / 1024 / 1024, "MiB")

    for i in range(len(CustomImageDataset(training_set))):
        resample(model, training_set, i,100)
