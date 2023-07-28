import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset import CustomImageDataset
from models.deeplabv3 import createDeepLabv3
from utils.utils import nanstd
from utils.visualize import plot_images

torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
# Define the device to be used for computation
device = torch.device("cuda" if use_cuda else "cpu")

def resample_output(query, path, index, n_samples=20, rich =True):
    with torch.no_grad():
        dataset = CustomImageDataset(path, train=False, geo_aug=False, color_aug=False)
        dataset_adapt = CustomImageDataset(path, train=False, geo_aug=True, color_aug=True)
        if rich:
            dataset_adapt.affineTransform.hfilp_prob = 0
            dataset_adapt.affineTransform.vfilp_prob = 0

        input, target = dataset[index]
        input = input.unsqueeze(0).to(device)
        output = query(input)

        # target = target.unsqueeze(0)
        # target = target.to(device)

        output_samples = torch.zeros((n_samples, *output.shape[1:]))
        output_masks = torch.zeros((n_samples, *output.shape[1:]))

        output_samples[0] = output
        output_masks[0] = torch.ones_like(output)
        for j in tqdm(range(1, n_samples)):
            input_sample, _ = dataset_adapt[index]
            # Move input and target tensors to the device (CPU or GPU)
            input_sample = input_sample.unsqueeze(0).to(device)
            output_sample = query(input_sample)
            if rich:
                # rotate direction
                output_sample[:, 3:4] = (output_sample[:, 3:4] - dataset_adapt.affineTransform.angle / 180) % 1
                # scale with
                output_sample[:, 2:3] = output_sample[:, 2:3] / dataset_adapt.affineTransform.scale

            output_samples[j] = dataset_adapt.affineTransform.backward(output_sample)
            output_masks[j] = torch.round(dataset_adapt.affineTransform.backward(torch.ones_like(output)))

        return output_samples, output_masks

def resample(query, path, index, n_samples=20, rich =True):
    output_samples, output_masks = resample_output(query, path, index, n_samples, rich)
    count = torch.sum(output_masks, dim=0, keepdim=True)
    output_adapt = torch.sum(output_samples, dim=0, keepdim=True) / count
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
    training_set = r"./data/training"

    # Check if GPU is available
    use_cuda = torch.cuda.is_available()

    # Define the device to be used for computation
    device = torch.device("cuda" if use_cuda else "cpu")


    model, preprocess, _ = createDeepLabv3(5, 400)

    state_dict = torch.load("out/model_best.pth.tar", map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()

    dataset = CustomImageDataset(training_set, train=False, color_aug=False, geo_aug=False)
    query = lambda input : model(preprocess(input))['out']
    for i in range(len(dataset)):
        output = F.sigmoid(resample(query, training_set, i, 1))
        output_samples, output_masks = resample_output(query, training_set, i, 10)
        output_samples[output_masks == 0] = torch.nan
        output_mode, _ = F.sigmoid(output_samples).nanmedian(keepdim=True, dim=0)
        output_mean = F.sigmoid(output_samples).nanmean(keepdim=True, dim=0)
        output_std = nanstd(F.sigmoid(output_samples), keepdim=True, dim=0)
        img = np.transpose(dataset[i][0].cpu().detach() / 255., (1, 2, 0))
        out = [output[0, 0], output_mean[0, 0], output_mode[0, 0], output_std[0, 0]] #, sdf, width, dir, tile]
        out = [o.cpu().detach().numpy() for o in out]
        images = [img]+out
        names = ["Input Image", "Prediction", "Mean Prediction TTA", "Median Prediction TTA", "Standard Deviation TTA"]
        plot_images(images, names, hpad=1.)
        plt.savefig(f"./figures/resample_{i}.png")