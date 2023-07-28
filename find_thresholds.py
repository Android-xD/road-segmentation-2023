import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from dataset import CustomImageDataset
from models.deeplabv3 import createDeepLabv3
from resample import resample
from utils.utils import accuracy_precision_and_recall, aggregate_tile
from utils.visualize import plot_images

torch.manual_seed(0)

def view_output(input, output, target):
    # normalize the output
    pred = output[:, :1]
    sdf = output[:, 1:2]
    width = output[:, 2:3]
    pred = F.sigmoid(pred)
    sdf = F.sigmoid(sdf)
    width = F.relu6(width)
    dir = output[:, 3:4]
    tile = F.sigmoid(output[:, :1])

    tiled = tile > 0.2
    gt = target[:,:1]> 0.5

    j = 0
    img = np.transpose(input.cpu().detach()[j] / 255., (1, 2, 0))
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

    model, preprocess, postprocess = createDeepLabv3(5, 400)
    state_dict = torch.load("out/model_best.pth.tar", map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()

    query = lambda input : postprocess(model(preprocess(input)))
    dataset = CustomImageDataset(training_set, geo_aug=False,color_aug=False, train=True)
    m = 10 #len(dataset)

    n_ticks = 101
    recall_space_16 = torch.zeros((m, n_ticks, n_ticks))
    precision_space_16 = torch.zeros((m, n_ticks, n_ticks))
    ticks = np.linspace(0, 1, n_ticks)

    for i in range(m):
        input, target = dataset[i]
        input = input.unsqueeze(0)
        target = target.unsqueeze(0)

        output = resample(query, training_set, i, 1)
        y_pred = F.sigmoid(output[:, :1])
        agg_target = aggregate_tile(target[:, :1])
        for r, th1 in enumerate(ticks):
            for c, th2 in enumerate(ticks):
                _, precision_space_16[i, r, c], recall_space_16[i, r, c] = \
                    accuracy_precision_and_recall(agg_target, aggregate_tile((y_pred > th1) * 1.0, thresh=th2))

        f1_space = 2. / (1 / torch.mean(recall_space_16[:i+1], dim=0) + 1 / torch.mean(precision_space_16[:i+1], dim=0))
        f1_space_cur = 2. / (
                    1 / recall_space_16[i] + 1 /precision_space_16[i])
        print(torch.max(f1_space), torch.max(f1_space_cur))

    plt.imshow(f1_space, cmap='jet')
    plt.colorbar()
    num_ticks = 11
    tick_locations = np.linspace(0, 1, num_ticks)
    tick_labels = ["{:.1f}".format(tick) for tick in tick_locations]
    plt.xlabel("Per Patch Threshold")
    plt.ylabel("Per Pixel Threshold")
    plt.xticks(tick_locations * (n_ticks-1), tick_labels)
    plt.yticks(tick_locations * (n_ticks-1), tick_labels)
    store_figures = r"./figures"
    plt.savefig(f"{store_figures}/thresholds.png")