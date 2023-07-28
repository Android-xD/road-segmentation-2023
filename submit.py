import cv2
import os
import matplotlib.pyplot as plt
from dataset import CustomImageDataset
import numpy as np
import torch
import utils.visualize as vis
import torchvision.transforms as T
from models.deeplabv3 import createDeepLabv3,load_model
from sklearn.metrics import f1_score, accuracy_score
import torch.nn.functional as F
from mask_to_submission import main
from resample import resample
from decoder import decoder, quantile_aggregate_tile
from utils.utils import un_aggregate_tile
from models.unet_backbone import get_Unet

torch.manual_seed(0)

if __name__ == '__main__':
    test_set = r"./data/test"

    # Check if GPU is available
    use_cuda = torch.cuda.is_available()

    # Define the device to be used for computation
    device = torch.device("cuda" if use_cuda else "cpu")
    dataset = CustomImageDataset(test_set, False, False, False, False)
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    model, preprocess, _ = get_Unet(1, 400)
    state_dict = torch.load("out/model_best.pth.tar", map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    query = lambda input : model(preprocess(input))['out']
    store_folder = "out/prediction"
    os.makedirs(store_folder, exist_ok=True)
    for i, (input, image_filenames) in enumerate(val_loader):
        # Move input and target tensors to the device (CPU or GPU)
        input = input.to(device)
        input = input.squeeze()
        # output = query(input.unsqueeze(0)) without resample
        output = resample(query, test_set, i, 50)
        # normalize the output
        output = F.sigmoid(output[:,:1])
        pred = (255*(output > 0.35)).detach().cpu().numpy().astype(np.uint8)
        j = 0
            # vis.output_target_heat(input.detach()[j] / 255, output.detach()[j, 1], 0.3, None)
            # plt.imshow(output[j, 1].detach().cpu().numpy())
            # plt.show()
        img_name = os.path.basename(image_filenames[j])
        print(img_name)
        cv2.imwrite(f"{store_folder}/mask_{img_name}", pred[j, 0])

    # now all masks are stored, convert it to the csv file
    # by running mask_to_submission.py
