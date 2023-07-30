import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from dataset import CustomImageDataset
from decoder import decoder, quantile_aggregate_tile
from models.deeplabv3 import createDeepLabv3
from resample import resample_output
from utils.utils import nanstd, quantile_aggregate_tile, un_aggregate_tile

if __name__ == '__main__':
    """
    This script is used to generate the submission file for the Kaggle competition. 
    """
    test_set = r"./data/test"

    # Define the device to be used for computation
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Create the dataset and data loader
    dataset = CustomImageDataset(test_set, False,color_aug=False, geo_aug=False)
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    # Load the best model
    model, preprocess, postprocess = createDeepLabv3(1, 400)
    state_dict = torch.load("out/model_best.pth.tar", map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    num_patches_per_image = (400 // 16) ** 2
    num_ticks = 11
    num_features = num_ticks * 4
    classifier = decoder(num_features)
    state_dict = torch.load("out/best_classifier.pth.tar", map_location=torch.device("cpu"))
    classifier.load_state_dict(state_dict)
    classifier = classifier.to(device)
    classifier.eval()

    # Define the query function
    query = lambda input : postprocess(model(preprocess(input)))
    store_folder = "out/prediction"
    os.makedirs(store_folder, exist_ok=True)

    # Iterate over the data in the validation set and compute the IoU score
    for i, (input, image_filenames) in enumerate(val_loader):
        # Move input and target tensors to the device (CPU or GPU)
        output_samples, output_masks = resample_output(query, test_set, i, 20)

        # Prepare the output
        output_i = F.sigmoid(output_samples[:1])
        output_samples[output_masks == 0] = torch.nan
        output_mode, _ = F.sigmoid(output_samples).nanmedian(keepdim=True, dim=0)
        output_mean = F.sigmoid(output_samples).nanmean(keepdim=True, dim=0)
        output_std = nanstd(F.sigmoid(output_samples), keepdim=True, dim=0)
        X = torch.zeros((num_patches_per_image,num_features))

        # Compute the aggregation quantiles for each output
        for j, output in enumerate([output_i, output_mean, output_mode, output_std]):
            quantiles = quantile_aggregate_tile(output, num_ticks)
            quantiles = torch.flatten(quantiles, start_dim=1, end_dim=4).T
            X[:, num_ticks * j:num_ticks * (j + 1)] = quantiles

        # Undo the aggregation
        agg = classifier(X.to(device))
        agg = agg.reshape(1,1,25, 25)
        output = un_aggregate_tile(agg)
        
        # Normalize the output store the image
        pred = (255*(output > 0.5)).detach().cpu().numpy().astype(np.uint8)
        j = 0
        # vis.output_target_heat(input.detach()[j] / 255, output.detach()[j, 1], 0.3, None)
        # plt.imshow(output[j, 1].detach().cpu().numpy())
        # plt.show()
        img_name = os.path.basename(image_filenames[j])
        print(img_name)
        cv2.imwrite(f"{store_folder}/mask_{img_name}", pred[j, 0])

    # now that all masks are stored, convert it to the csv file
    # by running mask_to_submission.py
