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
from mask_to_submission import main
from resample import resample
from unet import get_Unet
torch.manual_seed(0)

def aggregate_tile(tensor):
    """ takes a """
    b, _, h, w = tensor.shape
    patch_h = h // 16
    patch_w = w // 16

    # Reshape the tensor
    output_tensor = tensor.view(b, 1, patch_h, 16, patch_w, 16)

    # Permute the dimensions to get the desired shape
    #output_tensor = output_tensor.permute(0, 1, 2, 4, 3, 5)
    return torch.mean(output_tensor, dim=(3, 5)) > 0.25




if __name__ == '__main__':
    test_set = r"./data_/test"

    # Check if GPU is available
    use_cuda = torch.cuda.is_available()

    # Define the device to be used for computation
    device = torch.device("cuda" if use_cuda else "cpu")
    dataset = CustomImageDataset(test_set, False)
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    model, preprocess = get_Unet(1, 400)
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
        output = resample(query,test_set,i)
        # normalize the output
        output = F.sigmoid(output[:,:1])
        pred = (255* (output>0.5)).detach().cpu().numpy().astype(np.uint8)
        j = 0
            # vis.output_target_heat(input.detach()[j] / 255, output.detach()[j, 1], 0.3, None)
            # plt.imshow(output[j, 1].detach().cpu().numpy())
            # plt.show()
        img_name = os.path.basename(image_filenames[j])
        print(img_name)
        cv2.imwrite(f"{store_folder}/mask_{img_name}", pred[j, 0])

    # now all masks are stored, convert it to the csv file
    # by running mask_to_submission.py
