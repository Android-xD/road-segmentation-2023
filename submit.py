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
from mask_to_submission import main

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
    test_set = r"data/test"

    # Check if GPU is available
    use_cuda = torch.cuda.is_available()

    # Define the device to be used for computation
    device = torch.device("cuda" if use_cuda else "cpu")
    dataset = CustomImageDataset(test_set, False)
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    model, preprocess = createDeepLabv3(2, 400)
    state_dict = torch.load("out/model_best.pth.tar", map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)

    for i, (input, image_filenames) in enumerate(val_loader):
        # Move input and target tensors to the device (CPU or GPU)
        input = input.to(device)
        input = input.squeeze()
        output = model(preprocess(input))['out']
        # normalize the output
        output = F.softmax(output)
        pred = (255*(output[:, 1:2] > 0.2)).detach().cpu().numpy().astype(np.uint8)
        for j in range(input.shape[0]):
            # vis.output_target_heat(input.detach()[j] / 255, output.detach()[j, 1], 0.3, None)
            # plt.imshow(output[j, 1].detach().cpu().numpy())
            # plt.show()
            img_name = os.path.basename(image_filenames[j])
            print(img_name)
            cv2.imwrite(f"out/prediction/mask_{img_name}", pred[j, 0])

    # now all masks are stored, convert it to the csv file
    main(None)