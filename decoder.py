import cv2
import os
import matplotlib.pyplot as plt
from dataset import CustomImageDataset, test_train_split
import numpy as np
import torch
import visualize as vis
import torchvision.transforms as T
from deeplabv3 import createDeepLabv3, load_model
from sklearn.metrics import f1_score, accuracy_score
import torch.nn.functional as F
from mask_to_submission import main
from utils import aggregate_tile
from torch.autograd import Variable
import torch.nn.functional as func
from utils import f1_score, f1_loss
from resample import resample

torch.manual_seed(0)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def quantile_tile(tensor, n_quantiles=25, tile_size=16):
    """ takes a tensor of shape b, _, h, w and aggregates per 16x16 patch"""
    b, _, h, w = tensor.shape
    patch_h = h // tile_size
    patch_w = w // tile_size

    # Reshape the tensor
    output_tensor = tensor.view(b, 1, patch_h, tile_size, patch_w, tile_size)

    # Permute the dimensions to get the desired shape
    output_tensor = output_tensor.permute(0, 1, 2, 4, 3, 5)
    output_tensor = output_tensor.reshape(b, 1, patch_h, patch_w, tile_size * tile_size)
    quantiles = torch.quantile(output_tensor, torch.linspace(0, 1, n_quantiles).to(device), dim=4) * 1.
    # shape n_quantiles, b,1,25,25
    return quantiles

def balance(x,y):

    # Assuming your data is stored in 'x' and 'y' as PyTorch tensors
    positive_indices = torch.where(y == 1)[0]
    negative_indices = torch.where(y == 0)[0]

    positive_samples = x[positive_indices]
    negative_samples = x[negative_indices]

    num_samples = min(len(positive_samples), len(negative_samples))

    # Randomly select balanced samples
    positive_balanced_indices = torch.randperm(len(positive_samples))[:num_samples]
    negative_balanced_indices = torch.randperm(len(negative_samples))[:num_samples]

    balanced_indices = torch.cat(
        [positive_indices[positive_balanced_indices], negative_indices[negative_balanced_indices]])

    balanced_x = x[balanced_indices]
    balanced_y = y[balanced_indices]
    return balanced_x, balanced_y


class decoder(torch.nn.Module):

    def __init__(self, num_inputs):
        super(decoder, self).__init__()
        self.linear1 = torch.nn.Linear(num_inputs, 512)
        self.linear2 = torch.nn.Linear(512, 512)
        self.linear3 = torch.nn.Linear(512, 1)

    def forward(self, x):
        x = (func.relu(self.linear1(x)))
        x = (func.relu(self.linear2(x)))
        return (func.sigmoid(self.linear3(x)))


if __name__ == '__main__':
    test_set = r"data_/training"
    output_dir = "out"

    # Check if GPU is available
    use_cuda = torch.cuda.is_available()

    # Define the device to be used for computation
    device = torch.device("cuda" if use_cuda else "cpu")
    dataset = CustomImageDataset(test_set, True, False)
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    
    model, preprocess = createDeepLabv3(5, 400)
    state_dict = torch.load("out/model_best.pth.tar", map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    store_folder = "out/prediction"
    os.makedirs(store_folder, exist_ok=True)

    classifier = decoder(11)
    calssifier = classifier.to(device)
    loss = torch.nn.BCELoss()  # note that CrossEntropyLoss is for targets with more than 2 classes.
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.0005)
    query = lambda input: model(preprocess(input))['out']
    for e in range(10):
        for i, (input, target) in enumerate(val_loader):
            # Move input and target tensors to the device (CPU or GPU)
            input = input.to(device)
            target = target.to(device)
            input = input.squeeze()

            output = model(preprocess(input))['out']
            # normalize the output
            output = F.sigmoid(output[:, :1])
            quantiles = quantile_tile(output,11)
            quantiles = torch.flatten(quantiles, start_dim=1, end_dim=4)

            agg = aggregate_tile(target[:, :1], 0.25)
            agg = torch.flatten(agg, start_dim=0, end_dim=3).reshape(-1, 1)

        # agg = agg.detach().numpy()
        # quantiles = quantiles.detach().numpy()
        # plt.plot(quantiles[:, agg == 1], np.linspace(0, 1, 25), 'b', alpha=0.3)
        # plt.plot(quantiles[:, agg == 0], np.linspace(0, 1, 25), 'r', alpha=0.3)
        # plt.xlabel("quantile")
        # plt.ylabel("probability")
        # plt.show()
        # print(quantiles.shape, agg.shape)
        # plt.imshow(output[0, 0].detach().numpy())
        # plt.show()
            x_data = Variable(quantiles.T, requires_grad=False)
            y_data = Variable(agg, requires_grad=False)
            x_b, y_b = (x_data, y_data)
            baseline_pred = (torch.mean((x_b>0.25)*1.,dim=1)>0.25)*1.
            y = classifier(x_b)

            #l = loss(y, t_data)
            l = f1_loss(y_b, y)
            baseline_pred = (torch.mean((x_b>0.25)*1.,dim=1)>0.25)*1.
            learned_pred = (y > 0.5)*1.
            f1_learned = f1_score(y_b,learned_pred)
            f1_thresh = f1_score(y_b, baseline_pred) 
            print(f"itr:\t{i}\tloss:\t{l.item():.4f},\t pred:\t{f1_learned:.4f},\t baseline:\t{f1_thresh:.4f}\t impr:{f1_learned-f1_thresh:.4f}")
            optimizer.zero_grad()

            l.backward()

            optimizer.step()

    torch.save(classifier.state_dict(), os.path.join(output_dir, f'best_classifier.pth.tar'))


