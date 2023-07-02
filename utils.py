import torch
import numpy as np
from scipy.interpolate import griddata

def aggregate_tile(tensor, thresh=0.25, tile_size=16):
    """ takes a tensor of shape b, _, h, w and aggregates per 16x16 patch"""
    b, _, h, w = tensor.shape
    patch_h = h // tile_size
    patch_w = w // tile_size

    # Reshape the tensor
    output_tensor = tensor.view(b, 1, patch_h, tile_size, patch_w, tile_size)

    # Permute the dimensions to get the desired shape
    #output_tensor = output_tensor.permute(0, 1, 2, 4, 3, 5)
    return (torch.mean(output_tensor, dim=(3, 5)) > thresh)*1.

def un_aggregate_tile(tensor, tile_size=16):
    return tensor.repeat_interleave(tile_size, 2).repeat_interleave(tile_size, 3)

def f1_score(y_true, y_pred):
    TP = torch.count_nonzero(y_true[1 == y_pred])
    recall = TP / (torch.count_nonzero(y_true) + torch.finfo(torch.float32).eps)
    precision = TP / (torch.count_nonzero(y_pred) + torch.finfo(torch.float32).eps)
    return 2. / (1 / recall + 1 / precision)


def f1(y_true, y_pred):
    y_pred = torch.round(y_pred)
    tp = torch.sum(y_true * y_pred, axis=0)
    tn = torch.sum((1 - y_true) * (1 - y_pred), axis=0)
    fp = torch.sum((1 - y_true) * y_pred, axis=0)
    fn = torch.sum(y_true * (1 - y_pred), axis=0)

    p = tp / (tp + fp + torch.finfo(torch.float32).eps)
    r = tp / (tp + fn + torch.finfo(torch.float32).eps)

    f1 = 2 * p * r / (p + r + torch.finfo(torch.float32).eps)
    f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)
    return torch.mean(f1)

def f1_loss(y_true, y_pred):
    tp = torch.sum(y_true * y_pred, axis=0)
    tn = torch.sum((1 - y_true) * (1 - y_pred), axis=0)
    fp = torch.sum((1 - y_true) * y_pred, axis=0)
    fn = torch.sum(y_true * (1 - y_pred), axis=0)

    p = tp / (tp + fp + torch.finfo(torch.float32).eps)
    r = tp / (tp + fn + torch.finfo(torch.float32).eps)

    f1 = 2 * p * r / (p + r + torch.finfo(torch.float32).eps)
    f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)
    return 1 - torch.mean(f1)

def CircularMSELoss(y_true, y_pred):
    # Compute circular difference between predicted and target values
    diff = torch.abs(y_pred - y_true)
    circular_diff = torch.minimum(diff, 1 - diff)

    # Compute mean squared error loss
    loss = torch.mean(circular_diff**2)
    return loss

def flow_warp(angle, label):
    """angle in range  (0,1)"""
    angles = -angle * np.pi
    X, Y = np.meshgrid(np.arange(0, 400), np.arange(0, 400))
    points = np.column_stack((X.ravel(), Y.ravel()))
    angle = -angles * np.pi

    # Calculate U and V components from the vector field data
    V = 2 * np.cos(angle)  # x-component of vectors
    U = 2 * np.sin(angle)  # y-component of vectors

    # interpolate the error
    xi = np.arange(0, 400)
    yi = np.arange(0, 400)
    grid_x, grid_y = np.meshgrid(xi, yi)

    out = griddata(
        (grid_x.ravel(), grid_y.ravel()),
        label.ravel(),
        (grid_x + U, grid_y + V),
        method="linear",
    )
    return out


