import torch
import torch.nn.functional as F

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

def un_aggregate_tile(tensor, tile_size=16):
    return tensor.repeat_interleave(tile_size, 2).repeat_interleave(tile_size, 3)


def aggregate_tile(tensor,thresh=0.25):
    """ takes a tensor of shape b, _, h, w and aggregates per 16x16 patch"""
    b, _, h, w = tensor.shape
    patch_h = h // 16
    patch_w = w // 16

    # Reshape the tensor
    output_tensor = tensor.view(b, 1, patch_h, 16, patch_w, 16)

    # Permute the dimensions to get the desired shape
    #output_tensor = output_tensor.permute(0, 1, 2, 4, 3, 5)
    return (torch.mean(output_tensor, dim=(3, 5)) > thresh)*1.

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

def accuracy_precision_and_recall(y_true, y_pred):
    TP = torch.count_nonzero(y_true[1 == y_pred])
    accuracy = 1- torch.count_nonzero(y_true - y_pred) / torch.numel(y_true)
    recall = TP / (torch.count_nonzero(y_true) + torch.finfo(torch.float32).eps)
    precision = TP / (torch.count_nonzero(y_pred) + torch.finfo(torch.float32).eps)
    return accuracy, precision, recall

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

def rich_loss(output, target):
    BCELoss = torch.nn.BCELoss()
    MSELoss = torch.nn.MSELoss()
    pred = F.sigmoid(output[:, :1])
    sdf = F.sigmoid(output[:, 1:2])
    width = F.relu6(output[:, 2:3])
    dir = output[:, 3:4] % 1
    tiled = F.avg_pool2d(F.sigmoid(output[:, 4:5]), 16, 16, 0)

    mask_with = target[:, :1]
    mask_dir = target[:, 1:2] < 1

    return BCELoss(pred, target[:, :1]) \
           + MSELoss(sdf, target[:, 1:2]) \
           + MSELoss(width*mask_with, target[:, 2:3]) \
           + CircularMSELoss(dir*mask_dir, target[:, 3:4]) \
           + f1_loss(aggregate_tile(target[:, :1]), tiled)

def bce_loss(output, target):
    BCELoss = torch.nn.BCELoss()
    pred = F.sigmoid(output[:, :1])
    return BCELoss(pred, target[:, :1])


def nanstd(input_tensor, keepdim=False, dim=0):

    count_tensor = torch.sum(~torch.isnan(input_tensor), keepdim=keepdim, dim=dim)
    mean_tensor = torch.nanmean(input_tensor, keepdim=True, dim=dim)
    filled_tensor = torch.where(torch.isnan(input_tensor), mean_tensor, input_tensor)
    sample_variance = torch.nansum((filled_tensor - mean_tensor) ** 2, keepdim=keepdim, dim=dim)/ (count_tensor-1)
    std_tensor = torch.sqrt(sample_variance)

    return std_tensor
