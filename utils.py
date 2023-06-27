import torch


def aggregate_tile(tensor):
    """ takes a tensor of shape b, _, h, w and aggregates per 16x16 patch"""
    b, _, h, w = tensor.shape
    patch_h = h // 16
    patch_w = w // 16

    # Reshape the tensor
    output_tensor = tensor.view(b, 1, patch_h, 16, patch_w, 16)

    # Permute the dimensions to get the desired shape
    #output_tensor = output_tensor.permute(0, 1, 2, 4, 3, 5)
    return torch.mean(output_tensor, dim=(3, 5)) > 0.25


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
