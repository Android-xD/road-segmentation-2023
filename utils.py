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