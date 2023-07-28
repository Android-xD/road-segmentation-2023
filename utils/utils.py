import torch
import torch.nn.functional as F


def quantile_aggregate_tile(tensor, n_quantiles=25, tile_size=16):
    """ 
    Aggregates an image tensor into patches of specified size and computes quantile values for each patch.

    Args:
        tensor (torch.Tensor): Input image tensor of shape (b, c, h, w), where
                               b = batch size, c = channels, 
                               h = height, and w = width of the image.
        n_quantiles (int): The number of quantiles to compute for each patch. 
                           Default is 25.
        tile_size (int): The size of the square patches to aggregate over. 
                         Default is 16.

    Returns:
        torch.Tensor: A tensor containing quantile values for each patch. 
                      The shape of the output tensor is (n_quantiles, b, 1, patch_h, patch_w),
                      where patch_h and patch_w are the number of patches in the height 
                      and width directions, respectively.
    """
    # Extract the dimensions of the input tensor
    b, _, h, w = tensor.shape

    # Compute the number of patches in the height and width directions
    patch_h = h // tile_size
    patch_w = w // tile_size

    # Reshape the tensor into patches of size tile_size x tile_size
    output_tensor = tensor.view(b, 1, patch_h, tile_size, patch_w, tile_size)

    # Permute the dimensions to get the desired shape of b x 1 x patch_h x patch_w x tile_size * tile_size
    output_tensor = output_tensor.permute(0, 1, 2, 4, 3, 5)
    output_tensor = output_tensor.reshape(b, 1, patch_h, patch_w, tile_size * tile_size)

    # Compute the quantiles
    quantiles = torch.quantile(output_tensor, torch.linspace(0, 1, n_quantiles), dim=4) * 1.
    
    # shape n_quantiles,b,1,25,25
    return quantiles


def un_aggregate_tile(tensor, tile_size=16):
    """
    Un-aggregates a tensor for each patch back to the original image size.

    Args:
        tensor (torch.Tensor): Input image tensor of shape (b, 1, patch_h, patch_w, tile_size * tile_size), where
                               b = batch size, 1 = channels,
                               patch_h and patch_w = number of patches in the height and width directions, 
                               tile_size = size of the square patches used for aggregation
        tile_size (int): The size of the square patches used for aggregation.
                         Default is 16.

    Returns:
        torch.Tensor: The un-aggregated tensor of shape (b, 1, patch_h * tile_size, patch_w * tile_size),
                      representing the tensor expanded back to the original image size.
    """
    return tensor.repeat_interleave(tile_size, 2).repeat_interleave(tile_size, 3)


def aggregate_tile(tensor, thresh=0.25, tile_size=16):
    """
    Aggregates an image tensor into patches of a specified size and computes binary values for each patch.

    Args:
        tensor (torch.Tensor): The input image tensor with shape (b, c, h, w), where
                               b = batch size, c = number of channels,
                               h = height, and w = width of the image.
        thresh (float, optional): The threshold value for the binary computation.
                                  If the mean value of a patch is greater than this threshold, the binary value is 1,
                                  otherwise, it is 0. Default is 0.25.
        tile_size (int, optional): The size of the square patches to aggregate over. Default is 16.

    Returns:
        torch.Tensor: A binary tensor of shape (b, 1, patch_h, patch_w),
                      representing the aggregated binary values for each patch.
    """   
    # Extract the dimensions of the input tensor
    b, _, h, w = tensor.shape

    # Compute the number of patches in the height and width directions
    patch_h = h // tile_size
    patch_w = w // tile_size

    # Reshape the tensor into patches of size tile_size x tile_size
    output_tensor = tensor.view(b, 1, patch_h, tile_size, patch_w, tile_size)

    # Compute the mean value for each patch along dimensions 3 (height) and 5 (width)
    # Compare the mean values with the threshold and convert to binary values (0 or 1)
    return (torch.mean(output_tensor, dim=(3, 5)) > thresh)*1.


def f1_score(y_true, y_pred):
    """
    Calculates the F1 score for binary classification.

    Args:
        y_true (torch.Tensor): The true binary labels (ground truth) of shape (N,).
        y_pred (torch.Tensor): The predicted binary labels of shape (N,).

    Returns:
        float: The F1 score value.
    """
    # Calculate the precision, and recall
    _, precision, recall = accuracy_precision_and_recall(y_true, y_pred)

    # Calculate the F1 score
    return 2 * (precision * recall) / (precision + recall + torch.finfo(torch.float32).eps)


def accuracy_precision_and_recall(y_true, y_pred):
    """
    Calculates Accuracy, Precision, and Recall for binary classification.

    Args:
        y_true (torch.Tensor): The true binary labels (ground truth) of shape (N,).
        y_pred (torch.Tensor): The predicted binary labels of shape (N,).

    Returns:
        tuple: A tuple containing the calculated values of Accuracy, Precision, and Recall.
    """
    # Count the True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN)
    TP = torch.count_nonzero(y_true[1 == y_pred])
    TN = torch.count_nonzero(y_true[0 == y_pred])
    FP = torch.count_nonzero(y_true[0 == y_pred] - y_pred[0 == y_pred])
    FN = torch.count_nonzero(y_true[1 == y_pred] - y_pred[1 == y_pred])

    # Calculate Accuracy, Recall, and Precision
    accuracy = (TP + TN) / (TP + TN + FP + FN + torch.finfo(torch.float32).eps)
    recall = TP / (TP + FN + torch.finfo(torch.float32).eps)
    precision = TP / (TP + FP + torch.finfo(torch.float32).eps)

    return accuracy, precision, recall


def f1_loss(y_true, y_pred):
    """
    Calculates the F1 loss for binary classification.

    Args:
        y_true (torch.Tensor): The true binary labels (ground truth) of shape (N,).
        y_pred (torch.Tensor): The predicted binary labels of shape (N,).

    Returns:
        float: The F1 loss value.
    """
    # Calculate the F1 score
    f1 = f1_score(y_true, y_pred)
    
    # Calculate the F1 loss
    return 1 - torch.mean(f1)


def CircularMSELoss(y_true, y_pred):
    """
    Computes the Circular Mean Squared Error (MSE) Loss for circular regression tasks.

    Args:
        y_true (torch.Tensor): The true circular values (ground truth) of shape (N,).
        y_pred (torch.Tensor): The predicted circular values of shape (N,).

    Returns:
        torch.Tensor: The circular MSE loss value.
    """
    # Compute circular difference between predicted and target values
    diff = torch.abs(y_pred - y_true)
    circular_diff = torch.minimum(diff, 1 - diff)

    # Compute mean squared error loss
    loss = torch.mean(circular_diff**2)
    return loss


def rich_loss(output, target):
    """
    Computes the composite loss function for training a model with rich output.

    Args:
        output (torch.Tensor): The predicted output tensor of shape (N, num_properties),
                               where N is the number of samples and num_properties
                               is the total number of properties predicted by the model.
        target (torch.Tensor): The target tensor with ground truth of shape (N, num_properties).

    Returns:
        torch.Tensor: The composite loss value combining multiple loss components.
    """
    BCELoss = torch.nn.BCELoss()
    MSELoss = torch.nn.MSELoss()

    # Extract predicted values for different properties
    pred = F.sigmoid(output[:, :1])
    sdf = F.sigmoid(output[:, 1:2])
    width = F.relu6(output[:, 2:3])
    dir = output[:, 3:4] % 1
    tiled = F.avg_pool2d(F.sigmoid(output[:, 4:5]), 16, 16, 0)

    # Extract target values for different properties
    mask_with = target[:, :1]
    mask_dir = target[:, 1:2] < 1

    # Compute individual loss components and combine them
    return BCELoss(pred, target[:, :1]) \
           + MSELoss(sdf, target[:, 1:2]) \
           + MSELoss(width*mask_with, target[:, 2:3]) \
           + CircularMSELoss(dir*mask_dir, target[:, 3:4]) \
           + f1_loss(aggregate_tile(target[:, :1]), tiled)


def bce_loss(output, target):
    """
    Computes the Binary Cross Entropy (BCE) loss for binary classification.

    Args:
        output (torch.Tensor): The predicted output tensor of shape (N, num_properties),
                               where N is the number of samples and num_properties
                               is the total number of properties predicted by the model.
        target (torch.Tensor): The target tensor with ground truth of shape (N, num_properties).

    Returns:
        torch.Tensor: The BCE loss value for binary classification.
    """
    BCELoss = torch.nn.BCELoss()

    # Extract predicted values for binary classification
    pred = F.sigmoid(output[:, :1])

    # Compute Binary Cross Entropy loss
    return BCELoss(pred, target[:, :1])


def nanstd(input_tensor, keepdim=False, dim=0):
    """
    Calculates the standard deviation of the input tensor

    Args:
        input_tensor (torch.Tensor): The input tensor for which standard deviation needs to be calculated.
        keepdim (bool, optional): Whether to keep the dimensions of the output tensor as the input tensor. Default is False.
        dim (int or tuple of ints, optional): The dimensions along which the standard deviation is computed. Default is 0.

    Returns:
        torch.Tensor: The standard deviation tensor computed across the specified dimensions.
    """
    # Count the number of non-NaN elements in the input tensor along the specified dimensions
    count_tensor = torch.sum(~torch.isnan(input_tensor), keepdim=keepdim, dim=dim)

    # Compute the mean of the input tensor, ignoring NaN values
    mean_tensor = torch.nanmean(input_tensor, keepdim=True, dim=dim)

    # Replace NaN values in the input tensor with the corresponding mean value
    filled_tensor = torch.where(torch.isnan(input_tensor), mean_tensor, input_tensor)

    # Compute the sample variance, ignoring NaN values
    sample_variance = torch.nansum((filled_tensor - mean_tensor) ** 2, keepdim=keepdim, dim=dim)/ (count_tensor-1)
    
    # Compute the standard deviation as the square root of the sample variance
    std_tensor = torch.sqrt(sample_variance)

    return std_tensor