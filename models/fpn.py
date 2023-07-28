import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


def get_fpn(outputchannels=1, input_size=512):
    """
    Returns a segmentation model with a FPN encoder.
    """
    encoder_name = 'efficientnet-b7'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = None # could be None for logits or 'softmax2d' for multiclass segmentation

    # create segmentation model with pretrained encoder
    model = smp.FPN(
        encoder_name=encoder_name,
        encoder_weights=ENCODER_WEIGHTS,
        classes=outputchannels,
        activation=ACTIVATION,
    )

    # set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # set the model to train mode
    model.train()

    # define the preprocessing function
    pad = nn.ReflectionPad2d((input_size%32)//2)
    def pre(tensor):
        if torch.cuda.is_available():
            tensor = tensor.type('torch.cuda.FloatTensor')
        else:
            tensor = tensor.type('torch.FloatTensor')

        tensor = tensor /255.
        tensor -= torch.tensor([[[0.485]], [[0.456]], [[0.406]]]).to(device)
        tensor /= torch.tensor([[[0.229]], [[0.224]], [[0.225]]]).to(device)
        tensor = pad(tensor)
        return tensor

    # return the model, preprocessing function, and postprocessing function
    return model, pre, lambda x:TF.center_crop(x, input_size)
