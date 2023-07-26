import torch
import numpy as np
import segmentation_models_pytorch as smp



def get_resnext(outputchannels=1, input_size=512):
    ENCODER = 'se_resnext50_32x4d'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation

    # create segmentation model with pretrained encoder
    model = smp.FPN(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=outputchannels,
        activation=ACTIVATION,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    return model, preprocessing_fn, lambda x:x

def get_fpn(outputchannels=1, input_size=512):
    encoder_name = 'efficientnet-b7',
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = None # could be None for logits or 'softmax2d' for multiclass segmentation
    # create segmentation model with pretrained encoder
    model = smp.FPN(
        encoder_name=encoder_name,
        encoder_weights=ENCODER_WEIGHTS,
        classes=outputchannels,
        activation=ACTIVATION,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    def pre(tensor):
        if torch.cuda.is_available():
            tensor = tensor.type('torch.cuda.FloatTensor')
        else:
            tensor = tensor.type('torch.FloatTensor')

        tensor = tensor /255.
        tensor -= torch.tensor([[[0.485]], [[0.456]], [[0.406]]]).to(device)
        tensor /= torch.tensor([[[0.229]], [[0.224]], [[0.225]]]).to(device)
        return tensor

    return model, pre, lambda x:x
