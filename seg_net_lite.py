import torch.nn as nn
import torch

class SegNetLite(nn.Module):

    def __init__(self, kernel_sizes=[3, 3, 3, 3], down_filter_sizes=[32, 64, 128, 256],
            up_filter_sizes=[128, 64, 32, 32], conv_paddings=[1, 1, 1, 1],
            pooling_kernel_sizes=[2, 2, 2, 2], pooling_strides=[2, 2, 2, 2], **kwargs):
        """Initialize SegNet Module

        Args:
            kernel_sizes (list of ints): kernel sizes for each convolutional layer in downsample/upsample path.
            down_filter_sizes (list of ints): number of filters (out channels) of each convolutional layer in the downsample path.
            up_filter_sizes (list of ints): number of filters (out channels) of each convolutional layer in the upsample path.
            conv_paddings (list of ints): paddings for each convolutional layer in downsample/upsample path.
            pooling_kernel_sizes (list of ints): kernel sizes for each max-pooling layer and its max-unpooling layer.
            pooling_strides (list of ints): strides for each max-pooling layer and its max-unpooling layer.
        """
        super(SegNetLite, self).__init__()
        self.num_down_layers = len(kernel_sizes)
        self.num_up_layers = len(kernel_sizes)

        input_size = 3 # initial number of input channels
        # Construct downsampling layers.
        # As mentioned in the assignment, blocks of the downsampling path should have the
        # following output dimension (igoring batch dimension):
        # 3 x 64 x 64 (input) -> 32 x 32 x 32 -> 64 x 16 x 16 -> 128 x 8 x 8 -> 256 x 4 x 4
        # each block should consist of: Conv2d->BatchNorm2d->ReLU->MaxPool2d
        layers_conv_down = []
        layers_bn_down = []
        layers_pooling = []
        # kernel_sizes=[3, 3, 3, 3],
        # down_filter_sizes=[32, 64, 128, 256],
        # up_filter_sizes=[128, 64, 32, 32],
        # conv_paddings=[1, 1, 1, 1],
        # pooling_kernel_sizes=[2, 2, 2, 2],
        # pooling_strides=[2, 2, 2, 2]
        down_filter_sizes+=[input_size]
        for i in range(len(down_filter_sizes)-1):
            layers_conv_down += [nn.Conv2d(down_filter_sizes[i-1], down_filter_sizes[i], kernel_size=kernel_sizes[i], padding=conv_paddings[i])]
            layers_bn_down += [nn.BatchNorm2d(down_filter_sizes[i])]
            layers_pooling += [nn.MaxPool2d(kernel_size=pooling_kernel_sizes[i], stride=pooling_strides[i],return_indices=True)]
            self.__setattr__(f"conv_block_down_{i+1}", nn.Sequential(
                layers_conv_down[i],
                layers_bn_down[i],
                nn.ReLU(),
            )
                             )


        # Convert Python list to nn.ModuleList, so that PyTorch's autograd
        # package can track gradients and update parameters of these layers
        self.layers_conv_down = nn.ModuleList(layers_conv_down)
        self.layers_bn_down = nn.ModuleList(layers_bn_down)
        self.layers_pooling = nn.ModuleList(layers_pooling)

        # Construct upsampling layers
        # As mentioned in the assignment, blocks of the upsampling path should have the
        # following output dimension (igoring batch dimension):
        # 256 x 4 x 4 (input) -> 128 x 8 x 8 -> 64 x 16 x 16 -> 32 x 32 x 32 -> 32 x 64 x 64
        # each block should consist of: MaxUnpool2d->Conv2d->BatchNorm2d->ReLU
        layers_conv_up = []
        layers_bn_up = []
        layers_unpooling = []

        up_filter_sizes+=[down_filter_sizes[-2]]
        for i in range(len(up_filter_sizes)-1):
            layers_unpooling += [nn.MaxUnpool2d(kernel_size=pooling_kernel_sizes[i], stride=pooling_strides[i])]
            layers_bn_up += [nn.BatchNorm2d(num_features=up_filter_sizes[i])]
            layers_conv_up += [nn.Conv2d(up_filter_sizes[i-1], up_filter_sizes[i], kernel_size=kernel_sizes[i], padding=conv_paddings[i])]


            self.__setattr__(f"conv_block_up_{len(down_filter_sizes)+i}", nn.Sequential(
                layers_conv_up[i],
                layers_bn_up[i],
                nn.ReLU(),
                )
                             )

        # Convert Python list to nn.ModuleList, so that PyTorch's autograd
        # can track gradients and update parameters of these layers
        self.layers_conv_up = nn.ModuleList(layers_conv_up)
        self.layers_bn_up = nn.ModuleList(layers_bn_up)
        self.layers_unpooling = nn.ModuleList(layers_unpooling)

        self.relu = nn.ReLU(True)

        # Implement a final 1x1 convolution to to get the logits of 11 classes (background + 10 digits)
        self.final_conv = nn.Conv2d(up_filter_sizes[-2], 5, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.conv_block_down_1(x)
        x, i0 = self.layers_pooling[0](x)
        x = self.conv_block_down_2(x)
        x, i1 = self.layers_pooling[1](x)
        x = self.conv_block_down_3(x)
        x, i2 = self.layers_pooling[2](x)
        x = self.conv_block_down_4(x)
        x, i3 = self.layers_pooling[3](x)
        x = self.layers_unpooling[0](x, i3)
        x = self.conv_block_up_5(x)
        x = self.layers_unpooling[1](x, i2)
        x = self.conv_block_up_6(x)
        x = self.layers_unpooling[2](x, i1)
        x = self.conv_block_up_7(x)
        x = self.layers_unpooling[3](x, i0)
        x = self.conv_block_up_8(x)

        x = self.final_conv(x)
        return x


def get_seg_net(**kwargs):

    model = SegNetLite(**kwargs)

    return model
