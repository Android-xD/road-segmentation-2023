import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Downsample layers
        self.down1 = self.downsampling(in_channels, 64)
        self.down2 = self.downsampling(64, 128)
        self.down3 = self.downsampling(128, 256)
        self.down4 = self.downsampling(256, 512)

        # Upsample layers
        self.up1 = self.upsampling(512, 256)
        self.up2 = self.upsampling(256, 128)
        self.up3 = self.upsampling(128, 64)
        self.up4 = self.upsampling(64, 64)

        # Residual connections
        self.residual1 = self.residual_block(128, 256)
        self.residual2 = self.residual_block(64, 128)
        self.residual3 = self.residual_block(32, 64)
        self.residual4 = self.residual_block(32, out_channels)

        # Skip connections
        self.skip1 = self.skip_connection(256, 128)
        self.skip2 = self.skip_connection(128, 64)
        self.skip3 = self.skip_connection(64, 32)


    def downsampling(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def upsampling(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels // 2, kernel_size=2, stride=2)
        )

    def residual_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

    def skip_connection(self, in_channels, out_channels):
        return nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Downsample
        x1 = self.down1(x) # 8
        x2 = self.down2(x1) # 16
        x3 = self.down3(x2) # 32
        x4 = self.down4(x3) # 64

        # Upsample
        x = self.up1(x4)
        x = self.residual1(x + self.skip1(x3))
        x = self.up2(x)
        x = self.residual2(x + self.skip2(x2))
        x = self.up3(x)
        x = self.residual3(x + self.skip3(x1))
        x = self.up4(x)
        x = self.residual4(x)

        return x
