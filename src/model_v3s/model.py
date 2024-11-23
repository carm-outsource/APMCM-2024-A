import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic convolutional block with convolution, batch normalization, and activation"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, activation='relu'):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.act = nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        return out


class UWCNN(nn.Module):
    def __init__(self, num_channels=3):
        super(UWCNN, self).__init__()
        # Color correction branch
        self.color_branch = nn.Sequential(
            ConvBlock(num_channels, 64, kernel_size=3, padding=1),
            ConvBlock(64, 64, kernel_size=3, padding=1),
            ConvBlock(64, num_channels, kernel_size=3, padding=1, activation=None)
        )
        # Light enhancement branch
        self.light_branch = nn.Sequential(
            ConvBlock(num_channels, 64, kernel_size=3, padding=1),
            ConvBlock(64, 64, kernel_size=3, padding=1),
            ConvBlock(64, num_channels, kernel_size=3, padding=1, activation=None)
        )
        # Sharpness enhancement branch
        self.sharp_branch = nn.Sequential(
            ConvBlock(num_channels, 64, kernel_size=3, padding=1),
            ConvBlock(64, 64, kernel_size=3, padding=1),
            ConvBlock(64, num_channels, kernel_size=3, padding=1, activation=None)
        )
        # Dehazing branch
        self.dehaze_branch = nn.Sequential(
            ConvBlock(num_channels, 64, kernel_size=3, padding=1),
            ConvBlock(64, 64, kernel_size=3, padding=1),
            ConvBlock(64, num_channels, kernel_size=3, padding=1, activation=None)
        )
        # Feature fusion layer
        self.fusion = nn.Sequential(
            ConvBlock(num_channels * 4, 64, kernel_size=1, padding=0),
            ConvBlock(64, num_channels, kernel_size=3, padding=1, activation=None)
        )
        # Residual connection
        self.residual = nn.Identity()

    def forward(self, x):
        # Process each branch
        color_out = self.color_branch(x)
        light_out = self.light_branch(x)
        sharp_out = self.sharp_branch(x)
        dehaze_out = self.dehaze_branch(x)
        # Feature fusion
        combined = torch.cat([color_out, light_out, sharp_out, dehaze_out], dim=1)
        fusion_out = self.fusion(combined)
        # Residual connection
        out = self.residual(x) + fusion_out
        return out
