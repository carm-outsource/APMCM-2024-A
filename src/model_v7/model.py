import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class EFED(nn.Module):
    def __init__(self, in_channels):
        super(EFED, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        edge_features = self.conv1(x)
        edge_features = self.relu(edge_features)
        edge_features = self.conv2(edge_features)
        return x + edge_features

class ImprovedUWCNN(nn.Module):
    def __init__(self, num_residual_blocks=8):
        super(ImprovedUWCNN, self).__init__()
        self.input_conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # Add ConvBlock after the initial input convolution layer
        self.conv_block = ConvBlock(64, 64)

        # Residual Blocks
        self.residual_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_residual_blocks)])

        # Add EFED for edge feature enhancement
        self.efed = EFED(64)

        self.output_conv = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.relu(self.input_conv(x))

        # Pass through ConvBlock
        out = self.conv_block(out)

        # Pass through residual blocks
        out = self.residual_blocks(out)

        # Edge feature enhancement
        out = self.efed(out)

        # Final output
        out = self.output_conv(out)
        out = torch.clamp(out, 0, 1)
        return out

if __name__ == "__main__":
    model = ImprovedUWCNN()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(y.shape)