import torch
import torch.nn as nn


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


class BrightnessEnhancement(nn.Module):
    def __init__(self, in_channels):
        super(BrightnessEnhancement, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        return out


class SharpnessEnhancement(nn.Module):
    def __init__(self, in_channels):
        super(SharpnessEnhancement, self).__init__()
        self.laplacian_kernel = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels,
                                          bias=False)
        laplacian_filter = torch.tensor([[[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]], dtype=torch.float32)
        self.laplacian_kernel.weight = nn.Parameter(laplacian_filter.repeat(in_channels, 1, 1, 1), requires_grad=False)

    def forward(self, x):
        high_freq = self.laplacian_kernel(x)
        return x + high_freq


class ImprovedUWCNN(nn.Module):
    def __init__(self, num_residual_blocks=8):
        super(ImprovedUWCNN, self).__init__()
        self.input_conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.residual_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_residual_blocks)])
        self.brightness_enhancement = BrightnessEnhancement(64)
        self.sharpness_enhancement = SharpnessEnhancement(64)
        self.output_conv = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.relu(self.input_conv(x))
        out = self.residual_blocks(out)
        out = self.brightness_enhancement(out)
        out = self.sharpness_enhancement(out)
        out = self.output_conv(out)
        out = torch.clamp(out, 0, 1)
        return out


if __name__ == "__main__":
    model = ImprovedUWCNN()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(y.shape)
