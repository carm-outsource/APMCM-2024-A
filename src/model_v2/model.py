import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock,self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3,padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3,padding=1)

    def forward(self,x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return out

class CombinedNet(nn.Module):
    def __init__(self):
        super(CombinedNet, self).__init__()
        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, ceil_mode=True)
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, ceil_mode=True)
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.decoder3 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Residual Blocks for sharpening
        self.resblocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(5)]
        )

        # Output layer
        self.output_layer = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x_input = x

        # Encoding
        x1 = self.encoder1(x)  # [batch, 64, H, W]
        x1p = self.pool1(x1)   # [batch, 64, H1, W1]
        x2 = self.encoder2(x1p)  # [batch, 128, H1, W1]
        x2p = self.pool2(x2)     # [batch, 128, H2, W2]
        x3 = self.encoder3(x2p)  # [batch, 256, H2, W2]

        # Decoding
        x = F.interpolate(x3, size=x2.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, x2], dim=1)  # Concatenate along channel dimension
        x = self.decoder3(x)

        x = F.interpolate(x, size=x1.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, x1], dim=1)
        x = self.decoder2(x)

        # Sharpening
        x = self.resblocks(x)

        # Output
        x = self.output_layer(x)
        out = x + x_input  # Residual connection with input
        out = torch.clamp(out, 0.0, 1.0)  # Ensure output is in [0,1]

        return out
