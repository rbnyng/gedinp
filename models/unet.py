"""
UNet architecture for ConvCNP.

Encoder-decoder with skip connections for processing sparse spatial data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class DoubleConv(nn.Module):
    """Double convolution block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()

        # Use bilinear upsampling or transposed convolutions
        if bilinear:
            # Bilinear upsample doesn't change channels, so add 1x1 conv to reduce
            # from in_channels to in_channels // 2 before concatenation
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
            )
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Args:
            x1: Upsampled features from decoder
            x2: Skip connection from encoder
        """
        x1 = self.up(x1)

        # Handle size mismatch if input dimensions are not divisible by 2^depth
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    UNet architecture for dense spatial feature extraction.

    Processes sparse input (AGBD + mask + embeddings) and outputs
    dense feature maps at the same spatial resolution.
    """

    def __init__(
        self,
        in_channels: int = 130,  # 1 (AGBD) + 1 (mask) + 128 (embeddings)
        feature_dim: int = 128,  # Output feature dimension
        base_channels: int = 64,
        depth: int = 4,
        bilinear: bool = True
    ):
        """
        Initialize UNet.

        Args:
            in_channels: Number of input channels
            feature_dim: Dimension of output feature maps
            base_channels: Number of channels in first layer (doubles each level)
            depth: Number of downsampling/upsampling levels
            bilinear: Use bilinear upsampling (True) or transposed convs (False)
        """
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.depth = depth

        # Initial convolution
        self.inc = DoubleConv(in_channels, base_channels)

        # Encoder (downsampling path)
        self.downs = nn.ModuleList()
        ch = base_channels
        for i in range(depth):
            self.downs.append(Down(ch, ch * 2))
            ch *= 2

        # Decoder (upsampling path)
        self.ups = nn.ModuleList()
        for i in range(depth):
            self.ups.append(Up(ch, ch // 2, bilinear))
            ch //= 2

        # Output projection to feature_dim
        self.outc = nn.Conv2d(base_channels, feature_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, in_channels, H, W)

        Returns:
            Feature maps (B, feature_dim, H, W)
        """
        # Encoder
        x_enc = [self.inc(x)]

        for down in self.downs:
            x_enc.append(down(x_enc[-1]))

        # Decoder with skip connections
        x = x_enc[-1]
        for i, up in enumerate(self.ups):
            # Skip connection from encoder (in reverse order)
            skip_idx = -(i + 2)
            x = up(x, x_enc[skip_idx])

        # Output projection
        features = self.outc(x)

        return features


class UNetSmall(nn.Module):
    """
    Smaller UNet for faster training/inference.

    3-level architecture with fewer channels.
    """

    def __init__(
        self,
        in_channels: int = 130,
        feature_dim: int = 128,
        base_channels: int = 32
    ):
        super().__init__()

        self.inc = DoubleConv(in_channels, base_channels)

        # 3 levels
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)

        self.up1 = Up(base_channels * 8, base_channels * 4)
        self.up2 = Up(base_channels * 4, base_channels * 2)
        self.up3 = Up(base_channels * 2, base_channels)

        self.outc = nn.Conv2d(base_channels, feature_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Decoder
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        return self.outc(x)


if __name__ == "__main__":
    # Test UNet
    model = UNet(in_channels=130, feature_dim=128, base_channels=64, depth=3)

    # Test input: batch of 2, 130 channels, 256x256 spatial
    x = torch.randn(2, 130, 256, 256)
    out = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
