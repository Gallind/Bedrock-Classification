"""Lightweight U-Net for per-pixel seabed classification.

Classic encoder/decoder with skip connections (Ronneberger 2015), kept small in
the spirit of the lightweight variant (Leclerc 2019) used by the reference
seabed paper: configurable depth and base filter count (defaults: depth 4,
base 16 -> ~1.9M params), trained from scratch — the input bands are physical
measurements, not RGB, so there are no meaningful pretrained weights.
"""

from __future__ import annotations

import torch
from torch import nn


class DoubleConv(nn.Module):
    """(conv3x3 -> BN -> ReLU) x 2."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        base_filters: int = 16,
        depth: int = 4,
    ):
        super().__init__()
        self.depth = depth
        channels = [base_filters * 2**i for i in range(depth)]  # e.g. [16,32,64,128]

        self.encoders = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.encoders.append(DoubleConv(prev, ch))
            prev = ch
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(channels[-1], channels[-1] * 2)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev = channels[-1] * 2
        for ch in reversed(channels):
            self.upconvs.append(nn.ConvTranspose2d(prev, ch, kernel_size=2, stride=2))
            self.decoders.append(DoubleConv(ch * 2, ch))  # *2: skip concat
            prev = ch
        self.head = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        factor = 2**self.depth
        if h % factor or w % factor:
            raise ValueError(
                f"input {h}x{w} not divisible by 2^depth={factor}; "
                "use 128 px tiles or reduce model.depth"
            )

        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            x = decoder(torch.cat([skip, x], dim=1))
        return self.head(x)
