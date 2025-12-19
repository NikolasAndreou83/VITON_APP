from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class WarpNet(nn.Module):
    """
    Predicts a dense flow field (dx, dy) in *normalized grid units* for grid_sample.

    Input channels:
      agnostic (3) + pose_rendered (3) + cloth_mask (1) = 7
    Output:
      flow: [B,2,H,W], values ~[-flow_max, flow_max] in normalized coords
    """

    def __init__(self, in_ch: int = 7, base: int = 32, flow_max: float = 0.5):
        super().__init__()
        self.flow_max = float(flow_max)

        self.enc1 = conv_block(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = conv_block(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = conv_block(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = conv_block(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = conv_block(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = conv_block(base * 2, base)

        self.out = nn.Conv2d(base, 2, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        flow = torch.tanh(self.out(d1)) * self.flow_max
        return flow


def warp_with_flow(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    img:  [B,C,H,W]
    flow: [B,2,H,W] in normalized coords (dx,dy)
    Returns: warped img
    """
    b, c, h, w = img.shape
    # base grid in [-1,1]
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h, device=img.device, dtype=img.dtype),
        torch.linspace(-1, 1, w, device=img.device, dtype=img.dtype),
        indexing="ij",
    )
    base = torch.stack([xx, yy], dim=-1)  # H,W,2
    base = base.unsqueeze(0).repeat(b, 1, 1, 1)  # B,H,W,2

    grid = base + flow.permute(0, 2, 3, 1)  # B,H,W,2
    warped = F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return warped
