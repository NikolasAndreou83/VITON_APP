from __future__ import annotations

import torch


def tv_loss(flow: torch.Tensor) -> torch.Tensor:
    """
    Total variation loss to encourage smooth flow.
    flow: [B,2,H,W]
    """
    dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]).mean()
    dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1]).mean()
    return dx + dy
