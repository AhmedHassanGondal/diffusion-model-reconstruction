"""A compact time-conditioned U-Net that predicts the noise added to an image.

Designed for small (28x28) single-channel images so the whole project trains on
a CPU in minutes. Two downsample/upsample stages with skip connections and a
sinusoidal timestep embedding injected into every residual block.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of integer timesteps, shape (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ResBlock(nn.Module):
    """Conv -> GroupNorm -> SiLU, with a timestep bias and a residual."""

    def __init__(self, in_c: int, out_c: int, time_dim: int):
        super().__init__()
        self.time = nn.Linear(time_dim, out_c)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_c)
        self.norm2 = nn.GroupNorm(8, out_c)
        self.act = nn.SiLU()
        self.res = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.time(t_emb)[:, :, None, None]
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.res(x)


class UNet(nn.Module):
    def __init__(self, in_ch: int = 1, base: int = 32, time_dim: int = 128):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)

        self.d1 = ResBlock(base, base, time_dim)
        self.down1 = nn.Conv2d(base, base, 4, 2, 1)          # 28 -> 14
        self.d2 = ResBlock(base, base * 2, time_dim)
        self.down2 = nn.Conv2d(base * 2, base * 2, 4, 2, 1)  # 14 -> 7

        self.mid = ResBlock(base * 2, base * 2, time_dim)

        self.up2 = nn.ConvTranspose2d(base * 2, base * 2, 4, 2, 1)  # 7 -> 14
        self.u2 = ResBlock(base * 2 + base * 2, base, time_dim)
        self.up1 = nn.ConvTranspose2d(base, base, 4, 2, 1)          # 14 -> 28
        self.u1 = ResBlock(base + base, base, time_dim)

        self.out = nn.Conv2d(base, in_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(timestep_embedding(t, self.time_dim))
        x = self.in_conv(x)
        h1 = self.d1(x, t_emb)              # (base, 28, 28)
        x = self.down1(h1)                  # (base, 14, 14)
        h2 = self.d2(x, t_emb)              # (base*2, 14, 14)
        x = self.down2(h2)                  # (base*2, 7, 7)
        x = self.mid(x, t_emb)              # (base*2, 7, 7)
        x = self.up2(x)                     # (base*2, 14, 14)
        x = self.u2(torch.cat([x, h2], dim=1), t_emb)   # (base, 14, 14)
        x = self.up1(x)                     # (base, 28, 28)
        x = self.u1(torch.cat([x, h1], dim=1), t_emb)   # (base, 28, 28)
        return self.out(x)                  # (in_ch, 28, 28)
