"""Gaussian DDPM: forward noising, training loss, reverse sampling, reconstruction.

Implements the denoising-diffusion-probabilistic-model maths from Ho et al.
(2020): a fixed linear variance schedule, a closed-form forward process
``q(x_t | x_0)``, an epsilon-prediction training objective, and the ancestral
sampler for the reverse process.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _extract(values: torch.Tensor, t: torch.Tensor, shape) -> torch.Tensor:
    """Gather per-sample scalars and broadcast to ``shape``."""
    out = values.gather(0, t).float()
    return out.view(t.shape[0], *([1] * (len(shape) - 1)))


class GaussianDiffusion:
    def __init__(self, timesteps: int = 200, beta_start: float = 1e-4, beta_end: float = 0.02,
                 device: str = "cpu"):
        self.timesteps = timesteps
        self.device = device

        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    # ------------------------------------------------------------------ #
    # Forward process
    # ------------------------------------------------------------------ #
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None):
        """Sample x_t ~ q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_acp = _extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_1macp = _extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_acp * x0 + sqrt_1macp * noise

    def p_losses(self, model, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """MSE between the true and predicted noise (the training objective)."""
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        predicted = model(x_t, t)
        return F.mse_loss(predicted, noise)

    # ------------------------------------------------------------------ #
    # Reverse process
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def p_sample(self, model, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """One reverse step: sample x_{t-1} from x_t."""
        betas_t = _extract(self.betas, t, x_t.shape)
        sqrt_1macp_t = _extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_recip_alphas_t = _extract(self.sqrt_recip_alphas, t, x_t.shape)

        mean = sqrt_recip_alphas_t * (x_t - betas_t * model(x_t, t) / sqrt_1macp_t)
        if int(t[0]) == 0:
            return mean
        var = _extract(self.posterior_variance, t, x_t.shape)
        return mean + torch.sqrt(var) * torch.randn_like(x_t)

    @torch.no_grad()
    def sample(self, model, shape) -> torch.Tensor:
        """Generate images by running the full reverse chain from pure noise."""
        x = torch.randn(shape, device=self.device)
        for step in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), step, device=self.device, dtype=torch.long)
            x = self.p_sample(model, x, t)
        return x

    @torch.no_grad()
    def reconstruct(self, model, x0: torch.Tensor, t_start: int) -> torch.Tensor:
        """Noise ``x0`` up to ``t_start`` then denoise back — a reconstruction.

        Lower ``t_start`` keeps more of the original; higher values discard more
        structure before the model rebuilds it.
        """
        t_start = min(max(t_start, 1), self.timesteps)
        t = torch.full((x0.shape[0],), t_start - 1, device=self.device, dtype=torch.long)
        x = self.q_sample(x0, t)
        for step in reversed(range(t_start)):
            t = torch.full((x0.shape[0],), step, device=self.device, dtype=torch.long)
            x = self.p_sample(model, x, t)
        return x
