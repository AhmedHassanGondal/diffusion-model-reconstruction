"""DDPM maths + end-to-end shape tests."""

import torch

from diffusion.data import SyntheticShapes
from diffusion.ddpm import GaussianDiffusion
from diffusion.unet import UNet


def test_schedule_properties():
    d = GaussianDiffusion(timesteps=100)
    # alphas_cumprod is strictly decreasing from <1 toward 0.
    acp = d.alphas_cumprod
    assert acp[0] < 1.0
    assert torch.all(acp[1:] < acp[:-1])
    assert acp[-1] > 0.0
    assert torch.all(d.betas > 0) and torch.all(d.betas < 1)


def test_q_sample_shape_and_finiteness():
    d = GaussianDiffusion(timesteps=50)
    x0 = torch.randn(8, 1, 28, 28)
    t = torch.randint(0, 50, (8,))
    xt = d.q_sample(x0, t)
    assert xt.shape == x0.shape
    assert torch.isfinite(xt).all()


def test_p_losses_is_finite_scalar_and_backprops():
    d = GaussianDiffusion(timesteps=50)
    model = UNet()
    x0 = SyntheticShapes(n=8).data
    t = torch.randint(0, 50, (x0.size(0),))
    loss = d.p_losses(model, x0, t)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_sample_shape():
    d = GaussianDiffusion(timesteps=10)
    model = UNet().eval()
    out = d.sample(model, (2, 1, 28, 28))
    assert out.shape == (2, 1, 28, 28)
    assert torch.isfinite(out).all()


def test_reconstruct_shape():
    d = GaussianDiffusion(timesteps=10)
    model = UNet().eval()
    x0 = SyntheticShapes(n=3).data
    recon = d.reconstruct(model, x0, t_start=5)
    assert recon.shape == x0.shape
    assert torch.isfinite(recon).all()
