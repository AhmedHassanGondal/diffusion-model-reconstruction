"""U-Net shape tests."""

import torch

from diffusion.unet import UNet, timestep_embedding


def test_forward_preserves_shape():
    model = UNet()
    x = torch.randn(4, 1, 28, 28)
    t = torch.randint(0, 200, (4,))
    out = model(x, t)
    assert out.shape == x.shape


def test_timestep_embedding_shape():
    t = torch.arange(8)
    emb = timestep_embedding(t, 128)
    assert emb.shape == (8, 128)
    assert torch.isfinite(emb).all()
