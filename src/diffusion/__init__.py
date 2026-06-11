"""Denoising Diffusion Probabilistic Model for image reconstruction.

Public API:
    UNet               -> time-conditioned noise-prediction network
    GaussianDiffusion  -> forward/reverse diffusion, training loss, sampling
    SyntheticShapes    -> offline synthetic image dataset
"""

from diffusion.unet import UNet
from diffusion.ddpm import GaussianDiffusion
from diffusion.data import SyntheticShapes

__all__ = ["UNet", "GaussianDiffusion", "SyntheticShapes"]
__version__ = "1.0.0"
