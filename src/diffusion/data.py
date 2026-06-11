"""A small, deterministic synthetic image dataset.

Each image is a dark canvas with a few bright rectangles — enough structure for
a diffusion model to learn and reconstruct, while needing no downloads so the
project runs offline. Images are single-channel 28x28, normalised to [-1, 1].

To train on real images instead, swap this Dataset for e.g. torchvision MNIST;
the rest of the pipeline is unchanged.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

IMG_SIZE = 28


def _make_image(rng: np.random.Generator) -> np.ndarray:
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    for _ in range(rng.integers(1, 4)):
        h = rng.integers(4, 10)
        w = rng.integers(4, 10)
        y = rng.integers(0, IMG_SIZE - h)
        x = rng.integers(0, IMG_SIZE - w)
        img[y : y + h, x : x + w] = rng.uniform(0.6, 1.0)
    return img


class SyntheticShapes(Dataset):
    """Deterministic synthetic shapes dataset returning (1, 28, 28) tensors in [-1, 1]."""

    def __init__(self, n: int = 2000, seed: int = 42):
        rng = np.random.default_rng(seed)
        images = np.stack([_make_image(rng) for _ in range(n)])  # (n, 28, 28)
        images = images * 2.0 - 1.0  # [0,1] -> [-1,1]
        self.data = torch.from_numpy(images).unsqueeze(1).float()  # (n, 1, 28, 28)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]
