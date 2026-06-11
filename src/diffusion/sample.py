"""Generate new images from a trained DDPM and save a grid.

Usage:
    python -m diffusion.sample --n 16 --out outputs/samples.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from diffusion.ddpm import GaussianDiffusion
from diffusion.unet import UNet

CKPT_PATH = Path("models/ddpm.pt")


def load_model(device: str):
    if not CKPT_PATH.exists():
        raise FileNotFoundError(
            f"No checkpoint at {CKPT_PATH}. Run `python -m diffusion.train` first."
        )
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model = UNet().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["timesteps"]


def save_grid(images: torch.Tensor, path: Path, ncol: int = 4) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    images = ((images.clamp(-1, 1) + 1) / 2).cpu().numpy()  # -> [0,1]
    n = images.shape[0]
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.6, nrow * 1.6))
    for i, ax in enumerate(axes.flatten()):
        ax.axis("off")
        if i < n:
            ax.imshow(images[i, 0], cmap="gray", vmin=0, vmax=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample images from the DDPM.")
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--out", type=str, default="outputs/samples.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, timesteps = load_model(device)
    diffusion = GaussianDiffusion(timesteps=timesteps, device=device)
    images = diffusion.sample(model, (args.n, 1, 28, 28))
    save_grid(images, Path(args.out))
    print(f"Saved {args.n} samples -> {args.out}")


if __name__ == "__main__":
    main()
