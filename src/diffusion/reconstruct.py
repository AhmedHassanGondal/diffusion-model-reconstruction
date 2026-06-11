"""Reconstruct images: noise a real image to step t, then let the DDPM denoise it.

Saves a grid with rows [original | noised | reconstructed] for several images.

Usage:
    python -m diffusion.reconstruct --t-start 80 --out outputs/reconstruction.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from diffusion.data import SyntheticShapes
from diffusion.ddpm import GaussianDiffusion
from diffusion.sample import load_model


def save_triptych(original, noised, recon, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def prep(x):
        return ((x.clamp(-1, 1) + 1) / 2).cpu().numpy()

    original, noised, recon = prep(original), prep(noised), prep(recon)
    n = original.shape[0]
    fig, axes = plt.subplots(3, n, figsize=(n * 1.6, 5))
    rows = [("original", original), ("noised", noised), ("reconstructed", recon)]
    for r, (label, imgs) in enumerate(rows):
        for c in range(n):
            ax = axes[r, c]
            ax.axis("off")
            ax.imshow(imgs[c, 0], cmap="gray", vmin=0, vmax=1)
            if c == 0:
                ax.set_title(label, loc="left", fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct images with the DDPM.")
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--t-start", type=int, default=80)
    parser.add_argument("--out", type=str, default="outputs/reconstruction.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, timesteps = load_model(device)
    diffusion = GaussianDiffusion(timesteps=timesteps, device=device)

    originals = SyntheticShapes(n=args.n, seed=7).data.to(device)
    t = torch.full((args.n,), min(args.t_start, timesteps) - 1, device=device, dtype=torch.long)
    noised = diffusion.q_sample(originals, t)
    recon = diffusion.reconstruct(model, originals, t_start=args.t_start)

    save_triptych(originals, noised, recon, Path(args.out))
    print(f"Saved reconstruction (t_start={args.t_start}) -> {args.out}")


if __name__ == "__main__":
    main()
