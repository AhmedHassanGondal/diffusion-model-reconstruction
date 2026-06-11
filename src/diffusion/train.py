"""Train the DDPM on the synthetic shapes dataset.

Usage:
    python -m diffusion.train --epochs 20 --timesteps 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from diffusion.data import SyntheticShapes
from diffusion.ddpm import GaussianDiffusion
from diffusion.unet import UNet

CKPT_PATH = Path("models/ddpm.pt")


def train(epochs=20, timesteps=200, batch_size=64, n=2000, lr=2e-4, device=None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(SyntheticShapes(n=n), batch_size=batch_size, shuffle=True)

    model = UNet().to(device)
    diffusion = GaussianDiffusion(timesteps=timesteps, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    for epoch in range(epochs):
        running, batches = 0.0, 0
        for x in loader:
            x = x.to(device)
            t = torch.randint(0, timesteps, (x.size(0),), device=device)
            loss = diffusion.p_losses(model, x, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
            batches += 1
        epoch_loss = running / batches
        history.append(epoch_loss)
        print(f"epoch {epoch + 1:3d}/{epochs}  loss {epoch_loss:.4f}")

    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "timesteps": timesteps}, CKPT_PATH)
    print(f"Saved checkpoint -> {CKPT_PATH}")
    return {"history": history, "final_loss": history[-1]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the DDPM.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n", type=int, default=2000)
    args = parser.parse_args()
    train(args.epochs, args.timesteps, args.batch_size, args.n)


if __name__ == "__main__":
    main()
