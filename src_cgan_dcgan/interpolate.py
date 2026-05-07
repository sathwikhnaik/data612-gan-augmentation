"""
Latent space interpolation for the conditional GAN.

Generates a grid: 10 rows (one per class) x `steps` columns,
interpolating between two random noise vectors via spherical linear
interpolation (slerp) so the path stays on the latent hypersphere.
"""
import argparse
import os

import torch
from torchvision.utils import save_image

from .generate_synthetic import _load_generator
from .utils import ensure_dir, get_device, set_seed


def _slerp(z1: torch.Tensor, z2: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical linear interpolation between two 1-D latent vectors."""
    z1_n = z1 / z1.norm()
    z2_n = z2 / z2.norm()
    dot = (z1_n * z2_n).sum().clamp(-1.0, 1.0)
    omega = torch.acos(dot)
    if omega.abs().item() < 1e-6:
        return (1.0 - t) * z1 + t * z2
    sin_o = torch.sin(omega)
    return (torch.sin((1.0 - t) * omega) / sin_o) * z1 + (torch.sin(t * omega) / sin_o) * z2


def generate_interpolation_grid(
    generator_path: str,
    output_path: str,
    latent_dim: int = 100,
    steps: int = 10,
    seed: int = 42,
) -> str:
    """
    For each of the 10 classes, interpolate between two random latent vectors
    and save a grid image (10 rows × steps columns).
    """
    set_seed(seed)
    device = get_device()

    generator, latent_dim = _load_generator(generator_path, latent_dim, device)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        ensure_dir(out_dir)

    rows = []
    with torch.no_grad():
        for cls in range(10):
            z1 = torch.randn(1, latent_dim, device=device)
            z2 = torch.randn(1, latent_dim, device=device)
            label = torch.tensor([cls], device=device)
            frames = []
            for step in range(steps):
                t = step / max(steps - 1, 1)
                z = _slerp(z1, z2, t)
                img = generator(z, label)
                frames.append((img.clamp(-1, 1) + 1) / 2)
            rows.append(torch.cat(frames, dim=0))  # (steps, 1, 28, 28)

    grid = torch.cat(rows, dim=0)  # (10 * steps, 1, 28, 28)
    save_image(grid, output_path, nrow=steps, padding=2)
    print(f"Interpolation grid saved to: {output_path}")
    return output_path


def parse_args():
    p = argparse.ArgumentParser(description="Latent space interpolation for a trained cGAN.")
    p.add_argument("--generator-path", type=str, required=True)
    p.add_argument("--output-path", type=str, default="")
    p.add_argument("--latent-dim", type=int, default=100)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    out = args.output_path or os.path.join("outputs", "interpolations", "interpolation_grid.png")
    generate_interpolation_grid(
        generator_path=args.generator_path,
        output_path=out,
        latent_dim=args.latent_dim,
        steps=args.steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
