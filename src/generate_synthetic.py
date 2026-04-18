import argparse
import os

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from .models import ConditionalGenerator
from .utils import ensure_dir, get_device, set_seed, timestamp


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic images with a trained GAN.")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    parser.add_argument("--generator-path", type=str, default="")
    parser.add_argument("--num-samples", type=int, default=12000)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def generate_synthetic_core(
    generator_path: str,
    dataset: str,
    num_samples: int,
    latent_dim: int,
    seed: int,
    output_root: str | None = None,
) -> str:
    """
    Write class-balanced PNGs under output_root/<0-9>/. If output_root is None,
    uses outputs/synthetic/<dataset>_synthetic_<timestamp>/.
    """
    set_seed(seed)
    device = get_device()

    if not generator_path:
        raise ValueError("generator_path is required.")

    generator = ConditionalGenerator(latent_dim=latent_dim).to(device)
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    generator.eval()

    if output_root is None:
        run_id = f"{dataset}_synthetic_{timestamp()}"
        output_root = os.path.join("outputs", "synthetic", run_id)
    for cls in range(10):
        ensure_dir(os.path.join(output_root, str(cls)))

    per_class = num_samples // 10
    remainder = num_samples % 10

    with torch.no_grad():
        for cls in range(10):
            class_count = per_class + (1 if cls < remainder else 0)
            for i in tqdm(range(class_count), desc=f"Class {cls}"):
                noise = torch.randn(1, latent_dim, device=device)
                labels = torch.tensor([cls], device=device)
                img = generator(noise, labels)
                save_path = os.path.join(output_root, str(cls), f"{cls}_{i:06d}.png")
                save_image((img + 1) / 2, save_path)

    print(f"Synthetic images written to: {output_root}")
    return output_root


def generate():
    args = parse_args()
    if not args.generator_path:
        raise ValueError("Please provide --generator-path to a trained generator checkpoint.")
    generate_synthetic_core(
        generator_path=args.generator_path,
        dataset=args.dataset,
        num_samples=args.num_samples,
        latent_dim=args.latent_dim,
        seed=args.seed,
        output_root=None,
    )


if __name__ == "__main__":
    generate()
