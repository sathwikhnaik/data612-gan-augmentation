import argparse
import json
import os

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from .models import ConditionalGenerator, build_generator
from .utils import ensure_dir, get_device, set_seed, timestamp


def _load_generator(generator_path: str, latent_dim: int, device) -> tuple:
    """
    Load the generator using its saved generator_config.json if available,
    otherwise fall back to the default ConditionalGenerator.
    Returns (generator, latent_dim).
    """
    model_dir = os.path.dirname(generator_path)
    config_path = os.path.join(model_dir, "generator_config.json")

    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        G = build_generator(
            gen_type=cfg.get("gen_type", "mlp"),
            conditioning=cfg.get("conditioning", "embedding"),
            latent_dim=cfg.get("latent_dim", latent_dim),
            num_classes=cfg.get("num_classes", 10),
            embedding_dim=cfg.get("embedding_dim", 50),
            hidden_dims=cfg.get("hidden_dims"),
        )
        latent_dim = cfg.get("latent_dim", latent_dim)
    else:
        # backward-compat: assume default MLP cGAN
        G = ConditionalGenerator(latent_dim=latent_dim)

    G.load_state_dict(torch.load(generator_path, map_location=device, weights_only=True))
    G.to(device).eval()
    return G, latent_dim


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
    model_dir: str | None = None,
) -> str:
    """
    Write class-balanced PNGs under output_root/<0-9>/.
    If model_dir is given, generator_config.json is read from there to load
    the correct architecture. Otherwise generator_path's parent dir is checked.
    """
    set_seed(seed)
    device = get_device()

    if not generator_path:
        raise ValueError("generator_path is required.")

    # Allow caller to override which directory holds the config
    if model_dir:
        config_path = os.path.join(model_dir, "generator_config.json")
        if os.path.exists(config_path):
            # Re-point generator_path to the one in model_dir (should be the same)
            if not os.path.exists(generator_path):
                generator_path = os.path.join(model_dir, "generator.pt")

    G, latent_dim = _load_generator(generator_path, latent_dim, device)

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
                img = G(noise, labels)
                save_path = os.path.join(output_root, str(cls), f"{cls}_{i:06d}.png")
                x = (img.clamp(-1, 1) + 1) / 2
                save_image(x.repeat(1, 3, 1, 1), save_path)

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
