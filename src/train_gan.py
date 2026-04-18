import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from .config import GANConfig, TrainConfig
from .data import get_base_dataset
from .models import ConditionalDiscriminator, ConditionalGenerator
from .utils import ensure_dir, get_device, set_seed, timestamp


def parse_args():
    parser = argparse.ArgumentParser(description="Train conditional GAN.")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--latent-dim", type=int, default=GANConfig.latent_dim)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    return parser.parse_args()


def train_gan_core(
    dataset: str,
    epochs: int,
    batch_size: int,
    latent_dim: int,
    seed: int,
    num_workers: int,
    model_dir: str | None = None,
) -> str:
    """
    Train conditional GAN; return absolute-style path to saved generator.pt.
    If model_dir is None, uses outputs/models/<dataset>_<timestamp>/.
    """
    set_seed(seed)
    device = get_device()

    data = get_base_dataset(dataset, train=True)
    loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    G = ConditionalGenerator(latent_dim=latent_dim).to(device)
    D = ConditionalDiscriminator().to(device)

    criterion = nn.BCELoss()
    opt_g = optim.Adam(G.parameters(), lr=GANConfig.generator_lr, betas=GANConfig.betas)
    opt_d = optim.Adam(D.parameters(), lr=GANConfig.discriminator_lr, betas=GANConfig.betas)

    run_id = f"{dataset}_{timestamp()}"
    if model_dir is None:
        model_dir = os.path.join("outputs", "models", run_id)
    sample_dir = os.path.join(model_dir, "preview")
    ensure_dir(model_dir)
    ensure_dir(sample_dir)

    for epoch in range(epochs):
        g_loss_epoch = 0.0
        d_loss_epoch = 0.0
        for real_images, real_labels in tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            bs = real_images.size(0)
            real_images = real_images.to(device)
            real_labels = real_labels.to(device)

            valid = torch.ones(bs, 1, device=device)
            fake = torch.zeros(bs, 1, device=device)

            # Train discriminator
            opt_d.zero_grad()
            real_pred = D(real_images, real_labels)
            d_real_loss = criterion(real_pred, valid)

            noise = torch.randn(bs, latent_dim, device=device)
            sampled_labels = torch.randint(0, 10, (bs,), device=device)
            fake_images = G(noise, sampled_labels)
            fake_pred = D(fake_images.detach(), sampled_labels)
            d_fake_loss = criterion(fake_pred, fake)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            opt_d.step()

            # Train generator
            opt_g.zero_grad()
            gen_pred = D(fake_images, sampled_labels)
            g_loss = criterion(gen_pred, valid)
            g_loss.backward()
            opt_g.step()

            g_loss_epoch += g_loss.item()
            d_loss_epoch += d_loss.item()

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"D Loss: {d_loss_epoch / len(loader):.4f} "
            f"G Loss: {g_loss_epoch / len(loader):.4f}"
        )

        with torch.no_grad():
            noise = torch.randn(64, latent_dim, device=device)
            labels = torch.arange(0, 10, device=device).repeat(7)[:64]
            samples = G(noise, labels)
            save_image((samples + 1) / 2, os.path.join(sample_dir, f"epoch_{epoch + 1:03d}.png"), nrow=8)

    gen_path = os.path.join(model_dir, "generator.pt")
    torch.save(G.state_dict(), gen_path)
    torch.save(D.state_dict(), os.path.join(model_dir, "discriminator.pt"))
    print(f"Saved GAN models to: {model_dir}")
    return gen_path


def train():
    args = parse_args()
    train_gan_core(
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        seed=args.seed,
        num_workers=args.num_workers,
        model_dir=None,
    )


if __name__ == "__main__":
    train()
