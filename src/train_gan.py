import argparse
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from .config import GANConfig, TrainConfig
from .data import get_base_dataset, maybe_stratified_train_cap
from .models import build_discriminator, build_generator
from .utils import ensure_dir, get_device, save_json, set_seed, timestamp


# ── Loss helpers ───────────────────────────────────────────────────────────

def _gradient_penalty(D, real_imgs, fake_imgs, labels, device, lam: float = 10.0) -> torch.Tensor:
    """WGAN-GP gradient penalty. Interpolation happens in image space."""
    bs = real_imgs.size(0)
    alpha = torch.rand(bs, 1, 1, 1, device=device).expand_as(real_imgs)
    interp = (alpha * real_imgs + (1 - alpha) * fake_imgs.detach()).requires_grad_(True)
    d_interp = D(interp, labels)
    grads = torch.autograd.grad(
        outputs=d_interp, inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True, retain_graph=True,
    )[0]
    return lam * ((grads.view(bs, -1).norm(2, dim=1) - 1) ** 2).mean()


def _d_loss(loss_type, D, real_imgs, fake_imgs, real_labels, fake_labels,
            real_label_val, device, gp_lambda):
    if loss_type == "bce":
        crit = nn.BCELoss()
        return (
            crit(D(real_imgs, real_labels), torch.full((real_imgs.size(0), 1), real_label_val, device=device))
            + crit(D(fake_imgs.detach(), fake_labels), torch.zeros(fake_imgs.size(0), 1, device=device))
        )
    if loss_type == "wgan_gp":
        d_real = D(real_imgs, real_labels).mean()
        d_fake = D(fake_imgs.detach(), fake_labels).mean()
        gp = _gradient_penalty(D, real_imgs, fake_imgs, real_labels, device, lam=gp_lambda)
        return d_fake - d_real + gp
    # hinge
    return (
        torch.relu(1.0 - D(real_imgs, real_labels)).mean()
        + torch.relu(1.0 + D(fake_imgs.detach(), fake_labels)).mean()
    )


def _g_loss(loss_type, D, fake_imgs, fake_labels, device):
    if loss_type == "bce":
        return nn.BCELoss()(D(fake_imgs, fake_labels),
                            torch.ones(fake_imgs.size(0), 1, device=device))
    return -D(fake_imgs, fake_labels).mean()  # wgan_gp and hinge share this


# ── Plotting ───────────────────────────────────────────────────────────────

def _plot_loss_curves(g_losses: list, d_losses: list, output_path: str) -> None:
    epochs = range(1, len(g_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, g_losses, label="Generator", marker="o", markersize=3)
    ax.plot(epochs, d_losses, label="Discriminator", marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("GAN Training Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train conditional GAN.")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--latent-dim", type=int, default=GANConfig.latent_dim)
    parser.add_argument("--embedding-dim", type=int, default=GANConfig.embedding_dim)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--max-train-samples", type=int, default=0,
                        help="Stratified cap on GAN training data (0 = full).")
    # Architecture
    parser.add_argument("--gen-type", type=str, default="mlp", choices=["mlp", "dcgan"])
    parser.add_argument("--conditioning", type=str, default="embedding",
                        choices=["embedding", "onehot", "projection"])
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=None,
                        help="MLP hidden layer sizes, e.g. 256 512 1024")
    parser.add_argument("--disc-dropout", type=float, default=0.3)
    # Training strategy
    parser.add_argument("--loss-type", type=str, default="bce",
                        choices=["bce", "wgan_gp", "hinge"])
    parser.add_argument("--label-smoothing", action="store_true", default=False)
    parser.add_argument("--generator-lr", type=float, default=GANConfig.generator_lr)
    parser.add_argument("--discriminator-lr", type=float, default=GANConfig.discriminator_lr)
    parser.add_argument("--n-critic", type=int, default=1,
                        help="Discriminator updates per generator update (5 for WGAN-GP).")
    parser.add_argument("--gp-lambda", type=float, default=10.0,
                        help="Gradient penalty weight for WGAN-GP.")
    return parser.parse_args()


# ── Core ───────────────────────────────────────────────────────────────────

def train_gan_core(
    dataset: str,
    epochs: int,
    batch_size: int,
    latent_dim: int,
    seed: int,
    num_workers: int,
    model_dir: str | None = None,
    max_gan_train_samples: int | None = None,
    # Architecture
    gen_type: str = "mlp",
    conditioning: str = "embedding",
    embedding_dim: int = GANConfig.embedding_dim,
    hidden_dims: list | None = None,
    disc_dropout: float = 0.3,
    # Training strategy
    loss_type: str = "bce",
    label_smoothing: bool = False,
    generator_lr: float = GANConfig.generator_lr,
    discriminator_lr: float = GANConfig.discriminator_lr,
    n_critic: int = 1,
    gp_lambda: float = 10.0,
) -> str:
    """
    Train a conditional GAN and return the path to the saved generator.pt.
    Saves generator_config.json alongside so generate_synthetic can load the right arch.
    """
    set_seed(seed)
    device = get_device()

    data = get_base_dataset(dataset, train=True)
    data = maybe_stratified_train_cap(data, max_gan_train_samples, seed)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=torch.cuda.is_available())

    use_sigmoid = (loss_type == "bce")
    G = build_generator(gen_type=gen_type, conditioning=conditioning,
                        latent_dim=latent_dim, num_classes=10,
                        embedding_dim=embedding_dim, hidden_dims=hidden_dims).to(device)
    D = build_discriminator(gen_type=gen_type, conditioning=conditioning,
                            num_classes=10, embedding_dim=embedding_dim,
                            dropout=disc_dropout, use_sigmoid=use_sigmoid).to(device)

    opt_g = optim.Adam(G.parameters(), lr=generator_lr, betas=GANConfig.betas)
    opt_d = optim.Adam(D.parameters(), lr=discriminator_lr, betas=GANConfig.betas)

    if model_dir is None:
        model_dir = os.path.join("outputs", "models", f"{dataset}_{timestamp()}")
    sample_dir = os.path.join(model_dir, "preview")
    ensure_dir(model_dir)
    ensure_dir(sample_dir)

    real_label_val = 0.9 if label_smoothing else 1.0
    g_losses_hist: list = []
    d_losses_hist: list = []

    for epoch in range(epochs):
        g_loss_epoch = d_loss_epoch = 0.0
        n_batches = 0

        for real_images, real_labels in tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            bs = real_images.size(0)
            real_images = real_images.to(device)
            real_labels = real_labels.to(device)

            noise = torch.randn(bs, latent_dim, device=device)
            fake_labels = torch.randint(0, 10, (bs,), device=device)

            with torch.no_grad():
                fake_images_no_grad = G(noise, fake_labels)

            # ── Discriminator (n_critic steps) ────────────────────────────
            for _ in range(n_critic):
                noise_d = torch.randn(bs, latent_dim, device=device)
                fake_imgs_d = G(noise_d, fake_labels).detach()
                opt_d.zero_grad()
                dl = _d_loss(loss_type, D, real_images, fake_imgs_d.requires_grad_(False),
                             real_labels, fake_labels, real_label_val, device, gp_lambda)
                # WGAN-GP: use the non-detached version for gradient penalty
                if loss_type == "wgan_gp":
                    noise_gp = torch.randn(bs, latent_dim, device=device)
                    fake_imgs_gp = G(noise_gp, fake_labels)
                    dl = (D(fake_imgs_gp.detach(), fake_labels).mean()
                          - D(real_images, real_labels).mean()
                          + _gradient_penalty(D, real_images, fake_imgs_gp, real_labels, device, lam=gp_lambda))
                dl.backward()
                opt_d.step()
                d_loss_epoch += dl.item()

            # ── Generator ─────────────────────────────────────────────────
            opt_g.zero_grad()
            noise_g = torch.randn(bs, latent_dim, device=device)
            fake_imgs_g = G(noise_g, fake_labels)
            gl = _g_loss(loss_type, D, fake_imgs_g, fake_labels, device)
            gl.backward()
            opt_g.step()
            g_loss_epoch += gl.item()
            n_batches += 1

        g_losses_hist.append(g_loss_epoch / n_batches)
        d_losses_hist.append(d_loss_epoch / (n_batches * n_critic))
        print(f"Epoch {epoch + 1}/{epochs}  D: {d_losses_hist[-1]:.4f}  G: {g_losses_hist[-1]:.4f}")

        with torch.no_grad():
            preview_noise = torch.randn(64, latent_dim, device=device)
            preview_labels = torch.arange(0, 10, device=device).repeat(7)[:64]
            samples = G(preview_noise, preview_labels)
            save_image((samples + 1) / 2,
                       os.path.join(sample_dir, f"epoch_{epoch + 1:03d}.png"), nrow=8)

    gen_path = os.path.join(model_dir, "generator.pt")
    torch.save(G.state_dict(), gen_path)
    torch.save(D.state_dict(), os.path.join(model_dir, "discriminator.pt"))
    _plot_loss_curves(g_losses_hist, d_losses_hist, os.path.join(model_dir, "loss_curves.png"))

    history = {
        "g_loss": g_losses_hist,
        "d_loss": d_losses_hist,
        "config": {
            "gen_type": gen_type, "conditioning": conditioning,
            "loss_type": loss_type, "latent_dim": latent_dim,
            "embedding_dim": embedding_dim, "disc_dropout": disc_dropout,
            "label_smoothing": label_smoothing, "n_critic": n_critic,
        },
    }
    save_json(history, os.path.join(model_dir, "loss_history.json"))

    # Save arch config so generate_synthetic / interpolate can reload the right model
    gen_config = {
        "gen_type": gen_type,
        "conditioning": conditioning,
        "latent_dim": latent_dim,
        "num_classes": 10,
        "embedding_dim": embedding_dim,
        "hidden_dims": hidden_dims,
    }
    save_json(gen_config, os.path.join(model_dir, "generator_config.json"))

    print(f"Saved GAN models to: {model_dir}")
    return gen_path


# ── Entry point ────────────────────────────────────────────────────────────

def train():
    args = parse_args()
    cap = args.max_train_samples if args.max_train_samples > 0 else None
    n_critic = args.n_critic
    if args.loss_type == "wgan_gp" and n_critic == 1:
        n_critic = 5  # WGAN-GP default
    train_gan_core(
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        seed=args.seed,
        num_workers=args.num_workers,
        model_dir=None,
        max_gan_train_samples=cap,
        gen_type=args.gen_type,
        conditioning=args.conditioning,
        embedding_dim=args.embedding_dim,
        hidden_dims=args.hidden_dims,
        disc_dropout=args.disc_dropout,
        loss_type=args.loss_type,
        label_smoothing=args.label_smoothing,
        generator_lr=args.generator_lr,
        discriminator_lr=args.discriminator_lr,
        n_critic=n_critic,
        gp_lambda=args.gp_lambda,
    )


if __name__ == "__main__":
    train()
