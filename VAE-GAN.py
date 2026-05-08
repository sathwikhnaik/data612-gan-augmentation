"""
VAE-GAN Hybrid – 7-Goal Experimental Study
===========================================

Goal 1  – Base Conditional VAE-GAN
        Train a Conditional VAE-GAN on MNIST / Fashion-MNIST to generate
        class-conditioned synthetic images.

Goal 2  – Three-Way Classification Comparison
        CNN trained on: real-only | real+synthetic | synthetic-only
        Metrics: Accuracy, Precision, Recall, F1.

Goal 3  – Architecture Variants
        Vary latent dim, encoder/decoder depth, generator type (MLP vs CNN),
        and discriminator dropout.

Goal 4  – Training Strategies
        Vary reconstruction / KL / adversarial loss weights, LR, batch size.

Goal 5  – Data Scarcity & Synthetic Volume
        Train on 500 / 2k / 5k / full real images to study augmentation benefit.

Goal 6  – Label Injection Methods
        Compare: learned embedding | one-hot labels | generator-only conditioning.

Goal 7  – Extended Augmentation Scenarios
        Mix ratios | weighted sampling | progressive augmentation |
        synthetic pre-training → real fine-tuning.

Quick start
-----------
python VAE-GAN.py --dataset mnist --quick                 # 2-min sanity check
python VAE-GAN.py --dataset mnist --goals 2               # base 3-way comparison
python VAE-GAN.py --dataset mnist --goals all             # all goals
python VAE-GAN.py --dataset mnist --goals 2,3,7 --gan-epochs 30 --clf-epochs 10
"""

from __future__ import annotations

import argparse, csv, json, os, random, shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional, Tuple

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MNIST_CLASSES   = [str(i) for i in range(10)]
FASHION_CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal",  "Shirt",   "Sneaker",  "Bag",   "Ankle boot"]

def class_names(dataset: str) -> List[str]:
    return FASHION_CLASSES if dataset == "fashion_mnist" else MNIST_CLASSES


# ─────────────────────────────────────────────────────────────────────────────
# Config dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelCfg:
    latent_dim:    int   = 100
    embedding_dim: int   = 50
    num_classes:   int   = 10
    gen_arch:      str   = "mlp"        # "mlp" | "cnn"
    depth:         str   = "shallow"    # "shallow" | "deep"
    conditioning:  str   = "embedding"  # "embedding" | "onehot" | "gen_only"
    disc_dropout:  float = 0.3


@dataclass
class TrainCfg:
    epochs:            int            = 30
    batch_size:        int            = 128
    lr:                float          = 2e-4
    kl_weight:         float          = 0.001
    recon_weight:      float          = 1.0
    adv_weight:        float          = 0.1
    max_train_samples: Optional[int]  = None


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mkdir(p: str) -> str:
    os.makedirs(p, exist_ok=True); return p

def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_json(d: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(d, f, indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    (image, label) → (μ, log σ²)

    conditioning="embedding" : learned label embedding concatenated to flattened image
    conditioning="onehot"    : one-hot vector concatenated to flattened image
    conditioning="gen_only"  : encoder sees the image only (no label)
    """

    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cond = cfg.conditioning
        self.nc   = cfg.num_classes

        if cfg.conditioning == "embedding":
            self.emb = nn.Embedding(cfg.num_classes, cfg.embedding_dim)
            in_dim   = 784 + cfg.embedding_dim
        elif cfg.conditioning == "onehot":
            in_dim   = 784 + cfg.num_classes
        else:                               # gen_only
            in_dim   = 784

        sizes = [1024, 512, 256] if cfg.depth == "deep" else [512, 256]
        layers, prev = [], in_dim
        for h in sizes:
            layers += [nn.Linear(prev, h), nn.LeakyReLU(0.2, True)]
            prev = h
        self.net  = nn.Sequential(*layers)
        self.mu   = nn.Linear(prev, cfg.latent_dim)
        self.logv = nn.Linear(prev, cfg.latent_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        flat = x.view(x.size(0), -1)
        if self.cond == "embedding":
            h = self.net(torch.cat([flat, self.emb(y)], 1))
        elif self.cond == "onehot":
            h = self.net(torch.cat([flat, F.one_hot(y, self.nc).float()], 1))
        else:
            h = self.net(flat)
        return self.mu(h), self.logv(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logv: torch.Tensor) -> torch.Tensor:
        return mu + torch.exp(0.5 * logv) * torch.randn_like(mu)


class Generator(nn.Module):
    """
    (z, label) → 28×28 image in [−1, 1]

    gen_arch="mlp" : fully-connected decoder
    gen_arch="cnn" : transposed-conv decoder (7×7 → 14×14 → 28×28)

    Generator always receives the label (even when conditioning="gen_only").
    """

    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.arch = cfg.gen_arch
        self.cond = cfg.conditioning
        self.nc   = cfg.num_classes

        if cfg.conditioning == "onehot":
            cond_dim = cfg.num_classes
        else:
            self.emb = nn.Embedding(cfg.num_classes, cfg.embedding_dim)
            cond_dim = cfg.embedding_dim

        in_dim = cfg.latent_dim + cond_dim

        if cfg.gen_arch == "mlp":
            sizes = [256, 512, 1024, 1024] if cfg.depth == "deep" else [256, 512, 1024]
            layers, prev = [], in_dim
            for h in sizes:
                layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.LeakyReLU(0.2, True)]
                prev = h
            layers += [nn.Linear(prev, 784), nn.Tanh()]
            self.net = nn.Sequential(*layers)
        else:                               # cnn
            self.proj   = nn.Linear(in_dim, 128 * 7 * 7)
            self.deconv = nn.Sequential(
                nn.BatchNorm2d(128), nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                nn.BatchNorm2d(64),  nn.ReLU(True),
                nn.ConvTranspose2d(64,  1,  4, stride=2, padding=1),
                nn.Tanh(),
            )

    def _cond_vec(self, y: torch.Tensor) -> torch.Tensor:
        if self.cond == "onehot":
            return F.one_hot(y, self.nc).float()
        return self.emb(y)

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = torch.cat([z, self._cond_vec(y)], 1)
        if self.arch == "mlp":
            return self.net(h).view(-1, 1, 28, 28)
        return self.deconv(self.proj(h).view(-1, 128, 7, 7))


class Discriminator(nn.Module):
    """(image, label) → real/fake probability"""

    def __init__(self, cfg: ModelCfg):
        super().__init__()
        dr = cfg.disc_dropout
        self.emb = nn.Embedding(cfg.num_classes, cfg.embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(784 + cfg.embedding_dim, 512), nn.LeakyReLU(0.2, True), nn.Dropout(dr),
            nn.Linear(512, 256),                     nn.LeakyReLU(0.2, True), nn.Dropout(dr),
            nn.Linear(256, 1),                       nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x.view(x.size(0), -1), self.emb(y)], 1))


class CNNClassifier(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ─────────────────────────────────────────────────────────────────────────────
# Data utilities
# ─────────────────────────────────────────────────────────────────────────────

def norm_transform() -> transforms.Compose:
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


def load_dataset(name: str, train: bool = True):
    t = norm_transform()
    if name == "mnist":
        return datasets.MNIST("data", train=train, transform=t, download=True)
    if name == "fashion_mnist":
        return datasets.FashionMNIST("data", train=train, transform=t, download=True)
    raise ValueError(f"Unknown dataset: {name}")


def stratified_indices(ds, n: int, seed: int) -> List[int]:
    n    = min(n, len(ds))
    rng  = np.random.default_rng(seed)
    tgts = np.array(ds.targets)
    nc   = int(tgts.max()) + 1
    per  = n // nc; rem = n % nc
    idx: List[int] = []
    for c in range(nc):
        pool = np.where(tgts == c)[0]
        take = min(per + (1 if c < rem else 0), len(pool))
        if take > 0:
            idx.extend(rng.choice(pool, take, replace=False).tolist())
    return idx


def cap_dataset(ds, n: Optional[int], seed: int):
    if not n or n >= len(ds):
        return ds
    return Subset(ds, stratified_indices(ds, n, seed))


class SyntheticDataset(Dataset):
    """Loads labeled PNGs from root/<class_int>/ directories."""

    def __init__(self, root: str):
        self.samples: List[Tuple[str, int]] = []
        self.transform = norm_transform()
        for cls_dir in sorted(os.listdir(root)):
            full = os.path.join(root, cls_dir)
            if not os.path.isdir(full):
                continue
            label = int(cls_dir)
            for f in os.listdir(full):
                if f.lower().endswith((".png", ".jpg")):
                    self.samples.append((os.path.join(full, f), label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, label = self.samples[i]
        return self.transform(Image.open(path).convert("L")), label


class CombinedDataset(Dataset):
    """Concatenation of multiple datasets."""

    def __init__(self, *dsets):
        self.dsets = [d for d in dsets if d is not None and len(d) > 0]
        self._cum  = []
        n = 0
        for d in self.dsets:
            n += len(d); self._cum.append(n)

    def __len__(self) -> int:
        return self._cum[-1] if self._cum else 0

    def __getitem__(self, i: int):
        for j, end in enumerate(self._cum):
            if i < end:
                start = 0 if j == 0 else self._cum[j - 1]
                return self.dsets[j][i - start]
        raise IndexError(i)


def cap_synthetic(syn_ds, real_size: int, ratio: float, seed: int):
    """Return synthetic dataset capped to int(real_size * ratio) images."""
    target = int(real_size * ratio)
    if target >= len(syn_ds):
        return syn_ds
    idx = np.random.default_rng(seed).choice(len(syn_ds), target, replace=False).tolist()
    return Subset(syn_ds, idx)


# ─────────────────────────────────────────────────────────────────────────────
# VAE-GAN training  (Goal 1)
# ─────────────────────────────────────────────────────────────────────────────

def train_vaegan(
    dataset: str,
    mcfg: ModelCfg,
    tcfg: TrainCfg,
    out_dir: str,
    seed: int = 42,
    num_workers: int = 2,
    verbose: bool = True,
) -> dict:
    """
    Train one Conditional VAE-GAN variant.

    Loss:
      D:   BCE(D(real,y), 1) + ½[BCE(D(recon,y), 0) + BCE(D(gen,y), 0)]
      E+G: recon_w·MSE(x̂,x) + kl_w·KL(q‖p) + adv_w·½[adv_recon + adv_gen]
    """
    set_seed(seed)
    dev = get_device()

    data   = cap_dataset(load_dataset(dataset, True), tcfg.max_train_samples, seed)
    loader = DataLoader(data, tcfg.batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=True, drop_last=True)

    E = Encoder(mcfg).to(dev)
    G = Generator(mcfg).to(dev)
    D = Discriminator(mcfg).to(dev)

    bce    = nn.BCELoss()
    mse    = nn.MSELoss()
    opt_d  = optim.Adam(D.parameters(), tcfg.lr, betas=(0.5, 0.999))
    opt_eg = optim.Adam(list(E.parameters()) + list(G.parameters()), tcfg.lr, betas=(0.5, 0.999))

    mkdir(out_dir)
    prev_dir = mkdir(os.path.join(out_dir, "preview"))
    ld = mcfg.latent_dim

    if verbose:
        print(f"    device={dev}  n={len(data)}  epochs={tcfg.epochs}  "
              f"kl={tcfg.kl_weight}  recon={tcfg.recon_weight}  adv={tcfg.adv_weight}")

    for epoch in range(tcfg.epochs):
        d_tot = eg_tot = 0.0
        it = tqdm(loader, desc=f"    ep {epoch+1}/{tcfg.epochs}", leave=False) if verbose else loader

        for x_real, y_real in it:
            bs = x_real.size(0)
            x_real, y_real = x_real.to(dev), y_real.to(dev)
            ones  = torch.ones(bs,  1, device=dev)
            zeros = torch.zeros(bs, 1, device=dev)

            # ── Discriminator step ────────────────────────────────────────────
            opt_d.zero_grad()
            with torch.no_grad():
                mu, logv = E(x_real, y_real)
                z_rec    = Encoder.reparameterize(mu, logv)
                x_rec    = G(z_rec, y_real)
                z_noise  = torch.randn(bs, ld, device=dev)
                y_rand   = torch.randint(0, mcfg.num_classes, (bs,), device=dev)
                x_gen    = G(z_noise, y_rand)

            d_loss = (bce(D(x_real, y_real), ones) +
                      0.5 * (bce(D(x_rec.detach(), y_real), zeros) +
                             bce(D(x_gen.detach(), y_rand), zeros)))
            d_loss.backward(); opt_d.step()
            d_tot += d_loss.item()

            # ── Encoder + Generator step ──────────────────────────────────────
            opt_eg.zero_grad()
            mu, logv = E(x_real, y_real)
            z_rec    = Encoder.reparameterize(mu, logv)
            x_rec    = G(z_rec, y_real)

            l_recon = tcfg.recon_weight * mse(x_rec, x_real)
            l_kl    = tcfg.kl_weight   * (-0.5 * torch.mean(1 + logv - mu.pow(2) - logv.exp()))
            l_adv_r = tcfg.adv_weight  * bce(D(x_rec, y_real), ones)

            z_noise = torch.randn(bs, ld, device=dev)
            y_rand  = torch.randint(0, mcfg.num_classes, (bs,), device=dev)
            x_gen   = G(z_noise, y_rand)
            l_adv_g = tcfg.adv_weight * bce(D(x_gen, y_rand), ones)

            eg_loss = l_recon + l_kl + 0.5 * (l_adv_r + l_adv_g)
            eg_loss.backward(); opt_eg.step()
            eg_tot += eg_loss.item()

        nb = len(loader)
        if verbose:
            print(f"    ep {epoch+1:3d}/{tcfg.epochs}  D={d_tot/nb:.4f}  EG={eg_tot/nb:.4f}")

        with torch.no_grad():
            z  = torch.randn(80, ld, device=dev)
            lb = torch.arange(10, device=dev).repeat(8)
            save_image((G(z, lb) + 1) / 2,
                       os.path.join(prev_dir, f"ep{epoch+1:03d}.png"), nrow=10)

    gen_path = os.path.join(out_dir, "generator.pt")
    enc_path = os.path.join(out_dir, "encoder.pt")
    torch.save(G.state_dict(), gen_path)
    torch.save(E.state_dict(), enc_path)
    torch.save(D.state_dict(), os.path.join(out_dir, "discriminator.pt"))
    save_json(asdict(mcfg), os.path.join(out_dir, "model_cfg.json"))

    return {"generator_path": gen_path, "encoder_path": enc_path, "model_dir": out_dir}


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic image generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic(gen_path: str, mcfg: ModelCfg, n: int, seed: int, out_root: str) -> str:
    """Write n class-balanced PNG images to out_root/<class_int>/."""
    set_seed(seed)
    dev = get_device()
    G = Generator(mcfg).to(dev)
    G.load_state_dict(torch.load(gen_path, map_location=dev, weights_only=True))
    G.eval()

    per = n // mcfg.num_classes; rem = n % mcfg.num_classes
    with torch.no_grad():
        for c in range(mcfg.num_classes):
            cls_dir = mkdir(os.path.join(out_root, str(c)))
            for i in range(per + (1 if c < rem else 0)):
                z   = torch.randn(1, mcfg.latent_dim, device=dev)
                y   = torch.tensor([c], device=dev)
                img = G(z, y)
                save_image(((img.clamp(-1, 1) + 1) / 2).repeat(1, 3, 1, 1),
                           os.path.join(cls_dir, f"{c}_{i:06d}.png"))
    return out_root


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(p), "recall": float(r), "f1_score": float(f)}


# ─────────────────────────────────────────────────────────────────────────────
# Classifier training – four variants
# ─────────────────────────────────────────────────────────────────────────────

def _train_epoch(model, loader, opt, crit, dev) -> float:
    model.train(); total = 0.0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad(); l = crit(model(x), y); l.backward(); opt.step()
        total += l.item()
    return total / max(len(loader), 1)


def _evaluate(model, loader, dev) -> Tuple[np.ndarray, np.ndarray]:
    model.eval(); yt, yp = [], []
    with torch.no_grad():
        for x, y in loader:
            p = torch.argmax(model(x.to(dev)), 1).cpu().numpy()
            yp.extend(p.tolist()); yt.extend(y.numpy().tolist())
    return np.array(yt), np.array(yp)


def _save_cm(y_true, y_pred, path: str, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues"); fig.colorbar(im, ax=ax)
    ax.set(xticks=range(10), yticks=range(10),
           xlabel="Predicted", ylabel="True", title=title)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def train_classifier(
    dataset: str, train_ds, epochs: int, batch_size: int, lr: float,
    num_workers: int, seed: int, out_dir: str, dev, verbose: bool = False,
) -> dict:
    mkdir(out_dir); set_seed(seed)
    loader  = DataLoader(train_ds, batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    testset = DataLoader(load_dataset(dataset, False), batch_size, shuffle=False, num_workers=num_workers)
    model   = CNNClassifier().to(dev)
    crit    = nn.CrossEntropyLoss()
    opt     = optim.Adam(model.parameters(), lr)

    for ep in range(epochs):
        loss = _train_epoch(model, loader, opt, crit, dev)
        if verbose:
            print(f"      ep {ep+1}/{epochs}  loss={loss:.4f}")

    y_true, y_pred = _evaluate(model, testset, dev)
    m = compute_metrics(y_true, y_pred)
    if verbose:
        print(f"      → acc={m['accuracy']:.4f}  f1={m['f1_score']:.4f}  n={len(train_ds)}")

    save_json(m, os.path.join(out_dir, "metrics.json"))
    _save_cm(y_true, y_pred, os.path.join(out_dir, "confusion_matrix.png"), out_dir.split(os.sep)[-1])
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    return {"metrics": m, "train_size": len(train_ds), "out_dir": out_dir}


def train_classifier_weighted(
    dataset: str, real_ds, syn_ds, real_weight: float,
    epochs: int, batch_size: int, lr: float,
    num_workers: int, seed: int, out_dir: str, dev, verbose: bool = False,
) -> dict:
    """Real images sampled real_weight× more often than synthetic."""
    mkdir(out_dir); set_seed(seed)
    combined = CombinedDataset(real_ds, syn_ds)
    weights  = [real_weight] * len(real_ds) + [1.0] * len(syn_ds)
    sampler  = WeightedRandomSampler(weights, len(combined), replacement=True)
    loader   = DataLoader(combined, batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
    testset  = DataLoader(load_dataset(dataset, False), batch_size, shuffle=False, num_workers=num_workers)
    model    = CNNClassifier().to(dev)
    crit     = nn.CrossEntropyLoss()
    opt      = optim.Adam(model.parameters(), lr)

    for ep in range(epochs):
        loss = _train_epoch(model, loader, opt, crit, dev)
        if verbose:
            print(f"      ep {ep+1}/{epochs}  loss={loss:.4f}")

    y_true, y_pred = _evaluate(model, testset, dev)
    m = compute_metrics(y_true, y_pred)
    save_json(m, os.path.join(out_dir, "metrics.json"))
    _save_cm(y_true, y_pred, os.path.join(out_dir, "confusion_matrix.png"), "weighted_mix")
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    return {"metrics": m, "train_size": len(combined), "out_dir": out_dir}


def train_classifier_progressive(
    dataset: str, real_ds, syn_ds, stages: List[Tuple[int, float]],
    batch_size: int, lr: float, num_workers: int, seed: int,
    out_dir: str, dev, verbose: bool = False,
) -> dict:
    """
    Staged training with increasing synthetic ratio.
    stages = [(n_epochs, syn_ratio), ...], e.g. [(3, 0.0), (4, 0.5), (3, 1.0)]
    """
    mkdir(out_dir); set_seed(seed)
    testset = DataLoader(load_dataset(dataset, False), batch_size, shuffle=False, num_workers=num_workers)
    model   = CNNClassifier().to(dev)
    crit    = nn.CrossEntropyLoss()
    opt     = optim.Adam(model.parameters(), lr)
    last_n  = 0

    for n_epochs, ratio in stages:
        if ratio <= 0:
            tr_ds = real_ds
        else:
            capped = cap_synthetic(syn_ds, len(real_ds), ratio, seed)
            tr_ds  = CombinedDataset(real_ds, capped)
        last_n = len(tr_ds)
        loader = DataLoader(tr_ds, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        for ep in range(n_epochs):
            loss = _train_epoch(model, loader, opt, crit, dev)
            if verbose:
                print(f"      stage ratio={ratio}  ep {ep+1}  loss={loss:.4f}")

    y_true, y_pred = _evaluate(model, testset, dev)
    m = compute_metrics(y_true, y_pred)
    save_json(m, os.path.join(out_dir, "metrics.json"))
    _save_cm(y_true, y_pred, os.path.join(out_dir, "confusion_matrix.png"), "progressive")
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    return {"metrics": m, "train_size": last_n, "out_dir": out_dir}


def train_classifier_pretrain_finetune(
    dataset: str, real_ds, syn_ds, pretrain_epochs: int, finetune_epochs: int,
    batch_size: int, lr: float, num_workers: int, seed: int,
    out_dir: str, dev, verbose: bool = False,
) -> dict:
    """Pretrain on synthetic data, then fine-tune on real data at 0.1× LR."""
    mkdir(out_dir); set_seed(seed)
    testset = DataLoader(load_dataset(dataset, False), batch_size, shuffle=False, num_workers=num_workers)
    model   = CNNClassifier().to(dev)
    crit    = nn.CrossEntropyLoss()

    opt        = optim.Adam(model.parameters(), lr)
    syn_loader = DataLoader(syn_ds, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    for ep in range(pretrain_epochs):
        loss = _train_epoch(model, syn_loader, opt, crit, dev)
        if verbose:
            print(f"      pretrain ep {ep+1}  loss={loss:.4f}")

    opt         = optim.Adam(model.parameters(), lr * 0.1)
    real_loader = DataLoader(real_ds, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    for ep in range(finetune_epochs):
        loss = _train_epoch(model, real_loader, opt, crit, dev)
        if verbose:
            print(f"      finetune ep {ep+1}  loss={loss:.4f}")

    y_true, y_pred = _evaluate(model, testset, dev)
    m = compute_metrics(y_true, y_pred)
    save_json(m, os.path.join(out_dir, "metrics.json"))
    _save_cm(y_true, y_pred, os.path.join(out_dir, "confusion_matrix.png"), "pretrain_ft")
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    return {"metrics": m, "train_size": len(real_ds) + len(syn_ds), "out_dir": out_dir}


# ─────────────────────────────────────────────────────────────────────────────
# Reporting utilities
# ─────────────────────────────────────────────────────────────────────────────

def save_bar_chart(rows: List[dict], path: str, title: str) -> None:
    keys  = ["accuracy", "precision", "recall", "f1_score"]
    names = [r["scenario"] for r in rows]
    x = list(range(len(names))); w = 0.18
    fig, ax = plt.subplots(figsize=(max(9, len(names) * 1.4), 5))
    for i, k in enumerate(keys):
        ax.bar([xi + (i - 1.5) * w for xi in x],
               [r["metrics"][k] for r in rows], w,
               label=k.replace("_", " ").title())
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def save_delta_chart(rows: List[dict], path: str, title: str) -> None:
    """Bar chart of ΔF1 improvement over real-only baseline."""
    names  = [r["variant"] for r in rows]
    deltas = [r["f1_improvement"] for r in rows]
    colors = ["steelblue" if d >= 0 else "tomato" for d in deltas]
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.3), 4))
    ax.bar(range(len(names)), deltas, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_ylabel("ΔF1  (with synthetic − real only)"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def save_csv(rows: List[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = ["scenario", "accuracy", "precision", "recall", "f1_score", "train_size", "out_dir"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fields); w.writeheader()
        for r in rows:
            m = r["metrics"]
            w.writerow({"scenario": r["scenario"],
                        "accuracy": m["accuracy"], "precision": m["precision"],
                        "recall":   m["recall"],   "f1_score":  m["f1_score"],
                        "train_size": r.get("train_size", ""),
                        "out_dir":    r.get("out_dir", "")})


def print_table(rows: List[dict], header: str) -> None:
    print(f"\n  {header}")
    print(f"  {'Scenario':<32}  Acc     Prec    Rec     F1      N")
    print(f"  {'-'*75}")
    for r in rows:
        m = r["metrics"]
        print(f"  {r['scenario']:<32}  "
              f"{m['accuracy']:.4f}  {m['precision']:.4f}  "
              f"{m['recall']:.4f}  {m['f1_score']:.4f}  "
              f"{r.get('train_size', '?')}")


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def save_sample_grid(gen_path: str, mcfg: ModelCfg, cnames: List[str],
                     out_path: str, n_per_class: int = 8, seed: int = 42) -> None:
    dev = get_device()
    G   = Generator(mcfg).to(dev)
    G.load_state_dict(torch.load(gen_path, map_location=dev, weights_only=True))
    G.eval(); set_seed(seed)

    fig, axes = plt.subplots(10, n_per_class, figsize=(n_per_class * 0.9, 11))
    with torch.no_grad():
        for c in range(10):
            z    = torch.randn(n_per_class, mcfg.latent_dim, device=dev)
            y    = torch.full((n_per_class,), c, device=dev)
            imgs = ((G(z, y).clamp(-1, 1) + 1) / 2)[:, 0].cpu().numpy()
            for j in range(n_per_class):
                axes[c][j].imshow(imgs[j], cmap="gray", vmin=0, vmax=1)
                axes[c][j].axis("off")
            axes[c][0].set_ylabel(cnames[c], fontsize=7, rotation=0, labelpad=40, va="center")

    fig.suptitle("VAE-GAN Generated Samples  (rows = classes)", fontsize=10)
    fig.tight_layout(); fig.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  [vis] sample grid → {out_path}")


def save_recon_grid(gen_path: str, enc_path: str, mcfg: ModelCfg, dataset: str,
                    cnames: List[str], out_path: str, n_per_class: int = 2, seed: int = 42) -> None:
    dev = get_device()
    G   = Generator(mcfg).to(dev)
    G.load_state_dict(torch.load(gen_path, map_location=dev, weights_only=True)); G.eval()
    E   = Encoder(mcfg).to(dev)
    E.load_state_dict(torch.load(enc_path, map_location=dev, weights_only=True)); E.eval()
    ds  = load_dataset(dataset, False); tgts = np.array(ds.targets); set_seed(seed)

    cols = n_per_class * 2
    fig, axes = plt.subplots(10, cols, figsize=(cols * 0.9, 11))
    with torch.no_grad():
        for c in range(10):
            pool   = np.where(tgts == c)[0]
            chosen = np.random.default_rng(seed + c).choice(pool, min(n_per_class, len(pool)), replace=False)
            col = 0
            for idx in chosen:
                xi, yi = ds[int(idx)]
                xt = xi.unsqueeze(0).to(dev); yt = torch.tensor([yi], device=dev)
                mu, logv = E(xt, yt); z_r = Encoder.reparameterize(mu, logv); x_r = G(z_r, yt)
                real_img  = ((xt.clamp(-1, 1) + 1) / 2)[0, 0].cpu().numpy()
                recon_img = ((x_r.clamp(-1, 1) + 1) / 2)[0, 0].cpu().numpy()
                axes[c][col].imshow(real_img,  cmap="gray", vmin=0, vmax=1); axes[c][col].axis("off")
                axes[c][col+1].imshow(recon_img, cmap="gray", vmin=0, vmax=1); axes[c][col+1].axis("off")
                if c == 0:
                    axes[c][col].set_title("Real", fontsize=7)
                    axes[c][col+1].set_title("Recon", fontsize=7)
                col += 2
            axes[c][0].set_ylabel(cnames[c], fontsize=7, rotation=0, labelpad=40, va="center")

    fig.suptitle("VAE-GAN: Real vs Reconstructed  (rows = classes)", fontsize=10)
    fig.tight_layout(); fig.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  [vis] reconstruction grid → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FID (optional)
# ─────────────────────────────────────────────────────────────────────────────

def compute_fid(dataset: str, syn_root: str, work_dir: str,
                n_images: int, seed: int, batch_size: int = 64) -> Optional[float]:
    try:
        from torch_fidelity import calculate_metrics
    except ImportError:
        print("  [FID] install: pip install torch-fidelity"); return None

    real_dir = os.path.join(work_dir, "real_pngs")
    if os.path.isdir(real_dir):
        shutil.rmtree(real_dir)
    mkdir(real_dir)

    ds = load_dataset(dataset, True); n_images = min(n_images, len(ds))
    for k, si in enumerate(stratified_indices(ds, n_images, seed)):
        img, _ = ds[si]; x = ((img.clamp(-1, 1) + 1) / 2).repeat(3, 1, 1)
        save_image(x, os.path.join(real_dir, f"r{k:06d}.png"))

    res = calculate_metrics(input1=real_dir, input2=syn_root,
                            cuda=torch.cuda.is_available(), fid=True, isc=False, kid=False,
                            verbose=False, batch_size=batch_size)
    fid = float(res["frechet_inception_distance"])
    print(f"  [FID] {fid:.4f}  (lower = better)"); return fid


# ─────────────────────────────────────────────────────────────────────────────
# Shared variant helper (used by Goals 3, 4, 5, 6)
# ─────────────────────────────────────────────────────────────────────────────

def _variant_run(
    tag: str, dataset: str, mcfg: ModelCfg, tcfg: TrainCfg,
    num_syn: int, clf_epochs: int, clf_lr: float, clf_bs: int,
    n_workers: int, seed: int, root: str, verbose_clf: bool, dev,
) -> dict:
    """Train one VAE-GAN variant and compare real_only vs real+syn classifiers."""
    gan_dir = mkdir(os.path.join(root, tag, "gan"))
    print(f"\n  ── variant: {tag}")
    result  = train_vaegan(dataset, mcfg, tcfg, gan_dir, seed, n_workers, verbose=True)

    syn_dir = mkdir(os.path.join(root, tag, "synthetic"))
    print(f"     generating {num_syn} synthetic images …")
    generate_synthetic(result["generator_path"], mcfg, num_syn, seed, syn_dir)

    real_ds = cap_dataset(load_dataset(dataset, True), tcfg.max_train_samples, seed)
    syn_ds  = SyntheticDataset(syn_dir)

    out_r = train_classifier(dataset, real_ds,
                             clf_epochs, clf_bs, clf_lr, n_workers, seed,
                             os.path.join(root, tag, "clf_real_only"), dev, verbose_clf)
    out_s = train_classifier(dataset, CombinedDataset(real_ds, syn_ds),
                             clf_epochs, clf_bs, clf_lr, n_workers, seed,
                             os.path.join(root, tag, "clf_real_plus_syn"), dev, verbose_clf)
    delta = out_s["metrics"]["f1_score"] - out_r["metrics"]["f1_score"]

    return {
        "variant": tag, "f1_improvement": delta,
        "real_only":     {"scenario": "real_only",     **out_r},
        "real_plus_syn": {"scenario": "real_plus_syn", **out_s},
        "gen_path": result["generator_path"],
        "enc_path": result["encoder_path"],
        "syn_dir":  syn_dir,
    }


# ═════════════════════════════════════════════════════════════════════════════
# GOAL 2 – Three-Way Classification Comparison
# ═════════════════════════════════════════════════════════════════════════════

def run_goal2(args, exp_root: str) -> List[dict]:
    """
    Train the base VAE-GAN (Goal 1) and evaluate three classifiers:
      real_only | real+synthetic | synthetic_only
    """
    root = mkdir(os.path.join(exp_root, "goal2_three_cases"))
    dev  = get_device()
    mcfg = ModelCfg()
    tcfg = TrainCfg(epochs=args.gan_epochs)

    print("\n  [Goal 2] Training base Conditional VAE-GAN …")
    gan_dir = mkdir(os.path.join(root, "gan"))
    result  = train_vaegan(args.dataset, mcfg, tcfg, gan_dir, args.seed, args.num_workers)

    max_real = args.max_real_samples if args.max_real_samples > 0 else None
    real_ds  = cap_dataset(load_dataset(args.dataset, True), max_real, args.seed)

    syn_dir = mkdir(os.path.join(root, "synthetic"))
    print(f"  [Goal 2] Generating {args.num_synthetic} synthetic images …")
    generate_synthetic(result["generator_path"], mcfg, args.num_synthetic, args.seed, syn_dir)
    syn_ds = SyntheticDataset(syn_dir)

    vis_dir = mkdir(os.path.join(root, "visuals"))
    save_sample_grid(result["generator_path"], mcfg, class_names(args.dataset),
                     os.path.join(vis_dir, "sample_grid.png"), seed=args.seed)
    save_recon_grid(result["generator_path"], result["encoder_path"], mcfg,
                    args.dataset, class_names(args.dataset),
                    os.path.join(vis_dir, "reconstruction_grid.png"), seed=args.seed)

    if args.compute_fid:
        print("  [Goal 2] Computing FID …")
        fid = compute_fid(args.dataset, syn_dir,
                          os.path.join(root, "fid_work"),
                          args.fid_num_images, args.seed, args.fid_batch_size)
        save_json({"fid": fid}, os.path.join(root, "fid.json"))

    kw = dict(epochs=args.clf_epochs, batch_size=args.clf_batch_size, lr=args.clf_lr,
              num_workers=args.num_workers, seed=args.seed, dev=dev, verbose=args.verbose_clf)

    rows = []
    for name, ds in [("real_only",      real_ds),
                     ("real_plus_syn",  CombinedDataset(real_ds, syn_ds)),
                     ("synthetic_only", syn_ds)]:
        print(f"  [Goal 2] Training classifier: {name} …")
        r = train_classifier(args.dataset, ds,
                             out_dir=os.path.join(root, f"clf_{name}"), **kw)
        rows.append({"scenario": name, **r})
        m = r["metrics"]
        print(f"  ✓ {name:<22}  acc={m['accuracy']:.4f}  f1={m['f1_score']:.4f}")

    save_csv(rows, os.path.join(root, "three_cases.csv"))
    save_bar_chart(rows, os.path.join(root, "three_cases.png"),
                   f"Goal 2: Three-Way Comparison | {args.dataset}")
    save_json({"dataset": args.dataset, "runs": rows}, os.path.join(root, "results.json"))
    print_table(rows, f"Goal 2 – Three-Way Comparison ({args.dataset})")
    return rows


# ═════════════════════════════════════════════════════════════════════════════
# GOAL 3 – Architecture Variants
# ═════════════════════════════════════════════════════════════════════════════

ARCH_VARIANTS = [
    dict(name="base_mlp_shallow",   latent_dim=100, emb_dim=50,  gen_arch="mlp", depth="shallow", dropout=0.3),
    dict(name="large_latent_256",   latent_dim=256, emb_dim=100, gen_arch="mlp", depth="shallow", dropout=0.3),
    dict(name="small_latent_32",    latent_dim=32,  emb_dim=16,  gen_arch="mlp", depth="shallow", dropout=0.3),
    dict(name="deep_mlp",           latent_dim=100, emb_dim=50,  gen_arch="mlp", depth="deep",    dropout=0.3),
    dict(name="cnn_generator",      latent_dim=100, emb_dim=50,  gen_arch="cnn", depth="shallow", dropout=0.3),
    dict(name="no_disc_dropout",    latent_dim=100, emb_dim=50,  gen_arch="mlp", depth="shallow", dropout=0.0),
    dict(name="high_disc_dropout",  latent_dim=100, emb_dim=50,  gen_arch="mlp", depth="shallow", dropout=0.5),
]


def run_goal3(args, exp_root: str) -> List[dict]:
    """Architecture variant ablation."""
    root     = mkdir(os.path.join(exp_root, "goal3_architecture"))
    dev      = get_device()
    tcfg     = TrainCfg(epochs=args.gan_epochs)
    variants = ARCH_VARIANTS[:args.subset] if args.subset else ARCH_VARIANTS
    rows     = []

    for v in variants:
        mcfg = ModelCfg(latent_dim=v["latent_dim"], embedding_dim=v["emb_dim"],
                        gen_arch=v["gen_arch"], depth=v["depth"], disc_dropout=v["dropout"])
        r = _variant_run(v["name"], args.dataset, mcfg, tcfg,
                         args.num_synthetic, args.clf_epochs, args.clf_lr, args.clf_batch_size,
                         args.num_workers, args.seed, root, args.verbose_clf, dev)
        rows.append(r)

    flat = [{"scenario": r["variant"], **r["real_plus_syn"]} for r in rows]
    save_csv(flat, os.path.join(root, "architecture_variants.csv"))
    save_bar_chart(flat, os.path.join(root, "architecture_variants.png"),
                   f"Goal 3: Architecture Variants | {args.dataset}")
    save_delta_chart(rows, os.path.join(root, "architecture_delta.png"),
                     f"Goal 3: Architecture ΔF1 | {args.dataset}")
    save_json([{k: v for k, v in r.items() if k not in ("gen_path", "enc_path", "syn_dir")}
               for r in rows], os.path.join(root, "results.json"))

    print(f"\n  [Goal 3] {'Variant':<24}  real_only  real+syn   ΔF1")
    for r in rows:
        print(f"    {r['variant']:<24}  "
              f"{r['real_only']['metrics']['f1_score']:.4f}     "
              f"{r['real_plus_syn']['metrics']['f1_score']:.4f}     "
              f"{r['f1_improvement']:+.4f}")
    return rows


# ═════════════════════════════════════════════════════════════════════════════
# GOAL 4 – Training Strategies
# ═════════════════════════════════════════════════════════════════════════════

TRAIN_STRATEGIES = [
    dict(name="balanced",    kl=0.001, recon=1.0,  adv=0.1,  lr=2e-4, bs=128),
    dict(name="recon_heavy", kl=0.001, recon=10.0, adv=0.1,  lr=2e-4, bs=128),
    dict(name="adv_heavy",   kl=0.001, recon=1.0,  adv=1.0,  lr=2e-4, bs=128),
    dict(name="kl_heavy",    kl=0.1,   recon=1.0,  adv=0.1,  lr=2e-4, bs=128),
    dict(name="no_kl",       kl=0.0,   recon=1.0,  adv=0.1,  lr=2e-4, bs=128),
    dict(name="low_lr",      kl=0.001, recon=1.0,  adv=0.1,  lr=5e-5, bs=128),
    dict(name="high_lr",     kl=0.001, recon=1.0,  adv=0.1,  lr=5e-4, bs=128),
    dict(name="large_batch", kl=0.001, recon=1.0,  adv=0.1,  lr=2e-4, bs=256),
    dict(name="small_batch", kl=0.001, recon=1.0,  adv=0.1,  lr=2e-4, bs=64),
]


def run_goal4(args, exp_root: str) -> List[dict]:
    """Training strategy ablation: loss weights, LR, batch size."""
    root       = mkdir(os.path.join(exp_root, "goal4_training_strategy"))
    dev        = get_device(); mcfg = ModelCfg()
    strategies = TRAIN_STRATEGIES[:args.subset] if args.subset else TRAIN_STRATEGIES
    rows       = []

    for s in strategies:
        tcfg = TrainCfg(epochs=args.gan_epochs, batch_size=s["bs"], lr=s["lr"],
                        kl_weight=s["kl"], recon_weight=s["recon"], adv_weight=s["adv"])
        r = _variant_run(s["name"], args.dataset, mcfg, tcfg,
                         args.num_synthetic, args.clf_epochs, args.clf_lr, args.clf_batch_size,
                         args.num_workers, args.seed, root, args.verbose_clf, dev)
        rows.append(r)

    flat = [{"scenario": r["variant"], **r["real_plus_syn"]} for r in rows]
    save_csv(flat, os.path.join(root, "training_strategies.csv"))
    save_bar_chart(flat, os.path.join(root, "training_strategies.png"),
                   f"Goal 4: Training Strategies | {args.dataset}")
    save_delta_chart(rows, os.path.join(root, "training_strategies_delta.png"),
                     f"Goal 4: Training Strategy ΔF1 | {args.dataset}")
    save_json([{k: v for k, v in r.items() if k not in ("gen_path", "enc_path", "syn_dir")}
               for r in rows], os.path.join(root, "results.json"))

    print(f"\n  [Goal 4] {'Strategy':<14}  real_only  real+syn   ΔF1")
    for r in rows:
        print(f"    {r['variant']:<14}  "
              f"{r['real_only']['metrics']['f1_score']:.4f}     "
              f"{r['real_plus_syn']['metrics']['f1_score']:.4f}     "
              f"{r['f1_improvement']:+.4f}")
    return rows


# ═════════════════════════════════════════════════════════════════════════════
# GOAL 5 – Data Scarcity & Synthetic Volume
# ═════════════════════════════════════════════════════════════════════════════

SCARCITY_CONFIGS = [
    dict(name="full_real", max_real=None, num_syn=12000),
    dict(name="5k_real",   max_real=5000, num_syn=5000),
    dict(name="2k_real",   max_real=2000, num_syn=4000),
    dict(name="500_real",  max_real=500,  num_syn=2000),
]


def run_goal5(args, exp_root: str) -> List[dict]:
    """Data scarcity study: vary real training size and synthetic volume."""
    root    = mkdir(os.path.join(exp_root, "goal5_data_scarcity"))
    dev     = get_device(); mcfg = ModelCfg()
    configs = SCARCITY_CONFIGS[:args.subset] if args.subset else SCARCITY_CONFIGS

    if args.quick:
        configs = [dict(c, num_syn=min(c["num_syn"], 500)) for c in configs]

    rows = []
    for c in configs:
        tcfg = TrainCfg(epochs=args.gan_epochs, max_train_samples=c["max_real"])
        r = _variant_run(c["name"], args.dataset, mcfg, tcfg,
                         c["num_syn"], args.clf_epochs, args.clf_lr, args.clf_batch_size,
                         args.num_workers, args.seed, root, args.verbose_clf, dev)
        rows.append(r)

    scenarios = []
    for r in rows:
        scenarios.append({"scenario": r["variant"] + "_real_only", **r["real_only"]})
        scenarios.append({"scenario": r["variant"] + "_+syn",      **r["real_plus_syn"]})

    save_csv(scenarios, os.path.join(root, "data_scarcity.csv"))
    save_bar_chart(scenarios, os.path.join(root, "data_scarcity.png"),
                   f"Goal 5: Data Scarcity | {args.dataset}")
    save_delta_chart(rows, os.path.join(root, "data_scarcity_delta.png"),
                     f"Goal 5: Scarcity ΔF1 | {args.dataset}")
    save_json([{k: v for k, v in r.items() if k not in ("gen_path", "enc_path", "syn_dir")}
               for r in rows], os.path.join(root, "results.json"))

    print(f"\n  [Goal 5] {'Setting':<12}  real_only  real+syn   ΔF1")
    for r in rows:
        print(f"    {r['variant']:<12}  "
              f"{r['real_only']['metrics']['f1_score']:.4f}     "
              f"{r['real_plus_syn']['metrics']['f1_score']:.4f}     "
              f"{r['f1_improvement']:+.4f}")
    return rows


# ═════════════════════════════════════════════════════════════════════════════
# GOAL 6 – Label Injection Methods
# ═════════════════════════════════════════════════════════════════════════════

CONDITIONING_METHODS = [
    dict(name="learned_embedding",  conditioning="embedding", emb_dim=50),
    dict(name="onehot_labels",      conditioning="onehot",    emb_dim=10),
    dict(name="gen_only_embedding", conditioning="gen_only",  emb_dim=50),
]


def run_goal6(args, exp_root: str) -> List[dict]:
    """Label injection method comparison."""
    root    = mkdir(os.path.join(exp_root, "goal6_label_injection"))
    dev     = get_device()
    tcfg    = TrainCfg(epochs=args.gan_epochs)
    methods = CONDITIONING_METHODS[:args.subset] if args.subset else CONDITIONING_METHODS
    rows    = []

    for m in methods:
        mcfg = ModelCfg(conditioning=m["conditioning"], embedding_dim=m["emb_dim"])
        r = _variant_run(m["name"], args.dataset, mcfg, tcfg,
                         args.num_synthetic, args.clf_epochs, args.clf_lr, args.clf_batch_size,
                         args.num_workers, args.seed, root, args.verbose_clf, dev)
        rows.append(r)

    flat = [{"scenario": r["variant"], **r["real_plus_syn"]} for r in rows]
    save_csv(flat, os.path.join(root, "label_injection.csv"))
    save_bar_chart(flat, os.path.join(root, "label_injection.png"),
                   f"Goal 6: Label Injection Methods | {args.dataset}")
    save_delta_chart(rows, os.path.join(root, "label_injection_delta.png"),
                     f"Goal 6: Label Injection ΔF1 | {args.dataset}")
    save_json([{k: v for k, v in r.items() if k not in ("gen_path", "enc_path", "syn_dir")}
               for r in rows], os.path.join(root, "results.json"))

    print(f"\n  [Goal 6] {'Method':<24}  real_only  real+syn   ΔF1")
    for r in rows:
        print(f"    {r['variant']:<24}  "
              f"{r['real_only']['metrics']['f1_score']:.4f}     "
              f"{r['real_plus_syn']['metrics']['f1_score']:.4f}     "
              f"{r['f1_improvement']:+.4f}")
    return rows


# ═════════════════════════════════════════════════════════════════════════════
# GOAL 7 – Extended Augmentation Scenarios
# ═════════════════════════════════════════════════════════════════════════════

def run_goal7(args, exp_root: str) -> List[dict]:
    """
    Extended augmentation scenarios beyond simple real+syn:
      - 25 / 50 / 100 / 200% synthetic mix ratios
      - weighted sampling (real images sampled 2× more)
      - progressive augmentation (staged synthetic introduction)
      - synthetic pre-training → real fine-tuning
    """
    root = mkdir(os.path.join(exp_root, "goal7_augmentation_scenarios"))
    dev  = get_device()
    mcfg = ModelCfg()
    tcfg = TrainCfg(epochs=args.gan_epochs)

    print("\n  [Goal 7] Training base VAE-GAN …")
    gan_dir = mkdir(os.path.join(root, "gan"))
    result  = train_vaegan(args.dataset, mcfg, tcfg, gan_dir, args.seed, args.num_workers)

    max_real = args.max_real_samples if args.max_real_samples > 0 else None
    real_ds  = cap_dataset(load_dataset(args.dataset, True), max_real, args.seed)
    real_n   = len(real_ds)

    num_syn = max(args.num_synthetic, int(real_n * 2.0))
    syn_dir = mkdir(os.path.join(root, "synthetic"))
    print(f"  [Goal 7] Generating {num_syn} synthetic images …")
    generate_synthetic(result["generator_path"], mcfg, num_syn, args.seed, syn_dir)
    syn_full = SyntheticDataset(syn_dir)

    rows: List[dict] = []

    def add(name: str, res: dict) -> None:
        rows.append({"scenario": name, **res})
        m = res["metrics"]
        print(f"  ✓ {name:<32}  acc={m['accuracy']:.4f}  f1={m['f1_score']:.4f}")

    print("\n  [Goal 7] Running augmentation scenarios …")

    # Baseline
    add("real_only",
        train_classifier(args.dataset, real_ds,
                         args.clf_epochs, args.clf_batch_size, args.clf_lr,
                         args.num_workers, args.seed,
                         os.path.join(root, "clf_real_only"), dev, args.verbose_clf))

    # Mix ratios: 25 / 50 / 100 / 200%
    for pct, ratio in [(25, 0.25), (50, 0.50), (100, 1.00), (200, 2.00)]:
        syn_capped = cap_synthetic(syn_full, real_n, ratio, args.seed)
        add(f"real_plus_{pct}pct_syn",
            train_classifier(args.dataset, CombinedDataset(real_ds, syn_capped),
                             args.clf_epochs, args.clf_batch_size, args.clf_lr,
                             args.num_workers, args.seed,
                             os.path.join(root, f"clf_{pct}pct"), dev, args.verbose_clf))

    # Weighted sampling: real images weighted 2× over synthetic
    add("weighted_mix_real2x",
        train_classifier_weighted(
            args.dataset, real_ds,
            cap_synthetic(syn_full, real_n, 1.0, args.seed),
            2.0,
            args.clf_epochs, args.clf_batch_size, args.clf_lr,
            args.num_workers, args.seed,
            os.path.join(root, "clf_weighted"), dev, args.verbose_clf))

    # Progressive augmentation: real-only → 50% syn → 100% syn
    ep3    = max(1, args.clf_epochs // 3)
    stages = [(ep3, 0.0), (ep3, 0.5), (args.clf_epochs - 2 * ep3, 1.0)]
    add("progressive_aug",
        train_classifier_progressive(
            args.dataset, real_ds, syn_full, stages,
            args.clf_batch_size, args.clf_lr,
            args.num_workers, args.seed,
            os.path.join(root, "clf_progressive"), dev, args.verbose_clf))

    # Synthetic pre-training → real fine-tuning
    half = max(1, args.clf_epochs // 2)
    add("syn_pretrain_finetune",
        train_classifier_pretrain_finetune(
            args.dataset, real_ds, syn_full,
            half, args.clf_epochs - half,
            args.clf_batch_size, args.clf_lr,
            args.num_workers, args.seed,
            os.path.join(root, "clf_pretrain_ft"), dev, args.verbose_clf))

    save_csv(rows, os.path.join(root, "augmentation_scenarios.csv"))
    save_bar_chart(rows, os.path.join(root, "augmentation_scenarios.png"),
                   f"Goal 7: Extended Augmentation Scenarios | {args.dataset}")
    save_json({"dataset": args.dataset, "runs": rows}, os.path.join(root, "results.json"))
    print_table(rows, f"Goal 7 – Extended Augmentation Scenarios ({args.dataset})")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

GOAL_RUNNERS = {
    "2": (run_goal2, "Three-Way Classification Comparison"),
    "3": (run_goal3, "Architecture Variants"),
    "4": (run_goal4, "Training Strategies"),
    "5": (run_goal5, "Data Scarcity & Synthetic Volume"),
    "6": (run_goal6, "Label Injection Methods"),
    "7": (run_goal7, "Extended Augmentation Scenarios"),
}


def run(args: argparse.Namespace) -> None:
    root = args.output_dir or os.path.join("outputs", f"vaegan_{args.dataset}_{ts()}")
    mkdir(root)

    selected = (set(GOAL_RUNNERS.keys()) if args.goals.lower() == "all"
                else {g.strip() for g in args.goals.split(",")})

    print(f"\n{'='*68}")
    print(f"  VAE-GAN Hybrid – 7-Goal Experimental Study")
    print(f"  dataset={args.dataset}  goals={args.goals}  seed={args.seed}")
    print(f"  output → {root}")
    print(f"{'='*68}")

    for gid in sorted(selected):
        if gid not in GOAL_RUNNERS:
            print(f"  [skip] Goal {gid} not recognized (valid: 2–7; Goal 1 is implicit)")
            continue
        fn, name = GOAL_RUNNERS[gid]
        print(f"\n{'═'*68}\n  GOAL {gid} – {name}\n{'═'*68}")
        fn(args, root)

    print(f"\n{'='*68}\n  Done → {root}\n{'='*68}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="VAE-GAN Hybrid – 7-Goal Experiments (MNIST / Fashion-MNIST)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--goals",   default="2",
                   help="Comma-separated goal IDs (2-7) or 'all'. Goal 1 is always implicit.")
    p.add_argument("--subset",  type=int, default=0,
                   help="Run only first N configs per goal (0 = all).")
    p.add_argument("--quick",   action="store_true",
                   help="Minimal epochs/samples for fast end-to-end testing.")

    p.add_argument("--dataset",          default="mnist", choices=["mnist", "fashion_mnist"])
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--num-workers",      type=int,   default=2)
    p.add_argument("--max-real-samples", type=int,   default=0,
                   help="Cap on real training images for Goals 2 & 7 (0 = full dataset).")

    p.add_argument("--gan-epochs",     type=int,   default=30)
    p.add_argument("--gan-batch-size", type=int,   default=128)

    p.add_argument("--num-synthetic",  type=int,   default=12000)

    p.add_argument("--clf-epochs",     type=int,   default=10)
    p.add_argument("--clf-batch-size", type=int,   default=128)
    p.add_argument("--clf-lr",         type=float, default=1e-3)
    p.add_argument("--verbose-clf",    action="store_true",
                   help="Print per-epoch classifier training loss.")

    p.add_argument("--compute-fid",    action="store_true")
    p.add_argument("--fid-num-images", type=int,   default=5000)
    p.add_argument("--fid-batch-size", type=int,   default=64)

    p.add_argument("--output-dir",     type=str,   default="",
                   help="Root output directory (auto-generated if empty).")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()

    if args.quick:
        args.gan_epochs    = 2
        args.clf_epochs    = 1
        args.num_synthetic = 200
        if not args.subset:
            args.subset = 2
        print("[quick mode]  gan_epochs=2  clf_epochs=1  num_syn=200  subset=2")

    run(args)
