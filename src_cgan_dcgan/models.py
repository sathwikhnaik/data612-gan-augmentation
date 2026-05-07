"""
Generative models: conditional GAN (cGAN) [Mirza & Osindero, 2014].

Architectures
─────────────
MLP (default)
  ConditionalGenerator  — (z, y) → MLP → 28×28, learned embedding for y
  OneHotGenerator       — same but y is a raw 10-dim one-hot vector
  ConditionalDiscriminator — (x, y) → MLP → score, learned embedding for y
  OneHotDiscriminator      — same but y is a raw one-hot vector
  ProjectionDiscriminator  — (x, y) → feature MLP + inner-product projection [Miyato & Koyama 2018]

DCGAN (convolutional)
  DCGANGenerator       — (z, y) → FC → reshape 7×7 → ConvTranspose 28×28
  DCGANDiscriminator   — (x + projected_label_map) → Conv → score

Factory
  build_generator(gen_type, conditioning, ...)
  build_discriminator(gen_type, conditioning, ...)
"""
import torch
import torch.nn as nn


# ── Helpers ────────────────────────────────────────────────────────────────

def _mlp_block(in_dim: int, hidden_dims: list[int]) -> tuple[nn.Sequential, int]:
    """Build a LeakyReLU MLP body; returns (Sequential, out_dim)."""
    layers: list[nn.Module] = []
    prev = in_dim
    for i, h in enumerate(hidden_dims):
        layers.append(nn.Linear(prev, h))
        if i > 0:
            layers.append(nn.BatchNorm1d(h))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        prev = h
    return nn.Sequential(*layers), prev


# ── MLP cGAN — learned embedding ──────────────────────────────────────────

class ConditionalGenerator(nn.Module):
    def __init__(
        self,
        latent_dim: int = 100,
        num_classes: int = 10,
        embedding_dim: int = 50,
        hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 512, 1024]
        self.label_emb = nn.Embedding(num_classes, embedding_dim)
        body, out_dim = _mlp_block(latent_dim + embedding_dim, hidden_dims)
        self.net = nn.Sequential(body, nn.Linear(out_dim, 28 * 28), nn.Tanh())

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x = torch.cat([noise, self.label_emb(labels)], dim=1)
        return self.net(x).view(x.size(0), 1, 28, 28)


class ConditionalDiscriminator(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        embedding_dim: int = 50,
        dropout: float = 0.3,
        use_sigmoid: bool = True,
    ):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, embedding_dim)
        layers: list[nn.Module] = [
            nn.Linear(28 * 28 + embedding_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        ]
        if use_sigmoid:
            layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x = torch.cat([images.view(images.size(0), -1), self.label_emb(labels)], dim=1)
        return self.net(x)


# ── MLP cGAN — one-hot conditioning ───────────────────────────────────────

class OneHotGenerator(nn.Module):
    """y is encoded as a raw one-hot vector instead of a learned embedding."""

    def __init__(
        self,
        latent_dim: int = 100,
        num_classes: int = 10,
        hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        if hidden_dims is None:
            hidden_dims = [256, 512, 1024]
        body, out_dim = _mlp_block(latent_dim + num_classes, hidden_dims)
        self.net = nn.Sequential(body, nn.Linear(out_dim, 28 * 28), nn.Tanh())

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        one_hot = torch.zeros(labels.size(0), self.num_classes, device=labels.device)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        x = torch.cat([noise, one_hot], dim=1)
        return self.net(x).view(x.size(0), 1, 28, 28)


class OneHotDiscriminator(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        dropout: float = 0.3,
        use_sigmoid: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        layers: list[nn.Module] = [
            nn.Linear(28 * 28 + num_classes, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        ]
        if use_sigmoid:
            layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        flat = images.view(images.size(0), -1)
        one_hot = torch.zeros(labels.size(0), self.num_classes, device=labels.device)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        return self.net(torch.cat([flat, one_hot], dim=1))


# ── Projection discriminator [Miyato & Koyama, 2018] ──────────────────────

class ProjectionDiscriminator(nn.Module):
    """
    output(x, y) = v^T φ(x) + e_y^T φ(x)
    where φ is a feature MLP and e_y is the label embedding.
    Designed for hinge / WGAN-GP loss (use_sigmoid=False by default).
    """

    def __init__(
        self,
        num_classes: int = 10,
        feature_dim: int = 256,
        dropout: float = 0.3,
        use_sigmoid: bool = False,
    ):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, feature_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.linear = nn.Linear(feature_dim, 1)
        self.label_emb = nn.Embedding(num_classes, feature_dim)
        self.use_sigmoid = use_sigmoid

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        phi = self.phi(images.view(images.size(0), -1))
        out = self.linear(phi) + (phi * self.label_emb(labels)).sum(dim=1, keepdim=True)
        return torch.sigmoid(out) if self.use_sigmoid else out


# ── DCGAN — convolutional ──────────────────────────────────────────────────

class DCGANGenerator(nn.Module):
    """
    (z + label_emb) → Linear → (256, 7, 7) → ConvT(14×14) → ConvT(28×28).
    """

    def __init__(
        self,
        latent_dim: int = 100,
        num_classes: int = 10,
        embedding_dim: int = 50,
    ):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, embedding_dim)
        self.fc = nn.Linear(latent_dim + embedding_dim, 256 * 7 * 7)
        self.deconv = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # →14×14
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1),    # →28×28
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x = torch.cat([noise, self.label_emb(labels)], dim=1)
        return self.deconv(self.fc(x).view(-1, 256, 7, 7))


class DCGANDiscriminator(nn.Module):
    """
    Concatenates a spatially-projected label map as a second channel,
    then scores via strided convolutions.
    """

    def __init__(
        self,
        num_classes: int = 10,
        dropout: float = 0.3,
        use_sigmoid: bool = True,
    ):
        super().__init__()
        # Project label embedding to a full 28×28 spatial map
        self.label_spatial = nn.Embedding(num_classes, 28 * 28)
        self.conv = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1),        # →14×14
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),      # →7×7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),
        )
        self.fc = nn.Linear(128 * 7 * 7, 1)
        self.use_sigmoid = use_sigmoid

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_map = self.label_spatial(labels).view(-1, 1, 28, 28)
        x = self.conv(torch.cat([images, label_map], dim=1)).view(images.size(0), -1)
        out = self.fc(x)
        return torch.sigmoid(out) if self.use_sigmoid else out


# ── CNN Classifier ─────────────────────────────────────────────────────────

class SimpleCNNClassifier(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ── Factory functions ──────────────────────────────────────────────────────

def build_generator(
    gen_type: str = "mlp",
    conditioning: str = "embedding",
    latent_dim: int = 100,
    num_classes: int = 10,
    embedding_dim: int = 50,
    hidden_dims: list[int] | None = None,
) -> nn.Module:
    """
    gen_type    : 'mlp' | 'dcgan'
    conditioning: 'embedding' | 'onehot'   (dcgan always uses embedding)
    """
    if conditioning == "onehot":
        return OneHotGenerator(latent_dim=latent_dim, num_classes=num_classes,
                               hidden_dims=hidden_dims)
    if gen_type == "dcgan":
        return DCGANGenerator(latent_dim=latent_dim, num_classes=num_classes,
                              embedding_dim=embedding_dim)
    return ConditionalGenerator(latent_dim=latent_dim, num_classes=num_classes,
                                embedding_dim=embedding_dim, hidden_dims=hidden_dims)


def build_discriminator(
    gen_type: str = "mlp",
    conditioning: str = "embedding",
    num_classes: int = 10,
    embedding_dim: int = 50,
    dropout: float = 0.3,
    use_sigmoid: bool = True,
) -> nn.Module:
    """
    gen_type    : 'mlp' | 'dcgan'
    conditioning: 'embedding' | 'onehot' | 'projection'
    """
    if conditioning == "onehot":
        return OneHotDiscriminator(num_classes=num_classes, dropout=dropout,
                                   use_sigmoid=use_sigmoid)
    if conditioning == "projection":
        return ProjectionDiscriminator(num_classes=num_classes, dropout=dropout,
                                       use_sigmoid=use_sigmoid)
    if gen_type == "dcgan":
        return DCGANDiscriminator(num_classes=num_classes, dropout=dropout,
                                  use_sigmoid=use_sigmoid)
    return ConditionalDiscriminator(num_classes=num_classes, embedding_dim=embedding_dim,
                                    dropout=dropout, use_sigmoid=use_sigmoid)


# Backward-compat aliases
cGANGenerator = ConditionalGenerator
cGANDiscriminator = ConditionalDiscriminator
