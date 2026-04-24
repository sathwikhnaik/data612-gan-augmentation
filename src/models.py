"""
Generative models: conditional GAN (cGAN) [Mirza & Osindero, 2014].

Both networks take a class label y: the generator maps (z, y) -> x;
the discriminator scores (x, y). This matches the project proposal.
"""
import torch
import torch.nn as nn


class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim: int = 100, num_classes: int = 10, embedding_dim: int = 50):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, embedding_dim)
        input_dim = latent_dim + embedding_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 28 * 28),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_vec = self.label_emb(labels)
        x = torch.cat([noise, label_vec], dim=1)
        out = self.net(x)
        return out.view(out.size(0), 1, 28, 28)


class ConditionalDiscriminator(nn.Module):
    def __init__(self, num_classes: int = 10, embedding_dim: int = 50):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(28 * 28 + embedding_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        flat = images.view(images.size(0), -1)
        label_vec = self.label_emb(labels)
        x = torch.cat([flat, label_vec], dim=1)
        return self.net(x)


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


# Explicit aliases for papers / proposals that refer to "cGAN".
cGANGenerator = ConditionalGenerator
cGANDiscriminator = ConditionalDiscriminator
