from dataclasses import dataclass

# Human-readable names for per-class reporting (10 classes each).
MNIST_CLASS_NAMES = [str(i) for i in range(10)]
FASHION_MNIST_CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def class_names_for_dataset(dataset_name: str) -> list:
    if dataset_name == "fashion_mnist":
        return FASHION_MNIST_CLASS_NAMES
    return MNIST_CLASS_NAMES


@dataclass
class DataConfig:
    dataset: str = "mnist"  # mnist | fashion_mnist
    data_root: str = "data"
    image_size: int = 28
    channels: int = 1
    num_classes: int = 10


@dataclass
class GANConfig:
    latent_dim: int = 100
    embedding_dim: int = 50
    generator_lr: float = 2e-4
    discriminator_lr: float = 2e-4
    betas: tuple = (0.5, 0.999)


@dataclass
class TrainConfig:
    batch_size: int = 128
    epochs: int = 30
    seed: int = 42
    num_workers: int = 2
