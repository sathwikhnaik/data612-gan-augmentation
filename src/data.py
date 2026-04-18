import os
from typing import Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def _dataset_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


def get_base_dataset(dataset_name: str, root: str = "data", train: bool = True):
    transform = _dataset_transform()
    if dataset_name == "mnist":
        return datasets.MNIST(root=root, train=train, transform=transform, download=True)
    if dataset_name == "fashion_mnist":
        return datasets.FashionMNIST(root=root, train=train, transform=transform, download=True)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


class SyntheticImageFolderDataset(Dataset):
    def __init__(self, root_dir: str):
        self.samples = []
        self.transform = _dataset_transform()
        for class_name in sorted(os.listdir(root_dir)):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            label = int(class_name)
            for file_name in os.listdir(class_dir):
                if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(class_dir, file_name), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("L")
        image = self.transform(image)
        return image, label


class CombinedDataset(Dataset):
    def __init__(self, *datasets_):
        self.datasets_ = [d for d in datasets_ if d is not None]
        self.cumulative_sizes = []
        running = 0
        for d in self.datasets_:
            running += len(d)
            self.cumulative_sizes.append(running)

    def __len__(self):
        if not self.cumulative_sizes:
            return 0
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx: int):
        for dataset_idx, end in enumerate(self.cumulative_sizes):
            if idx < end:
                start = 0 if dataset_idx == 0 else self.cumulative_sizes[dataset_idx - 1]
                return self.datasets_[dataset_idx][idx - start]
        raise IndexError("Index out of bounds")


def get_dataloaders(
    dataset_name: str,
    batch_size: int,
    num_workers: int = 2,
    synthetic_root: str = "",
    scenario: str = "real_only",
) -> Tuple[DataLoader, DataLoader]:
    real_train = get_base_dataset(dataset_name=dataset_name, train=True)
    real_test = get_base_dataset(dataset_name=dataset_name, train=False)
    synthetic_train = None

    if synthetic_root and os.path.isdir(synthetic_root):
        synthetic_train = SyntheticImageFolderDataset(synthetic_root)

    if scenario == "real_only":
        train_dataset = real_train
    elif scenario == "real_plus_synthetic":
        train_dataset = CombinedDataset(real_train, synthetic_train)
    elif scenario == "synthetic_only":
        if synthetic_train is None or len(synthetic_train) == 0:
            raise ValueError("Synthetic-only scenario requires non-empty synthetic dataset.")
        train_dataset = synthetic_train
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        real_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader
