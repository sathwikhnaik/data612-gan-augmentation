import os
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
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


def num_classes(dataset_name: str) -> int:
    _ = dataset_name  # both benchmarks use 10 classes
    return 10


def stratified_subset_indices(dataset, max_samples: int, seed: int) -> List[int]:
    """
    Balanced scarcity: as equal as possible per class from dataset.targets.
    """
    if max_samples <= 0:
        return []
    max_samples = min(max_samples, len(dataset))
    rng = np.random.default_rng(seed)
    targets = np.array(dataset.targets)
    n_classes = int(targets.max()) + 1
    per = max_samples // n_classes
    rem = max_samples % n_classes
    indices: List[int] = []
    for c in range(n_classes):
        pool = np.where(targets == c)[0]
        take = per + (1 if c < rem else 0)
        take = min(take, len(pool))
        if take > 0:
            chosen = rng.choice(pool, size=take, replace=False)
            indices.extend(chosen.tolist())
    return indices


def maybe_stratified_train_cap(dataset, max_samples: Optional[int], seed: int):
    """Return original dataset or a Subset capped to max_samples (stratified by class)."""
    if max_samples is None or max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    idx = stratified_subset_indices(dataset, max_samples, seed)
    return Subset(dataset, idx)


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
    max_real_train_samples: Optional[int] = None,
    train_subset_seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    real_train = get_base_dataset(dataset_name=dataset_name, train=True)
    if scenario in ("real_only", "real_plus_synthetic"):
        real_train = maybe_stratified_train_cap(
            real_train, max_real_train_samples, train_subset_seed
        )
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
