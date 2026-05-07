import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import datasets, transforms

# Scenarios handled by get_dataloaders (single static loader)
STATIC_SCENARIOS = {
    "real_only",
    "augmented_real",
    "real_plus_synthetic",
    "real_plus_synthetic_weighted",
    "synthetic_only",
}

# Scenarios that require two loaders (phase switch mid-training)
PHASED_SCENARIOS = {"synthetic_pretrain", "progressive"}


def _dataset_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


def _augmented_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


def get_base_dataset(dataset_name: str, root: str = "data", train: bool = True, augment: bool = False):
    transform = _augmented_transform() if (augment and train) else _dataset_transform()
    if dataset_name == "mnist":
        return datasets.MNIST(root=root, train=train, transform=transform, download=True)
    if dataset_name == "fashion_mnist":
        return datasets.FashionMNIST(root=root, train=train, transform=transform, download=True)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def num_classes(dataset_name: str) -> int:
    _ = dataset_name  # both benchmarks use 10 classes
    return 10


def stratified_subset_indices(dataset, max_samples: int, seed: int) -> List[int]:
    """Balanced scarcity: as equal as possible per class from dataset.targets."""
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
        take = min(per + (1 if c < rem else 0), len(pool))
        if take > 0:
            indices.extend(rng.choice(pool, size=take, replace=False).tolist())
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
        return self.transform(image), label


class CombinedDataset(Dataset):
    def __init__(self, *datasets_):
        self.datasets_ = [d for d in datasets_ if d is not None]
        self.cumulative_sizes = []
        running = 0
        for d in self.datasets_:
            running += len(d)
            self.cumulative_sizes.append(running)

    def __len__(self):
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx: int):
        for dataset_idx, end in enumerate(self.cumulative_sizes):
            if idx < end:
                start = 0 if dataset_idx == 0 else self.cumulative_sizes[dataset_idx - 1]
                return self.datasets_[dataset_idx][idx - start]
        raise IndexError("Index out of bounds")


def _make_loader(dataset, batch_size: int, num_workers: int,
                 sampler=None, shuffle: bool = True) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def get_dataloaders(
    dataset_name: str,
    batch_size: int,
    num_workers: int = 2,
    synthetic_root: str = "",
    scenario: str = "real_only",
    max_real_train_samples: Optional[int] = None,
    train_subset_seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns (train_loader, test_loader) for static scenarios.
    For phased scenarios (synthetic_pretrain, progressive) use get_phased_loaders().
    """
    real_train = None
    if scenario in ("real_only", "real_plus_synthetic", "augmented_real",
                    "real_plus_synthetic_weighted"):
        real_train = get_base_dataset(
            dataset_name=dataset_name, train=True,
            augment=(scenario == "augmented_real"),
        )
        real_train = maybe_stratified_train_cap(real_train, max_real_train_samples, train_subset_seed)

    real_test = get_base_dataset(dataset_name=dataset_name, train=False)

    synthetic_train = None
    if synthetic_root and os.path.isdir(synthetic_root):
        synthetic_train = SyntheticImageFolderDataset(synthetic_root)

    sampler = None

    if scenario in ("real_only", "augmented_real"):
        train_dataset = real_train

    elif scenario == "real_plus_synthetic":
        train_dataset = CombinedDataset(real_train, synthetic_train)

    elif scenario == "real_plus_synthetic_weighted":
        if synthetic_train is None or len(synthetic_train) == 0:
            raise ValueError("real_plus_synthetic_weighted requires a non-empty synthetic dataset.")
        train_dataset = CombinedDataset(real_train, synthetic_train)
        # Real images receive 2× weight relative to synthetic
        weights = [2.0] * len(real_train) + [1.0] * len(synthetic_train)
        sampler = WeightedRandomSampler(
            torch.tensor(weights, dtype=torch.float),
            num_samples=len(weights),
            replacement=True,
        )

    elif scenario == "synthetic_only":
        if synthetic_train is None or len(synthetic_train) == 0:
            raise ValueError("synthetic_only scenario requires a non-empty synthetic dataset.")
        train_dataset = synthetic_train

    else:
        raise ValueError(f"Unknown scenario for get_dataloaders: {scenario}. "
                         f"Use get_phased_loaders() for {PHASED_SCENARIOS}.")

    train_loader = _make_loader(train_dataset, batch_size, num_workers, sampler=sampler)
    test_loader = _make_loader(real_test, batch_size, num_workers, shuffle=False)
    return train_loader, test_loader


def get_phased_loaders(
    dataset_name: str,
    batch_size: int,
    num_workers: int,
    synthetic_root: str,
    scenario: str,
    max_real_train_samples: Optional[int] = None,
    train_subset_seed: int = 42,
) -> dict:
    """
    Returns {"phase1": DataLoader, "phase2": DataLoader, "test": DataLoader}.

    synthetic_pretrain — phase1: synthetic only  → phase2: real only (fine-tune)
    progressive        — phase1: real only        → phase2: real + synthetic
    """
    if scenario not in PHASED_SCENARIOS:
        raise ValueError(f"get_phased_loaders only handles {PHASED_SCENARIOS}.")

    real_train = get_base_dataset(dataset_name=dataset_name, train=True)
    real_train = maybe_stratified_train_cap(real_train, max_real_train_samples, train_subset_seed)
    real_test = get_base_dataset(dataset_name=dataset_name, train=False)

    synthetic_train = None
    if synthetic_root and os.path.isdir(synthetic_root):
        synthetic_train = SyntheticImageFolderDataset(synthetic_root)

    if synthetic_train is None or len(synthetic_train) == 0:
        raise ValueError(f"{scenario} requires a non-empty synthetic dataset.")

    def loader(ds):
        return _make_loader(ds, batch_size, num_workers)

    test_loader = _make_loader(real_test, batch_size, num_workers, shuffle=False)

    if scenario == "synthetic_pretrain":
        return {
            "phase1": loader(synthetic_train),
            "phase2": loader(real_train),
            "test": test_loader,
        }
    # progressive
    return {
        "phase1": loader(real_train),
        "phase2": loader(CombinedDataset(real_train, synthetic_train)),
        "test": test_loader,
    }
