import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .config import class_names_for_dataset
from .data import get_base_dataset, get_dataloaders
from .evaluate import (
    plot_confusion_matrix,
    plot_per_class_f1,
    summarize_metrics_full,
)
from .models import SimpleCNNClassifier
from .utils import ensure_dir, get_device, save_json, set_seed, timestamp


def parse_args():
    parser = argparse.ArgumentParser(description="Train classifier in one of three scenarios.")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    parser.add_argument(
        "--scenario",
        type=str,
        default="real_only",
        choices=["real_only", "real_plus_synthetic", "synthetic_only"],
    )
    parser.add_argument("--synthetic-root", type=str, default="")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-real-train-samples",
        type=int,
        default=0,
        help="Stratified cap on real training images (0 = use full training set). Simulates label scarcity.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.0,
        help="If >0 and --max-real-train-samples is 0, cap real training set to this fraction of the full train split.",
    )
    return parser.parse_args()


def train_classifier_core(
    dataset: str,
    scenario: str,
    synthetic_root: str,
    epochs: int,
    batch_size: int,
    lr: float,
    num_workers: int,
    seed: int,
    run_dir: str | None = None,
    quiet: bool = False,
    max_real_train_samples: int | None = None,
    train_fraction: float | None = None,
    train_subset_seed: int | None = None,
) -> dict:
    """
    Train and evaluate one classifier; save metrics, confusion matrix, and weights.
    Returns dict with keys: metrics, per_class, run_dir, classifier_path, confusion_matrix_path.
    """
    set_seed(seed)
    device = get_device()
    subset_seed = train_subset_seed if train_subset_seed is not None else seed

    cap = max_real_train_samples
    if (cap is None or cap <= 0) and train_fraction is not None and train_fraction > 0:
        full_train = get_base_dataset(dataset, train=True)
        cap = max(1, int(len(full_train) * train_fraction))

    train_loader, test_loader = get_dataloaders(
        dataset_name=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        synthetic_root=synthetic_root,
        scenario=scenario,
        max_real_train_samples=cap if cap and cap > 0 else None,
        train_subset_seed=subset_seed,
    )

    model = SimpleCNNClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if run_dir is None:
        run_dir = os.path.join("outputs", "classifier", f"{dataset}_{scenario}_{timestamp()}")
    ensure_dir(run_dir)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        iterator = train_loader
        if not quiet:
            iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for images, labels in iterator:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if not quiet:
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {running_loss / len(train_loader):.4f}")

    y_true, y_pred = evaluate_model(model, test_loader, device)
    class_names = class_names_for_dataset(dataset)
    metrics, cm, per_class = summarize_metrics_full(
        y_true, y_pred, class_names=class_names, num_classes=len(class_names)
    )
    if not quiet:
        print("Evaluation metrics:", metrics)

    save_json(metrics, os.path.join(run_dir, "metrics.json"))
    save_json({"per_class": per_class}, os.path.join(run_dir, "per_class_metrics.json"))
    cm_path = os.path.join(run_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, cm_path, title=f"Confusion matrix ({dataset}, {scenario})")
    pc_path = os.path.join(run_dir, "per_class_f1.png")
    plot_per_class_f1(per_class, pc_path, title=f"Per-class F1 ({dataset}, {scenario})")
    clf_path = os.path.join(run_dir, "classifier.pt")
    torch.save(model.state_dict(), clf_path)
    if not quiet:
        print(f"Saved outputs to: {run_dir}")

    return {
        "metrics": metrics,
        "per_class": per_class,
        "run_dir": run_dir,
        "classifier_path": clf_path,
        "confusion_matrix_path": cm_path,
        "per_class_f1_plot": pc_path,
    }


def evaluate_model(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(labels.numpy().tolist())
    return np.array(y_true), np.array(y_pred)


def train():
    args = parse_args()
    max_cap = args.max_real_train_samples if args.max_real_train_samples > 0 else None
    frac = args.train_fraction if args.train_fraction > 0 else None
    train_classifier_core(
        dataset=args.dataset,
        scenario=args.scenario,
        synthetic_root=args.synthetic_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        seed=args.seed,
        run_dir=None,
        quiet=False,
        max_real_train_samples=max_cap,
        train_fraction=frac,
    )


if __name__ == "__main__":
    train()
