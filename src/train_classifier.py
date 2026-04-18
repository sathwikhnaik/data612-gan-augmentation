import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .data import get_dataloaders
from .evaluate import plot_confusion_matrix, summarize_metrics
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
) -> dict:
    """
    Train and evaluate one classifier; save metrics, confusion matrix, and weights.
    Returns dict with keys: metrics, run_dir, classifier_path, confusion_matrix_path.
    """
    set_seed(seed)
    device = get_device()

    train_loader, test_loader = get_dataloaders(
        dataset_name=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        synthetic_root=synthetic_root,
        scenario=scenario,
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
    metrics, cm = summarize_metrics(y_true, y_pred)
    if not quiet:
        print("Evaluation metrics:", metrics)

    save_json(metrics, os.path.join(run_dir, "metrics.json"))
    cm_path = os.path.join(run_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, cm_path)
    clf_path = os.path.join(run_dir, "classifier.pt")
    torch.save(model.state_dict(), clf_path)
    if not quiet:
        print(f"Saved outputs to: {run_dir}")

    return {
        "metrics": metrics,
        "run_dir": run_dir,
        "classifier_path": clf_path,
        "confusion_matrix_path": cm_path,
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
    )


if __name__ == "__main__":
    train()
