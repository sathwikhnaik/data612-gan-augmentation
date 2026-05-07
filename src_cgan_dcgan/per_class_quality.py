"""
Per-class GAN quality analysis.

Uses a classifier trained on real data only to score synthetic images:
for each class, what fraction of synthetic images of that class are
correctly identified? High recognizability means the GAN generates
realistic, class-consistent images; low means mode collapse or blur.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import class_names_for_dataset
from .data import SyntheticImageFolderDataset
from .evaluate import per_class_metrics
from .models import SimpleCNNClassifier
from .utils import ensure_dir, get_device, save_json


def evaluate_synthetic_quality(
    classifier_path: str,
    synthetic_root: str,
    dataset: str,
    output_dir: str,
    batch_size: int = 128,
) -> dict:
    """
    Run a real-trained classifier over synthetic images and report
    per-class recall (= recognizability: % of class-N images labelled N).
    """
    device = get_device()

    model = SimpleCNNClassifier().to(device)
    model.load_state_dict(torch.load(classifier_path, map_location=device, weights_only=True))
    model.eval()

    syn_dataset = SyntheticImageFolderDataset(synthetic_root)
    if len(syn_dataset) == 0:
        print("[per_class_quality] Synthetic dataset is empty — skipping.")
        return {}

    loader = DataLoader(syn_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds = torch.argmax(model(images), dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(labels.numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    class_names = class_names_for_dataset(dataset)
    per_class = per_class_metrics(y_true, y_pred, class_names=class_names, num_classes=10)
    overall = float((y_true == y_pred).mean())

    ensure_dir(output_dir)
    result = {"overall_recognizability": overall, "per_class": per_class}
    save_json(result, os.path.join(output_dir, "synthetic_quality.json"))
    _plot_recognizability(per_class, os.path.join(output_dir, "synthetic_recognizability.png"), dataset)

    print(f"[per_class_quality] Overall synthetic recognizability: {overall:.4f}")
    return result


def _plot_recognizability(per_class: list, output_path: str, dataset: str) -> None:
    names = [r["class_name"] for r in per_class]
    recalls = [r["recall"] for r in per_class]
    mean_r = sum(recalls) / len(recalls) if recalls else 0.0

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(names))
    ax.bar(x, recalls, color="coral")
    ax.axhline(mean_r, color="gray", linestyle="--", alpha=0.8, label=f"mean={mean_r:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Recognizability (recall by real-trained classifier)")
    ax.set_title(f"{dataset}: per-class GAN image recognizability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Recognizability plot saved to: {output_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate per-class GAN quality using a real-trained classifier."
    )
    p.add_argument("--classifier-path", type=str, required=True)
    p.add_argument("--synthetic-root", type=str, required=True)
    p.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    p.add_argument("--output-dir", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.output_dir or os.path.join("outputs", "per_class_quality")
    evaluate_synthetic_quality(
        classifier_path=args.classifier_path,
        synthetic_root=args.synthetic_root,
        dataset=args.dataset,
        output_dir=out_dir,
    )


if __name__ == "__main__":
    main()
