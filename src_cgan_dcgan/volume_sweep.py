"""
Synthetic data volume sweep.

Trains a real_plus_synthetic classifier for each value in num_synthetic_values
and plots accuracy / F1 as a function of synthetic dataset size. This reveals
the diminishing-returns curve and the minimum useful synthetic count.
"""
import argparse
import os
from typing import List, Optional

import matplotlib.pyplot as plt

from .generate_synthetic import generate_synthetic_core
from .train_classifier import train_classifier_core
from .utils import ensure_dir, save_json


def run_volume_sweep(
    generator_path: str,
    dataset: str,
    num_synthetic_values: List[int],
    classifier_epochs: int,
    seed: int,
    output_dir: str,
    latent_dim: int = 100,
    max_real_train_samples: Optional[int] = None,
    train_fraction: Optional[float] = None,
    quiet: bool = True,
) -> List[dict]:
    ensure_dir(output_dir)
    results = []

    for n_syn in num_synthetic_values:
        print(f"[volume_sweep] num_synthetic={n_syn}")
        syn_dir = os.path.join(output_dir, f"synthetic_{n_syn}")
        generate_synthetic_core(
            generator_path=generator_path,
            dataset=dataset,
            num_samples=n_syn,
            latent_dim=latent_dim,
            seed=seed,
            output_root=syn_dir,
        )
        clf_dir = os.path.join(output_dir, f"clf_{n_syn}")
        out = train_classifier_core(
            dataset=dataset,
            scenario="real_plus_synthetic",
            synthetic_root=syn_dir,
            epochs=classifier_epochs,
            batch_size=128,
            lr=1e-3,
            num_workers=2,
            seed=seed,
            run_dir=clf_dir,
            quiet=quiet,
            max_real_train_samples=max_real_train_samples,
            train_fraction=train_fraction,
        )
        results.append(
            {
                "num_synthetic": n_syn,
                "metrics": out["metrics"],
                "per_class": out.get("per_class", []),
            }
        )

    save_json(results, os.path.join(output_dir, "volume_sweep.json"))
    _plot_sweep(results, os.path.join(output_dir, "volume_sweep.png"), dataset)
    return results


def _plot_sweep(results: List[dict], output_path: str, dataset: str) -> None:
    x = [r["num_synthetic"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, label in [("accuracy", "Accuracy"), ("f1_score", "F1 Score")]:
        y = [r["metrics"][key] for r in results]
        ax.plot(x, y, marker="o", label=label)
    ax.set_xlabel("Number of synthetic images")
    ax.set_ylabel("Score")
    ax.set_title(f"{dataset}: classifier score vs. synthetic data volume")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Volume sweep plot saved to: {output_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Sweep synthetic data volume for real_plus_synthetic classifier."
    )
    p.add_argument("--generator-path", type=str, required=True)
    p.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    p.add_argument(
        "--num-synthetic-values",
        type=int,
        nargs="+",
        default=[500, 1000, 3000, 6000, 12000],
    )
    p.add_argument("--classifier-epochs", type=int, default=10)
    p.add_argument("--max-real-train-samples", type=int, default=0)
    p.add_argument("--train-fraction", type=float, default=0.0)
    p.add_argument("--latent-dim", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.output_dir or os.path.join("outputs", "volume_sweep", args.dataset)
    max_real = args.max_real_train_samples if args.max_real_train_samples > 0 else None
    frac = args.train_fraction if args.train_fraction > 0 else None
    run_volume_sweep(
        generator_path=args.generator_path,
        dataset=args.dataset,
        num_synthetic_values=args.num_synthetic_values,
        classifier_epochs=args.classifier_epochs,
        seed=args.seed,
        output_dir=out_dir,
        latent_dim=args.latent_dim,
        max_real_train_samples=max_real,
        train_fraction=frac,
        quiet=False,
    )


if __name__ == "__main__":
    main()
