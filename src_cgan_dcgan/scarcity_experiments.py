"""
Comprehensive scarcity experiment runner.

Reuses trained generators and synthetic images from prior run_experiments output
(no GAN retraining needed). Sweeps over real training data caps across both
datasets and generates a full comparative analysis.

Usage:
  python -m src.scarcity_experiments \\
    --mnist-exp-dir   outputs/experiments/mnist_full \\
    --fmnist-exp-dir  outputs/experiments/fashion_mnist_full \\
    --output-dir      outputs/scarcity_experiments \\
    --scarcity-levels 100 300 600 1200 3000 6000 12000 30000 60000 \\
    --seeds 42 123 \\
    --mnist-clf-epochs 10 \\
    --fmnist-clf-epochs 15
"""
import argparse
import json
import os

from .scarcity_sweep import (
    plot_crossdataset_gap,
    plot_crossdataset_scarcity,
    plot_f1_gain_heatmap,
    plot_gap_chart,
    plot_per_class_at_scarcity,
    plot_scarcity_curves,
    plot_synthetic_only_crossover,
    print_scarcity_table,
    run_scarcity_sweep,
)
from .utils import ensure_dir, save_json


DEFAULT_SCARCITY_LEVELS = [100, 300, 600, 1200, 3000, 6000, 12000, 30000, 60000]


def parse_args():
    p = argparse.ArgumentParser(
        description="Run comprehensive scarcity experiments on MNIST and Fashion-MNIST."
    )
    p.add_argument("--mnist-exp-dir", type=str, default="outputs/experiments/mnist_full",
                   help="Path to completed MNIST run_experiments output directory.")
    p.add_argument("--fmnist-exp-dir", type=str, default="outputs/experiments/fashion_mnist_full",
                   help="Path to completed Fashion-MNIST run_experiments output directory.")
    p.add_argument("--output-dir", type=str, default="outputs/scarcity_experiments")
    p.add_argument("--scarcity-levels", type=int, nargs="+", default=DEFAULT_SCARCITY_LEVELS)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123],
                   help="Random seeds — multiple seeds produce error bands on plots.")
    p.add_argument("--mnist-clf-epochs", type=int, default=10)
    p.add_argument("--fmnist-clf-epochs", type=int, default=15)
    p.add_argument("--quiet", action="store_true", default=True,
                   help="Suppress per-epoch classifier output.")
    p.add_argument("--skip-mnist", action="store_true",
                   help="Skip MNIST sweep (use if already completed).")
    p.add_argument("--skip-fmnist", action="store_true",
                   help="Skip Fashion-MNIST sweep (use if already completed).")
    return p.parse_args()


def _load_results(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_and_plot_dataset(
    exp_dir: str,
    dataset: str,
    scarcity_levels: list,
    seeds: list,
    clf_epochs: int,
    output_dir: str,
    quiet: bool,
    skip: bool,
) -> dict:
    ensure_dir(output_dir)
    results_path = os.path.join(output_dir, "scarcity_results.json")

    synthetic_root = os.path.join(exp_dir, "synthetic")
    if not os.path.isdir(synthetic_root):
        raise FileNotFoundError(
            f"Synthetic images not found at {synthetic_root}. "
            f"Run run_experiments first for {dataset}."
        )

    if skip:
        if os.path.exists(results_path):
            print(f"[scarcity] Skipping {dataset} — loading existing results from {results_path}")
            return _load_results(results_path)
        print(f"[scarcity] Skipping {dataset} — no results file found, returning empty.")
        return {}

    print(f"\n{'='*60}")
    print(f"[scarcity] Starting sweep: {dataset}")
    print(f"  Scarcity levels : {scarcity_levels}")
    print(f"  Seeds           : {seeds}")
    print(f"  Classifier epochs: {clf_epochs}")
    print(f"  Synthetic root  : {synthetic_root}")
    print(f"{'='*60}\n")

    results = run_scarcity_sweep(
        synthetic_root=synthetic_root,
        dataset=dataset,
        scarcity_levels=scarcity_levels,
        seeds=seeds,
        classifier_epochs=clf_epochs,
        output_dir=output_dir,
        quiet=quiet,
    )

    # ── Per-dataset plots ──────────────────────────────────────────────────
    plot_scarcity_curves(
        results, os.path.join(output_dir, "scarcity_accuracy.png"), dataset, metric="accuracy"
    )
    plot_scarcity_curves(
        results, os.path.join(output_dir, "scarcity_f1.png"), dataset, metric="f1_score"
    )
    plot_gap_chart(results, os.path.join(output_dir, "gap_vs_baseline.png"), dataset)
    plot_synthetic_only_crossover(results, os.path.join(output_dir, "synthetic_crossover.png"), dataset)
    plot_per_class_at_scarcity(results, os.path.join(output_dir, "per_class_scarcity.png"), dataset)
    plot_f1_gain_heatmap(results, os.path.join(output_dir, "f1_gain_heatmap.png"), dataset)
    print_scarcity_table(results, dataset)

    return results


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    # ── MNIST sweep ────────────────────────────────────────────────────────
    mnist_out = os.path.join(args.output_dir, "mnist")
    mnist_results = run_and_plot_dataset(
        exp_dir=args.mnist_exp_dir,
        dataset="mnist",
        scarcity_levels=args.scarcity_levels,
        seeds=args.seeds,
        clf_epochs=args.mnist_clf_epochs,
        output_dir=mnist_out,
        quiet=args.quiet,
        skip=args.skip_mnist,
    )

    # ── Fashion-MNIST sweep ────────────────────────────────────────────────
    fmnist_out = os.path.join(args.output_dir, "fashion_mnist")
    fmnist_results = run_and_plot_dataset(
        exp_dir=args.fmnist_exp_dir,
        dataset="fashion_mnist",
        scarcity_levels=args.scarcity_levels,
        seeds=args.seeds,
        clf_epochs=args.fmnist_clf_epochs,
        output_dir=fmnist_out,
        quiet=args.quiet,
        skip=args.skip_fmnist,
    )

    # ── Cross-dataset comparison plots (only if both datasets ran) ─────────
    cross_dir = os.path.join(args.output_dir, "cross_dataset")
    if mnist_results and fmnist_results:
        ensure_dir(cross_dir)
        plot_crossdataset_scarcity(
            mnist_results, fmnist_results,
            os.path.join(cross_dir, "scarcity_accuracy_comparison.png"),
            metric="accuracy",
        )
        plot_crossdataset_scarcity(
            mnist_results, fmnist_results,
            os.path.join(cross_dir, "scarcity_f1_comparison.png"),
            metric="f1_score",
        )
        plot_crossdataset_gap(
            mnist_results, fmnist_results,
            os.path.join(cross_dir, "gap_comparison.png"),
        )
    else:
        print("[scarcity] Skipping cross-dataset plots — only one dataset available.")

    # ── Save combined summary ──────────────────────────────────────────────
    save_json(
        {"mnist": mnist_results, "fashion_mnist": fmnist_results},
        os.path.join(args.output_dir, "all_scarcity_results.json"),
    )

    print(f"\n[scarcity] All done. Artifacts under: {args.output_dir}")
    print(f"  mnist/          — per-dataset plots + results")
    print(f"  fashion_mnist/  — per-dataset plots + results")
    print(f"  cross_dataset/  — comparison plots")


if __name__ == "__main__":
    main()
