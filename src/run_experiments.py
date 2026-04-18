import argparse
import csv
import os

import matplotlib.pyplot as plt

from .generate_synthetic import generate_synthetic_core
from .train_classifier import train_classifier_core
from .train_gan import train_gan_core
from .utils import ensure_dir, save_json, timestamp


def parse_args():
    p = argparse.ArgumentParser(
        description="Run full pipeline: GAN (optional), synthetic data, three classifier scenarios, comparison table and plot."
    )
    p.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gan-epochs", type=int, default=30)
    p.add_argument("--gan-batch-size", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=100)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--num-synthetic", type=int, default=12000)
    p.add_argument("--classifier-epochs", type=int, default=10)
    p.add_argument("--classifier-batch-size", type=int, default=128)
    p.add_argument("--classifier-lr", type=float, default=1e-3)
    p.add_argument("--skip-gan", action="store_true", help="Use --generator-path (do not train GAN).")
    p.add_argument("--generator-path", type=str, default="", help="Trained generator .pt (required if --skip-gan).")
    p.add_argument("--synthetic-root", type=str, default="", help="Use existing synthetic folder; skip generation.")
    p.add_argument("--experiment-dir", type=str, default="", help="Root for this run; default outputs/experiments/<timestamp>.")
    p.add_argument("--quiet-classifiers", action="store_true", help="Less tqdm output during classifier training.")
    return p.parse_args()


def plot_comparison(rows, output_path: str, title: str) -> None:
    scenarios = [r["scenario"] for r in rows]
    metric_keys = ["accuracy", "precision", "recall", "f1_score"]
    labels = [k.replace("_", " ").title() for k in metric_keys]
    x = list(range(len(scenarios)))
    n_met = len(metric_keys)
    width = 0.8 / max(n_met, 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, key in enumerate(metric_keys):
        vals = [r["metrics"][key] for r in rows]
        offset = (i - (n_met - 1) / 2) * width
        ax.bar([xi + offset for xi in x], vals, width, label=labels[i])
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_csv(rows: list[dict], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    fieldnames = ["scenario", "accuracy", "precision", "recall", "f1_score", "run_dir"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            m = r["metrics"]
            w.writerow(
                {
                    "scenario": r["scenario"],
                    "accuracy": m["accuracy"],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "f1_score": m["f1_score"],
                    "run_dir": r["run_dir"],
                }
            )


def main():
    args = parse_args()
    if args.skip_gan and not args.generator_path:
        raise SystemExit("--skip-gan requires --generator-path.")

    exp_root = args.experiment_dir or os.path.join("outputs", "experiments", f"{args.dataset}_{timestamp()}")
    ensure_dir(exp_root)
    gan_dir = os.path.join(exp_root, "gan")

    if args.skip_gan:
        generator_path = args.generator_path
    else:
        print(f"[experiment] Training GAN -> {gan_dir}")
        generator_path = train_gan_core(
            dataset=args.dataset,
            epochs=args.gan_epochs,
            batch_size=args.gan_batch_size,
            latent_dim=args.latent_dim,
            seed=args.seed,
            num_workers=args.num_workers,
            model_dir=gan_dir,
        )

    if args.synthetic_root:
        synthetic_root = args.synthetic_root
        print(f"[experiment] Using synthetic data at: {synthetic_root}")
    else:
        syn_out = os.path.join(exp_root, "synthetic")
        print(f"[experiment] Generating synthetic images -> {syn_out}")
        synthetic_root = generate_synthetic_core(
            generator_path=generator_path,
            dataset=args.dataset,
            num_samples=args.num_synthetic,
            latent_dim=args.latent_dim,
            seed=args.seed,
            output_root=syn_out,
        )

    scenarios = ["real_only", "real_plus_synthetic", "synthetic_only"]
    rows = []

    for scenario in scenarios:
        syn_arg = "" if scenario == "real_only" else synthetic_root
        clf_dir = os.path.join(exp_root, "classifiers", scenario)
        print(f"[experiment] Classifier: {scenario}")
        out = train_classifier_core(
            dataset=args.dataset,
            scenario=scenario,
            synthetic_root=syn_arg,
            epochs=args.classifier_epochs,
            batch_size=args.classifier_batch_size,
            lr=args.classifier_lr,
            num_workers=args.num_workers,
            seed=args.seed,
            run_dir=clf_dir,
            quiet=args.quiet_classifiers,
        )
        rows.append(
            {
                "scenario": scenario,
                "metrics": out["metrics"],
                "run_dir": out["run_dir"],
            }
        )

    csv_path = os.path.join(exp_root, "comparison.csv")
    json_path = os.path.join(exp_root, "comparison.json")
    plot_path = os.path.join(exp_root, "comparison_metrics.png")

    write_csv(rows, csv_path)
    save_json({"dataset": args.dataset, "generator_path": generator_path, "synthetic_root": synthetic_root, "runs": rows}, json_path)
    plot_comparison(rows, plot_path, title=f"{args.dataset} classifier comparison")

    print(f"[experiment] Done. Artifacts under:\n  {exp_root}")
    print(f"  - {csv_path}")
    print(f"  - {json_path}")
    print(f"  - {plot_path}")


if __name__ == "__main__":
    main()
