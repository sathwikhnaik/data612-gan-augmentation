"""
Group 1: GAN Architecture sweep — one-factor-at-a-time (OFAT) ablation.

Each factor is swept while all others are held at the baseline value.
For every config: train GAN → generate synthetics → train real_plus_synthetic classifier.
The primary outcome metric is real_plus_synthetic classifier accuracy on the test set.
FID and recognizability are also recorded when available.

Usage:
    python -m src.arch_sweep --dataset mnist --gan-epochs 30 --classifier-epochs 10
    python -m src.arch_sweep --dataset fashion_mnist --gan-epochs 30 --num-synthetic 6000
    # Quick sanity check (weak metrics, fast):
    python -m src.arch_sweep --dataset mnist --gan-epochs 2 --classifier-epochs 1 \\
        --num-synthetic 500 --quiet
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from .generate_synthetic import generate_synthetic_core
from .per_class_quality import evaluate_synthetic_quality
from .train_classifier import train_classifier_core
from .train_gan import train_gan_core
from .utils import ensure_dir, save_json, timestamp

# ── Baseline configuration ─────────────────────────────────────────────────

BASELINE = {
    "gen_type": "mlp",
    "conditioning": "embedding",
    "latent_dim": 100,
    "embedding_dim": 50,
    "hidden_dims": [256, 512, 1024],
    "disc_dropout": 0.3,
    # Training params held fixed during arch sweep
    "loss_type": "bce",
    "label_smoothing": False,
    "generator_lr": 2e-4,
    "discriminator_lr": 2e-4,
    "n_critic": 1,
    "gp_lambda": 10.0,
}

# ── Factors to sweep (one at a time) ──────────────────────────────────────

FACTORS: dict = {
    "gen_type": ["mlp", "dcgan"],
    "conditioning": ["embedding", "onehot", "projection"],
    "latent_dim": [32, 64, 100, 200],
    "embedding_dim": [10, 32, 50, 100],
    # MLP depth: shallow → medium (baseline) → deep
    "hidden_dims": [
        [256, 512],
        [256, 512, 1024],
        [256, 512, 1024, 1024],
    ],
    "disc_dropout": [0.0, 0.1, 0.3, 0.5],
}

# dcgan + onehot/projection are incompatible (DCGAN always uses embedding)
_SKIP = {
    ("gen_type", "dcgan"): {"conditioning": ["onehot", "projection"]},
}


def _label(factor: str, value) -> str:
    if isinstance(value, list):
        return "x".join(str(v) for v in value)
    return str(value)


def run_arch_sweep(
    dataset: str,
    gan_epochs: int,
    classifier_epochs: int,
    num_synthetic: int,
    seed: int,
    num_workers: int,
    output_dir: str,
    quiet: bool = True,
    batch_size: int = 128,
) -> list:
    """
    OFAT ablation over FACTORS. Returns list of result dicts.
    Skips (resumes from) already-completed runs.
    """
    ensure_dir(output_dir)
    results = []

    for factor, values in FACTORS.items():
        for value in values:
            run_label = f"{factor}__{_label(factor, value)}"
            run_dir = os.path.join(output_dir, run_label)
            result_path = os.path.join(run_dir, "result.json")

            if os.path.exists(result_path):
                with open(result_path, encoding="utf-8") as f:
                    results.append(json.load(f))
                print(f"[arch_sweep] CACHED  {run_label}")
                continue

            print(f"\n[arch_sweep] RUNNING {run_label}")
            ensure_dir(run_dir)

            cfg = {**BASELINE, factor: value}

            # Skip incompatible combos
            skip = False
            for (fk, fv), constraints in _SKIP.items():
                if cfg.get(fk) == fv:
                    for ck, cv_list in constraints.items():
                        if cfg.get(ck) in cv_list:
                            print(f"[arch_sweep] SKIPPED {run_label} (incompatible with {fk}={fv})")
                            skip = True
                            break
            if skip:
                continue

            # 1. Train GAN
            gan_dir = os.path.join(run_dir, "gan")
            try:
                gen_path = train_gan_core(
                    dataset=dataset,
                    epochs=gan_epochs,
                    batch_size=batch_size,
                    latent_dim=cfg["latent_dim"],
                    seed=seed,
                    num_workers=num_workers,
                    model_dir=gan_dir,
                    gen_type=cfg["gen_type"],
                    conditioning=cfg["conditioning"],
                    embedding_dim=cfg["embedding_dim"],
                    hidden_dims=cfg["hidden_dims"],
                    disc_dropout=cfg["disc_dropout"],
                    loss_type=cfg["loss_type"],
                    label_smoothing=cfg["label_smoothing"],
                    generator_lr=cfg["generator_lr"],
                    discriminator_lr=cfg["discriminator_lr"],
                    n_critic=cfg["n_critic"],
                    gp_lambda=cfg["gp_lambda"],
                )
            except Exception as e:
                print(f"[arch_sweep] GAN training FAILED for {run_label}: {e}")
                continue

            # 2. Generate synthetic images
            syn_dir = os.path.join(run_dir, "synthetic")
            try:
                generate_synthetic_core(
                    generator_path=gen_path,
                    dataset=dataset,
                    num_samples=num_synthetic,
                    latent_dim=cfg["latent_dim"],
                    seed=seed,
                    output_root=syn_dir,
                    model_dir=gan_dir,
                )
            except Exception as e:
                print(f"[arch_sweep] Synthetic generation FAILED for {run_label}: {e}")
                continue

            # 3. Train real_plus_synthetic classifier
            clf_dir = os.path.join(run_dir, "classifier")
            try:
                clf_out = train_classifier_core(
                    dataset=dataset,
                    scenario="real_plus_synthetic",
                    synthetic_root=syn_dir,
                    epochs=classifier_epochs,
                    batch_size=batch_size,
                    lr=1e-3,
                    num_workers=num_workers,
                    seed=seed,
                    run_dir=clf_dir,
                    quiet=quiet,
                )
            except Exception as e:
                print(f"[arch_sweep] Classifier FAILED for {run_label}: {e}")
                continue

            # 4. Recognizability (optional but fast)
            recognizability = None
            try:
                qual = evaluate_synthetic_quality(
                    classifier_path=clf_out["classifier_path"],
                    synthetic_root=syn_dir,
                    dataset=dataset,
                    output_dir=os.path.join(run_dir, "quality"),
                )
                recognizability = qual.get("overall_recognizability")
            except Exception:
                pass

            entry = {
                "factor": factor,
                "value": _label(factor, value),
                "value_raw": value,
                "config": cfg,
                "metrics": clf_out["metrics"],
                "recognizability": recognizability,
                "run_dir": run_dir,
            }
            save_json(entry, result_path)
            results.append(entry)
            rec_str = f"{recognizability:.4f}" if recognizability is not None else "N/A"
            print(f"[arch_sweep] accuracy={clf_out['metrics']['accuracy']:.4f}  "
                  f"f1={clf_out['metrics']['f1_score']:.4f}  "
                  f"recognizability={rec_str}")

    save_json(results, os.path.join(output_dir, "arch_sweep_results.json"))
    _plot_arch_sweep(results, output_dir, dataset)
    return results


def _plot_arch_sweep(results: list, output_dir: str, dataset: str) -> None:
    """One bar chart per factor showing accuracy vs. factor value."""
    by_factor: dict = {}
    for r in results:
        by_factor.setdefault(r["factor"], []).append(r)

    for factor, entries in by_factor.items():
        labels = [e["value"] for e in entries]
        acc = [e["metrics"]["accuracy"] for e in entries]
        f1 = [e["metrics"]["f1_score"] for e in entries]
        rec = [e["recognizability"] if e["recognizability"] is not None else 0.0
               for e in entries]

        x = np.arange(len(labels))
        w = 0.25
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 5))
        ax.bar(x - w, acc, w, label="Accuracy", color="steelblue", alpha=0.85)
        ax.bar(x, f1, w, label="F1 Score", color="darkorange", alpha=0.85)
        ax.bar(x + w, rec, w, label="Recognizability", color="seagreen", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title(f"{dataset}: Architecture ablation — {factor}")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        out_path = os.path.join(output_dir, f"arch_ablation_{factor}.png")
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved plot: {out_path}")

    # Summary table in stdout
    print(f"\n{'='*80}")
    print(f"ARCHITECTURE SWEEP SUMMARY — {dataset}")
    print(f"{'='*80}")
    hdr = f"{'Factor':<20} {'Value':<25} {'Accuracy':>10} {'F1':>10} {'Recogn.':>10}"
    print(hdr)
    print("-" * 80)
    for r in results:
        rec_str = f"{r['recognizability']:>10.4f}" if r["recognizability"] is not None else "       N/A"
        print(f"{r['factor']:<20} {r['value']:<25} "
              f"{r['metrics']['accuracy']:>10.4f} {r['metrics']['f1_score']:>10.4f} {rec_str}")
    print("=" * 80)


def parse_args():
    p = argparse.ArgumentParser(
        description="OFAT architecture ablation sweep over GAN design choices."
    )
    p.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    p.add_argument("--gan-epochs", type=int, default=30)
    p.add_argument("--classifier-epochs", type=int, default=10)
    p.add_argument("--num-synthetic", type=int, default=6000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--output-dir", type=str, default="")
    p.add_argument("--quiet", action="store_true", default=False)
    p.add_argument("--factors", type=str, nargs="*", default=None,
                   help="Restrict sweep to specific factors, e.g. gen_type latent_dim")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.output_dir or os.path.join(
        "outputs", "arch_sweep", f"{args.dataset}_{timestamp()}"
    )

    global FACTORS
    if args.factors:
        FACTORS = {k: v for k, v in FACTORS.items() if k in args.factors}

    run_arch_sweep(
        dataset=args.dataset,
        gan_epochs=args.gan_epochs,
        classifier_epochs=args.classifier_epochs,
        num_synthetic=args.num_synthetic,
        seed=args.seed,
        num_workers=args.num_workers,
        output_dir=out_dir,
        quiet=args.quiet,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
