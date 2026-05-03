"""
Group 2: GAN Training Strategy sweep — one-factor-at-a-time (OFAT) ablation.

Architecture is held fixed at the MLP+embedding baseline. Each training
hyperparameter is swept independently:
  - loss_type  : bce | hinge | wgan_gp
  - lr_ratio   : discriminator_lr / generator_lr  (1:1, 2:1, 0.5:1)
  - label_smoothing: False | True
  - gan_epochs : 10 | 20 | 30 | 50

For every config: train GAN → generate synthetics → train real_plus_synthetic classifier.

Usage:
    python -m src.training_sweep --dataset mnist --classifier-epochs 10
    python -m src.training_sweep --dataset fashion_mnist --num-synthetic 6000
    # Quick sanity check:
    python -m src.training_sweep --dataset mnist --base-epochs 2 --classifier-epochs 1 \\
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

# ── Baseline (MLP+embedding architecture, default training) ───────────────

BASELINE = {
    "gen_type": "mlp",
    "conditioning": "embedding",
    "latent_dim": 100,
    "embedding_dim": 50,
    "hidden_dims": [256, 512, 1024],
    "disc_dropout": 0.3,
    "loss_type": "bce",
    "label_smoothing": False,
    "generator_lr": 2e-4,
    "discriminator_lr": 2e-4,
    "n_critic": 1,
    "gp_lambda": 10.0,
}

# Base epoch count; the epochs factor sweeps multiples of this.
BASE_EPOCHS = 30

# ── Factors ────────────────────────────────────────────────────────────────
# Each entry is (display_label, config_overrides_dict)

def _build_factors(base_epochs: int) -> dict:
    return {
        "loss_type": [
            ("bce",     {"loss_type": "bce",     "n_critic": 1}),
            ("hinge",   {"loss_type": "hinge",   "n_critic": 1}),
            ("wgan_gp", {"loss_type": "wgan_gp", "n_critic": 5}),
        ],
        "lr_ratio": [
            ("g=4e-4,d=2e-4",  {"generator_lr": 4e-4, "discriminator_lr": 2e-4}),
            ("g=2e-4,d=2e-4",  {"generator_lr": 2e-4, "discriminator_lr": 2e-4}),   # baseline
            ("g=2e-4,d=4e-4",  {"generator_lr": 2e-4, "discriminator_lr": 4e-4}),
            ("g=1e-4,d=4e-4",  {"generator_lr": 1e-4, "discriminator_lr": 4e-4}),
        ],
        "label_smoothing": [
            ("no_smoothing",   {"label_smoothing": False}),
            ("smoothing_0.9",  {"label_smoothing": True}),
        ],
        "gan_epochs": [
            (f"ep{n}", {"_epochs_override": n})
            for n in [base_epochs // 3, base_epochs // 2, base_epochs, base_epochs * 2]
            if n > 0
        ],
    }


def run_training_sweep(
    dataset: str,
    base_epochs: int,
    classifier_epochs: int,
    num_synthetic: int,
    seed: int,
    num_workers: int,
    output_dir: str,
    quiet: bool = True,
    batch_size: int = 128,
) -> list:
    """
    OFAT ablation over training strategy factors. Returns list of result dicts.
    Skips (resumes from) already-completed runs.
    """
    ensure_dir(output_dir)
    results = []
    factors = _build_factors(base_epochs)

    for factor, variants in factors.items():
        for label, overrides in variants:
            run_label = f"{factor}__{label}"
            run_dir = os.path.join(output_dir, run_label)
            result_path = os.path.join(run_dir, "result.json")

            if os.path.exists(result_path):
                with open(result_path, encoding="utf-8") as f:
                    results.append(json.load(f))
                print(f"[training_sweep] CACHED  {run_label}")
                continue

            print(f"\n[training_sweep] RUNNING {run_label}")
            ensure_dir(run_dir)

            cfg = {**BASELINE}
            epochs = base_epochs
            for k, v in overrides.items():
                if k == "_epochs_override":
                    epochs = v
                else:
                    cfg[k] = v

            # 1. Train GAN
            gan_dir = os.path.join(run_dir, "gan")
            try:
                gen_path = train_gan_core(
                    dataset=dataset,
                    epochs=epochs,
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
                print(f"[training_sweep] GAN training FAILED for {run_label}: {e}")
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
                print(f"[training_sweep] Synthetic generation FAILED for {run_label}: {e}")
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
                print(f"[training_sweep] Classifier FAILED for {run_label}: {e}")
                continue

            # 4. Recognizability
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
                "label": label,
                "gan_epochs": epochs,
                "config": cfg,
                "metrics": clf_out["metrics"],
                "recognizability": recognizability,
                "run_dir": run_dir,
            }
            save_json(entry, result_path)
            results.append(entry)
            rec_str = f"{recognizability:.4f}" if recognizability is not None else "N/A"
            print(f"[training_sweep] accuracy={clf_out['metrics']['accuracy']:.4f}  "
                  f"f1={clf_out['metrics']['f1_score']:.4f}  recognizability={rec_str}")

    save_json(results, os.path.join(output_dir, "training_sweep_results.json"))
    _plot_training_sweep(results, output_dir, dataset)
    return results


def _plot_training_sweep(results: list, output_dir: str, dataset: str) -> None:
    """One bar chart per factor."""
    by_factor: dict = {}
    for r in results:
        by_factor.setdefault(r["factor"], []).append(r)

    for factor, entries in by_factor.items():
        labels = [e["label"] for e in entries]
        acc = [e["metrics"]["accuracy"] for e in entries]
        f1 = [e["metrics"]["f1_score"] for e in entries]
        rec = [e["recognizability"] if e["recognizability"] is not None else 0.0
               for e in entries]

        x = np.arange(len(labels))
        w = 0.25
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.8), 5))
        ax.bar(x - w, acc, w, label="Accuracy", color="steelblue", alpha=0.85)
        ax.bar(x, f1, w, label="F1 Score", color="darkorange", alpha=0.85)
        ax.bar(x + w, rec, w, label="Recognizability", color="seagreen", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title(f"{dataset}: Training strategy ablation — {factor}")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        out_path = os.path.join(output_dir, f"training_ablation_{factor}.png")
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved plot: {out_path}")

    # Summary table
    print(f"\n{'='*80}")
    print(f"TRAINING STRATEGY SWEEP SUMMARY — {dataset}")
    print(f"{'='*80}")
    hdr = f"{'Factor':<18} {'Variant':<25} {'Accuracy':>10} {'F1':>10} {'Recogn.':>10}"
    print(hdr)
    print("-" * 80)
    for r in results:
        rec_str = f"{r['recognizability']:>10.4f}" if r["recognizability"] is not None else "       N/A"
        print(f"{r['factor']:<18} {r['label']:<25} "
              f"{r['metrics']['accuracy']:>10.4f} {r['metrics']['f1_score']:>10.4f} {rec_str}")
    print("=" * 80)


def parse_args():
    p = argparse.ArgumentParser(
        description="OFAT training strategy ablation sweep over GAN hyperparameters."
    )
    p.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    p.add_argument("--base-epochs", type=int, default=30,
                   help="Base GAN epoch count; epoch sweep uses fractions/multiples of this.")
    p.add_argument("--classifier-epochs", type=int, default=10)
    p.add_argument("--num-synthetic", type=int, default=6000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--output-dir", type=str, default="")
    p.add_argument("--quiet", action="store_true", default=False)
    p.add_argument("--factors", type=str, nargs="*", default=None,
                   help="Restrict to specific factors, e.g. loss_type lr_ratio")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.output_dir or os.path.join(
        "outputs", "training_sweep", f"{args.dataset}_{timestamp()}"
    )

    global _build_factors
    if args.factors:
        original_build = _build_factors

        def _build_factors(base_epochs):
            return {k: v for k, v in original_build(base_epochs).items()
                    if k in args.factors}

    run_training_sweep(
        dataset=args.dataset,
        base_epochs=args.base_epochs,
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
