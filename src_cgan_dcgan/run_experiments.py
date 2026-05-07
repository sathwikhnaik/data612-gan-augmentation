import argparse
import csv
import os

import matplotlib.pyplot as plt

from .fid_eval import compute_fid_for_experiment
from .generate_synthetic import generate_synthetic_core
from .interpolate import generate_interpolation_grid
from .per_class_quality import evaluate_synthetic_quality
from .train_classifier import train_classifier_core
from .train_gan import train_gan_core
from .utils import ensure_dir, save_json, timestamp
from .volume_sweep import run_volume_sweep


def parse_args():
    p = argparse.ArgumentParser(
        description="Run full pipeline: cGAN (optional), synthetic data, four classifier scenarios, comparison table and plot."
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
    p.add_argument(
        "--max-real-train-samples",
        type=int,
        default=0,
        help="Stratified cap on real training images for classifiers (0 = full train set). Simulates label scarcity.",
    )
    p.add_argument(
        "--train-fraction",
        type=float,
        default=0.0,
        help="If >0 and max-real-train-samples is 0, use this fraction of the full real training set for classifiers.",
    )
    p.add_argument(
        "--max-gan-train-samples",
        type=int,
        default=0,
        help="Stratified cap on GAN training data (0 = full train set). Can differ from classifier scarcity.",
    )
    p.add_argument("--skip-gan", action="store_true", help="Use --generator-path (do not train GAN).")
    p.add_argument("--generator-path", type=str, default="", help="Trained generator .pt (required if --skip-gan).")
    p.add_argument("--synthetic-root", type=str, default="", help="Use existing synthetic folder; skip generation.")
    p.add_argument("--experiment-dir", type=str, default="", help="Root for this run; default outputs/experiments/<timestamp>.")
    p.add_argument("--quiet-classifiers", action="store_true", help="Less tqdm output during classifier training.")
    p.add_argument(
        "--compute-fid",
        action="store_true",
        help="Compute FID vs stratified real reference (downloads Inception weights on first run; GPU recommended).",
    )
    p.add_argument(
        "--fid-num-images",
        type=int,
        default=10000,
        help="Number of real train images to export for FID.",
    )
    p.add_argument("--fid-batch-size", type=int, default=64)
    p.add_argument(
        "--interpolation-steps",
        type=int,
        default=10,
        help="Number of steps for latent space interpolation grid.",
    )
    p.add_argument(
        "--sweep-volume",
        action="store_true",
        help="Run a volume sweep: train real_plus_synthetic classifier across multiple synthetic counts.",
    )
    p.add_argument(
        "--volume-sweep-values",
        type=int,
        nargs="+",
        default=[500, 1000, 3000, 6000, 12000],
        help="Synthetic counts to sweep over when --sweep-volume is set.",
    )
    # ── GAN Architecture ──────────────────────────────────────────────────
    p.add_argument("--gen-type", type=str, default="mlp", choices=["mlp", "dcgan"],
                   help="Generator architecture: MLP (default) or DCGAN (convolutional).")
    p.add_argument("--conditioning", type=str, default="embedding",
                   choices=["embedding", "onehot", "projection"],
                   help="How class labels are injected into the GAN.")
    p.add_argument("--embedding-dim", type=int, default=50)
    p.add_argument("--hidden-dims", type=int, nargs="+", default=None,
                   help="MLP generator hidden layer sizes, e.g. 256 512 1024.")
    p.add_argument("--disc-dropout", type=float, default=0.3)
    # ── GAN Training Strategy ─────────────────────────────────────────────
    p.add_argument("--loss-type", type=str, default="bce",
                   choices=["bce", "wgan_gp", "hinge"])
    p.add_argument("--label-smoothing", action="store_true", default=False)
    p.add_argument("--generator-lr", type=float, default=2e-4)
    p.add_argument("--discriminator-lr", type=float, default=2e-4)
    p.add_argument("--n-critic", type=int, default=1,
                   help="D updates per G update (auto-set to 5 for wgan_gp).")
    p.add_argument("--gp-lambda", type=float, default=10.0)
    # ── Extra classifier scenarios ────────────────────────────────────────
    p.add_argument("--extra-scenarios", type=str, nargs="*", default=[],
                   choices=["real_plus_synthetic_weighted", "synthetic_pretrain", "progressive"],
                   help="Additional classifier scenarios beyond the core four.")
    return p.parse_args()


def plot_comparison(rows, output_path: str, title: str) -> None:
    scenarios = [r["scenario"] for r in rows]
    metric_keys = ["accuracy", "precision", "recall", "f1_score"]
    labels = [k.replace("_", " ").title() for k in metric_keys]
    x = list(range(len(scenarios)))
    n_met = len(metric_keys)
    width = 0.8 / max(n_met, 1)
    fig, ax = plt.subplots(figsize=(10, 5))
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


def _aggregate_per_class(rows):
    return {r["scenario"]: r.get("per_class", []) for r in rows}


def write_csv(rows, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    fieldnames = [
        "scenario",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "mean_per_class_f1",
        "min_per_class_f1",
        "run_dir",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            m = r["metrics"]
            pcs = r.get("per_class") or []
            f1s = [p["f1_score"] for p in pcs] if pcs else [m["f1_score"]]
            w.writerow(
                {
                    "scenario": r["scenario"],
                    "accuracy": m["accuracy"],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "f1_score": m["f1_score"],
                    "mean_per_class_f1": sum(f1s) / len(f1s) if f1s else 0.0,
                    "min_per_class_f1": min(f1s) if f1s else 0.0,
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

    max_gan = args.max_gan_train_samples if args.max_gan_train_samples > 0 else None

    # ── 1. Train or load cGAN ──────────────────────────────────────────────
    if args.skip_gan:
        generator_path = args.generator_path
    else:
        print(f"[experiment] Training cGAN -> {gan_dir}")
        n_critic = args.n_critic
        if args.loss_type == "wgan_gp" and n_critic == 1:
            n_critic = 5
        generator_path = train_gan_core(
            dataset=args.dataset,
            epochs=args.gan_epochs,
            batch_size=args.gan_batch_size,
            latent_dim=args.latent_dim,
            seed=args.seed,
            num_workers=args.num_workers,
            model_dir=gan_dir,
            max_gan_train_samples=max_gan,
            gen_type=args.gen_type,
            conditioning=args.conditioning,
            embedding_dim=args.embedding_dim,
            hidden_dims=args.hidden_dims,
            disc_dropout=args.disc_dropout,
            loss_type=args.loss_type,
            label_smoothing=args.label_smoothing,
            generator_lr=args.generator_lr,
            discriminator_lr=args.discriminator_lr,
            n_critic=n_critic,
            gp_lambda=args.gp_lambda,
        )

    # ── 2. Latent space interpolation ─────────────────────────────────────
    interp_path = os.path.join(gan_dir, "interpolation_grid.png")
    print(f"[experiment] Generating interpolation grid -> {interp_path}")
    generate_interpolation_grid(
        generator_path=generator_path,
        output_path=interp_path,
        latent_dim=args.latent_dim,
        steps=args.interpolation_steps,
        seed=args.seed,
    )

    # ── 3. Generate synthetic images ───────────────────────────────────────
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
            model_dir=gan_dir,
        )

    max_real = args.max_real_train_samples if args.max_real_train_samples > 0 else None
    train_frac = args.train_fraction if args.train_fraction > 0 else None

    # ── 4. Train classifier scenarios ─────────────────────────────────────
    scenarios = ["real_only", "augmented_real", "real_plus_synthetic", "synthetic_only"]
    scenarios += [s for s in (args.extra_scenarios or []) if s not in scenarios]
    rows = []

    for scenario in scenarios:
        syn_arg = "" if scenario in ("real_only", "augmented_real") else synthetic_root
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
            max_real_train_samples=max_real,
            train_fraction=train_frac,
            train_subset_seed=args.seed,
        )
        rows.append(
            {
                "scenario": scenario,
                "metrics": out["metrics"],
                "per_class": out.get("per_class", []),
                "run_dir": out["run_dir"],
                "classifier_path": out["classifier_path"],
            }
        )

    # ── 5. Per-class GAN quality (uses real_only classifier) ───────────────
    real_only_row = next((r for r in rows if r["scenario"] == "real_only"), None)
    synthetic_quality = None
    if real_only_row:
        print("[experiment] Evaluating per-class GAN image quality...")
        synthetic_quality = evaluate_synthetic_quality(
            classifier_path=real_only_row["classifier_path"],
            synthetic_root=synthetic_root,
            dataset=args.dataset,
            output_dir=os.path.join(exp_root, "per_class_quality"),
        )

    # ── 6. Optional: FID ──────────────────────────────────────────────────
    fid_info = None
    if args.compute_fid:
        fid_dir = os.path.join(exp_root, "fid_work")
        print("[experiment] Computing FID (may take several minutes)...")
        fid_info = compute_fid_for_experiment(
            dataset_name=args.dataset,
            synthetic_root=synthetic_root,
            work_dir=fid_dir,
            num_images=args.fid_num_images,
            seed=args.seed,
            batch_size=args.fid_batch_size,
        )
        print(f"[experiment] FID: {fid_info['fid']:.4f}")

    # ── 7. Optional: volume sweep ─────────────────────────────────────────
    volume_sweep_results = None
    if args.sweep_volume:
        print("[experiment] Running synthetic data volume sweep...")
        volume_sweep_results = run_volume_sweep(
            generator_path=generator_path,
            dataset=args.dataset,
            num_synthetic_values=args.volume_sweep_values,
            classifier_epochs=args.classifier_epochs,
            seed=args.seed,
            output_dir=os.path.join(exp_root, "volume_sweep"),
            latent_dim=args.latent_dim,
            max_real_train_samples=max_real,
            train_fraction=train_frac,
            quiet=args.quiet_classifiers,
        )

    # ── 8. Write comparison outputs ───────────────────────────────────────
    csv_path = os.path.join(exp_root, "comparison.csv")
    json_path = os.path.join(exp_root, "comparison.json")
    plot_path = os.path.join(exp_root, "comparison_metrics.png")
    per_class_path = os.path.join(exp_root, "per_class_by_scenario.json")

    write_csv(rows, csv_path)
    summary = {
        "dataset": args.dataset,
        "generator_path": generator_path,
        "synthetic_root": synthetic_root,
        "interpolation_grid": interp_path,
        "max_real_train_samples": max_real,
        "train_fraction": train_frac,
        "max_gan_train_samples": max_gan,
        "fid": fid_info,
        "synthetic_quality": synthetic_quality,
        "volume_sweep": volume_sweep_results,
        "runs": rows,
        "per_class_by_scenario": _aggregate_per_class(rows),
    }
    save_json(summary, json_path)
    save_json(summary["per_class_by_scenario"], per_class_path)

    title = (
        f"{args.dataset} classifier comparison"
        + (f" (real train cap={max_real})" if max_real else "")
        + (f" (train fraction={train_frac})" if train_frac and not max_real else "")
    )
    plot_comparison(rows, plot_path, title=title)

    print(f"[experiment] Done. Artifacts under:\n  {exp_root}")
    print(f"  - {csv_path}")
    print(f"  - {json_path}")
    print(f"  - {per_class_path}")
    print(f"  - {plot_path}")
    print(f"  - {interp_path}")
    if synthetic_quality:
        print(f"  - {os.path.join(exp_root, 'per_class_quality', 'synthetic_recognizability.png')}")
    if args.sweep_volume:
        print(f"  - {os.path.join(exp_root, 'volume_sweep', 'volume_sweep.png')}")


if __name__ == "__main__":
    main()
