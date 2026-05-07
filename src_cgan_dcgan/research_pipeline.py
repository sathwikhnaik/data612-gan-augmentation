"""
Master research pipeline — runs all experiments and generates a professor-ready report.

Stages
──────
1  Main comparison  : MNIST + Fashion-MNIST (core 4 + 3 extra scenarios)
2  Architecture     : OFAT ablation over gen_type, conditioning, latent_dim, hidden_dims
3  Training strategy: OFAT ablation over loss_type, lr_ratio, label_smoothing
4  Scarcity         : sweep real-data caps on Fashion-MNIST (uses Stage-1 generator)
5  Volume sweep     : synthetic count vs accuracy on Fashion-MNIST
6  Cross-dataset    : side-by-side MNIST vs Fashion-MNIST comparison plots
7  Report           : compile HTML report from all artifacts

Usage
──────
python -m src.research_pipeline                     # run all stages
python -m src.research_pipeline --dry-run           # print plan, no runs
python -m src.research_pipeline --stages 1 2 3      # run specific stages
python -m src.research_pipeline --resume            # skip completed stages
"""
import argparse
import os
import shutil
import sys
import time
import traceback
from datetime import datetime

from .arch_sweep import run_arch_sweep
from .compare_datasets import (
    plot_delta_vs_baseline,
    plot_recognizability_comparison,
    plot_scenario_comparison,
    print_summary_table,
)
from .evaluate import plot_per_class_f1
from .fid_eval import compute_fid_for_experiment
from .generate_synthetic import generate_synthetic_core
from .interpolate import generate_interpolation_grid
from .per_class_quality import evaluate_synthetic_quality
from .scarcity_sweep import (
    plot_crossdataset_gap,
    plot_crossdataset_scarcity,
    plot_f1_gain_heatmap,
    plot_gap_chart,
    plot_scarcity_curves,
    plot_synthetic_only_crossover,
    print_scarcity_table,
    run_scarcity_sweep,
)
from .train_classifier import train_classifier_core
from .train_gan import train_gan_core
from .training_sweep import run_training_sweep
from .utils import ensure_dir, save_json, set_seed, timestamp
from .volume_sweep import run_volume_sweep

import json
import matplotlib.pyplot as plt
import numpy as np


# ── Output root ────────────────────────────────────────────────────────────
REPORT_ROOT = os.path.join("outputs", "research_report")
LOG_PATH = os.path.join(REPORT_ROOT, "pipeline.log")
STATE_PATH = os.path.join(REPORT_ROOT, "pipeline_state.json")

# ── Experiment settings ────────────────────────────────────────────────────
CFG = {
    # Main comparison
    "main_gan_epochs":     30,
    "main_clf_epochs":     10,
    "main_num_synthetic":  6000,
    # Ablation (faster; still meaningful with MPS)
    "abl_gan_epochs":      15,
    "abl_clf_epochs":      8,
    "abl_num_synthetic":   3000,
    # Scarcity
    "scarcity_levels":     [100, 300, 600, 1200, 3000, 6000, 12000],
    "scarcity_seeds":      [42, 123],
    "scarcity_clf_epochs": 10,
    # Volume sweep
    "volume_values":       [500, 1000, 3000, 6000, 12000],
    # Shared
    "batch_size": 128,
    "num_workers": 2,
    "seed": 42,
    "latent_dim": 100,
    "embedding_dim": 50,
}

CORE_SCENARIOS = [
    "real_only",
    "augmented_real",
    "real_plus_synthetic",
    "synthetic_only",
]
EXTRA_SCENARIOS = [
    "real_plus_synthetic_weighted",
    "synthetic_pretrain",
    "progressive",
]
ALL_SCENARIOS = CORE_SCENARIOS + EXTRA_SCENARIOS


# ── Logging ────────────────────────────────────────────────────────────────

class Logger:
    def __init__(self, path: str):
        ensure_dir(os.path.dirname(path))
        self._fh = open(path, "a", encoding="utf-8")

    def log(self, msg: str, level: str = "INFO"):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] [{level}] {msg}"
        print(line, flush=True)
        self._fh.write(line + "\n")
        self._fh.flush()

    def section(self, title: str):
        bar = "=" * 70
        self.log(bar)
        self.log(f"  {title}")
        self.log(bar)

    def close(self):
        self._fh.close()


_logger: Logger | None = None


def log(msg, level="INFO"):
    if _logger:
        _logger.log(msg, level)
    else:
        print(f"[{level}] {msg}")


# ── State management ───────────────────────────────────────────────────────

def load_state() -> dict:
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(state: dict):
    ensure_dir(REPORT_ROOT)
    save_json(state, STATE_PATH)


def stage_done(state: dict, name: str) -> bool:
    return state.get(f"stage_{name}_done", False)


def mark_done(state: dict, name: str, **kwargs):
    state[f"stage_{name}_done"] = True
    state.update(kwargs)
    save_state(state)


# ── Helper: run one classifier scenario ───────────────────────────────────

def _run_clf(
    dataset, scenario, synthetic_root, run_dir, epochs,
    quiet=True, max_real=None, clf_epochs=None
):
    epochs_ = clf_epochs or CFG["main_clf_epochs"]
    return train_classifier_core(
        dataset=dataset,
        scenario=scenario,
        synthetic_root=synthetic_root,
        epochs=epochs_,
        batch_size=CFG["batch_size"],
        lr=1e-3,
        num_workers=CFG["num_workers"],
        seed=CFG["seed"],
        run_dir=run_dir,
        quiet=quiet,
        max_real_train_samples=max_real,
    )


def _write_comparison(rows, out_dir, dataset, title_suffix=""):
    """Write CSV + bar chart comparing all scenarios."""
    import csv
    ensure_dir(out_dir)
    csv_path = os.path.join(out_dir, "comparison.csv")
    fieldnames = ["scenario", "accuracy", "precision", "recall", "f1_score",
                  "mean_per_class_f1", "min_per_class_f1"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            m = r["metrics"]
            pcs = r.get("per_class") or []
            f1s = [p["f1_score"] for p in pcs] if pcs else [m["f1_score"]]
            w.writerow({
                "scenario": r["scenario"],
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1_score": m["f1_score"],
                "mean_per_class_f1": sum(f1s) / len(f1s) if f1s else 0.0,
                "min_per_class_f1": min(f1s) if f1s else 0.0,
            })

    # Bar chart
    scenarios = [r["scenario"] for r in rows]
    metric_keys = ["accuracy", "precision", "recall", "f1_score"]
    labels = [k.replace("_", " ").title() for k in metric_keys]
    x = list(range(len(scenarios)))
    w_bar = 0.8 / len(metric_keys)
    fig, ax = plt.subplots(figsize=(max(10, len(scenarios) * 1.5), 5))
    for i, (key, lbl) in enumerate(zip(metric_keys, labels)):
        vals = [r["metrics"][key] for r in rows]
        offset = (i - (len(metric_keys) - 1) / 2) * w_bar
        ax.bar([xi + offset for xi in x], vals, w_bar, label=lbl)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"{dataset} — Classifier Scenario Comparison{title_suffix}")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "comparison_metrics.png"), dpi=200)
    plt.close(fig)
    return rows


# ── STAGE 1: Main comparison ───────────────────────────────────────────────

def stage_main_comparison(state: dict, dataset: str, stage_key: str, dry_run: bool):
    out_dir = os.path.join(REPORT_ROOT, "1_main_comparison", dataset)
    ensure_dir(out_dir)
    gan_dir = os.path.join(out_dir, "gan")
    syn_dir = os.path.join(out_dir, "synthetic")
    clf_dir = os.path.join(out_dir, "classifiers")

    if dry_run:
        log(f"[DRY-RUN] Would train cGAN on {dataset} for {CFG['main_gan_epochs']} epochs")
        log(f"[DRY-RUN] Would generate {CFG['main_num_synthetic']} synthetic images")
        log(f"[DRY-RUN] Would train {len(ALL_SCENARIOS)} classifiers")
        return

    # 1a. Train GAN
    gen_path_file = os.path.join(gan_dir, "generator.pt")
    if not os.path.exists(gen_path_file):
        log(f"[{dataset}] Training cGAN ({CFG['main_gan_epochs']} epochs)...")
        t0 = time.time()
        gen_path = train_gan_core(
            dataset=dataset,
            epochs=CFG["main_gan_epochs"],
            batch_size=CFG["batch_size"],
            latent_dim=CFG["latent_dim"],
            seed=CFG["seed"],
            num_workers=CFG["num_workers"],
            model_dir=gan_dir,
        )
        log(f"[{dataset}] GAN done in {(time.time()-t0)/60:.1f} min → {gen_path}")
    else:
        gen_path = gen_path_file
        log(f"[{dataset}] GAN already trained, skipping.")

    # 1b. Latent interpolation
    interp_path = os.path.join(gan_dir, "interpolation_grid.png")
    if not os.path.exists(interp_path):
        generate_interpolation_grid(
            generator_path=gen_path,
            output_path=interp_path,
            latent_dim=CFG["latent_dim"],
            steps=10,
            seed=CFG["seed"],
        )

    # 1c. Generate synthetic images
    if not os.path.isdir(syn_dir) or not os.listdir(syn_dir):
        log(f"[{dataset}] Generating {CFG['main_num_synthetic']} synthetic images...")
        generate_synthetic_core(
            generator_path=gen_path,
            dataset=dataset,
            num_samples=CFG["main_num_synthetic"],
            latent_dim=CFG["latent_dim"],
            seed=CFG["seed"],
            output_root=syn_dir,
            model_dir=gan_dir,
        )

    # 1d. Train all scenarios
    rows = []
    for scenario in ALL_SCENARIOS:
        s_dir = os.path.join(clf_dir, scenario)
        metric_file = os.path.join(s_dir, "metrics.json")
        if os.path.exists(metric_file):
            with open(metric_file) as f:
                metrics = json.load(f)
            pc_file = os.path.join(s_dir, "per_class_metrics.json")
            per_class = json.load(open(pc_file)).get("per_class", []) if os.path.exists(pc_file) else []
            clf_path = os.path.join(s_dir, "classifier.pt")
            log(f"[{dataset}] Scenario '{scenario}' cached (acc={metrics['accuracy']:.4f})")
        else:
            log(f"[{dataset}] Training classifier: {scenario}")
            t0 = time.time()
            syn_arg = "" if scenario in ("real_only", "augmented_real") else syn_dir
            out = _run_clf(dataset, scenario, syn_arg, s_dir,
                           epochs=CFG["main_clf_epochs"])
            metrics = out["metrics"]
            per_class = out.get("per_class", [])
            clf_path = out["classifier_path"]
            log(f"[{dataset}] '{scenario}' done in {(time.time()-t0)/60:.1f} min "
                f"— acc={metrics['accuracy']:.4f} f1={metrics['f1_score']:.4f}")

        rows.append({"scenario": scenario, "metrics": metrics,
                     "per_class": per_class, "classifier_path": clf_path})

    # 1e. Synthetic quality (recognizability)
    real_only_clf = next(r["classifier_path"] for r in rows if r["scenario"] == "real_only")
    qual_dir = os.path.join(out_dir, "synthetic_quality")
    qual_path = os.path.join(qual_dir, "synthetic_quality.json")
    if not os.path.exists(qual_path):
        log(f"[{dataset}] Computing per-class GAN recognizability...")
        evaluate_synthetic_quality(
            classifier_path=real_only_clf,
            synthetic_root=syn_dir,
            dataset=dataset,
            output_dir=qual_dir,
        )

    # 1f. Comparison chart + CSV
    _write_comparison(rows, out_dir, dataset)
    save_json(
        {"dataset": dataset, "runs": [
            {"scenario": r["scenario"], "metrics": r["metrics"],
             "per_class": r["per_class"], "run_dir": os.path.join(clf_dir, r["scenario"])}
            for r in rows
        ]},
        os.path.join(out_dir, "comparison.json"),
    )
    log(f"[{dataset}] All scenarios done. Artifacts at: {out_dir}")
    mark_done(state, stage_key,
              **{f"{stage_key}_dir": out_dir,
                 f"{stage_key}_gen_path": gen_path,
                 f"{stage_key}_syn_dir": syn_dir})


# ── STAGE 2: Architecture ablation ────────────────────────────────────────

def stage_arch_ablation(state: dict, dry_run: bool):
    out_dir = os.path.join(REPORT_ROOT, "2_arch_ablation")
    if dry_run:
        log("[DRY-RUN] Would run OFAT architecture ablation on Fashion-MNIST")
        return
    log("Starting architecture ablation on Fashion-MNIST...")
    run_arch_sweep(
        dataset="fashion_mnist",
        gan_epochs=CFG["abl_gan_epochs"],
        classifier_epochs=CFG["abl_clf_epochs"],
        num_synthetic=CFG["abl_num_synthetic"],
        seed=CFG["seed"],
        num_workers=CFG["num_workers"],
        output_dir=out_dir,
        quiet=True,
        batch_size=CFG["batch_size"],
    )
    mark_done(state, "arch_ablation", arch_ablation_dir=out_dir)


# ── STAGE 3: Training strategy ablation ───────────────────────────────────

def stage_training_ablation(state: dict, dry_run: bool):
    out_dir = os.path.join(REPORT_ROOT, "3_training_ablation")
    if dry_run:
        log("[DRY-RUN] Would run OFAT training strategy ablation on Fashion-MNIST")
        return
    log("Starting training strategy ablation on Fashion-MNIST...")
    run_training_sweep(
        dataset="fashion_mnist",
        base_epochs=CFG["abl_gan_epochs"],
        classifier_epochs=CFG["abl_clf_epochs"],
        num_synthetic=CFG["abl_num_synthetic"],
        seed=CFG["seed"],
        num_workers=CFG["num_workers"],
        output_dir=out_dir,
        quiet=True,
        batch_size=CFG["batch_size"],
    )
    mark_done(state, "training_ablation", training_ablation_dir=out_dir)


# ── STAGE 4: Scarcity sweep ────────────────────────────────────────────────

def stage_scarcity(state: dict, dry_run: bool):
    fmnist_syn_dir = state.get("fmnist_main_syn_dir", "")
    out_dir = os.path.join(REPORT_ROOT, "4_scarcity")

    if dry_run:
        log(f"[DRY-RUN] Would run scarcity sweep using synthetic from {fmnist_syn_dir}")
        return
    if not fmnist_syn_dir or not os.path.isdir(fmnist_syn_dir):
        log("Scarcity: Fashion-MNIST synthetic dir not found — skipping.", "WARN")
        return

    log(f"Starting scarcity sweep (levels={CFG['scarcity_levels']}, seeds={CFG['scarcity_seeds']})...")
    results = run_scarcity_sweep(
        synthetic_root=fmnist_syn_dir,
        dataset="fashion_mnist",
        scarcity_levels=CFG["scarcity_levels"],
        seeds=CFG["scarcity_seeds"],
        classifier_epochs=CFG["scarcity_clf_epochs"],
        output_dir=out_dir,
        quiet=True,
    )

    # Generate all scarcity plots
    plot_scarcity_curves(results, os.path.join(out_dir, "scarcity_accuracy.png"),
                         "fashion_mnist", metric="accuracy")
    plot_scarcity_curves(results, os.path.join(out_dir, "scarcity_f1.png"),
                         "fashion_mnist", metric="f1_score")
    plot_gap_chart(results, os.path.join(out_dir, "gap_vs_baseline.png"), "fashion_mnist")
    plot_synthetic_only_crossover(results, os.path.join(out_dir, "synthetic_crossover.png"),
                                  "fashion_mnist")
    plot_f1_gain_heatmap(results, os.path.join(out_dir, "f1_gain_heatmap.png"), "fashion_mnist")
    print_scarcity_table(results, "fashion_mnist")
    mark_done(state, "scarcity", scarcity_dir=out_dir)


# ── STAGE 5: Volume sweep ──────────────────────────────────────────────────

def stage_volume(state: dict, dry_run: bool):
    gen_path = state.get("fmnist_main_gen_path", "")
    out_dir = os.path.join(REPORT_ROOT, "5_volume_sweep")

    if dry_run:
        log(f"[DRY-RUN] Would run volume sweep with {CFG['volume_values']} synthetic counts")
        return
    if not gen_path or not os.path.exists(gen_path):
        log("Volume sweep: generator not found — skipping.", "WARN")
        return

    log(f"Starting volume sweep: {CFG['volume_values']} synthetic images...")
    run_volume_sweep(
        generator_path=gen_path,
        dataset="fashion_mnist",
        num_synthetic_values=CFG["volume_values"],
        classifier_epochs=CFG["abl_clf_epochs"],
        seed=CFG["seed"],
        output_dir=out_dir,
        latent_dim=CFG["latent_dim"],
        quiet=True,
    )
    mark_done(state, "volume", volume_dir=out_dir)


# ── STAGE 6: Cross-dataset comparison ─────────────────────────────────────

def stage_cross_dataset(state: dict, dry_run: bool):
    mnist_dir = state.get("mnist_main_dir", "")
    fmnist_dir = state.get("fmnist_main_dir", "")
    out_dir = os.path.join(REPORT_ROOT, "6_cross_dataset")

    if dry_run:
        log("[DRY-RUN] Would generate cross-dataset comparison plots")
        return

    for d, label in [(mnist_dir, "MNIST"), (fmnist_dir, "Fashion-MNIST")]:
        if not d or not os.path.exists(os.path.join(d, "comparison.json")):
            log(f"Cross-dataset: {label} comparison.json not found — skipping.", "WARN")
            return

    ensure_dir(out_dir)
    with open(os.path.join(mnist_dir, "comparison.json")) as f:
        ms = json.load(f)
    with open(os.path.join(fmnist_dir, "comparison.json")) as f:
        fs = json.load(f)

    print_summary_table(ms, fs)
    plot_scenario_comparison(ms, fs, os.path.join(out_dir, "scenario_comparison.png"))
    plot_delta_vs_baseline(ms, fs, os.path.join(out_dir, "delta_vs_baseline.png"))
    plot_recognizability_comparison(ms, fs, os.path.join(out_dir, "recognizability_comparison.png"))

    # Cross-dataset scarcity (if scarcity ran for both — here just FMNIST, so skip MNIST)
    log("Cross-dataset comparison plots saved.")
    mark_done(state, "cross_dataset", cross_dataset_dir=out_dir)


# ── STAGE 7: Generate HTML report ─────────────────────────────────────────

def stage_report(state: dict, dry_run: bool):
    if dry_run:
        log("[DRY-RUN] Would generate HTML report at outputs/research_report/index.html")
        return
    log("Generating HTML report...")
    from .report_generator import generate_report
    generate_report(REPORT_ROOT, state, CFG)
    mark_done(state, "report")
    log(f"Report ready: {os.path.join(REPORT_ROOT, 'index.html')}")


# ── Main ───────────────────────────────────────────────────────────────────

STAGE_MAP = {
    1: ("mnist_main",       "Stage 1a: Main comparison — MNIST"),
    2: ("fmnist_main",      "Stage 1b: Main comparison — Fashion-MNIST"),
    3: ("arch_ablation",    "Stage 2:  Architecture ablation"),
    4: ("training_ablation","Stage 3:  Training strategy ablation"),
    5: ("scarcity",         "Stage 4:  Scarcity sweep (Fashion-MNIST)"),
    6: ("volume",           "Stage 5:  Volume sweep (Fashion-MNIST)"),
    7: ("cross_dataset",    "Stage 6:  Cross-dataset comparison"),
    8: ("report",           "Stage 7:  HTML report generation"),
}


def parse_args():
    p = argparse.ArgumentParser(description="Master research pipeline.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print plan without running any experiments.")
    p.add_argument("--resume", action="store_true", default=True,
                   help="Skip already-completed stages (default: on).")
    p.add_argument("--no-resume", action="store_false", dest="resume",
                   help="Re-run all stages from scratch.")
    p.add_argument("--stages", type=int, nargs="*", default=None,
                   help="Run only these stage numbers (1-8). Default: all.")
    return p.parse_args()


def main():
    global _logger
    ensure_dir(REPORT_ROOT)
    _logger = Logger(LOG_PATH)
    args = parse_args()

    state = load_state() if args.resume else {}
    stages_to_run = args.stages or list(STAGE_MAP.keys())

    _logger.section("GAN Research Pipeline — Group 8")
    log(f"Output root : {os.path.abspath(REPORT_ROOT)}")
    log(f"Log file    : {os.path.abspath(LOG_PATH)}")
    log(f"Stages      : {stages_to_run}")
    log(f"Dry run     : {args.dry_run}")
    log(f"Resume      : {args.resume}")
    log(f"Compute     : {'MPS (Apple GPU)' if _mps_available() else 'CPU'}")
    log("")

    t_total = time.time()
    for stage_num, (stage_key, stage_title) in STAGE_MAP.items():
        if stage_num not in stages_to_run:
            continue
        if args.resume and stage_done(state, stage_key) and not args.dry_run:
            log(f"✓ {stage_title} — already done, skipping.")
            continue

        _logger.section(stage_title)
        t0 = time.time()
        try:
            if stage_num == 1:
                stage_main_comparison(state, "mnist", "mnist_main", args.dry_run)
            elif stage_num == 2:
                stage_main_comparison(state, "fashion_mnist", "fmnist_main", args.dry_run)
            elif stage_num == 3:
                stage_arch_ablation(state, args.dry_run)
            elif stage_num == 4:
                stage_training_ablation(state, args.dry_run)
            elif stage_num == 5:
                stage_scarcity(state, args.dry_run)
            elif stage_num == 6:
                stage_volume(state, args.dry_run)
            elif stage_num == 7:
                stage_cross_dataset(state, args.dry_run)
            elif stage_num == 8:
                stage_report(state, args.dry_run)

            elapsed = (time.time() - t0) / 60
            log(f"✓ {stage_title} completed in {elapsed:.1f} min")
        except Exception:
            log(f"✗ {stage_title} FAILED:", "ERROR")
            log(traceback.format_exc(), "ERROR")
            log("Continuing to next stage...", "WARN")

    total_min = (time.time() - t_total) / 60
    _logger.section(f"Pipeline complete — {total_min:.1f} min total")
    log(f"Report: {os.path.abspath(os.path.join(REPORT_ROOT, 'index.html'))}")
    _logger.close()


def _mps_available():
    try:
        import torch
        return torch.backends.mps.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
