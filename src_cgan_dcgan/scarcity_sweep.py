"""
Scarcity sweep: train all classifier scenarios across a range of real training
data caps and answer two research questions:
  1. Does real_plus_synthetic beat real_only when real data is scarce?
  2. Does synthetic_only ever compete with real_only?

Supports multiple random seeds so plots show mean ± std error bands.
"""
import json
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from .config import class_names_for_dataset
from .train_classifier import train_classifier_core
from .utils import ensure_dir, save_json

SCENARIOS = ["real_only", "augmented_real", "real_plus_synthetic", "synthetic_only"]

SCENARIO_COLORS = {
    "real_only": "steelblue",
    "augmented_real": "seagreen",
    "real_plus_synthetic": "darkorange",
    "synthetic_only": "crimson",
}
SCENARIO_LABELS = {
    "real_only": "Real only",
    "augmented_real": "Augmented real (classical)",
    "real_plus_synthetic": "Real + Synthetic (GAN)",
    "synthetic_only": "Synthetic only (GAN)",
}


# ── Core sweep ─────────────────────────────────────────────────────────────

def run_scarcity_sweep(
    synthetic_root: str,
    dataset: str,
    scarcity_levels: List[int],
    seeds: List[int],
    classifier_epochs: int,
    output_dir: str,
    classifier_lr: float = 1e-3,
    num_workers: int = 2,
    quiet: bool = True,
) -> Dict:
    """
    For every (seed × scarcity_level × scenario) combination, train a classifier
    and record metrics.

    Returns nested dict: {str(level): {scenario: [{"seed": int, "metrics": {}, "per_class": []}]}}
    """
    ensure_dir(output_dir)
    results: Dict = {str(level): {s: [] for s in SCENARIOS} for level in scarcity_levels}
    total = len(seeds) * len(scarcity_levels) * len(SCENARIOS)
    done = 0

    for seed in seeds:
        for level in scarcity_levels:
            for scenario in SCENARIOS:
                done += 1
                run_dir = os.path.join(output_dir, "runs", f"seed{seed}", f"cap{level}", scenario)
                cached = os.path.join(run_dir, "metrics.json")
                pc_cached = os.path.join(run_dir, "per_class_metrics.json")

                if os.path.exists(cached):
                    # Resume: load from disk instead of retraining
                    with open(cached) as f:
                        metrics = json.load(f)
                    per_class = []
                    if os.path.exists(pc_cached):
                        with open(pc_cached) as f:
                            per_class = json.load(f).get("per_class", [])
                    print(
                        f"[scarcity_sweep] ({done}/{total}) CACHED  "
                        f"dataset={dataset}  cap={level:>6}  seed={seed}  scenario={scenario}"
                    )
                else:
                    print(
                        f"[scarcity_sweep] ({done}/{total}) RUNNING "
                        f"dataset={dataset}  cap={level:>6}  seed={seed}  scenario={scenario}"
                    )
                    syn_arg = "" if scenario in ("real_only", "augmented_real") else synthetic_root
                    out = train_classifier_core(
                        dataset=dataset,
                        scenario=scenario,
                        synthetic_root=syn_arg,
                        epochs=classifier_epochs,
                        batch_size=128,
                        lr=classifier_lr,
                        num_workers=num_workers,
                        seed=seed,
                        run_dir=run_dir,
                        quiet=quiet,
                        max_real_train_samples=level,
                        train_fraction=None,
                        train_subset_seed=seed,
                    )
                    metrics = out["metrics"]
                    per_class = out.get("per_class", [])

                results[str(level)][scenario].append(
                    {"seed": seed, "metrics": metrics, "per_class": per_class}
                )

            # Save after every level so partial results survive interruption
            save_json(results, os.path.join(output_dir, "scarcity_results.json"))

    return results


# ── Aggregation helpers ────────────────────────────────────────────────────

def _sorted_levels(results: Dict) -> List[int]:
    return sorted(int(k) for k in results.keys())


def _agg(values: List[float]):
    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(arr.std())


def _extract_metric(results: Dict, metric: str):
    """Returns {scenario: (levels_list, means_list, stds_list)}"""
    levels = _sorted_levels(results)
    out = {}
    for s in SCENARIOS:
        means, stds = [], []
        for level in levels:
            vals = [r["metrics"][metric] for r in results[str(level)][s]]
            m, sd = _agg(vals)
            means.append(m)
            stds.append(sd)
        out[s] = (levels, np.array(means), np.array(stds))
    return out


# ── Individual plots ───────────────────────────────────────────────────────

def plot_scarcity_curves(results: Dict, output_path: str, dataset: str, metric: str = "accuracy") -> None:
    """Main result: metric vs #real_samples, one line per scenario, with error bands."""
    data = _extract_metric(results, metric)
    fig, ax = plt.subplots(figsize=(11, 6))

    for s in SCENARIOS:
        levels, means, stds = data[s]
        ax.plot(levels, means, marker="o", color=SCENARIO_COLORS[s],
                label=SCENARIO_LABELS[s], linewidth=2.2, zorder=3)
        if stds.max() > 0:
            ax.fill_between(levels, means - stds, means + stds,
                            alpha=0.15, color=SCENARIO_COLORS[s])

    ax.set_xscale("log")
    ax.set_xlabel("Number of real training samples (log scale)", fontsize=11)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=11)
    ax.set_title(
        f"{dataset}: {metric.replace('_', ' ').title()} vs Real Training Data Size\n"
        f"(shaded = ±1 std across seeds)",
        fontsize=12,
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Scarcity curve saved: {output_path}")


def plot_gap_chart(results: Dict, output_path: str, dataset: str) -> None:
    """
    Accuracy delta vs real_only (in percentage points) for every other scenario.
    Positive = beats baseline, negative = worse.
    """
    data = _extract_metric(results, "accuracy")
    levels = _sorted_levels(results)
    _, real_means, _ = data["real_only"]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axhline(0, color=SCENARIO_COLORS["real_only"], linewidth=1.8,
               linestyle="-", label=SCENARIO_LABELS["real_only"] + " (baseline)")

    for s in [sc for sc in SCENARIOS if sc != "real_only"]:
        _, means, stds = data[s]
        gap_pp = (means - real_means) * 100
        std_pp = stds * 100
        ax.plot(levels, gap_pp, marker="o", color=SCENARIO_COLORS[s],
                label=SCENARIO_LABELS[s], linewidth=2.2, zorder=3)
        if std_pp.max() > 0:
            ax.fill_between(levels, gap_pp - std_pp, gap_pp + std_pp,
                            alpha=0.15, color=SCENARIO_COLORS[s])

    ax.set_xscale("log")
    ax.set_xlabel("Number of real training samples (log scale)", fontsize=11)
    ax.set_ylabel("Accuracy delta vs real_only (pp)", fontsize=11)
    ax.set_title(f"{dataset}: Augmentation benefit relative to real_only baseline", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Gap chart saved: {output_path}")


def plot_synthetic_only_crossover(results: Dict, output_path: str, dataset: str) -> None:
    """
    Shows whether/where synthetic_only matches or exceeds real_only as data gets scarce.
    Highlights the crossover region.
    """
    data = _extract_metric(results, "accuracy")
    levels = _sorted_levels(results)

    fig, ax = plt.subplots(figsize=(11, 5))

    for s in ["real_only", "synthetic_only", "real_plus_synthetic"]:
        _, means, stds = data[s]
        ax.plot(levels, means, marker="o", color=SCENARIO_COLORS[s],
                label=SCENARIO_LABELS[s], linewidth=2.2, zorder=3)
        if stds.max() > 0:
            ax.fill_between(levels, means - stds, means + stds,
                            alpha=0.15, color=SCENARIO_COLORS[s])

    # Shade region where synthetic_only >= real_only
    _, real_means, _ = data["real_only"]
    _, syn_means, _ = data["synthetic_only"]
    crossover_levels = [levels[i] for i in range(len(levels)) if syn_means[i] >= real_means[i]]
    if crossover_levels:
        ax.axvspan(min(crossover_levels) * 0.7, max(crossover_levels) * 1.3,
                   alpha=0.07, color="crimson",
                   label=f"synthetic_only ≥ real_only  (≤ {max(crossover_levels):,} real samples)")
    else:
        ax.text(0.5, 0.5, "synthetic_only never reaches real_only",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color="gray", style="italic")

    ax.set_xscale("log")
    ax.set_xlabel("Number of real training samples (log scale)", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title(f"{dataset}: Can synthetic-only compete with real-only?", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Crossover plot saved: {output_path}")


def plot_per_class_at_scarcity(
    results: Dict,
    output_path: str,
    dataset: str,
    highlight_levels: Optional[List[int]] = None,
) -> None:
    """Per-class F1 for all scenarios at two or three selected scarcity levels."""
    available = _sorted_levels(results)
    if highlight_levels is None:
        # Pick low, medium, and the largest available level
        highlight_levels = []
        for target in [300, 1200, available[-1]]:
            closest = min(available, key=lambda x: abs(x - target))
            if closest not in highlight_levels:
                highlight_levels.append(closest)

    class_names = class_names_for_dataset(dataset)
    n = len(highlight_levels)
    fig, axes = plt.subplots(n, 1, figsize=(13, 4.5 * n), squeeze=False)

    for row, level in enumerate(highlight_levels):
        ax = axes[row][0]
        x = np.arange(len(class_names))
        w = 0.20
        offsets = {"real_only": -1.5, "augmented_real": -0.5,
                   "real_plus_synthetic": 0.5, "synthetic_only": 1.5}
        for s, off in offsets.items():
            runs = results.get(str(level), {}).get(s, [])
            if not runs:
                continue
            all_f1 = [[rc["f1_score"] for rc in run["per_class"]] for run in runs]
            mean_f1 = np.array(all_f1).mean(axis=0)
            ax.bar(x + off * w, mean_f1, w, label=SCENARIO_LABELS[s],
                   color=SCENARIO_COLORS[s], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("F1 Score")
        ax.set_title(f"{dataset} — Per-class F1 at {level:,} real training samples")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Per-class scarcity plot saved: {output_path}")


def plot_f1_gain_heatmap(results: Dict, output_path: str, dataset: str) -> None:
    """
    Heatmap: rows = classes, cols = scarcity levels,
    colour = (real_plus_synthetic F1 - real_only F1).
    Reveals which classes benefit most from GAN augmentation under scarcity.
    """
    class_names = class_names_for_dataset(dataset)
    levels = _sorted_levels(results)

    gain_matrix = np.zeros((len(class_names), len(levels)))
    for col, level in enumerate(levels):
        for s_name, s_data in [("real_only", results[str(level)]["real_only"]),
                                ("real_plus_synthetic", results[str(level)]["real_plus_synthetic"])]:
            if not s_data:
                continue
            f1s = np.array([[rc["f1_score"] for rc in run["per_class"]] for run in s_data]).mean(axis=0)
            if s_name == "real_plus_synthetic":
                gain_matrix[:, col] += f1s
            else:
                gain_matrix[:, col] -= f1s  # subtract real_only

    vmax = max(abs(gain_matrix).max(), 0.01)
    fig, ax = plt.subplots(figsize=(13, 5))
    im = ax.imshow(gain_matrix, aspect="auto", cmap="RdYlGn",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([f"{l:,}" for l in levels], rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Real training samples")
    ax.set_title(
        f"{dataset}: Per-class F1 gain of real_plus_synthetic over real_only\n"
        "(green = GAN helps, red = GAN hurts)"
    )
    plt.colorbar(im, ax=ax, label="F1 gain (pp)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"F1 gain heatmap saved: {output_path}")


# ── Summary table ──────────────────────────────────────────────────────────

def print_scarcity_table(results: Dict, dataset: str) -> None:
    levels = _sorted_levels(results)
    data = _extract_metric(results, "accuracy")

    header_parts = [f"{'Cap':>8}"]
    for s in SCENARIOS:
        short = s.replace("real_plus_synthetic", "real+syn").replace("augmented_real", "aug_real")
        header_parts.append(f"{short:>14}")
    header_parts.append(f"{'GAN gain':>10}")
    header = "  ".join(header_parts)

    print(f"\n{'='*len(header)}")
    print(f"SCARCITY SWEEP — {dataset}")
    print(f"{'='*len(header)}")
    print(header)
    print(f"{'-'*len(header)}")

    _, real_means, _ = data["real_only"]
    _, rps_means, _ = data["real_plus_synthetic"]

    for i, level in enumerate(levels):
        row = [f"{level:>8,}"]
        for s in SCENARIOS:
            _, means, stds = data[s]
            row.append(f"{means[i]:>12.4f}" + (f"±{stds[i]:.3f}" if stds[i] > 0 else "      "))
        gain_pp = (rps_means[i] - real_means[i]) * 100
        sign = "+" if gain_pp >= 0 else ""
        row.append(f"{sign}{gain_pp:>7.2f}pp")
        print("  ".join(row))

    print(f"{'='*len(header)}\n")


# ── Cross-dataset comparison ───────────────────────────────────────────────

def plot_crossdataset_scarcity(
    mnist_results: Dict,
    fmnist_results: Dict,
    output_path: str,
    metric: str = "accuracy",
) -> None:
    """Side-by-side scarcity curves for both datasets."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    for ax, (results, dname) in zip(axes, [(mnist_results, "MNIST"), (fmnist_results, "Fashion-MNIST")]):
        data = _extract_metric(results, metric)
        for s in SCENARIOS:
            levels, means, stds = data[s]
            ax.plot(levels, means, marker="o", color=SCENARIO_COLORS[s],
                    label=SCENARIO_LABELS[s], linewidth=2.2, zorder=3)
            if stds.max() > 0:
                ax.fill_between(levels, means - stds, means + stds,
                                alpha=0.15, color=SCENARIO_COLORS[s])
        ax.set_xscale("log")
        ax.set_xlabel("Real training samples (log scale)", fontsize=10)
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=10)
        ax.set_title(dname, fontsize=12)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Scarcity sweep: {metric.replace('_', ' ').title()} — MNIST vs Fashion-MNIST\n"
        "(shaded = ±1 std across seeds)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Cross-dataset scarcity plot saved: {output_path}")


def plot_crossdataset_gap(mnist_results: Dict, fmnist_results: Dict, output_path: str) -> None:
    """Side-by-side gap charts (real_plus_synthetic - real_only) for both datasets."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    for ax, (results, dname) in zip(axes, [(mnist_results, "MNIST"), (fmnist_results, "Fashion-MNIST")]):
        data = _extract_metric(results, "accuracy")
        levels = _sorted_levels(results)
        _, real_means, _ = data["real_only"]
        ax.axhline(0, color=SCENARIO_COLORS["real_only"], linewidth=1.8,
                   linestyle="-", label="real_only (baseline)")
        for s in [sc for sc in SCENARIOS if sc != "real_only"]:
            _, means, stds = data[s]
            gap_pp = (means - real_means) * 100
            std_pp = stds * 100
            ax.plot(levels, gap_pp, marker="o", color=SCENARIO_COLORS[s],
                    label=SCENARIO_LABELS[s], linewidth=2.2, zorder=3)
            if std_pp.max() > 0:
                ax.fill_between(levels, gap_pp - std_pp, gap_pp + std_pp,
                                alpha=0.15, color=SCENARIO_COLORS[s])
        ax.set_xscale("log")
        ax.set_xlabel("Real training samples (log scale)", fontsize=10)
        ax.set_ylabel("Accuracy delta vs real_only (pp)", fontsize=10)
        ax.set_title(dname, fontsize=12)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "GAN augmentation benefit vs real_only — MNIST vs Fashion-MNIST\n"
        "(positive = beats baseline, negative = worse)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Cross-dataset gap chart saved: {output_path}")
