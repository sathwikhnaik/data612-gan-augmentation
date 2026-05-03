"""
Cross-dataset comparison: MNIST vs Fashion-MNIST.

Reads comparison.json from two experiment directories and produces:
  - A side-by-side grouped bar chart (all scenarios, both datasets)
  - A delta table showing gain/loss vs real_only baseline per dataset
  - A per-class GAN quality comparison (recognizability) if available
  - A printed summary table
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def _load(exp_dir: str) -> dict:
    path = os.path.join(exp_dir, "comparison.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _scenario_metrics(summary: dict) -> dict:
    return {r["scenario"]: r["metrics"] for r in summary["runs"]}


def plot_scenario_comparison(mnist_summary, fmnist_summary, output_path: str) -> None:
    sm = _scenario_metrics(mnist_summary)
    sf = _scenario_metrics(fmnist_summary)
    scenarios = list(sm.keys())

    metrics = [("accuracy", "Accuracy"), ("f1_score", "F1 Score"),
               ("precision", "Precision"), ("recall", "Recall")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    for ax, (key, label) in zip(axes.flat, metrics):
        x = np.arange(len(scenarios))
        w = 0.35
        mnist_vals = [sm.get(s, {}).get(key, 0) for s in scenarios]
        fmnist_vals = [sf.get(s, {}).get(key, 0) for s in scenarios]
        ax.bar(x - w / 2, mnist_vals, w, label="MNIST", color="steelblue", alpha=0.85)
        ax.bar(x + w / 2, fmnist_vals, w, label="Fashion-MNIST", color="coral", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("MNIST vs Fashion-MNIST — all scenarios", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Scenario comparison plot saved to: {output_path}")


def plot_delta_vs_baseline(mnist_summary, fmnist_summary, output_path: str) -> None:
    """Show accuracy gain/loss vs real_only for every non-baseline scenario."""
    sm = _scenario_metrics(mnist_summary)
    sf = _scenario_metrics(fmnist_summary)
    baseline = "real_only"
    scenarios = [s for s in sm if s != baseline]

    x = np.arange(len(scenarios))
    w = 0.35
    mnist_deltas = [(sm[s]["accuracy"] - sm[baseline]["accuracy"]) * 100 for s in scenarios]
    fmnist_deltas = [(sf[s]["accuracy"] - sf[baseline]["accuracy"]) * 100 for s in scenarios]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, mnist_deltas, w, label="MNIST", color="steelblue", alpha=0.85)
    ax.bar(x + w / 2, fmnist_deltas, w, label="Fashion-MNIST", color="coral", alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=15, ha="right")
    ax.set_ylabel("Accuracy delta vs real_only (pp)")
    ax.set_title("Accuracy gain / loss relative to real_only baseline")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Delta plot saved to: {output_path}")


def plot_recognizability_comparison(mnist_summary, fmnist_summary, output_path: str) -> None:
    """Per-class GAN recognizability side by side."""
    mq = mnist_summary.get("synthetic_quality")
    fq = fmnist_summary.get("synthetic_quality")
    if not mq or not fq:
        print("[compare] Per-class quality data not found — skipping recognizability plot.")
        return

    mnist_pc = {r["class_name"]: r["recall"] for r in mq["per_class"]}
    fmnist_pc = {r["class_name"]: r["recall"] for r in fq["per_class"]}

    # Use MNIST class indices as x-axis (0-9 for both datasets)
    mnist_labels = [r["class_name"] for r in mq["per_class"]]
    fmnist_labels = [r["class_name"] for r in fq["per_class"]]
    x = np.arange(10)
    w = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w / 2, list(mnist_pc.values()), w, label="MNIST", color="steelblue", alpha=0.85)
    ax.bar(x + w / 2, list(fmnist_pc.values()), w, label="Fashion-MNIST", color="coral", alpha=0.85)

    # Dual x-tick labels: MNIST class / Fashion class
    tick_labels = [f"{m}\n{f}" for m, f in zip(mnist_labels, fmnist_labels)]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Recognizability (recall by real-trained classifier)")
    ax.set_title("Per-class GAN image recognizability: MNIST vs Fashion-MNIST")
    ax.legend()
    ax.axhline(mq["overall_recognizability"], color="steelblue", linestyle="--", alpha=0.5,
               label=f"MNIST mean={mq['overall_recognizability']:.3f}")
    ax.axhline(fq["overall_recognizability"], color="coral", linestyle="--", alpha=0.5,
               label=f"FM mean={fq['overall_recognizability']:.3f}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Recognizability comparison plot saved to: {output_path}")


def print_summary_table(mnist_summary, fmnist_summary) -> None:
    sm = _scenario_metrics(mnist_summary)
    sf = _scenario_metrics(fmnist_summary)
    scenarios = list(sm.keys())

    col_w = 14
    header = f"{'Scenario':<22} {'MNIST Acc':>{col_w}} {'FM Acc':>{col_w}} {'MNIST F1':>{col_w}} {'FM F1':>{col_w}}"
    print("\n" + "=" * len(header))
    print("CROSS-DATASET COMPARISON SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for s in scenarios:
        ma = sm.get(s, {}).get("accuracy", 0)
        fa = sf.get(s, {}).get("accuracy", 0)
        mf = sm.get(s, {}).get("f1_score", 0)
        ff = sf.get(s, {}).get("f1_score", 0)
        print(f"{s:<22} {ma:>{col_w}.4f} {fa:>{col_w}.4f} {mf:>{col_w}.4f} {ff:>{col_w}.4f}")
    print("-" * len(header))

    baseline = "real_only"
    if baseline in sm and baseline in sf:
        print(f"\nDelta vs '{baseline}':")
        for s in [sc for sc in scenarios if sc != baseline]:
            md = (sm.get(s, {}).get("accuracy", 0) - sm[baseline]["accuracy"]) * 100
            fd = (sf.get(s, {}).get("accuracy", 0) - sf[baseline]["accuracy"]) * 100
            print(f"  {s:<20}  MNIST: {md:+.2f}pp   Fashion-MNIST: {fd:+.2f}pp")

    mq = mnist_summary.get("synthetic_quality")
    fq = fmnist_summary.get("synthetic_quality")
    if mq and fq:
        print(f"\nGAN recognizability (overall):")
        print(f"  MNIST:         {mq['overall_recognizability']:.4f}")
        print(f"  Fashion-MNIST: {fq['overall_recognizability']:.4f}")
    print("=" * len(header) + "\n")


def main():
    p = argparse.ArgumentParser(description="Compare MNIST vs Fashion-MNIST experiment results.")
    p.add_argument("--mnist-dir", type=str, required=True, help="MNIST experiment output directory.")
    p.add_argument("--fmnist-dir", type=str, required=True, help="Fashion-MNIST experiment output directory.")
    p.add_argument("--output-dir", type=str, default="", help="Where to save comparison plots.")
    args = p.parse_args()

    out_dir = args.output_dir or os.path.join("outputs", "cross_dataset_comparison")
    os.makedirs(out_dir, exist_ok=True)

    mnist_summary = _load(args.mnist_dir)
    fmnist_summary = _load(args.fmnist_dir)

    print_summary_table(mnist_summary, fmnist_summary)
    plot_scenario_comparison(mnist_summary, fmnist_summary,
                             os.path.join(out_dir, "scenario_comparison.png"))
    plot_delta_vs_baseline(mnist_summary, fmnist_summary,
                           os.path.join(out_dir, "delta_vs_baseline.png"))
    plot_recognizability_comparison(mnist_summary, fmnist_summary,
                                    os.path.join(out_dir, "recognizability_comparison.png"))
    print(f"All comparison artifacts written to: {out_dir}")


if __name__ == "__main__":
    main()
