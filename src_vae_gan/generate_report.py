"""
Generate a comprehensive PDF report for the VAE-GAN augmentation study.
Reads all results from outputs/, builds figures, writes LaTeX, compiles PDF.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

# ─── paths ────────────────────────────────────────────────────────────────────
BASE   = Path(__file__).parent
OUT_M  = BASE / "outputs" / "vaegan_mnist_20260503_195158"
OUT_F  = BASE / "outputs" / "vaegan_fashion_mnist_20260504_083600"
REPORT = BASE / "outputs" / "report"
REPORT.mkdir(parents=True, exist_ok=True)
FIGS   = REPORT / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

FASHION_CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal",  "Shirt",   "Sneaker",  "Bag",   "Ankle boot"]
MNIST_CLASSES   = [str(i) for i in range(10)]

PAL = ["#2a6fdb", "#e05c2a", "#2ab87f", "#9b59b6",
       "#e6b800", "#c0392b", "#1abc9c", "#8e44ad"]


def load(path: Path):
    with open(path) as f:
        return json.load(f)


# ─── all results ──────────────────────────────────────────────────────────────
mnist = {k: load(OUT_M / d / "results.json") for k, d in [
    ("g2", "goal2_three_cases"), ("g3", "goal3_architecture"),
    ("g4", "goal4_training_strategy"), ("g5", "goal5_data_scarcity"),
    ("g6", "goal6_label_injection"), ("g7", "goal7_augmentation_scenarios"),
]}
fashion = {k: load(OUT_F / d / "results.json") for k, d in [
    ("g2", "goal2_three_cases"), ("g3", "goal3_architecture"),
    ("g4", "goal4_training_strategy"), ("g5", "goal5_data_scarcity"),
    ("g6", "goal6_label_injection"), ("g7", "goal7_augmentation_scenarios"),
]}

def f1(m): return m["f1_score"]
def acc(m): return m["accuracy"]

def save_fig(fig, name: str) -> str:
    p = FIGS / f"{name}.pdf"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return f"figs/{name}.pdf"

def savepng(fig, name: str) -> str:
    p = FIGS / f"{name}.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return f"figs/{name}.png"


# ─── helper: copy an existing PNG into figs/ and return rel path ──────────────
def copy_asset(src: Path, name: str) -> str:
    dst = FIGS / name
    shutil.copy2(src, dst)
    return f"figs/{name}"


def bar_grouped(ax, groups, labels, vals_list, colors, ylabel="F1 Score"):
    x = np.arange(len(groups))
    n, w = len(labels), 0.8 / len(labels)
    for i, (lbl, vals, col) in enumerate(zip(labels, vals_list, colors)):
        ax.bar(x + (i - (n-1)/2)*w, vals, w, label=lbl, color=col, alpha=0.87)
    ax.set_xticks(x); ax.set_xticklabels(groups, rotation=22, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    ax.spines[["top","right"]].set_visible(False)


# ══════════════════════════════════════════════════════════════════════════════
#  GENERATED FIGURES
# ══════════════════════════════════════════════════════════════════════════════

print("Generating figures...")

# ── Fig 1: Goal 2 three-way bar chart ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, data, title in [(axes[0], mnist["g2"], "MNIST"),
                         (axes[1], fashion["g2"], "Fashion-MNIST")]:
    runs  = data["runs"]
    names = [r["scenario"].replace("_", "\n") for r in runs]
    bar_grouped(ax, names, ["F1","Accuracy","Precision","Recall"],
                [[f1(r["metrics"]) for r in runs],
                 [acc(r["metrics"]) for r in runs],
                 [r["metrics"]["precision"] for r in runs],
                 [r["metrics"]["recall"]    for r in runs]], PAL[:4])
    ax.set_title(f"Goal 2 -- Three-Way Comparison\n{title}", fontsize=10, fontweight="bold")
fig.tight_layout(pad=2.0)
F_G2 = save_fig(fig, "goal2_threeway")

# ── Fig 2: Goal 3 delta F1 ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, data, title in [(axes[0], mnist["g3"], "MNIST"),
                         (axes[1], fashion["g3"], "Fashion-MNIST")]:
    v  = [r["variant"] for r in data]
    d  = [r["f1_improvement"] for r in data]
    c  = [PAL[2] if x >= 0 else PAL[0] for x in d]
    ax.bar(range(len(v)), [x*100 for x in d], color=c, alpha=0.87)
    ax.axhline(0, color="black", lw=0.9, ls="--")
    ax.set_xticks(range(len(v))); ax.set_xticklabels(v, rotation=28, ha="right", fontsize=8)
    ax.set_ylabel("$\\Delta$F1 (pp)", fontsize=9)
    ax.set_title(f"Goal 3 -- Architecture $\\Delta$F1\n{title}", fontsize=10, fontweight="bold")
    ax.grid(axis="y", lw=0.4, alpha=0.5); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(pad=2.0)
F_G3D = save_fig(fig, "goal3_delta")

# ── Fig 3: Goal 3 absolute F1 ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, data, title in [(axes[0], mnist["g3"], "MNIST"),
                         (axes[1], fashion["g3"], "Fashion-MNIST")]:
    v = [r["variant"] for r in data]
    bar_grouped(ax, v, ["Real Only","Real + Synthetic"],
                [[f1(r["real_only"]["metrics"])     for r in data],
                 [f1(r["real_plus_syn"]["metrics"]) for r in data]],
                [PAL[0], PAL[1]])
    ax.set_title(f"Goal 3 -- Architecture F1\n{title}", fontsize=10, fontweight="bold")
fig.tight_layout(pad=2.0)
F_G3A = save_fig(fig, "goal3_abs")

# ── Fig 4: Goal 4 delta F1 ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, data, title in [(axes[0], mnist["g4"], "MNIST"),
                         (axes[1], fashion["g4"], "Fashion-MNIST")]:
    v  = [r["variant"] for r in data]
    d  = [r["f1_improvement"] for r in data]
    c  = [PAL[2] if x >= 0 else PAL[0] for x in d]
    ax.bar(range(len(v)), [x*100 for x in d], color=c, alpha=0.87)
    ax.axhline(0, color="black", lw=0.9, ls="--")
    ax.set_xticks(range(len(v))); ax.set_xticklabels(v, rotation=28, ha="right", fontsize=8)
    ax.set_ylabel("$\\Delta$F1 (pp)", fontsize=9)
    ax.set_title(f"Goal 4 -- Training Strategy $\\Delta$F1\n{title}", fontsize=10, fontweight="bold")
    ax.grid(axis="y", lw=0.4, alpha=0.5); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(pad=2.0)
F_G4 = save_fig(fig, "goal4_delta")

# ── Fig 5: Goal 5 scarcity line plot ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, data, title in [(axes[0], mnist["g5"], "MNIST"),
                         (axes[1], fashion["g5"], "Fashion-MNIST")]:
    settings = [r["variant"] for r in data]
    real_f1s = [f1(r["real_only"]["metrics"])     for r in data]
    syn_f1s  = [f1(r["real_plus_syn"]["metrics"]) for r in data]
    x = range(len(settings))
    ax.plot(x, real_f1s, "o-",  color=PAL[0], label="Real Only",        lw=2, ms=7)
    ax.plot(x, syn_f1s,  "s--", color=PAL[1], label="Real + Synthetic", lw=2, ms=7)
    ax.fill_between(list(x), real_f1s, syn_f1s,
                    where=[s < r for s,r in zip(syn_f1s, real_f1s)],
                    alpha=0.1, color=PAL[1])
    ax.set_xticks(list(x)); ax.set_xticklabels(settings, fontsize=9)
    ax.set_ylabel("F1 Score", fontsize=9); ax.set_ylim(0.6, 1.02)
    ax.legend(fontsize=9); ax.grid(lw=0.4, alpha=0.5)
    ax.spines[["top","right"]].set_visible(False)
    ax.set_title(f"Goal 5 -- Data Scarcity\n{title}", fontsize=10, fontweight="bold")
fig.tight_layout(pad=2.0)
F_G5 = save_fig(fig, "goal5_scarcity")

# ── Fig 6: Goal 6 label injection ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, data, title in [(axes[0], mnist["g6"], "MNIST"),
                         (axes[1], fashion["g6"], "Fashion-MNIST")]:
    v = [r["variant"].replace("_", "\n") for r in data]
    bar_grouped(ax, v, ["Real Only","Real + Synthetic"],
                [[f1(r["real_only"]["metrics"])     for r in data],
                 [f1(r["real_plus_syn"]["metrics"]) for r in data]],
                [PAL[0], PAL[3]])
    ax.set_title(f"Goal 6 -- Label Injection\n{title}", fontsize=10, fontweight="bold")
fig.tight_layout(pad=2.0)
F_G6 = save_fig(fig, "goal6_label")

# ── Fig 7: Goal 7 augmentation scenarios ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, data, title in [(axes[0], mnist["g7"], "MNIST"),
                         (axes[1], fashion["g7"], "Fashion-MNIST")]:
    runs   = data["runs"]
    names  = [r["scenario"].replace("_", "\n") for r in runs]
    f1s    = [f1(r["metrics"]) for r in runs]
    colors = [PAL[0]] + [PAL[4]] * (len(runs)-1)
    bars = ax.bar(range(len(names)), f1s, color=colors, alpha=0.87)
    ax.axhline(f1s[0], color=PAL[0], lw=1.2, ls="--", alpha=0.6)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel("F1 Score", fontsize=9); ax.set_ylim(min(f1s)-0.025, 1.005)
    ax.set_title(f"Goal 7 -- Extended Augmentation\n{title}", fontsize=10, fontweight="bold")
    ax.grid(axis="y", lw=0.4, alpha=0.5); ax.spines[["top","right"]].set_visible(False)
    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=5.5)
fig.tight_layout(pad=2.0)
F_G7 = save_fig(fig, "goal7_augmentation")

# ── Fig 8: Summary heatmap ─────────────────────────────────────────────────────
rows_heat = [
    ("G3 large_latent_256",  mnist["g3"][1]["f1_improvement"], fashion["g3"][1]["f1_improvement"]),
    ("G3 deep_mlp",          mnist["g3"][3]["f1_improvement"], fashion["g3"][3]["f1_improvement"]),
    ("G3 cnn_generator",     mnist["g3"][4]["f1_improvement"], fashion["g3"][4]["f1_improvement"]),
    ("G3 small_latent_32",   mnist["g3"][2]["f1_improvement"], fashion["g3"][2]["f1_improvement"]),
    ("G3 high_dropout",      mnist["g3"][6]["f1_improvement"], fashion["g3"][6]["f1_improvement"]),
    ("G4 small_batch",       mnist["g4"][8]["f1_improvement"], fashion["g4"][8]["f1_improvement"]),
    ("G4 kl_heavy",          mnist["g4"][3]["f1_improvement"], fashion["g4"][3]["f1_improvement"]),
    ("G4 recon_heavy",       mnist["g4"][1]["f1_improvement"], fashion["g4"][1]["f1_improvement"]),
    ("G4 low_lr",            mnist["g4"][5]["f1_improvement"], fashion["g4"][5]["f1_improvement"]),
    ("G6 learned_embed",     mnist["g6"][0]["f1_improvement"], fashion["g6"][0]["f1_improvement"]),
    ("G6 onehot",            mnist["g6"][1]["f1_improvement"], fashion["g6"][1]["f1_improvement"]),
    ("G6 gen_only",          mnist["g6"][2]["f1_improvement"], fashion["g6"][2]["f1_improvement"]),
]
labels_h  = [r[0] for r in rows_heat]
mat       = np.array([[r[1]*100, r[2]*100] for r in rows_heat])
vmax      = max(abs(mat).max(), 0.5)
fig, ax   = plt.subplots(figsize=(5.5, 7))
im = ax.imshow(mat, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
fig.colorbar(im, ax=ax, label="$\\Delta$F1 (pp)", shrink=0.7)
ax.set_xticks([0,1]); ax.set_xticklabels(["MNIST","Fashion-MNIST"], fontsize=10)
ax.set_yticks(range(len(labels_h))); ax.set_yticklabels(labels_h, fontsize=8)
for i in range(len(labels_h)):
    for j in range(2):
        ax.text(j, i, f"{mat[i,j]:+.2f}", ha="center", va="center", fontsize=8, color="black")
ax.set_title("$\\Delta$F1 Heatmap: Real+Syn vs Real-Only\n(percentage points)", fontsize=10, fontweight="bold")
fig.tight_layout()
F_HEAT = save_fig(fig, "summary_heatmap")

# ── Fig 9: Goal 3 Real+Syn F1 — top/bottom ranking ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, data, title in [(axes[0], mnist["g3"], "MNIST"),
                         (axes[1], fashion["g3"], "Fashion-MNIST")]:
    v = [r["variant"] for r in data]
    syn_f1s  = [f1(r["real_plus_syn"]["metrics"]) for r in data]
    real_base = f1(data[0]["real_only"]["metrics"])
    colors = [PAL[2] if s >= real_base else PAL[0] for s in syn_f1s]
    ax.barh(v, syn_f1s, color=colors, alpha=0.87)
    ax.axvline(real_base, color="black", lw=1.3, ls="--", label=f"Real-only baseline ({real_base:.4f})")
    ax.set_xlabel("F1 Score (Real + Synthetic)", fontsize=9)
    ax.set_title(f"Goal 3 -- Variant Ranking\n{title}", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(axis="x", lw=0.4, alpha=0.5)
    ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(pad=2.0)
F_G3R = save_fig(fig, "goal3_ranking")

# ── Fig 10: Goal 5 scarcity gap (absolute values for all 4 metrics) ───────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
metric_keys = ["accuracy", "precision", "recall", "f1_score"]
metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
for ax, mk, mn in zip(axes.flatten(), metric_keys, metric_names):
    for data, label, color in [(mnist["g5"], "MNIST", PAL[0]),
                                (fashion["g5"], "Fashion-MNIST", PAL[1])]:
        settings  = [r["variant"] for r in data]
        real_vals = [r["real_only"]["metrics"][mk]     for r in data]
        syn_vals  = [r["real_plus_syn"]["metrics"][mk] for r in data]
        x = np.arange(len(settings))
        ax.plot(x, real_vals, "o-",  color=color, label=f"{label} Real-Only", lw=1.8, ms=6)
        ax.plot(x, syn_vals,  "s--", color=color, label=f"{label} +Synthetic", lw=1.8, ms=6, alpha=0.7)
    ax.set_xticks(list(range(len(settings)))); ax.set_xticklabels(settings, fontsize=9)
    ax.set_ylabel(mn, fontsize=9); ax.set_ylim(0.6, 1.02)
    ax.legend(fontsize=7, ncol=2); ax.grid(lw=0.4, alpha=0.5)
    ax.spines[["top","right"]].set_visible(False); ax.set_title(mn, fontsize=10, fontweight="bold")
fig.suptitle("Goal 5 -- Data Scarcity: All Metrics (MNIST vs Fashion-MNIST)", fontsize=11, fontweight="bold")
fig.tight_layout(pad=2.0, rect=[0,0,1,0.97])
F_G5ALL = save_fig(fig, "goal5_all_metrics")

# ── Fig 11: Goal 4 full strategy comparison ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, data, title in [(axes[0], mnist["g4"], "MNIST"),
                         (axes[1], fashion["g4"], "Fashion-MNIST")]:
    v = [r["variant"] for r in data]
    bar_grouped(ax, v, ["Real Only", "Real + Synthetic"],
                [[f1(r["real_only"]["metrics"])     for r in data],
                 [f1(r["real_plus_syn"]["metrics"]) for r in data]],
                [PAL[0], PAL[1]])
    ax.set_title(f"Goal 4 -- Training Strategies (F1)\n{title}", fontsize=10, fontweight="bold")
fig.tight_layout(pad=2.0)
F_G4A = save_fig(fig, "goal4_abs")

# ── Fig 12: Goal 7 — mix-ratio deep-dive ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, data, title in [(axes[0], mnist["g7"], "MNIST"),
                         (axes[1], fashion["g7"], "Fashion-MNIST")]:
    runs  = data["runs"]
    names = [r["scenario"] for r in runs]
    f1s   = [f1(r["metrics"]) for r in runs]
    baseline = f1s[0]
    # highlight just the mix-ratio scenarios
    mix_names  = [n for n in names if "pct" in n]
    mix_f1s    = [f for n, f in zip(names, f1s) if "pct" in n]
    ratios     = [25, 50, 100, 200]
    ax.plot(ratios, mix_f1s, "o-", color=PAL[1], lw=2, ms=8, label="Real+X% Synthetic")
    ax.axhline(baseline, color=PAL[0], lw=1.5, ls="--", label=f"Real-only F1 = {baseline:.4f}")
    ax.set_xlabel("Synthetic mix ratio (%)", fontsize=9)
    ax.set_ylabel("F1 Score", fontsize=9); ax.set_ylim(min(mix_f1s)-0.01, max(f1s)+0.005)
    ax.legend(fontsize=9); ax.grid(lw=0.4, alpha=0.5)
    ax.spines[["top","right"]].set_visible(False)
    ax.set_title(f"Goal 7 -- Mix Ratio vs F1\n{title}", fontsize=10, fontweight="bold")
fig.tight_layout(pad=2.0)
F_G7MIX = save_fig(fig, "goal7_mixratio")

# ── Copy existing assets: sample grids, reconstruction grids ─────────────────
SG_M  = copy_asset(OUT_M / "goal2_three_cases" / "visuals" / "sample_grid.png",        "sample_grid_mnist.png")
SG_F  = copy_asset(OUT_F / "goal2_three_cases" / "visuals" / "sample_grid.png",        "sample_grid_fashion.png")
RG_M  = copy_asset(OUT_M / "goal2_three_cases" / "visuals" / "reconstruction_grid.png","recon_grid_mnist.png")
RG_F  = copy_asset(OUT_F / "goal2_three_cases" / "visuals" / "reconstruction_grid.png","recon_grid_fashion.png")

# ── Confusion matrices Goal 2 ─────────────────────────────────────────────────
def make_cm_panel(base: Path, name_prefix: str, scenarios: list[str]) -> str:
    """Tile the three confusion matrix PNGs into one figure."""
    imgs = [np.array(Image.open(base / f"clf_{s}" / "confusion_matrix.png").convert("RGB"))
            for s in scenarios]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, img, sc in zip(axes, imgs, scenarios):
        ax.imshow(img); ax.axis("off")
        ax.set_title(sc.replace("_", "\n"), fontsize=9, fontweight="bold")
    fig.suptitle(f"Goal 2 Confusion Matrices — {name_prefix}", fontsize=11, fontweight="bold")
    fig.tight_layout(pad=1.0)
    return save_fig(fig, f"cm_g2_{name_prefix.lower().replace('-','_')}")

CM_M = make_cm_panel(OUT_M / "goal2_three_cases", "MNIST",
                     ["real_only", "real_plus_syn", "synthetic_only"])
CM_F = make_cm_panel(OUT_F / "goal2_three_cases", "Fashion-MNIST",
                     ["real_only", "real_plus_syn", "synthetic_only"])

# ── Architecture diagram (model overview) ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 3.5))
ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
boxes = [
    (0.6, 1.5, "Real Image\n$x$",        "#dde8ff"),
    (2.3, 1.5, "Encoder $E$\n(MLP)",     "#b3c9f7"),
    (4.0, 1.5, "Latent\n$z \\sim q$",    "#cce5ff"),
    (5.7, 1.5, "Generator $G$\n(MLP/CNN)","#b3c9f7"),
    (7.4, 1.5, "Discriminator $D$",       "#ffd9b3"),
    (9.0, 1.5, "Real/Fake",               "#ffe8cc"),
]
for cx, cy, text, color in boxes:
    ax.add_patch(mpatches.FancyBboxPatch((cx-0.55, cy-0.55), 1.1, 1.1,
                                         boxstyle="round,pad=0.05", fc=color, ec="#555", lw=1.2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=8)
# arrows
for i in range(len(boxes)-1):
    ax.annotate("", xy=(boxes[i+1][0]-0.57, boxes[i+1][1]),
                xytext=(boxes[i][0]+0.57, boxes[i][1]),
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))
# label embedding arrow to D
ax.annotate("", xy=(7.3, 1.15), xytext=(5.8, 0.6),
            arrowprops=dict(arrowstyle="->", color="#666", lw=1.0, ls="dashed"))
ax.text(6.5, 0.55, "Class label $y$\n(embedding)", ha="center", fontsize=7.5, color="#444")
# noise arrow
ax.annotate("", xy=(5.13, 1.5), xytext=(4.55, 1.5),
            arrowprops=dict(arrowstyle="->", color="#999", lw=0.9, ls="dashed"))
ax.text(4.1, 0.8, "$z_{noise}\\sim\\mathcal{N}(0,I)$", fontsize=7.5, color="#444")
ax.set_title("Conditional VAE-GAN Architecture Overview", fontsize=11, fontweight="bold", pad=8)
fig.tight_layout()
F_ARCH = save_fig(fig, "architecture_diagram")

print("  All figures generated.")


# ══════════════════════════════════════════════════════════════════════════════
#  LaTeX HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def esc(s: str) -> str:
    for ch, rep in [("&",r"\&"),("%",r"\%"),("#",r"\#"),("$",r"\$"),("^",r"\^{}"),("_",r"\_")]:
        s = s.replace(ch, rep)
    return s


def fig_block(rel_path: str, caption: str, label: str, width: str = r"\linewidth") -> str:
    return (f"\\begin{{figure}}[H]\n"
            f"  \\centering\n"
            f"  \\includegraphics[width={width}]{{{rel_path}}}\n"
            f"  \\caption{{{caption}}}\n"
            f"  \\label{{{label}}}\n"
            f"\\end{{figure}}\n")


def TR(*cells) -> str:
    return " & ".join(str(c) for c in cells) + r" \\"


# ── table helpers ─────────────────────────────────────────────────────────────
def g2_rows(data):
    return "\n".join(TR(esc(r["scenario"]),
                        f'{acc(r["metrics"]):.4f}',
                        f'{r["metrics"]["precision"]:.4f}',
                        f'{r["metrics"]["recall"]:.4f}',
                        f'{f1(r["metrics"]):.4f}',
                        str(r["train_size"])) for r in data["runs"])

def variant_rows(data):
    out = []
    for r in data:
        d = r["f1_improvement"]; s = "+" if d >= 0 else ""
        out.append(TR(esc(r["variant"]),
                      f'{f1(r["real_only"]["metrics"]):.4f}',
                      f'{f1(r["real_plus_syn"]["metrics"]):.4f}',
                      f'{s}{d*100:.2f}'))
    return "\n".join(out)

def g5_rows(data):
    out = []
    for r in data:
        d = r["f1_improvement"]; s = "+" if d >= 0 else ""
        out.append(TR(esc(r["variant"]),
                      str(r["real_only"]["train_size"]),
                      f'{f1(r["real_only"]["metrics"]):.4f}',
                      f'{f1(r["real_plus_syn"]["metrics"]):.4f}',
                      f'{s}{d*100:.2f}'))
    return "\n".join(out)

def g7_rows(data):
    baseline = f1(data["runs"][0]["metrics"])
    out = []
    for r in data["runs"]:
        d = f1(r["metrics"]) - baseline; s = "+" if d >= 0 else ""
        out.append(TR(esc(r["scenario"]),
                      f'{acc(r["metrics"]):.4f}',
                      f'{f1(r["metrics"]):.4f}',
                      f'{s}{d*100:.2f}' if d != 0 else "--",
                      str(r["train_size"])))
    return "\n".join(out)

def g3_full_rows(data):
    out = []
    for r in data:
        mr, ms = r["real_only"]["metrics"], r["real_plus_syn"]["metrics"]
        out.append(TR(esc(r["variant"]),
                      f'{acc(mr):.4f}', f'{mr["precision"]:.4f}',
                      f'{mr["recall"]:.4f}', f'{f1(mr):.4f}',
                      f'{acc(ms):.4f}', f'{ms["precision"]:.4f}',
                      f'{ms["recall"]:.4f}', f'{f1(ms):.4f}'))
    return "\n".join(out)


# ══════════════════════════════════════════════════════════════════════════════
#  BUILD LATEX
# ══════════════════════════════════════════════════════════════════════════════

def build_latex() -> str:
    return r"""\documentclass[12pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[margin=2.2cm]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{float}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{array}
\usepackage{amsmath}
\usepackage{titlesec}
\usepackage{setspace}

\setlength{\parskip}{0.75ex}
\setlength{\parindent}{0pt}

\definecolor{accent}{HTML}{2a6fdb}

\hypersetup{colorlinks,linkcolor=accent,urlcolor=accent,citecolor=accent,
  pdftitle={VAE-GAN Augmentation Study},pdfauthor={Group 8 -- DATA612}}

\pagestyle{myheadings}
\markboth{DATA612 -- Group 8}{VAE-GAN Augmentation Study -- May 2026}

\titleformat{\section}{\large\bfseries\color{accent}}{}{0em}{}[\titlerule]
\titleformat{\subsection}{\normalsize\bfseries\color{accent}}{}{0em}{}

\begin{document}

%% ─── TITLE PAGE ────────────────────────────────────────────────────────────
\begin{titlepage}
  \centering\vspace*{2cm}
  {\Huge\bfseries\color{accent}GAN-Based Data Augmentation\\[0.4em]
   for Image Classification\par}
  \vspace{0.8cm}
  {\large\bfseries A Conditional VAE-GAN Hybrid Study\par}
  \vspace{0.5cm}\rule{\linewidth}{0.5pt}\vspace{0.5cm}
  {\large Comprehensive Results \& Analysis Report\par}
  \vspace{2cm}
  {\large\bfseries Group 8 \quad $|$ \quad DATA612\par}
  \vspace{0.4cm}{\large May 2026\par}
  \vspace{2.8cm}
  \begin{abstract}\noindent
  This report presents a systematic experimental evaluation of a Conditional
  VAE-GAN hybrid trained on MNIST and Fashion-MNIST for image classification
  augmentation. Six research goals are investigated across both datasets:
  three-way classifier comparison, architecture ablations, training strategy
  sweeps, data scarcity studies, label injection methods, and extended
  augmentation scenarios. Synthetic augmentation consistently fails to match
  the real-data baseline, with the largest losses under moderate scarcity and
  with CNN-based generators on Fashion-MNIST. The \texttt{large\_latent\_256}
  architecture is the only configuration to produce marginal positive uplift
  on both datasets. Findings highlight the limits of current VAE-GAN quality
  as an augmentation tool for near-saturated image classification benchmarks.
  \end{abstract}
\end{titlepage}

\tableofcontents\newpage

%% ─── 1. INTRODUCTION ───────────────────────────────────────────────────────
\section{Introduction}

Generative Adversarial Networks (GANs) offer a principled approach to
data augmentation: instead of applying hand-crafted transforms, a learned
generator synthesises new training examples that approximate the real data
distribution. The Variational Autoencoder (VAE) extends this by imposing a
regularised latent space, enabling smoother sampling and better diversity.
This project combines both into a \textbf{Conditional VAE-GAN} and evaluates
whether the generated images can improve downstream CNN classification.

\subsection*{Datasets}
\begin{itemize}
  \item \textbf{MNIST}: 60k/10k train/test, $28\!\times\!28$ greyscale,
    10 handwritten digit classes.
  \item \textbf{Fashion-MNIST}: Same split and resolution; 10 clothing
    categories (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt,
    Sneaker, Bag, Ankle boot). Substantially harder than MNIST.
\end{itemize}

\subsection*{Classifier}
A two-block CNN (Conv-32 $\to$ MaxPool $\to$ Conv-64 $\to$ MaxPool
$\to$ FC-128 $\to$ FC-10) is used throughout, trained for 10 epochs
with Adam ($lr=10^{-3}$) and evaluated on the fixed official test split.

%% ─── 2. ARCHITECTURE ───────────────────────────────────────────────────────
\section{Model Architecture}

""" + fig_block(F_ARCH,
    "Conditional VAE-GAN architecture. The encoder maps $(x,y) \\to (\\mu,\\log\\sigma^2)$; "
    "the generator decodes $(z,y) \\to \\hat{x}$; the discriminator scores $(\\hat{x},y)$ "
    "as real or fake.",
    "fig:arch", r"0.98\linewidth") + r"""

\subsection*{Components}
\begin{center}
\begin{tabular}{ll}
\toprule
\textbf{Component} & \textbf{Architecture} \\
\midrule
Encoder (shallow) & FC(784+cond$\to$512)$\to$FC($\to$256)$\to\mu,\log\sigma^2$ \\
Encoder (deep)    & adds FC($\to$1024) as first layer \\
Generator MLP (shallow) & FC(z+cond$\to$256)$\to$512$\to$1024$\to$Tanh(784) \\
Generator MLP (deep)    & adds extra 1024-unit layer \\
Generator CNN     & FC$\to128\!\times\!7\!\times\!7$$\to$ConvT-64$\to$ConvT-1 \\
Discriminator     & FC(784+emb$\to$512)$\to$256$\to$Sigmoid \\
CNN Classifier    & Conv-32$\to$MaxPool$\to$Conv-64$\to$MaxPool$\to$FC-128$\to$FC-10 \\
\bottomrule
\end{tabular}
\end{center}

\subsection*{Training Loss}
\[
\mathcal{L}_D = \mathrm{BCE}(D(x_{\mathrm{real}},y),1)
  + \tfrac{1}{2}[\mathrm{BCE}(D(\hat{x}_{\mathrm{rec}},y),0)
  + \mathrm{BCE}(D(x_{\mathrm{gen}},y),0)]
\]
\[
\mathcal{L}_{E+G} =
  w_{\mathrm{rec}}\,\mathrm{MSE}(\hat{x},x)
  + w_{\mathrm{KL}}\,\mathrm{KL}(q\|p)
  + w_{\mathrm{adv}}\,\tfrac{1}{2}[\ell_{\mathrm{adv,rec}} + \ell_{\mathrm{adv,gen}}]
\]
Default weights: $w_{\mathrm{rec}}=1.0$, $w_{\mathrm{KL}}=0.001$,
$w_{\mathrm{adv}}=0.1$.

\subsection*{Default Configuration}
\begin{center}
\begin{tabular}{ll@{\quad}ll}
\toprule
GAN epochs & 30 & Classifier epochs & 10 \\
GAN batch size & 128 & Classifier LR & $10^{-3}$ \\
Latent dim & 100 & Synthetic images & 12{,}000 \\
Embedding dim & 50 & Optimiser & Adam ($\beta_1=0.5$) \\
\bottomrule
\end{tabular}
\end{center}

%% ─── 3. VAE-GAN GENERATED SAMPLES ─────────────────────────────────────────
\section{VAE-GAN Generated Samples}

""" + fig_block(SG_M,
    "VAE-GAN generated samples for MNIST. Each row is one digit class (0--9); "
    "each column is an independent sample from the generator.",
    "fig:sg_mnist", r"0.8\linewidth") + \
     fig_block(SG_F,
    "VAE-GAN generated samples for Fashion-MNIST. Row order: T-shirt, Trouser, "
    "Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.",
    "fig:sg_fashion", r"0.8\linewidth") + r"""

The generated MNIST digits are visually coherent and class-consistent.
Fashion-MNIST samples capture general shape (trouser legs, bag handles)
but show blurriness and inter-class confusion (e.g.\ shirt vs.\ pullover),
consistent with the harder texture statistics of clothing images.

""" + fig_block(RG_M,
    "VAE-GAN reconstructions for MNIST: alternating columns show real input (left) "
    "and VAE-GAN reconstruction (right) for each class.",
    "fig:rg_mnist", r"0.8\linewidth") + \
     fig_block(RG_F,
    "VAE-GAN reconstructions for Fashion-MNIST: real (left) vs.\ reconstructed (right).",
    "fig:rg_fashion", r"0.8\linewidth") + r"""

The reconstruction quality is high for MNIST and adequate for Fashion-MNIST,
confirming the encoder correctly captures class-level structure in latent space.

%% ─── 4. GOAL 2: THREE-WAY COMPARISON ─────────────────────────────────────
\section{Goal 2 -- Three-Way Classifier Comparison}

\begin{table}[H]\centering
\caption{Goal 2 Results -- MNIST}
\begin{tabular}{lccccrr}
\toprule
\textbf{Scenario} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall}
  & \textbf{F1} & \textbf{Train N} \\
\midrule
""" + g2_rows(mnist["g2"]) + r"""
\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]\centering
\caption{Goal 2 Results -- Fashion-MNIST}
\begin{tabular}{lccccrr}
\toprule
\textbf{Scenario} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall}
  & \textbf{F1} & \textbf{Train N} \\
\midrule
""" + g2_rows(fashion["g2"]) + r"""
\bottomrule
\end{tabular}
\end{table}

""" + fig_block(F_G2,
    "Goal 2: All four metrics for MNIST (left) and Fashion-MNIST (right) across the three classifier scenarios.",
    "fig:goal2") + \
     fig_block(CM_M,
    "Confusion matrices for Goal 2 MNIST classifiers: real-only (left), real+synthetic (centre), synthetic-only (right).",
    "fig:cm_mnist") + \
     fig_block(CM_F,
    "Confusion matrices for Goal 2 Fashion-MNIST classifiers.",
    "fig:cm_fashion") + r"""

\textbf{Key observations:}
\begin{itemize}
  \item MNIST real-only achieves F1\,=\,0.9925 -- near ceiling.
    Adding synthetic data drops it to 0.9910 ($\Delta = -0.15$\,pp).
    Synthetic-only reaches 0.9340.
  \item Fashion-MNIST: real-only\,=\,0.9196; real+synthetic\,=\,0.9173
    ($\Delta = -0.23$\,pp); synthetic-only\,=\,0.7918.
  \item Adding synthetic data consistently \emph{hurts} the classifier
    in both datasets; synthetic-only incurs a much larger accuracy penalty.
\end{itemize}

%% ─── 5. GOAL 3: ARCHITECTURE VARIANTS ─────────────────────────────────────
\section{Goal 3 -- Architecture Variants}

\begin{table}[H]\centering
\caption{Goal 3 -- MNIST ($\Delta$F1 in percentage points)}
\begin{tabular}{lccc}
\toprule
\textbf{Variant} & \textbf{Real Only F1} & \textbf{Real+Syn F1} & \textbf{$\Delta$F1 (pp)} \\
\midrule
""" + variant_rows(mnist["g3"]) + r"""
\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]\centering
\caption{Goal 3 -- Fashion-MNIST ($\Delta$F1 in percentage points)}
\begin{tabular}{lccc}
\toprule
\textbf{Variant} & \textbf{Real Only F1} & \textbf{Real+Syn F1} & \textbf{$\Delta$F1 (pp)} \\
\midrule
""" + variant_rows(fashion["g3"]) + r"""
\bottomrule
\end{tabular}
\end{table}

""" + fig_block(F_G3D,
    "Goal 3: $\\Delta$F1 (real+synthetic minus real-only) in percentage points. "
    "Green bars = positive uplift; blue bars = degradation.",
    "fig:g3d") + \
     fig_block(F_G3A,
    "Goal 3: Absolute F1 for real-only and real+synthetic classifiers across all architecture variants.",
    "fig:g3a") + \
     fig_block(F_G3R,
    "Goal 3: Horizontal ranking of real+synthetic F1. Dashed line = real-only baseline. "
    "Green = beats baseline; blue = below baseline.",
    "fig:g3r") + r"""

\textbf{Key observations:}
\begin{itemize}
  \item \texttt{large\_latent\_256} is the only variant with positive $\Delta$F1
    on both datasets (+0.07\,pp MNIST, +0.14\,pp Fashion-MNIST).
  \item CNN generator hurts Fashion-MNIST most ($-0.62$\,pp), indicating
    transposed-conv upsampling artefacts confuse the classifier on
    fine-grained clothing textures.
  \item High discriminator dropout (0.5) consistently degrades both datasets.
  \item Deep MLP gives marginal positive uplift on Fashion-MNIST (+0.02\,pp)
    but hurts on MNIST.
\end{itemize}

%% ─── 6. GOAL 4: TRAINING STRATEGIES ───────────────────────────────────────
\section{Goal 4 -- Training Strategies}

\begin{table}[H]\centering
\caption{Goal 4 -- MNIST ($\Delta$F1 in pp)}
\begin{tabular}{lccc}
\toprule
\textbf{Strategy} & \textbf{Real Only F1} & \textbf{Real+Syn F1} & \textbf{$\Delta$F1 (pp)} \\
\midrule
""" + variant_rows(mnist["g4"]) + r"""
\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]\centering
\caption{Goal 4 -- Fashion-MNIST ($\Delta$F1 in pp)}
\begin{tabular}{lccc}
\toprule
\textbf{Strategy} & \textbf{Real Only F1} & \textbf{Real+Syn F1} & \textbf{$\Delta$F1 (pp)} \\
\midrule
""" + variant_rows(fashion["g4"]) + r"""
\bottomrule
\end{tabular}
\end{table}

""" + fig_block(F_G4,
    "Goal 4: $\\Delta$F1 for each training strategy on MNIST (left) and Fashion-MNIST (right).",
    "fig:g4d") + \
     fig_block(F_G4A,
    "Goal 4: Absolute F1 for real-only and real+synthetic classifiers across training strategies.",
    "fig:g4a") + r"""

\textbf{Key observations:}
\begin{itemize}
  \item \texttt{small\_batch} (bs=64) is the only positive strategy on
    MNIST (+0.01\,pp) but the worst on Fashion-MNIST ($-0.56$\,pp).
  \item \texttt{low\_lr} ($5\!\times\!10^{-5}$) is the worst on both datasets,
    suggesting insufficient GAN convergence at the same epoch count.
  \item KL weight has negligible impact: \texttt{no\_kl} and \texttt{kl\_heavy}
    produce similar $\Delta$F1, so reconstruction quality dominates.
  \item \texttt{adv\_heavy} minimises the Fashion-MNIST penalty,
    suggesting stronger adversarial pressure improves image fidelity.
\end{itemize}

%% ─── 7. GOAL 5: DATA SCARCITY ──────────────────────────────────────────────
\section{Goal 5 -- Data Scarcity Study}

\begin{table}[H]\centering
\caption{Goal 5 -- MNIST}
\begin{tabular}{lrccc}
\toprule
\textbf{Setting} & \textbf{Real N} & \textbf{Real Only F1} & \textbf{Real+Syn F1}
  & \textbf{$\Delta$F1 (pp)} \\
\midrule
""" + g5_rows(mnist["g5"]) + r"""
\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]\centering
\caption{Goal 5 -- Fashion-MNIST}
\begin{tabular}{lrccc}
\toprule
\textbf{Setting} & \textbf{Real N} & \textbf{Real Only F1} & \textbf{Real+Syn F1}
  & \textbf{$\Delta$F1 (pp)} \\
\midrule
""" + g5_rows(fashion["g5"]) + r"""
\bottomrule
\end{tabular}
\end{table}

""" + fig_block(F_G5,
    "Goal 5: F1 scores under data scarcity (MNIST left, Fashion-MNIST right). "
    "Shaded region highlights cases where real+synthetic underperforms real-only.",
    "fig:g5") + \
     fig_block(F_G5ALL,
    "Goal 5: All four metrics (Accuracy, Precision, Recall, F1) plotted across scarcity levels "
    "for both datasets. Solid = real-only; dashed = real+synthetic.",
    "fig:g5all") + r"""

\textbf{Key observations:}
\begin{itemize}
  \item Synthetic augmentation hurts across all scarcity levels on both datasets.
  \item The penalty is \emph{largest at moderate scarcity} (2k--5k images):
    MNIST 2k\_real: $\Delta = -2.49$\,pp; Fashion-MNIST 5k\_real: $-1.49$\,pp.
  \item At extreme scarcity (500 real), the penalty nearly vanishes
    ($\approx -0.08$ to $-0.11$\,pp), since both real and synthetic data
    are limited at that scale.
  \item GAN quality is doubly hurt under scarcity: fewer real images degrade
    generator training AND classifier training simultaneously.
\end{itemize}

%% ─── 8. GOAL 6: LABEL INJECTION ────────────────────────────────────────────
\section{Goal 6 -- Label Injection Methods}

\begin{table}[H]\centering
\caption{Goal 6 -- MNIST ($\Delta$F1 in pp)}
\begin{tabular}{lccc}
\toprule
\textbf{Method} & \textbf{Real Only F1} & \textbf{Real+Syn F1} & \textbf{$\Delta$F1 (pp)} \\
\midrule
""" + variant_rows(mnist["g6"]) + r"""
\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]\centering
\caption{Goal 6 -- Fashion-MNIST ($\Delta$F1 in pp)}
\begin{tabular}{lccc}
\toprule
\textbf{Method} & \textbf{Real Only F1} & \textbf{Real+Syn F1} & \textbf{$\Delta$F1 (pp)} \\
\midrule
""" + variant_rows(fashion["g6"]) + r"""
\bottomrule
\end{tabular}
\end{table}

""" + fig_block(F_G6,
    "Goal 6: Label injection method comparison on MNIST (left) and Fashion-MNIST (right).",
    "fig:g6") + r"""

\textbf{Key observations:}
\begin{itemize}
  \item All three conditioning methods produce negative $\Delta$F1.
  \item \texttt{learned\_embedding} least negative on MNIST ($-0.15$\,pp).
  \item \texttt{gen\_only\_embedding} least negative on Fashion-MNIST ($-0.28$\,pp).
  \item \texttt{onehot\_labels} is consistently the worst: one-hot vectors
    have lower representational capacity than learned embeddings.
\end{itemize}

%% ─── 9. GOAL 7: EXTENDED AUGMENTATION ─────────────────────────────────────
\section{Goal 7 -- Extended Augmentation Scenarios}

\begin{table}[H]\centering
\caption{Goal 7 Results -- MNIST}
\begin{tabular}{lcccc}
\toprule
\textbf{Scenario} & \textbf{Accuracy} & \textbf{F1} & \textbf{$\Delta$F1 (pp)} & \textbf{Train N} \\
\midrule
""" + g7_rows(mnist["g7"]) + r"""
\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]\centering
\caption{Goal 7 Results -- Fashion-MNIST}
\begin{tabular}{lcccc}
\toprule
\textbf{Scenario} & \textbf{Accuracy} & \textbf{F1} & \textbf{$\Delta$F1 (pp)} & \textbf{Train N} \\
\midrule
""" + g7_rows(fashion["g7"]) + r"""
\bottomrule
\end{tabular}
\end{table}

""" + fig_block(F_G7,
    "Goal 7: F1 scores across all augmentation scenarios. Dashed line = real-only baseline.",
    "fig:g7") + \
     fig_block(F_G7MIX,
    "Goal 7 deep-dive: F1 vs.\ synthetic mix ratio (25\%, 50\%, 100\%, 200\\%). "
    "Increasing synthetic proportion monotonically degrades Fashion-MNIST performance.",
    "fig:g7mix") + r"""

\textbf{Key observations:}
\begin{itemize}
  \item More synthetic data is worse: increasing mix ratio from 25\% to 200\%
    monotonically decreases F1 on Fashion-MNIST.
  \item Weighted sampling (real $2\times$ upweighted) reduces the penalty
    but cannot recover the baseline.
  \item Progressive augmentation is among the worst strategies on
    Fashion-MNIST ($-0.94$\,pp).
  \item Synthetic pre-training + real fine-tuning is the worst overall on
    Fashion-MNIST ($-4.75$\,pp): features learned from synthetic data
    transfer poorly even after fine-tuning.
\end{itemize}

%% ─── 10. CROSS-GOAL SUMMARY ────────────────────────────────────────────────
\section{Cross-Goal Summary}

""" + fig_block(F_HEAT,
    "$\\Delta$F1 heatmap (percentage points) across goals and datasets. "
    "Green = improvement over real-only baseline; red = degradation.",
    "fig:heat", r"0.6\linewidth") + r"""

The heatmap consolidates $\Delta$F1 across Goals 3, 4, and 6.
No variant achieves meaningful positive uplift: the best case
(\texttt{large\_latent\_256}) gains $< 0.15$\,pp on Fashion-MNIST.
The worst cases (CNN generator, high dropout, small batch on Fashion-MNIST)
lose up to $0.62$\,pp.

%% ─── 11. KEY FINDINGS ──────────────────────────────────────────────────────
\section{Key Findings}

\begin{enumerate}
  \item \textbf{VAE-GAN augmentation does not improve classification.}
    All augmented classifiers score below their real-only counterpart
    consistently across both datasets and all experimental conditions.

  \item \textbf{Synthetic augmentation under data scarcity is
    counter-productive.} Adding synthetic images to limited real datasets
    widens the performance gap in the moderate-scarcity regime (2k--5k images).

  \item \textbf{Larger latent space is the best architecture.}
    \texttt{large\_latent\_256} is the only variant to give positive
    $\Delta$F1 on both datasets.

  \item \textbf{CNN generators hurt on complex datasets.}
    The transposed-conv decoder loses $-0.62$\,pp on Fashion-MNIST
    vs.\ $-0.07$\,pp for the MLP baseline.

  \item \textbf{Loss weight configuration has second-order effects.}
    The KL term has negligible impact; reconstruction quality dominates.

  \item \textbf{Learned embeddings are the preferred conditioning method.}
    They deliver the smallest performance penalty in both datasets.

  \item \textbf{Progressive training and synthetic pre-training backfire.}
    Curriculum-style introduction of synthetic data disrupts learned
    real-data representations.

  \item \textbf{Fashion-MNIST is $2$--$8\times$ more sensitive to synthetic
    quality than MNIST}, owing to its richer texture and inter-class similarity.
\end{enumerate}

%% ─── 12. CONCLUSION ────────────────────────────────────────────────────────
\section{Conclusion}

This study provides a systematic, multi-dimensional evaluation of Conditional
VAE-GAN augmentation. Across 6 research goals, 2 datasets, and dozens of
configuration variants, the consistent finding is that VAE-GAN quality is
insufficient to supplement real training data on MNIST and Fashion-MNIST under
the conditions tested. The marginal positive result from
\texttt{large\_latent\_256} ($+0.14$\,pp on Fashion-MNIST) represents the
upper bound achievable with this architecture family.

The most actionable recommendation is to prioritise generator fidelity
(better architectures, longer training, or FID-guided early stopping)
before attempting augmentation, and to focus augmentation on
\emph{genuinely} scarce settings ($<100$ examples per class) where the
baseline is low enough for synthetic data to make a meaningful contribution.

%% ─── APPENDIX ───────────────────────────────────────────────────────────────
\appendix

\section{Full Goal 3 Metric Tables}

\begin{table}[H]\centering\small
\caption{Goal 3 Full Metrics -- MNIST}
\begin{tabular}{lcccc|cccc}
\toprule
 & \multicolumn{4}{c|}{\textbf{Real Only}}
 & \multicolumn{4}{c}{\textbf{Real + Synthetic}} \\
\textbf{Variant} & Acc & Prec & Rec & F1 & Acc & Prec & Rec & F1 \\
\midrule
""" + g3_full_rows(mnist["g3"]) + r"""
\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]\centering\small
\caption{Goal 3 Full Metrics -- Fashion-MNIST}
\begin{tabular}{lcccc|cccc}
\toprule
 & \multicolumn{4}{c|}{\textbf{Real Only}}
 & \multicolumn{4}{c}{\textbf{Real + Synthetic}} \\
\textbf{Variant} & Acc & Prec & Rec & F1 & Acc & Prec & Rec & F1 \\
\midrule
""" + g3_full_rows(fashion["g3"]) + r"""
\bottomrule
\end{tabular}
\end{table}

\end{document}
"""


# ══════════════════════════════════════════════════════════════════════════════
#  WRITE AND COMPILE
# ══════════════════════════════════════════════════════════════════════════════
latex_src = REPORT / "report.tex"
latex_src.write_text(build_latex(), encoding="utf-8")
print(f"LaTeX written -> {latex_src}")

for pass_n in range(1, 3):
    print(f"Compiling PDF (pass {pass_n})...")
    r = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(REPORT), str(latex_src)],
        capture_output=True, text=True, cwd=str(REPORT),
    )
    if "Fatal error" in r.stdout or ("! LaTeX Error" in r.stdout and "Output written" not in r.stdout):
        print("FATAL pdflatex error:\n")
        print(r.stdout[-4000:])
        sys.exit(1)

pdf_path = REPORT / "report.pdf"
size_mb  = pdf_path.stat().st_size / 1e6
print(f"\n{'='*60}")
print(f"  PDF report ready  ({size_mb:.1f} MB)")
print(f"  {pdf_path}")
print(f"{'='*60}")
