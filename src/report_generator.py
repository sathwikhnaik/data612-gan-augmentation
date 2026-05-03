"""
HTML report generator — reads all pipeline outputs and produces a
single self-contained index.html that can be presented to a professor.
"""
import base64
import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Image helpers ──────────────────────────────────────────────────────────

def _img_b64(path: str) -> str:
    """Return base64-encoded PNG for embedding in HTML."""
    if not path or not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _img_tag(path: str, caption: str = "", width: str = "100%") -> str:
    b64 = _img_b64(path)
    if not b64:
        return f'<div class="alert alert-secondary">Image not found: {os.path.basename(path)}</div>'
    cap_html = f'<figcaption class="text-muted small mt-1">{caption}</figcaption>' if caption else ""
    return (
        f'<figure class="text-center my-3">'
        f'<img src="data:image/png;base64,{b64}" style="max-width:{width};border-radius:6px;'
        f'box-shadow:0 2px 8px rgba(0,0,0,0.12);" class="img-fluid">'
        f'{cap_html}'
        f"</figure>"
    )


# ── Metric table helpers ───────────────────────────────────────────────────

def _metric_table(rows: list, highlight: bool = True) -> str:
    """
    rows: list of dicts with keys 'scenario', 'accuracy', 'f1_score', 'precision', 'recall'
    Green = best per column, orange = worst.
    """
    keys = ["accuracy", "f1_score", "precision", "recall"]
    if not rows:
        return "<p class='text-muted'>No data</p>"

    best = {k: max(r.get(k, 0) for r in rows) for k in keys}
    worst = {k: min(r.get(k, 0) for r in rows) for k in keys}

    header = "<thead class='table-dark'><tr><th>Scenario</th>"
    header += "".join(f"<th>{k.replace('_',' ').title()}</th>" for k in keys)
    header += "</tr></thead>"

    body = "<tbody>"
    for r in rows:
        body += "<tr>"
        body += f"<td><code>{r.get('scenario','')}</code></td>"
        for k in keys:
            val = r.get(k, 0)
            style = ""
            if highlight:
                if abs(val - best[k]) < 1e-6:
                    style = " class='table-success fw-bold'"
                elif abs(val - worst[k]) < 1e-6 and best[k] != worst[k]:
                    style = " class='table-warning'"
            body += f"<td{style}>{val:.4f}</td>"
        body += "</tr>"
    body += "</tbody>"

    return f'<div class="table-responsive"><table class="table table-bordered table-hover table-sm">{header}{body}</table></div>'


def _per_class_table(per_class: list) -> str:
    if not per_class:
        return "<p class='text-muted'>No per-class data.</p>"
    header = "<thead class='table-dark'><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>"
    body = "<tbody>"
    for r in per_class:
        body += f"<tr><td>{r.get('class_name','')}</td><td>{r.get('precision',0):.3f}</td>"
        body += f"<td>{r.get('recall',0):.3f}</td><td>{r.get('f1_score',0):.3f}</td>"
        body += f"<td>{r.get('support',0)}</td></tr>"
    body += "</tbody>"
    return f'<div class="table-responsive"><table class="table table-sm table-bordered">{header}{body}</table></div>'


# ── Section builders ───────────────────────────────────────────────────────

def _section(title: str, content: str, level: int = 2) -> str:
    hn = f"h{level}"
    badge = f'<span class="badge bg-secondary ms-2 align-middle" style="font-size:0.6em">§</span>'
    return (
        f'<section class="mb-5">'
        f'<{hn} class="border-bottom pb-2 mb-3">{title}{badge}</{hn}>'
        f'{content}'
        f'</section>'
    )


def _card(title: str, body: str, color: str = "primary") -> str:
    return (
        f'<div class="card mb-4 border-{color}">'
        f'<div class="card-header bg-{color} text-white"><strong>{title}</strong></div>'
        f'<div class="card-body">{body}</div>'
        f'</div>'
    )


def _col2(left: str, right: str) -> str:
    return (
        '<div class="row g-3">'
        f'<div class="col-md-6">{left}</div>'
        f'<div class="col-md-6">{right}</div>'
        '</div>'
    )


def _col3(a: str, b: str, c: str) -> str:
    return (
        '<div class="row g-3">'
        f'<div class="col-md-4">{a}</div>'
        f'<div class="col-md-4">{b}</div>'
        f'<div class="col-md-4">{c}</div>'
        '</div>'
    )


# ── Load helpers ───────────────────────────────────────────────────────────

def _load_json(path: str) -> dict:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _find(root: str, filename: str) -> str:
    """Find first occurrence of filename under root."""
    for dirpath, _, files in os.walk(root):
        if filename in files:
            return os.path.join(dirpath, filename)
    return ""


# ── Per-section HTML builders ──────────────────────────────────────────────

def _build_exec_summary(state: dict, cfg: dict) -> str:
    """Key findings box at the top of the report."""
    findings = []

    # Check if main comparison data exists
    for dataset, key in [("MNIST", "stage_mnist_main"), ("Fashion-MNIST", "stage_fmnist_main")]:
        d = state.get(f"{key}_dir", "")
        comp = _load_json(os.path.join(d, "comparison.json")) if d else {}
        if comp and comp.get("runs"):
            runs = {r["scenario"]: r["metrics"] for r in comp["runs"]}
            if "real_only" in runs and "real_plus_synthetic" in runs:
                gain = (runs["real_plus_synthetic"]["accuracy"] - runs["real_only"]["accuracy"]) * 100
                sign = "+" if gain >= 0 else ""
                findings.append(
                    f"<li>On <strong>{dataset}</strong>, GAN augmentation (<code>real_plus_synthetic</code>) "
                    f"achieved accuracy <strong>{runs['real_plus_synthetic']['accuracy']:.4f}</strong> "
                    f"vs. real-only <strong>{runs['real_only']['accuracy']:.4f}</strong> "
                    f"(<em>{sign}{gain:.2f} pp delta</em>).</li>"
                )

    if not findings:
        findings = ["<li>Run the pipeline to populate findings.</li>"]

    body = (
        '<div class="alert alert-info mb-3">'
        '<h5 class="alert-heading">Key Findings</h5>'
        f'<ul class="mb-0">{"".join(findings)}</ul>'
        '</div>'
        '<p class="text-muted small">All classifiers are evaluated on the official test split '
        '(10,000 images for both MNIST and Fashion-MNIST) to ensure fair comparison across scenarios.</p>'
    )
    return body


def _build_methodology() -> str:
    return """
<div class="row g-4">
  <div class="col-md-6">
    <div class="card h-100">
      <div class="card-header bg-dark text-white">Datasets</div>
      <div class="card-body">
        <ul>
          <li><strong>MNIST</strong> — 60k train / 10k test, 28×28 grayscale, 10 digit classes</li>
          <li><strong>Fashion-MNIST</strong> — same size/structure, 10 clothing categories (harder)</li>
        </ul>
      </div>
    </div>
  </div>
  <div class="col-md-6">
    <div class="card h-100">
      <div class="card-header bg-dark text-white">GAN Architecture (cGAN)</div>
      <div class="card-body">
        <ul>
          <li><strong>Generator</strong>: (z ∈ ℝ¹⁰⁰, y) → MLP [256→512→1024] → 28×28 image</li>
          <li><strong>Discriminator</strong>: (x, y) → MLP [512→256→1] → real/fake score</li>
          <li>Class label y injected via learned embedding (dim=50)</li>
          <li>Optimizer: Adam (β₁=0.5, β₂=0.999, lr=2×10⁻⁴)</li>
        </ul>
      </div>
    </div>
  </div>
  <div class="col-md-6">
    <div class="card h-100">
      <div class="card-header bg-dark text-white">Classifier</div>
      <div class="card-body">
        <ul>
          <li>CNN: Conv(32) → MaxPool → Conv(64) → MaxPool → FC(128) → FC(10)</li>
          <li>Optimizer: Adam (lr=1×10⁻³), CrossEntropyLoss</li>
          <li>All scenarios use the <em>same</em> classifier architecture</li>
        </ul>
      </div>
    </div>
  </div>
  <div class="col-md-6">
    <div class="card h-100">
      <div class="card-header bg-dark text-white">Experimental Scenarios</div>
      <div class="card-body">
        <table class="table table-sm">
          <tr><td><code>real_only</code></td><td>Baseline — real images only</td></tr>
          <tr><td><code>augmented_real</code></td><td>Real + classical augmentation</td></tr>
          <tr><td><code>real_plus_synthetic</code></td><td>Real + GAN-generated images</td></tr>
          <tr><td><code>synthetic_only</code></td><td>GAN images only (no real)</td></tr>
          <tr><td><code>real_plus_synthetic_weighted</code></td><td>Real (2×weight) + synthetic</td></tr>
          <tr><td><code>synthetic_pretrain</code></td><td>Pre-train on synthetic, fine-tune on real</td></tr>
          <tr><td><code>progressive</code></td><td>Start real-only, add synthetic midway</td></tr>
        </table>
      </div>
    </div>
  </div>
</div>
"""


def _build_main_comparison(report_root: str, state: dict) -> str:
    html = ""
    for dataset, key, color in [
        ("MNIST", "stage_mnist_main", "primary"),
        ("Fashion-MNIST", "stage_fmnist_main", "danger"),
    ]:
        d = state.get(f"{key}_dir", "")
        comp = _load_json(os.path.join(d, "comparison.json")) if d else {}
        if not comp or not comp.get("runs"):
            html += f'<p class="text-muted">{dataset}: not yet available.</p>'
            continue

        runs = comp["runs"]
        rows = [{"scenario": r["scenario"], **r["metrics"]} for r in runs]
        table = _metric_table(rows)

        core_runs = [r for r in runs if r["scenario"] in
                     ["real_only", "augmented_real", "real_plus_synthetic", "synthetic_only"]]
        extra_rows = [{"scenario": r["scenario"], **r["metrics"]} for r in runs
                      if r["scenario"] not in
                      ["real_only", "augmented_real", "real_plus_synthetic", "synthetic_only"]]

        img_comp = _img_tag(os.path.join(d, "comparison_metrics.png"),
                             f"{dataset}: all scenarios comparison")
        img_interp = _img_tag(os.path.join(d, "gan", "interpolation_grid.png"),
                               f"{dataset}: GAN latent space interpolation (10 classes × 10 steps)")
        img_recog = _img_tag(os.path.join(d, "synthetic_quality", "synthetic_recognizability.png"),
                              f"{dataset}: per-class GAN recognizability (higher = better quality)")
        img_loss = _img_tag(os.path.join(d, "gan", "loss_curves.png"),
                             f"{dataset}: GAN training loss curves")

        per_class_html = ""
        for scenario in ["real_only", "real_plus_synthetic"]:
            r = next((x for x in runs if x["scenario"] == scenario), None)
            if r:
                pc = r.get("per_class", [])
                img_pc = _img_tag(
                    _find(os.path.join(d, "classifiers", scenario), "per_class_f1.png"),
                    f"Per-class F1 — {scenario}"
                )
                per_class_html += f"<h6 class='mt-3'><code>{scenario}</code></h6>{img_pc}"

        inner = (
            f"<h4>{dataset}</h4>"
            + table
            + _col2(img_comp, img_loss)
            + "<h5 class='mt-4'>GAN Quality</h5>"
            + _col2(img_interp, img_recog)
            + "<h5 class='mt-4'>Per-class F1</h5>"
            + per_class_html
        )
        if extra_rows:
            inner += "<h5 class='mt-4'>Extended Scenarios</h5>" + _metric_table(extra_rows)

        html += _card(dataset, inner, color) + "<hr>"

    return html


def _build_arch_ablation(report_root: str, state: dict) -> str:
    d = state.get("arch_ablation_dir", os.path.join(report_root, "2_arch_ablation"))
    results_path = os.path.join(d, "arch_sweep_results.json")
    if not os.path.exists(results_path):
        return '<p class="text-muted">Architecture ablation not yet run.</p>'

    with open(results_path) as f:
        results = json.load(f)

    by_factor: dict = {}
    for r in results:
        by_factor.setdefault(r["factor"], []).append(r)

    html = ""
    for factor, entries in by_factor.items():
        rows = [{"scenario": e["value"],
                 "accuracy": e["metrics"]["accuracy"],
                 "f1_score": e["metrics"]["f1_score"],
                 "precision": e["metrics"]["precision"],
                 "recall": e["metrics"]["recall"]}
                for e in entries]
        table = _metric_table(rows, highlight=True)
        img = _img_tag(os.path.join(d, f"arch_ablation_{factor}.png"),
                        f"Ablation: {factor}")
        html += f"<h5 class='mt-4'><code>{factor}</code></h5>"
        html += _col2(img, table)

    return html or '<p class="text-muted">No ablation data found.</p>'


def _build_training_ablation(report_root: str, state: dict) -> str:
    d = state.get("training_ablation_dir", os.path.join(report_root, "3_training_ablation"))
    results_path = os.path.join(d, "training_sweep_results.json")
    if not os.path.exists(results_path):
        return '<p class="text-muted">Training strategy ablation not yet run.</p>'

    with open(results_path) as f:
        results = json.load(f)

    by_factor: dict = {}
    for r in results:
        by_factor.setdefault(r["factor"], []).append(r)

    html = ""
    for factor, entries in by_factor.items():
        rows = [{"scenario": e["label"],
                 "accuracy": e["metrics"]["accuracy"],
                 "f1_score": e["metrics"]["f1_score"],
                 "precision": e["metrics"]["precision"],
                 "recall": e["metrics"]["recall"]}
                for e in entries]
        table = _metric_table(rows, highlight=True)
        img = _img_tag(os.path.join(d, f"training_ablation_{factor}.png"),
                        f"Ablation: {factor}")
        html += f"<h5 class='mt-4'><code>{factor}</code></h5>"
        html += _col2(img, table)

    return html or '<p class="text-muted">No training ablation data found.</p>'


def _build_scarcity(report_root: str, state: dict) -> str:
    d = state.get("scarcity_dir", os.path.join(report_root, "4_scarcity"))
    results_path = os.path.join(d, "scarcity_results.json")
    if not os.path.exists(results_path):
        return '<p class="text-muted">Scarcity analysis not yet run.</p>'

    img_acc = _img_tag(os.path.join(d, "scarcity_accuracy.png"),
                        "Accuracy vs. real training samples (log scale)")
    img_f1 = _img_tag(os.path.join(d, "scarcity_f1.png"),
                       "F1 score vs. real training samples")
    img_gap = _img_tag(os.path.join(d, "gap_vs_baseline.png"),
                        "Accuracy gain/loss vs. real_only baseline (pp)")
    img_cross = _img_tag(os.path.join(d, "synthetic_crossover.png"),
                          "Where does synthetic_only cross real_only?")
    img_heat = _img_tag(os.path.join(d, "f1_gain_heatmap.png"),
                         "Per-class F1 gain of real+synthetic over real-only (green = GAN helps)")

    return (
        "<p>Sweeps the cap on real training images to answer: <em>when does GAN augmentation help most?</em></p>"
        + _col2(img_acc, img_f1)
        + _col2(img_gap, img_cross)
        + img_heat
    )


def _build_volume(report_root: str, state: dict) -> str:
    d = state.get("volume_dir", os.path.join(report_root, "5_volume_sweep"))
    img = _img_tag(os.path.join(d, "volume_sweep.png"),
                   "Accuracy and F1 vs. number of synthetic images added")
    results_path = os.path.join(d, "volume_sweep.json")
    table = ""
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        rows = [{"scenario": str(r["num_synthetic"]) + " synthetic",
                 "accuracy": r["metrics"]["accuracy"],
                 "f1_score": r["metrics"]["f1_score"],
                 "precision": r["metrics"]["precision"],
                 "recall": r["metrics"]["recall"]}
                for r in results]
        table = _metric_table(rows, highlight=True)

    if not img and not table:
        return '<p class="text-muted">Volume sweep not yet run.</p>'
    return (
        "<p>Trains <code>real_plus_synthetic</code> classifier with increasing synthetic dataset sizes "
        "to find the diminishing-returns knee.</p>" + _col2(img, table)
    )


def _build_cross_dataset(report_root: str, state: dict) -> str:
    d = state.get("cross_dataset_dir", os.path.join(report_root, "6_cross_dataset"))
    img_scen = _img_tag(os.path.join(d, "scenario_comparison.png"),
                         "All scenarios: MNIST vs Fashion-MNIST")
    img_delta = _img_tag(os.path.join(d, "delta_vs_baseline.png"),
                          "Accuracy delta vs. real_only baseline")
    img_recog = _img_tag(os.path.join(d, "recognizability_comparison.png"),
                          "Per-class GAN recognizability: MNIST vs Fashion-MNIST")

    if not any(os.path.exists(os.path.join(d, f))
               for f in ["scenario_comparison.png", "delta_vs_baseline.png"]):
        return '<p class="text-muted">Cross-dataset comparison not yet run.</p>'

    return (
        _col2(img_scen, img_delta)
        + img_recog
    )


def _build_conclusions(state: dict) -> str:
    bullets = []

    for dataset, key in [("MNIST", "stage_mnist_main"), ("Fashion-MNIST", "stage_fmnist_main")]:
        d = state.get(f"{key}_dir", "")
        comp = _load_json(os.path.join(d, "comparison.json")) if d else {}
        if comp and comp.get("runs"):
            runs = {r["scenario"]: r["metrics"] for r in comp["runs"]}
            if "real_only" in runs and "real_plus_synthetic" in runs:
                gain = (runs["real_plus_synthetic"]["accuracy"] - runs["real_only"]["accuracy"]) * 100
                bullets.append(
                    f"<strong>{dataset}</strong>: GAN augmentation yields "
                    f"{'improvement' if gain >= 0 else 'degradation'} of <strong>{gain:+.2f} pp</strong> "
                    f"vs. real-only baseline."
                )
            if "augmented_real" in runs and "real_plus_synthetic" in runs:
                diff = (runs["real_plus_synthetic"]["accuracy"] - runs["augmented_real"]["accuracy"]) * 100
                winner = "GAN augmentation" if diff > 0 else "classical augmentation"
                bullets.append(
                    f"<strong>{dataset}</strong>: {winner} outperforms the other by {abs(diff):.2f} pp."
                )

    if not bullets:
        bullets = ["Results will appear here after the pipeline completes."]

    return (
        "<ul class='mb-4'>"
        + "".join(f"<li class='mb-1'>{b}</li>" for b in bullets)
        + "</ul>"
        "<h5>Limitations</h5>"
        "<ul>"
        "<li>Both datasets are grayscale 28×28; findings may not generalize to natural images.</li>"
        "<li>FID uses Inception-v3 pretrained on ImageNet — treat as a <em>relative</em> metric only.</li>"
        "<li>MLP generator; a deeper convolutional generator (e.g., DCGAN) may improve quality.</li>"
        "<li>Single random seed for most ablation runs; error bars require multiple seeds.</li>"
        "</ul>"
    )


# ── Main entry point ───────────────────────────────────────────────────────

def generate_report(report_root: str, state: dict, cfg: dict):
    """Generate self-contained HTML report at report_root/index.html."""

    sections = [
        ("Executive Summary",      _build_exec_summary(state, cfg)),
        ("Methodology",            _build_methodology()),
        ("Main Scenario Comparison",
         _build_main_comparison(report_root, state)),
        ("Group 1 — GAN Architecture Ablation",
         _build_arch_ablation(report_root, state)),
        ("Group 2 — Training Strategy Ablation",
         _build_training_ablation(report_root, state)),
        ("Group 3 — Data Scarcity Analysis",
         _build_scarcity(report_root, state)),
        ("Group 3 — Synthetic Volume Sweep",
         _build_volume(report_root, state)),
        ("Cross-Dataset Comparison: MNIST vs Fashion-MNIST",
         _build_cross_dataset(report_root, state)),
        ("Conclusions & Limitations",
         _build_conclusions(state)),
    ]

    toc = "<ul class='list-group list-group-flush mb-4'>"
    body_sections = ""
    for i, (title, content) in enumerate(sections, 1):
        anchor = f"sec{i}"
        toc += f'<li class="list-group-item"><a href="#{anchor}">{i}. {title}</a></li>'
        body_sections += (
            f'<section id="{anchor}" class="mb-5">'
            f'<h2 class="border-bottom pb-2 mb-3">{i}. {title}</h2>'
            f'{content}'
            f'</section>'
        )
    toc += "</ul>"

    import torch
    device_str = "MPS (Apple GPU)" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "CPU"
    from datetime import datetime
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GAN-Based Data Augmentation — Research Report (Group 8)</title>
<link rel="stylesheet"
  href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css">
<style>
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background:#f8f9fa; }}
  .container-xl {{ max-width:1100px; }}
  .hero {{ background: linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);
           color:#fff; padding:3rem 2rem; border-radius:12px; margin-bottom:2rem; }}
  .hero h1 {{ font-size:1.9rem; font-weight:700; }}
  .hero .subtitle {{ opacity:.8; font-size:1.05rem; }}
  .sidebar {{ position:sticky; top:1rem; }}
  h2 {{ color:#0f3460; font-size:1.4rem; }}
  h5 {{ color:#333; }}
  code {{ color:#e83e8c; background:#f8f8f8; padding:1px 4px; border-radius:3px; }}
  .table-success td {{ background-color:#d4edda !important; font-weight:600; }}
  .table-warning td {{ background-color:#fff3cd !important; }}
  section {{ background:#fff; border-radius:10px; padding:2rem; box-shadow:0 1px 6px rgba(0,0,0,.07); }}
  @media print {{ .sidebar {{ display:none; }} body {{ background:#fff; }} section {{ box-shadow:none; }} }}
</style>
</head>
<body>
<div class="container-xl py-4">

  <!-- Hero -->
  <div class="hero">
    <h1>GAN-Based Data Augmentation for Image Classification</h1>
    <p class="subtitle">
      Group 8 &nbsp;·&nbsp; DATA612 &nbsp;·&nbsp;
      Generated: {generated_at} &nbsp;·&nbsp; Device: {device_str}
    </p>
    <p class="small mb-0 mt-2" style="opacity:.7">
      Datasets: MNIST &amp; Fashion-MNIST &nbsp;|&nbsp;
      Model: Conditional GAN (Mirza &amp; Osindero, 2014) &nbsp;|&nbsp;
      Metrics: Accuracy, F1, Precision, Recall (weighted) + per-class
    </p>
  </div>

  <div class="row g-4">
    <!-- Sidebar TOC -->
    <div class="col-lg-3 d-none d-lg-block">
      <div class="sidebar">
        <div class="card shadow-sm">
          <div class="card-header bg-dark text-white">Contents</div>
          {toc}
        </div>
      </div>
    </div>

    <!-- Main content -->
    <div class="col-lg-9">
      {body_sections}
    </div>
  </div>

  <footer class="text-center text-muted small py-4 mt-4">
    Generated by <strong>GAN Research Pipeline</strong> &mdash; Group 8
  </footer>
</div>
</body>
</html>"""

    out_path = os.path.join(report_root, "index.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report written to: {os.path.abspath(out_path)}")
    return out_path
