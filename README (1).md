# GAN-Based Data Augmentation for Image Classification

This repository implements the project proposed in `Project Proposal(group8).pdf`:

- **Conditional GAN (cGAN)** [Mirza & Osindero, 2014]: generator and discriminator are conditioned on class labels (`src/models.py`: `ConditionalGenerator` / `ConditionalDiscriminator`; aliases `cGANGenerator` / `cGANDiscriminator`).
- **Synthetic data** for augmentation and **three classifier training regimes**: real-only, real + synthetic, synthetic-only.
- **Data scarcity**: stratified caps on how many *real* training images the classifier sees (`--max-real-train-samples` or `--train-fraction`), and optional caps on **cGAN** training data (`--max-train-samples` on `train_gan`, `--max-gan-train-samples` on `run_experiments`).
- **Datasets**: MNIST and **Fashion-MNIST** (`--dataset fashion_mnist`).
- **Metrics**: accuracy, precision, recall, F1 (weighted + **per-class** tables and plots), confusion matrices (fixed 10×10 labels).
- **Image quality**: **FID** (Fréchet Inception Distance) via `torch-fidelity` (optional; first run downloads Inception weights).

## Project structure

| Path | Role |
|------|------|
| `src/models.py` | cGAN + CNN classifier |
| `src/data.py` | Datasets, stratified subsetting, synthetic folder loader |
| `src/train_gan.py` | Train cGAN |
| `src/generate_synthetic.py` | Export synthetic RGB PNGs per class |
| `src/train_classifier.py` | Train one classifier scenario + per-class analysis |
| `src/evaluate.py` | Metrics, confusion matrix, per-class F1 plot |
| `src/fid_eval.py` | Export real reference PNGs + compute FID vs synthetic tree |
| `src/run_experiments.py` | End-to-end pipeline, comparison CSV/JSON/plots, optional FID |
| `outputs/` | Models, samples, experiment bundles (gitignored) |

## Setup

```bash
cd "path/to/DATA612 Project"
python -m venv .venv
# Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
```

## 1) Full experiment (recommended)

Trains the **cGAN**, generates synthetic images, runs **all three** classifiers, writes `comparison.csv`, `comparison.json`, `comparison_metrics.png`, and **`per_class_by_scenario.json`**. Add **`--compute-fid`** for FID (slower; GPU helps).

Under each mode below, the *Quick sanity check* command uses fewer epochs and smaller sample counts so you can confirm the pipeline end-to-end in minutes (metrics will not be meaningful).

**MNIST — full real training data**

```bash
python -m src.run_experiments --dataset mnist --gan-epochs 30 --classifier-epochs 10 --num-synthetic 12000
```

*Quick sanity check:* few GAN/classifier epochs and few synthetic images (expect weak metrics).

```bash
python -m src.run_experiments --dataset mnist --gan-epochs 2 --classifier-epochs 1 --num-synthetic 500 --quiet-classifiers
```

**Fashion-MNIST (harder, more “complex” appearance)**

```bash
python -m src.run_experiments --dataset fashion_mnist --gan-epochs 40 --classifier-epochs 15 --num-synthetic 12000
```

*Quick sanity check:*

```bash
python -m src.run_experiments --dataset fashion_mnist --gan-epochs 2 --classifier-epochs 1 --num-synthetic 500 --quiet-classifiers
```

**Simulate label scarcity** (stratified cap: e.g. 600 real train images ≈ 60/class)

```bash
python -m src.run_experiments --dataset mnist --max-real-train-samples 600 --gan-epochs 30 --classifier-epochs 15 --num-synthetic 12000
```

*Quick sanity check:*

```bash
python -m src.run_experiments --dataset mnist --max-real-train-samples 200 --gan-epochs 2 --classifier-epochs 1 --num-synthetic 500 --quiet-classifiers
```

**Scarcity via fraction** (e.g. 10% of the 60k MNIST train split)

```bash
python -m src.run_experiments --dataset mnist --train-fraction 0.1 --classifier-epochs 15 --num-synthetic 12000
```

*Quick sanity check:*

```bash
python -m src.run_experiments --dataset mnist --train-fraction 0.05 --gan-epochs 2 --classifier-epochs 1 --num-synthetic 500 --quiet-classifiers
```

**Restrict cGAN training data** (e.g. GAN sees only 5k real images; classifiers still follow `--max-real-train-samples` / `--train-fraction`)

```bash
python -m src.run_experiments --dataset mnist --max-gan-train-samples 5000 --max-real-train-samples 2000 --gan-epochs 30 --classifier-epochs 15
```

*Quick sanity check:*

```bash
python -m src.run_experiments --dataset mnist --max-gan-train-samples 800 --max-real-train-samples 300 --gan-epochs 2 --classifier-epochs 1 --num-synthetic 500 --quiet-classifiers
```

**FID + scarcity + Fashion-MNIST**

```bash
python -m src.run_experiments --dataset fashion_mnist --max-real-train-samples 3000 --compute-fid --fid-num-images 8000 --classifier-epochs 12
```

*Quick sanity check:* smaller FID sample count and fewer epochs (FID still does a full Inception pass over the exported sets).

```bash
python -m src.run_experiments --dataset fashion_mnist --max-real-train-samples 500 --compute-fid --fid-num-images 512 --fid-batch-size 32 --gan-epochs 2 --classifier-epochs 1 --num-synthetic 500 --quiet-classifiers
```

Reuse a trained generator (skip GAN training):

```bash
python -m src.run_experiments --dataset mnist --skip-gan --generator-path "outputs/experiments/.../gan/generator.pt" --classifier-epochs 10
```

*Quick sanity check:* replace the path with a real `generator.pt` on your machine; keeps a short classifier pass and small synthetic set.

```bash
python -m src.run_experiments --dataset mnist --skip-gan --generator-path "outputs/experiments/YOUR_RUN/gan/generator.pt" --classifier-epochs 1 --num-synthetic 500 --quiet-classifiers
```

Artifacts for each run live under `outputs/experiments/<dataset>_<timestamp>/`:

- `classifiers/<scenario>/` — `metrics.json`, **`per_class_metrics.json`**, **`per_class_f1.png`**, `confusion_matrix.png`, `classifier.pt`
- `comparison.json` — includes optional **`fid`** block when `--compute-fid` is set
- `per_class_by_scenario.json` — per-class precision/recall/F1/support per scenario

## 2) FID only (custom folders)

Compares stratified **real train** exports to your **synthetic** root (class subfolders with PNGs, as produced by `generate_synthetic`).

```bash
python -m src.fid_eval --dataset fashion_mnist --synthetic-root "outputs/experiments/my_run/synthetic" --num-images 10000
```

*Quick sanity check:*

```bash
python -m src.fid_eval --dataset mnist --synthetic-root "outputs/experiments/my_run/synthetic" --num-images 256 --batch-size 32
```

## 3) Step-by-step CLIs

**Train cGAN**

```bash
python -m src.train_gan --dataset mnist --epochs 30 --batch-size 128
# Optional scarcity for the GAN:
python -m src.train_gan --dataset mnist --epochs 30 --max-train-samples 10000
```

*Quick sanity check:*

```bash
python -m src.train_gan --dataset mnist --epochs 2 --batch-size 128
python -m src.train_gan --dataset mnist --epochs 2 --max-train-samples 2000 --batch-size 128
```

**Generate synthetic images**

```bash
python -m src.generate_synthetic --dataset mnist --generator-path "outputs/experiments/.../gan/generator.pt" --num-samples 12000
```

*Quick sanity check:*

```bash
python -m src.generate_synthetic --dataset mnist --generator-path "outputs/experiments/.../gan/generator.pt" --num-samples 200
```

**Train one classifier** (with scarcity + per-class outputs)

```bash
python -m src.train_classifier --dataset mnist --scenario real_plus_synthetic --synthetic-root "path/to/synthetic" --epochs 10 --max-real-train-samples 1000
# or:
python -m src.train_classifier --dataset fashion_mnist --scenario real_only --epochs 10 --train-fraction 0.2
```

*Quick sanity check:*

```bash
python -m src.train_classifier --dataset mnist --scenario real_plus_synthetic --synthetic-root "path/to/synthetic" --epochs 1 --max-real-train-samples 200
python -m src.train_classifier --dataset fashion_mnist --scenario real_only --epochs 1 --train-fraction 0.1
```

---

## VAE-GAN Hybrid (`VAE-GAN.py`)

A self-contained single-file pipeline that implements a **Conditional VAE-GAN** with a 7-goal experimental study. Everything — models, training, evaluation, and reporting — lives in `VAE-GAN.py`.

### Goals overview

| Goal | Description |
|------|-------------|
| 1 (implicit) | Train a Conditional VAE-GAN (Encoder + Generator + Discriminator) |
| 2 | Three-way classifier comparison: real-only \| real+synthetic \| synthetic-only |
| 3 | Architecture variants (latent dim, encoder depth, MLP vs CNN generator, discriminator dropout) |
| 4 | Training strategy ablation (loss weights, learning rate, batch size) |
| 5 | Data scarcity study (500 / 2k / 5k / full real images) |
| 6 | Label injection method comparison (learned embedding \| one-hot \| generator-only) |
| 7 | Extended augmentation scenarios (mix ratios, weighted sampling, progressive augmentation, pretrain→finetune) |

Goal 1 is always run implicitly as part of whichever goals you select.

### Quick start

```bash
# 2-minute sanity check (2 GAN epochs, 1 classifier epoch, 200 synthetic images)
python VAE-GAN.py --dataset mnist --quick

# Base 3-way comparison only (Goal 2)
python VAE-GAN.py --dataset mnist --goals 2

# Run all 7 goals
python VAE-GAN.py --dataset mnist --goals all

# Specific goals with custom epochs
python VAE-GAN.py --dataset mnist --goals 2,3,7 --gan-epochs 30 --clf-epochs 10

# Fashion-MNIST (harder dataset)
python VAE-GAN.py --dataset fashion_mnist --goals all --gan-epochs 40 --clf-epochs 15
```

### Per-goal commands

**Goal 2 — Three-Way Comparison**

```bash
python VAE-GAN.py --dataset mnist --goals 2 --gan-epochs 30 --num-synthetic 12000 --clf-epochs 10
```

*Quick sanity check:*

```bash
python VAE-GAN.py --dataset mnist --goals 2 --quick
```

**Goal 3 — Architecture Variants** (7 configs by default; use `--subset N` to limit)

```bash
python VAE-GAN.py --dataset mnist --goals 3 --gan-epochs 30 --num-synthetic 5000
# Run only first 2 configs:
python VAE-GAN.py --dataset mnist --goals 3 --subset 2 --quick
```

**Goal 4 — Training Strategies** (9 configs by default)

```bash
python VAE-GAN.py --dataset mnist --goals 4 --gan-epochs 30
# Quick subset:
python VAE-GAN.py --dataset mnist --goals 4 --subset 3 --quick
```

**Goal 5 — Data Scarcity**

```bash
python VAE-GAN.py --dataset mnist --goals 5 --gan-epochs 30 --clf-epochs 10
# With explicit real-data cap:
python VAE-GAN.py --dataset mnist --goals 2 --max-real-samples 2000 --gan-epochs 30
```

*Quick sanity check:*

```bash
python VAE-GAN.py --dataset mnist --goals 5 --subset 2 --quick
```

**Goal 6 — Label Injection Methods**

```bash
python VAE-GAN.py --dataset mnist --goals 6 --gan-epochs 30 --num-synthetic 5000
```

**Goal 7 — Extended Augmentation Scenarios**

```bash
python VAE-GAN.py --dataset mnist --goals 7 --gan-epochs 30 --clf-epochs 10
```

**FID evaluation** (requires `pip install torch-fidelity`)

```bash
python VAE-GAN.py --dataset mnist --goals 2 --compute-fid --fid-num-images 5000
# Quick FID check:
python VAE-GAN.py --dataset mnist --goals 2 --compute-fid --fid-num-images 512 --fid-batch-size 32 --quick
```

### Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--goals` | `2` | Comma-separated goal IDs (2–7) or `all` |
| `--dataset` | `mnist` | `mnist` or `fashion_mnist` |
| `--quick` | off | Override: 2 GAN epochs, 1 classifier epoch, 200 synthetic images, subset=2 |
| `--subset` | 0 (all) | Run only the first N configs per goal (Goals 3–6) |
| `--gan-epochs` | 30 | Training epochs for the VAE-GAN |
| `--num-synthetic` | 12000 | Synthetic images to generate per run |
| `--clf-epochs` | 10 | Training epochs for each CNN classifier |
| `--clf-lr` | 1e-3 | Classifier learning rate |
| `--max-real-samples` | 0 (full) | Cap real training images for Goals 2 & 7 |
| `--compute-fid` | off | Compute FID score (Goal 2 only) |
| `--fid-num-images` | 5000 | Images used for FID calculation |
| `--verbose-clf` | off | Print per-epoch classifier loss |
| `--seed` | 42 | Global random seed |
| `--output-dir` | auto | Root output directory (auto-named if empty) |

### Output structure

Each run writes to `outputs/vaegan_<dataset>_<timestamp>/`:

```
outputs/vaegan_mnist_<timestamp>/
  goal2_three_cases/
    gan/                  ← generator.pt, encoder.pt, discriminator.pt, preview/
    synthetic/            ← class subfolders of generated PNGs
    visuals/              ← sample_grid.png, reconstruction_grid.png
    clf_real_only/        ← metrics.json, confusion_matrix.png, model.pt
    clf_real_plus_syn/
    clf_synthetic_only/
    three_cases.csv
    three_cases.png
    results.json
    fid.json              ← (only if --compute-fid)
  goal3_architecture/     ← per-variant subdirs + architecture_variants.csv/png
  goal4_training_strategy/
  goal5_data_scarcity/
  goal6_label_injection/
  goal7_augmentation_scenarios/
```

---

## Notes

- **cGAN** is the default generative model throughout; there is no unconditional GAN in this codebase.
- **Stratified scarcity** keeps classes as balanced as possible when subsampling real training data.
- **FID** uses Inception-v3 features (`torch-fidelity`). It is most informative when you have enough images and a GPU; MNIST/Fashion-MNIST are out-of-domain for ImageNet features, so treat FID as a **relative** comparison across runs rather than an absolute quality score.
- Classifiers always evaluate on the **official test split** for fair comparison across scenarios.
