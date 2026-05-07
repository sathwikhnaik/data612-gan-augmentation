# GAN-Based Data Augmentation for Image Classification

This repository implements the project proposed in `Project Proposal(group8).pdf`:

- **Conditional GAN (cGAN)** [Mirza & Osindero, 2014]: generator and discriminator conditioned on class labels. Multiple architectures supported: MLP with learned embedding (default), MLP with one-hot conditioning, DCGAN (convolutional), and projection discriminator [Miyato & Koyama, 2018].
- **Seven classifier training scenarios**: real-only, augmented-real (classical), real + synthetic, real + synthetic weighted, synthetic-only, synthetic pretrain → fine-tune, and progressive mixing.
- **Data scarcity experiments**: stratified caps on real training images (`--max-real-train-samples` or `--train-fraction`), and optional caps on cGAN training data (`--max-gan-train-samples`).
- **Datasets**: MNIST and **Fashion-MNIST** (`--dataset fashion_mnist`).
- **Metrics**: accuracy, precision, recall, F1 (weighted + **per-class** tables and plots), confusion matrices.
- **Image quality**: **FID** (Fréchet Inception Distance) via `torch-fidelity` (optional), and per-class **GAN recognizability** (fraction of synthetic images correctly classified by a real-trained model).
- **Hyperparameter studies**: OFAT architecture ablation, training strategy ablation, scarcity sweep, synthetic volume sweep.
- **Research pipeline**: single command runs all experiments and generates a professor-ready HTML report.

---

## Project Structure

| Path | Role |
|------|------|
| `src/models.py` | cGAN architectures (MLP, DCGAN, one-hot, projection) + CNN classifier |
| `src/config.py` | Hyperparameter dataclasses and class name mappings |
| `src/data.py` | Datasets, stratified subsetting, all 7 scenario data loaders |
| `src/train_gan.py` | Train cGAN (BCE / hinge / WGAN-GP loss, configurable arch) |
| `src/generate_synthetic.py` | Export synthetic PNGs per class from a trained generator |
| `src/train_classifier.py` | Train one classifier scenario + per-class analysis |
| `src/evaluate.py` | Metrics, confusion matrix, per-class F1 plot |
| `src/fid_eval.py` | Export real reference PNGs + compute FID vs synthetic tree |
| `src/run_experiments.py` | End-to-end pipeline: GAN → synthetic → all classifiers → comparison |
| `src/interpolate.py` | Latent space slerp interpolation grid |
| `src/per_class_quality.py` | Per-class GAN recognizability evaluation |
| `src/arch_sweep.py` | OFAT architecture ablation sweep |
| `src/training_sweep.py` | OFAT training strategy ablation sweep |
| `src/scarcity_sweep.py` | Classifier accuracy vs real-data cap (multi-seed) |
| `src/scarcity_experiments.py` | Standalone scarcity experiment runner |
| `src/volume_sweep.py` | Accuracy vs number of synthetic images |
| `src/compare_datasets.py` | Cross-dataset MNIST vs Fashion-MNIST comparison plots |
| `src/research_pipeline.py` | Master 8-stage pipeline with resume support |
| `src/report_generator.py` | Auto-generates Bootstrap HTML report from all artifacts |
| `outputs/` | All generated artifacts (gitignored) |

---

## Setup

```bash
cd "path/to/GAN"
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows:     .\.venv\Scripts\activate
pip install -r requirements.txt
```

Device priority is **CUDA → MPS (Apple Silicon) → CPU**, selected automatically.

---

## Classifier Training Scenarios

| Scenario | Training data | Notes |
|---|---|---|
| `real_only` | Real images only | Baseline |
| `augmented_real` | Real + RandomCrop(pad=4) + RandomRotation(±15°) | Classical augmentation baseline |
| `real_plus_synthetic` | Real + GAN synthetic (concatenated) | Main GAN augmentation scenario |
| `real_plus_synthetic_weighted` | Same, real images sampled 2× more often | Uses `WeightedRandomSampler` |
| `synthetic_only` | GAN synthetic images only | Tests GAN distributional quality |
| `synthetic_pretrain` | Phase 1: synthetic → Phase 2: fine-tune on real | Two-phase training |
| `progressive` | Phase 1: real only → Phase 2: real + synthetic | Introduces synthetic gradually |

---

## 1) Full Research Pipeline (recommended)

Runs all 8 stages automatically and generates `outputs/research_report/index.html`.

```bash
python -m src.research_pipeline
```

Stages:
1. Main comparison — MNIST (all 7 scenarios)
2. Main comparison — Fashion-MNIST (all 7 scenarios)
3. Architecture ablation (OFAT over gen_type, conditioning, latent_dim, embedding_dim, MLP depth, dropout)
4. Training strategy ablation (OFAT over loss_type, lr_ratio, label_smoothing, gan_epochs)
5. Scarcity sweep on Fashion-MNIST (real-data caps: 100 → 12,000, 2 seeds)
6. Synthetic volume sweep on Fashion-MNIST (500 → 12,000 synthetic images)
7. Cross-dataset comparison plots
8. HTML report generation

**Options:**

```bash
python -m src.research_pipeline --dry-run          # print plan, run nothing
python -m src.research_pipeline --stages 1 2 7 8   # run specific stages only
python -m src.research_pipeline --no-resume        # re-run all stages from scratch
```

The pipeline is fully resumable — completed stages are skipped by default. State is saved to `outputs/research_report/pipeline_state.json`.

---

## 2) Single End-to-End Experiment

Trains the cGAN, generates synthetic images, runs all classifiers, and writes comparison CSV/JSON/plots.

**MNIST — full training data**

```bash
python -m src.run_experiments --dataset mnist --gan-epochs 30 --classifier-epochs 10 --num-synthetic 12000
```

*Quick sanity check (weak metrics, completes in minutes):*

```bash
python -m src.run_experiments --dataset mnist --gan-epochs 2 --classifier-epochs 1 --num-synthetic 500 --quiet-classifiers
```

**Fashion-MNIST**

```bash
python -m src.run_experiments --dataset fashion_mnist --gan-epochs 40 --classifier-epochs 15 --num-synthetic 12000
```

**Simulate label scarcity** (stratified cap: e.g. 600 real training images ≈ 60/class)

```bash
python -m src.run_experiments --dataset mnist --max-real-train-samples 600 --gan-epochs 30 --classifier-epochs 15 --num-synthetic 12000
```

**Scarcity via fraction** (e.g. 10% of 60k MNIST train split)

```bash
python -m src.run_experiments --dataset mnist --train-fraction 0.1 --classifier-epochs 15 --num-synthetic 12000
```

**Restrict cGAN training data** (GAN sees only 5k images; classifiers follow their own scarcity setting)

```bash
python -m src.run_experiments --dataset mnist --max-gan-train-samples 5000 --max-real-train-samples 2000 --gan-epochs 30 --classifier-epochs 15
```

**With FID evaluation**

```bash
python -m src.run_experiments --dataset fashion_mnist --max-real-train-samples 3000 --compute-fid --fid-num-images 8000 --classifier-epochs 12
```

**Reuse a trained generator (skip GAN training)**

```bash
python -m src.run_experiments --dataset mnist --skip-gan --generator-path "outputs/experiments/.../gan/generator.pt" --classifier-epochs 10
```

Artifacts for each run live under `outputs/experiments/<dataset>_<timestamp>/`:
- `classifiers/<scenario>/` — `metrics.json`, `per_class_metrics.json`, `per_class_f1.png`, `confusion_matrix.png`, `classifier.pt`
- `gan/` — `generator.pt`, `discriminator.pt`, `loss_curves.png`, `loss_history.json`, `interpolation_grid.png`
- `comparison.csv`, `comparison.json`, `comparison_metrics.png`
- `per_class_by_scenario.json`

---

## 3) Hyperparameter Studies

### Architecture Ablation

OFAT sweep over GAN architecture choices on Fashion-MNIST. For each configuration: train GAN → generate synthetic → train `real_plus_synthetic` classifier → record accuracy, F1, recognizability.

Factors swept (one at a time, others held at baseline):

| Factor | Values |
|---|---|
| `gen_type` | mlp, dcgan |
| `conditioning` | embedding, onehot, projection |
| `latent_dim` | 32, 64, **100**, 200 |
| `embedding_dim` | 10, 32, **50**, 100 |
| `hidden_dims` | [256,512], **[256,512,1024]**, [256,512,1024,1024] |
| `disc_dropout` | 0.0, 0.1, **0.3**, 0.5 |

**Bold** = baseline value.

```bash
python -m src.arch_sweep --dataset fashion_mnist --gan-epochs 15 --classifier-epochs 8 --num-synthetic 3000
```

*Quick sanity check:*

```bash
python -m src.arch_sweep --dataset mnist --gan-epochs 2 --classifier-epochs 1 --num-synthetic 500 --quiet
```

### Training Strategy Ablation

OFAT sweep over training hyperparameters, architecture held at MLP+embedding baseline.

| Factor | Variants |
|---|---|
| `loss_type` | **bce**, hinge, wgan_gp (auto n_critic=5) |
| `lr_ratio` | G=4e-4/D=2e-4, **G=D=2e-4**, G=2e-4/D=4e-4, G=1e-4/D=4e-4 |
| `label_smoothing` | **False**, True (real label = 0.9) |
| `gan_epochs` | base//3, base//2, **base**, base×2 |

```bash
python -m src.training_sweep --dataset fashion_mnist --base-epochs 30 --classifier-epochs 8 --num-synthetic 3000
```

*Quick sanity check:*

```bash
python -m src.training_sweep --dataset mnist --base-epochs 6 --classifier-epochs 1 --num-synthetic 500 --quiet
```

### Scarcity Sweep

Trains all 4 core scenarios at multiple real-data caps to answer: *does GAN augmentation help when real data is scarce?*

```bash
python -m src.scarcity_sweep --dataset fashion_mnist --synthetic-root "outputs/.../synthetic" --classifier-epochs 10
```

Default caps: 100, 300, 600, 1200, 3000, 6000, 12000 images. Runs 2 seeds by default for mean ± std error bands.

Outputs: scarcity curves, gap chart (real+synthetic vs real-only), synthetic-only crossover plot, F1 gain heatmap.

### Synthetic Volume Sweep

Fixes real training data at full size, varies the number of synthetic images.

```bash
python -m src.volume_sweep --generator-path "outputs/.../gan/generator.pt" --dataset fashion_mnist
```

Default values: 500, 1000, 3000, 6000, 12000. Plots accuracy and F1 vs synthetic count to find the diminishing-returns curve.

---

## 4) Individual Tools

### FID Evaluation

```bash
python -m src.fid_eval --dataset fashion_mnist --synthetic-root "outputs/experiments/my_run/synthetic" --num-images 10000
```

*Quick sanity check:*

```bash
python -m src.fid_eval --dataset mnist --synthetic-root "outputs/experiments/my_run/synthetic" --num-images 256 --batch-size 32
```

### Latent Space Interpolation

Generates a 10×steps grid: each row is one class, each column is a step along a spherical linear interpolation (slerp) between two random noise vectors.

```bash
python -m src.interpolate --generator-path "outputs/.../gan/generator.pt" --steps 10
```

### Per-Class GAN Quality

Runs a real-trained classifier over synthetic images and reports the fraction correctly identified per class (recognizability).

```bash
python -m src.per_class_quality --classifier-path "outputs/.../classifier.pt" --synthetic-root "outputs/.../synthetic" --dataset fashion_mnist
```

### Cross-Dataset Comparison

```bash
python -m src.compare_datasets --mnist-dir "outputs/.../mnist_run" --fmnist-dir "outputs/.../fashion_mnist_run"
```

Produces: `scenario_comparison.png`, `delta_vs_baseline.png`, `recognizability_comparison.png`.

### Step-by-Step CLIs

**Train cGAN**

```bash
# MLP baseline
python -m src.train_gan --dataset mnist --epochs 30 --batch-size 128

# DCGAN with WGAN-GP loss
python -m src.train_gan --dataset fashion_mnist --gen-type dcgan --loss-type wgan_gp --epochs 30

# MLP with hinge loss + label smoothing + slower discriminator
python -m src.train_gan --dataset mnist --loss-type hinge --label-smoothing --generator-lr 4e-4 --discriminator-lr 2e-4 --epochs 30

# Restrict GAN training data
python -m src.train_gan --dataset mnist --epochs 30 --max-train-samples 10000
```

**Generate synthetic images**

```bash
python -m src.generate_synthetic --dataset mnist --generator-path "outputs/.../gan/generator.pt" --num-samples 12000
```

**Train one classifier**

```bash
python -m src.train_classifier --dataset mnist --scenario real_plus_synthetic --synthetic-root "path/to/synthetic" --epochs 10
python -m src.train_classifier --dataset fashion_mnist --scenario augmented_real --epochs 15
python -m src.train_classifier --dataset mnist --scenario progressive --synthetic-root "path/to/synthetic" --epochs 10 --max-real-train-samples 2000
```

---

## Key Results

| Scenario | MNIST Accuracy | Fashion-MNIST Accuracy |
|---|---|---|
| real_only | **99.27%** | **91.94%** |
| augmented_real | 99.13% (−0.14 pp) | 89.33% (−2.61 pp) |
| real_plus_synthetic | 99.00% (−0.27 pp) | 91.90% (−0.04 pp) |
| synthetic_only | 45.27% (−54.00 pp) | 73.73% (−18.21 pp) |

GAN recognizability (how often a real-trained classifier correctly identifies synthetic images):
- MNIST overall: **0.617** — digit `9` was worst at 6.1%
- Fashion-MNIST overall: **0.771** — Pullover was worst at 29.1%

All classifiers evaluated on the **official 10,000-image test split** for fair comparison.

---

## Notes

- **cGAN** is the default generative model throughout; there is no unconditional GAN in this codebase.
- **Stratified scarcity** keeps classes as balanced as possible when subsampling real training data.
- **FID** uses Inception-v3 features (`torch-fidelity`). MNIST/Fashion-MNIST are out-of-domain for ImageNet features — treat FID as a **relative** comparison across runs, not an absolute quality score.
- **Slerp interpolation**: spherical linear interpolation is used instead of linear blending because the Gaussian latent space is approximately a hypersphere; linear blending cuts through low-probability regions and produces blurry midpoints.
- **WGAN-GP** automatically sets `n_critic=5` (discriminator updates per generator update) when selected.
- **`generator_config.json`** is saved alongside every trained generator so that `generate_synthetic` and `interpolate` can reconstruct the exact architecture without requiring CLI flags.
- The research pipeline saves state to `pipeline_state.json` and is safe to interrupt and resume at any time.
