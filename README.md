# GAN-Based Data Augmentation for Image Classification

This repository implements the project proposed in `Project Proposal(group8).pdf`:
- Train a GAN on MNIST or Fashion-MNIST
- Generate synthetic images
- Train and compare classifiers under three setups:
  1. Real-only data
  2. Real + synthetic data
  3. Synthetic-only data
- Evaluate with accuracy, precision, recall, F1-score, and confusion matrix

## Project Structure

`src/`
- `config.py` - central configuration defaults
- `data.py` - dataset loading and synthetic dataset helpers
- `models.py` - GAN and classifier model definitions
- `train_gan.py` - GAN training script
- `generate_synthetic.py` - export synthetic samples to disk
- `train_classifier.py` - classifier training for 3 scenarios
- `run_experiments.py` - **end-to-end**: GAN, synthetic export, all three classifiers, CSV + comparison plot
- `evaluate.py` - evaluation metrics and confusion matrix plotting
- `utils.py` - training utilities (seeding, logging, directories)

`outputs/`
- Generated during runs (models, samples, metrics, plots)

## Environment Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### All-in-one experiment (recommended)

Runs GAN training (unless you skip it), generates synthetic images, trains all three classifiers, and writes `comparison.csv`, `comparison.json`, and `comparison_metrics.png` under `outputs/experiments/<run_id>/`.

```bash
python -m src.run_experiments --dataset mnist --gan-epochs 30 --classifier-epochs 10 --num-synthetic 12000
```

Reuse a trained generator and only run classifiers + comparison:

```bash
python -m src.run_experiments --dataset mnist --skip-gan --generator-path "path/to/generator.pt" --classifier-epochs 10
```

### 1) Train GAN

```bash
python -m src.train_gan --dataset mnist --epochs 30 --batch-size 128
```

### 2) Generate Synthetic Images

```bash
python -m src.generate_synthetic --dataset mnist --generator-path "outputs/models/<run_id>/generator.pt" --num-samples 12000
```

### 3) Train Classifier (Real-only)

```bash
python -m src.train_classifier --dataset mnist --scenario real_only --epochs 10
```

### 4) Train Classifier (Real + Synthetic)

```bash
python -m src.train_classifier --dataset mnist --scenario real_plus_synthetic --epochs 10
```

### 5) Train Classifier (Synthetic-only)

```bash
python -m src.train_classifier --dataset mnist --scenario synthetic_only --epochs 10
```

## Supported Datasets

- `mnist`
- `fashion_mnist`

## Notes

- This project uses a conditional GAN so synthetic images can be generated per class label.
- If GPU is available, scripts will use CUDA automatically.
- For reproducibility, all scripts expose `--seed`.
