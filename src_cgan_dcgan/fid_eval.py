"""
Fréchet Inception Distance (FID) between real and synthetic image folders.

Uses torch-fidelity (Inception-v3 features). Grayscale exports are saved as RGB
so the default Inception pipeline applies consistently.
"""
import argparse
import os
import shutil
from typing import Optional

import numpy as np
import torch
from torchvision.utils import save_image

from .data import get_base_dataset, stratified_subset_indices
from .utils import ensure_dir


def export_real_images_for_fid(
    dataset_name: str,
    out_dir: str,
    num_images: int,
    seed: int,
    data_root: str = "data",
    stratified: bool = True,
) -> str:
    """
    Write PNGs under out_dir (flat) from the official *training* split for FID reference.
    """
    ensure_dir(out_dir)
    ds = get_base_dataset(dataset_name, root=data_root, train=True)
    num_images = min(num_images, len(ds))
    if stratified:
        idx = stratified_subset_indices(ds, num_images, seed)
    else:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(ds), size=num_images, replace=False).tolist()
    for k, sample_idx in enumerate(idx):
        img_tensor, _ = ds[sample_idx]
        path = os.path.join(out_dir, f"real_{k:06d}.png")
        x = (img_tensor.clamp(-1, 1) + 1) / 2
        x3 = x.repeat(3, 1, 1)
        save_image(x3, path)
    return out_dir


def compute_fid(
    real_dir: str,
    synthetic_dir: str,
    cuda: Optional[bool] = None,
    batch_size: int = 64,
) -> float:
    from torch_fidelity import calculate_metrics

    if cuda is None:
        cuda = torch.cuda.is_available()
    metrics = calculate_metrics(
        input1=real_dir,
        input2=synthetic_dir,
        cuda=cuda,
        fid=True,
        isc=False,
        kid=False,
        verbose=False,
        batch_size=batch_size,
    )
    return float(metrics["frechet_inception_distance"])


def compute_fid_for_experiment(
    dataset_name: str,
    synthetic_root: str,
    work_dir: str,
    num_images: int,
    seed: int,
    batch_size: int = 64,
) -> dict:
    """
    Export a matched-size real reference set and compute FID vs synthetic folder tree.
    """
    ensure_dir(work_dir)
    real_dir = os.path.join(work_dir, "fid_real_pngs")
    if os.path.isdir(real_dir):
        shutil.rmtree(real_dir)
    export_real_images_for_fid(dataset_name, real_dir, num_images=num_images, seed=seed)
    fid = compute_fid(real_dir, synthetic_root, batch_size=batch_size)
    return {"fid": fid, "real_export_dir": real_dir, "num_real_images": num_images}


def parse_args():
    p = argparse.ArgumentParser(description="Compute FID between real (exported) and synthetic image folders.")
    p.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    p.add_argument("--synthetic-root", type=str, required=True, help="Root folder containing generated class subdirs.")
    p.add_argument("--num-images", type=int, default=10000, help="How many real train images to export (capped by dataset size).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--work-dir", type=str, default="", help="Where to write exported real PNGs; default temp under outputs/fid_runs.")
    return p.parse_args()


def main():
    args = parse_args()
    work = args.work_dir or os.path.join("outputs", "fid_runs", f"{args.dataset}_fid")
    out = compute_fid_for_experiment(
        dataset_name=args.dataset,
        synthetic_root=args.synthetic_root,
        work_dir=work,
        num_images=args.num_images,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    print(f"FID (lower is better): {out['fid']:.4f}")
    print(f"Real PNGs: {out['real_export_dir']}")


if __name__ == "__main__":
    main()
