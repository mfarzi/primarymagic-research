#!/usr/bin/env python
"""Diagnostic script for D→S misclassification in tetrapeptide evaluation.

Investigates why the differential classifier predicts S (serine) instead of
D (aspartic acid) for FARD vs FAR pairs. The tetrapeptide evaluation for
seq_step26_tp095 shows D has ~1% recall — 99/100 samples predicted as S.

Steps:
    1. Confirm the misclassification on FARD vs FAR and FARS vs FAR pairs
    2. Examine the latent space (differentials, PCA, softmax distributions)
    3. Compare raw spectra (mean overlays and differential spectra)

Usage:
    python scripts/diagnose_d_vs_s.py
    python scripts/diagnose_d_vs_s.py --checkpoint path/to/checkpoint.pt
    python scripts/diagnose_d_vs_s.py --no-show  # save plots only
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from primarymagic.data.spectra_dataset import SpectraDataset
from primarymagic.models.autoencoder import DifferentialClassifierWithPretrainedEncoder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AMINO_ACIDS_6 = ["A", "D", "F", "G", "R", "S"]
D_IDX = AMINO_ACIDS_6.index("D")  # 1
S_IDX = AMINO_ACIDS_6.index("S")  # 5

DEFAULT_CHECKPOINT = "checkpoints/decoupled_v1/seq_step26_tp095/differential_classifier.pt"
DEFAULT_DATA_ROOT = "data/processed/primary_magic"
DEFAULT_OUTPUT_DIR = "results/decoupled_v1/seq_step26_tp095/d_vs_s_diagnosis"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose D→S misclassification in tetrapeptide evaluation"
    )
    parser.add_argument(
        "--checkpoint", default=DEFAULT_CHECKPOINT,
        help="Path to differential_classifier.pt checkpoint",
    )
    parser.add_argument(
        "--data-root", default=DEFAULT_DATA_ROOT,
        help="Path to processed data directory",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help="Directory to save output plots",
    )
    parser.add_argument(
        "--n-samples", type=int, default=100,
        help="Number of random spectra to sample per sequence",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Save plots only, do not call plt.show()",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Step 1: Confirm the misclassification
# ---------------------------------------------------------------------------
def confirm_misclassification(model, spectra_data, n_samples, rng, device):
    """Predict on FARD vs FAR and FARS vs FAR pairs, print results."""
    print("=" * 60)
    print("STEP 1: Confirm misclassification")
    print("=" * 60)

    fard = spectra_data.get_spectra("FARD")
    fars = spectra_data.get_spectra("FARS")
    far = spectra_data.get_spectra("FAR")

    if fard is None or fars is None or far is None:
        missing = [s for s, v in [("FARD", fard), ("FARS", fars), ("FAR", far)] if v is None]
        print(f"ERROR: Missing spectra for: {missing}")
        return None, None
    print(f"FARD: {len(fard)} spectra, FARS: {len(fars)} spectra, FAR: {len(far)} spectra")

    model.eval()
    results = {}

    for name, longer_data, expected_label in [
        ("FARD vs FAR (expect D)", fard, D_IDX),
        ("FARS vs FAR (expect S)", fars, S_IDX),
    ]:
        # Sample spectra (with replacement if n_samples > available)
        replace_longer = n_samples > len(longer_data)
        replace_shorter = n_samples > len(far)
        longer_idx = rng.choice(len(longer_data), size=n_samples, replace=replace_longer)
        shorter_idx = rng.choice(len(far), size=n_samples, replace=replace_shorter)

        longer_t = torch.tensor(longer_data[longer_idx], dtype=torch.float32).to(device)
        shorter_t = torch.tensor(far[shorter_idx], dtype=torch.float32).to(device)

        with torch.no_grad():
            logits = model(longer_t, shorter_t)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1).cpu().numpy()

        expected_name = AMINO_ACIDS_6[expected_label]
        correct = (preds == expected_label).sum()
        total = len(preds)

        print(f"\n{name}:")
        print(f"  Correct: {correct}/{total}")

        # Per-class prediction breakdown
        pred_counts = {}
        for p in preds:
            aa = AMINO_ACIDS_6[p]
            pred_counts[aa] = pred_counts.get(aa, 0) + 1
        for aa in AMINO_ACIDS_6:
            count = pred_counts.get(aa, 0)
            if count > 0:
                print(f"  Predicted {aa}: {count}/{total}")

        results[name] = {
            "preds": preds,
            "probs": probs.cpu().numpy(),
            "expected": expected_label,
            "longer_idx": longer_idx,
            "shorter_idx": shorter_idx,
        }

    return results, far


# ---------------------------------------------------------------------------
# Step 2: Examine the latent space
# ---------------------------------------------------------------------------
def examine_latent_space(model, spectra_data, results_step1, output_dir, device):
    """Encode spectra, compute differentials, plot analysis."""
    print("\n" + "=" * 60)
    print("STEP 2: Latent space analysis")
    print("=" * 60)

    fard = spectra_data.get_spectra("FARD")
    fars = spectra_data.get_spectra("FARS")
    far = spectra_data.get_spectra("FAR")

    encoder = model.encoder

    # Encode all spectra
    with torch.no_grad():
        z_fard = encoder(torch.tensor(fard, dtype=torch.float32).to(device)).cpu().numpy()
        z_fars = encoder(torch.tensor(fars, dtype=torch.float32).to(device)).cpu().numpy()
        z_far = encoder(torch.tensor(far, dtype=torch.float32).to(device)).cpu().numpy()

    print(f"Latent dim: {z_fard.shape[1]}")
    print(f"z_FARD: {z_fard.shape}, z_FARS: {z_fars.shape}, z_FAR: {z_far.shape}")

    # Compute mean differentials
    mean_z_fard = z_fard.mean(axis=0)
    mean_z_fars = z_fars.mean(axis=0)
    mean_z_far = z_far.mean(axis=0)

    z_diff_D = mean_z_fard - mean_z_far  # mean differential for D
    z_diff_S = mean_z_fars - mean_z_far  # mean differential for S

    # Statistics
    print(f"\nMean differential (D): mean={z_diff_D.mean():.4f}, std={z_diff_D.std():.4f}")
    print(f"Mean differential (S): mean={z_diff_S.mean():.4f}, std={z_diff_S.std():.4f}")

    # Cosine similarity
    cos_sim = np.dot(z_diff_D, z_diff_S) / (
        np.linalg.norm(z_diff_D) * np.linalg.norm(z_diff_S) + 1e-8
    )
    print(f"Cosine similarity(z_diff_D, z_diff_S): {cos_sim:.4f}")

    # L2 distance
    l2_dist = np.linalg.norm(z_diff_D - z_diff_S)
    print(f"L2 distance(z_diff_D, z_diff_S): {l2_dist:.4f}")

    # Per-dimension stats
    print(f"\nPer-dimension comparison (top 10 by |D-S| difference):")
    diff_abs = np.abs(z_diff_D - z_diff_S)
    top_dims = np.argsort(diff_abs)[::-1][:10]
    print(f"  {'Dim':>4s}  {'D_diff':>8s}  {'S_diff':>8s}  {'|D-S|':>8s}")
    for d in top_dims:
        print(f"  {d:4d}  {z_diff_D[d]:8.4f}  {z_diff_S[d]:8.4f}  {diff_abs[d]:8.4f}")

    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Subplot 1: Bar chart of mean differentials per latent dimension
    ax = axes[0]
    latent_dim = len(z_diff_D)
    x = np.arange(latent_dim)
    width = 0.35
    ax.bar(x - width / 2, z_diff_D, width, label="D (FARD-FAR)", alpha=0.8)
    ax.bar(x + width / 2, z_diff_S, width, label="S (FARS-FAR)", alpha=0.8)
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel("Mean differential")
    ax.set_title("Mean latent differential per dimension")
    ax.legend()

    # Subplot 2: PCA scatter of per-sample differentials
    ax = axes[1]
    # Compute per-sample differentials by pairing each longer spectrum
    # with the mean FAR encoding
    z_diff_D_samples = z_fard - mean_z_far  # (n_fard, latent_dim)
    z_diff_S_samples = z_fars - mean_z_far  # (n_fars, latent_dim)

    all_diffs = np.vstack([z_diff_D_samples, z_diff_S_samples])
    pca = PCA(n_components=2)
    all_2d = pca.fit_transform(all_diffs)
    n_d = len(z_diff_D_samples)

    ax.scatter(all_2d[:n_d, 0], all_2d[:n_d, 1], alpha=0.4, s=10, label="D (FARD)")
    ax.scatter(all_2d[n_d:, 0], all_2d[n_d:, 1], alpha=0.4, s=10, label="S (FARS)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("PCA of latent differentials")
    ax.legend()

    # Subplot 3: Softmax probability histograms for FARD samples
    ax = axes[2]
    if results_step1 is not None:
        fard_key = "FARD vs FAR (expect D)"
        if fard_key in results_step1:
            probs = results_step1[fard_key]["probs"]
            ax.hist(probs[:, D_IDX], bins=20, alpha=0.6, label=f"P(D) [idx={D_IDX}]")
            ax.hist(probs[:, S_IDX], bins=20, alpha=0.6, label=f"P(S) [idx={S_IDX}]")
            ax.set_xlabel("Softmax probability")
            ax.set_ylabel("Count")
            ax.set_title("Classifier output on FARD samples")
            ax.legend()
    else:
        ax.text(0.5, 0.5, "No predictions available", ha="center", va="center",
                transform=ax.transAxes)

    plt.tight_layout()
    out_path = output_dir / "latent_diff_analysis.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")

    return fig


# ---------------------------------------------------------------------------
# Step 3: Compare raw spectra
# ---------------------------------------------------------------------------
def compare_raw_spectra(spectra_data, output_dir):
    """Plot mean spectra and differential spectra."""
    print("\n" + "=" * 60)
    print("STEP 3: Raw spectra comparison")
    print("=" * 60)

    fard = spectra_data.get_spectra("FARD")
    fars = spectra_data.get_spectra("FARS")
    far = spectra_data.get_spectra("FAR")
    wavelengths = spectra_data.wavelengths

    mean_fard = fard.mean(axis=0)
    mean_fars = fars.mean(axis=0)
    mean_far = far.mean(axis=0)

    # Differential spectra
    diff_d = mean_fard - mean_far
    diff_s = mean_fars - mean_far

    # Correlation between differential spectra
    corr = np.corrcoef(diff_d, diff_s)[0, 1]
    print(f"Correlation between mean differential spectra (D vs S): {corr:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Mean spectra overlay
    ax = axes[0]
    ax.plot(wavelengths, mean_fard, label="FARD (mean)", alpha=0.8)
    ax.plot(wavelengths, mean_fars, label="FARS (mean)", alpha=0.8)
    ax.plot(wavelengths, mean_far, label="FAR (mean)", alpha=0.8, linestyle="--")
    ax.set_xlabel("Wavenumber (1/cm)")
    ax.set_ylabel("Intensity")
    ax.set_title("Mean spectra: FARD vs FARS vs FAR")
    ax.legend()

    # Subplot 2: Differential spectra
    ax = axes[1]
    ax.plot(wavelengths, diff_d, label="FARD - FAR (->D)", alpha=0.8)
    ax.plot(wavelengths, diff_s, label="FARS - FAR (->S)", alpha=0.8)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Wavenumber (1/cm)")
    ax.set_ylabel("Intensity difference")
    ax.set_title(f"Differential spectra (corr={corr:.3f})")
    ax.legend()

    plt.tight_layout()
    out_path = output_dir / "spectra_comparison.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(42)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model, checkpoint = DifferentialClassifierWithPretrainedEncoder.from_checkpoint(
        args.checkpoint, device=device,
    )
    model = model.to(device)
    model.eval()
    print(f"Model loaded — {model.num_classes} classes, latent_dim={model.latent_dim}")

    # Load data
    print(f"Loading data: {args.data_root}")
    spectra_data = SpectraDataset(args.data_root, min_spectra=40)
    print(spectra_data)

    # Step 1
    results_step1, far_spectra = confirm_misclassification(
        model, spectra_data, args.n_samples, rng, device,
    )

    # Step 2
    fig1 = examine_latent_space(model, spectra_data, results_step1, output_dir, device)

    # Step 3
    fig2 = compare_raw_spectra(spectra_data, output_dir)

    if not args.no_show:
        plt.show()

    print("\nDone.")


if __name__ == "__main__":
    main()
