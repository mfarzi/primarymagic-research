#!/usr/bin/env python
"""Diagnostic script for X->S misclassification in tetrapeptide evaluation.

Investigates why the differential classifier predicts S (serine) instead of
a target amino acid X for FAR+X vs FAR pairs.

Steps:
    1. Confirm the misclassification on FAR+X vs FAR and FARS vs FAR pairs
    2. Examine the latent space (differentials, PCA, softmax distributions)
    3. Compare raw spectra (mean overlays and differential spectra)

Usage:
    python scripts/diagnose_vs_s.py --target G
    python scripts/diagnose_vs_s.py --target D
    python scripts/diagnose_vs_s.py --target G --no-show
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
S_IDX = AMINO_ACIDS_6.index("S")  # 5

DEFAULT_CHECKPOINT = "checkpoints/decoupled_v1/seq_step26_tp095/differential_classifier.pt"
DEFAULT_DATA_ROOT = "data/processed/primary_magic"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose X->S misclassification in tetrapeptide evaluation"
    )
    parser.add_argument(
        "--target", required=True, choices=["A", "D", "F", "G", "R"],
        help="Target amino acid to diagnose (the one being misclassified as S)",
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
        "--output-dir", default=None,
        help="Directory to save output plots (auto-generated if not set)",
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
def confirm_misclassification(model, spectra_data, target, target_idx, n_samples, rng, device):
    """Predict on FAR+target vs FAR and FARS vs FAR pairs, print results."""
    print("=" * 60)
    print("STEP 1: Confirm misclassification")
    print("=" * 60)

    target_code = "FAR" + target  # e.g. FARG
    target_spectra = spectra_data.get_spectra(target_code)
    fars = spectra_data.get_spectra("FARS")
    far = spectra_data.get_spectra("FAR")

    if target_spectra is None or fars is None or far is None:
        missing = [s for s, v in [(target_code, target_spectra), ("FARS", fars), ("FAR", far)] if v is None]
        print(f"ERROR: Missing spectra for: {missing}")
        return None
    print(f"{target_code}: {len(target_spectra)} spectra, FARS: {len(fars)} spectra, FAR: {len(far)} spectra")

    model.eval()
    results = {}

    for name, longer_data, expected_label in [
        (f"{target_code} vs FAR (expect {target})", target_spectra, target_idx),
        ("FARS vs FAR (expect S)", fars, S_IDX),
    ]:
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

        correct = (preds == expected_label).sum()
        total = len(preds)

        print(f"\n{name}:")
        print(f"  Correct: {correct}/{total}")

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
        }

    # Mean fingerprint predictions
    print("\nMean fingerprint predictions:")
    with torch.no_grad():
        for name, longer, expected in [(target_code, target_spectra, target), ("FARS", fars, "S")]:
            logits = model(
                torch.tensor(longer.mean(axis=0), dtype=torch.float32).unsqueeze(0).to(device),
                torch.tensor(far.mean(axis=0), dtype=torch.float32).unsqueeze(0).to(device),
            )
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
            print(f"  Mean {name} vs Mean FAR (expect {expected}): Predicted {AMINO_ACIDS_6[probs.argmax()]}")
            for i, aa in enumerate(AMINO_ACIDS_6):
                print(f"    P({aa}) = {probs[i]:.4f}")

    return results


# ---------------------------------------------------------------------------
# Step 2: Examine the latent space
# ---------------------------------------------------------------------------
def examine_latent_space(model, spectra_data, target, target_idx, results_step1, output_dir, device):
    """Encode spectra, compute differentials, plot analysis."""
    print("\n" + "=" * 60)
    print("STEP 2: Latent space analysis")
    print("=" * 60)

    target_code = "FAR" + target
    target_spectra = spectra_data.get_spectra(target_code)
    fars = spectra_data.get_spectra("FARS")
    far = spectra_data.get_spectra("FAR")

    encoder = model.encoder

    with torch.no_grad():
        z_target = encoder(torch.tensor(target_spectra, dtype=torch.float32).to(device)).cpu().numpy()
        z_fars = encoder(torch.tensor(fars, dtype=torch.float32).to(device)).cpu().numpy()
        z_far = encoder(torch.tensor(far, dtype=torch.float32).to(device)).cpu().numpy()

    print(f"Latent dim: {z_target.shape[1]}")
    print(f"z_{target_code}: {z_target.shape}, z_FARS: {z_fars.shape}, z_FAR: {z_far.shape}")

    mean_z_target = z_target.mean(axis=0)
    mean_z_fars = z_fars.mean(axis=0)
    mean_z_far = z_far.mean(axis=0)

    z_diff_T = mean_z_target - mean_z_far
    z_diff_S = mean_z_fars - mean_z_far

    print(f"\nMean differential ({target}): mean={z_diff_T.mean():.4f}, std={z_diff_T.std():.4f}")
    print(f"Mean differential (S): mean={z_diff_S.mean():.4f}, std={z_diff_S.std():.4f}")

    cos_sim = np.dot(z_diff_T, z_diff_S) / (
        np.linalg.norm(z_diff_T) * np.linalg.norm(z_diff_S) + 1e-8
    )
    l2_dist = np.linalg.norm(z_diff_T - z_diff_S)
    print(f"Cosine similarity(z_diff_{target}, z_diff_S): {cos_sim:.4f}")
    print(f"L2 distance(z_diff_{target}, z_diff_S): {l2_dist:.4f}")

    # Cross-correlation lag on raw spectra
    mean_target_raw = target_spectra.mean(axis=0)
    mean_far_raw = far.mean(axis=0)
    xcorr = np.correlate(mean_target_raw, mean_far_raw, mode="full")
    lag = np.arange(-len(mean_far_raw) + 1, len(mean_target_raw))
    best_lag = lag[np.argmax(xcorr)]
    print(f"Cross-correlation lag ({target_code} vs FAR): {best_lag}")

    print(f"\nPer-dimension comparison (top 10 by |{target}-S| difference):")
    diff_abs = np.abs(z_diff_T - z_diff_S)
    top_dims = np.argsort(diff_abs)[::-1][:10]
    print(f"  {'Dim':>4s}  {target + '_diff':>8s}  {'S_diff':>8s}  {'|' + target + '-S|':>8s}")
    for d in top_dims:
        print(f"  {d:4d}  {z_diff_T[d]:8.4f}  {z_diff_S[d]:8.4f}  {diff_abs[d]:8.4f}")

    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Bar chart
    ax = axes[0]
    latent_dim = len(z_diff_T)
    x = np.arange(latent_dim)
    width = 0.35
    ax.bar(x - width / 2, z_diff_T, width, label=f"{target} ({target_code}-FAR)", alpha=0.8)
    ax.bar(x + width / 2, z_diff_S, width, label="S (FARS-FAR)", alpha=0.8)
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel("Mean differential")
    ax.set_title("Mean latent differential per dimension")
    ax.legend()

    # PCA scatter
    ax = axes[1]
    z_diff_T_samples = z_target - mean_z_far
    z_diff_S_samples = z_fars - mean_z_far
    all_diffs = np.vstack([z_diff_T_samples, z_diff_S_samples])
    pca = PCA(n_components=2)
    all_2d = pca.fit_transform(all_diffs)
    n_t = len(z_diff_T_samples)
    ax.scatter(all_2d[:n_t, 0], all_2d[:n_t, 1], alpha=0.4, s=10, label=f"{target} ({target_code})")
    ax.scatter(all_2d[n_t:, 0], all_2d[n_t:, 1], alpha=0.4, s=10, label="S (FARS)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("PCA of latent differentials")
    ax.legend()

    # Softmax histogram
    ax = axes[2]
    if results_step1 is not None:
        target_key = f"{target_code} vs FAR (expect {target})"
        if target_key in results_step1:
            probs = results_step1[target_key]["probs"]
            ax.hist(probs[:, target_idx], bins=20, alpha=0.6, label=f"P({target}) [idx={target_idx}]")
            ax.hist(probs[:, S_IDX], bins=20, alpha=0.6, label=f"P(S) [idx={S_IDX}]")
            ax.set_xlabel("Softmax probability")
            ax.set_ylabel("Count")
            ax.set_title(f"Classifier output on {target_code} samples")
            ax.legend()

    plt.tight_layout()
    out_path = output_dir / "latent_diff_analysis.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")

    return fig


# ---------------------------------------------------------------------------
# Step 3: Compare raw spectra
# ---------------------------------------------------------------------------
def compare_raw_spectra(spectra_data, target, output_dir):
    """Plot mean spectra and differential spectra."""
    print("\n" + "=" * 60)
    print("STEP 3: Raw spectra comparison")
    print("=" * 60)

    target_code = "FAR" + target
    target_spectra = spectra_data.get_spectra(target_code)
    fars = spectra_data.get_spectra("FARS")
    far = spectra_data.get_spectra("FAR")
    wavelengths = spectra_data.wavelengths

    mean_target = target_spectra.mean(axis=0)
    mean_fars = fars.mean(axis=0)
    mean_far = far.mean(axis=0)

    diff_t = mean_target - mean_far
    diff_s = mean_fars - mean_far

    corr = np.corrcoef(diff_t, diff_s)[0, 1]
    print(f"Correlation between mean differential spectra ({target} vs S): {corr:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(wavelengths, mean_target, label=f"{target_code} (mean)", alpha=0.8)
    ax.plot(wavelengths, mean_fars, label="FARS (mean)", alpha=0.8)
    ax.plot(wavelengths, mean_far, label="FAR (mean)", alpha=0.8, linestyle="--")
    ax.set_xlabel("Wavenumber (1/cm)")
    ax.set_ylabel("Intensity")
    ax.set_title(f"Mean spectra: {target_code} vs FARS vs FAR")
    ax.legend()

    ax = axes[1]
    ax.plot(wavelengths, diff_t, label=f"{target_code} - FAR (->{target})", alpha=0.8)
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

    target = args.target
    target_idx = AMINO_ACIDS_6.index(target)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(42)

    if args.output_dir is None:
        output_dir = Path(f"results/decoupled_v1/seq_step26_tp095/{target.lower()}_vs_s_diagnosis")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model, checkpoint = DifferentialClassifierWithPretrainedEncoder.from_checkpoint(
        args.checkpoint, device=device,
    )
    model = model.to(device)
    model.eval()
    print(f"Model loaded - {model.num_classes} classes, latent_dim={model.latent_dim}")

    # Load data
    print(f"Loading data: {args.data_root}")
    spectra_data = SpectraDataset(args.data_root, min_spectra=40)
    print(spectra_data)

    # Step 1
    results_step1 = confirm_misclassification(
        model, spectra_data, target, target_idx, args.n_samples, rng, device,
    )

    # Step 2
    fig1 = examine_latent_space(model, spectra_data, target, target_idx, results_step1, output_dir, device)

    # Step 3
    fig2 = compare_raw_spectra(spectra_data, target, output_dir)

    if not args.no_show:
        plt.show()

    print("\nDone.")


if __name__ == "__main__":
    main()
