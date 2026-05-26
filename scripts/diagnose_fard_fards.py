#!/usr/bin/env python
"""Compare FARD and FARDS spectra relative to FAR.

Checks whether the wavenumber calibration offset found in FARD (relative to
FAR) is also present in FARDS.  Produces plots analogous to the d_vs_s
diagnosis for seq_step26_tp095.

Outputs (saved to --output-dir):
    - fard_vs_far_and_fards_vs_far.png  : side-by-side mean spectra comparison
    - fard_shifted_fards_far.png        : FARD (shifted), FARDS, FAR overlay
    - cross_correlation.png             : cross-correlation lag analysis
    - shift_sweep.png                   : correlation vs shift for both sequences

Usage:
    python scripts/diagnose_fard_fards.py
    python scripts/diagnose_fard_fards.py --shift -8
    python scripts/diagnose_fard_fards.py --no-show
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


DEFAULT_DATA_ROOT = "data/processed/primary_magic"
DEFAULT_OUTPUT_DIR = "results/decoupled_v1/fard_fards_diagnosis"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare FARD and FARDS spectra relative to FAR"
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
        "--shift", type=int, default=-8,
        help="Shift to apply to FARD spectra (default: -8)",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Save plots only, do not call plt.show()",
    )
    return parser.parse_args()


def load_spectra(data_root, n_letter, sequence):
    """Load and concatenate clean_data.npz from all reps."""
    seq_dir = Path(data_root) / f"{n_letter}-letter" / sequence
    arrays = []
    wavelengths = None
    for rep_dir in sorted(seq_dir.iterdir()):
        if not rep_dir.is_dir() or not rep_dir.name.startswith("rep"):
            continue
        clean_path = rep_dir / "clean_data.npz"
        if clean_path.exists():
            d = np.load(clean_path, allow_pickle=True)
            arrays.append(d["intensities"])
            if wavelengths is None:
                wavelengths = d["wavelengths"]
    return np.concatenate(arrays, axis=0), wavelengths


def shift_spectra(spectra, shift):
    """Shift spectra along the wavelength axis (positive = right)."""
    shifted = np.zeros_like(spectra)
    if shift > 0:
        shifted[:, shift:] = spectra[:, :-shift]
    elif shift < 0:
        shifted[:, :shift] = spectra[:, -shift:]
    else:
        shifted = spectra.copy()
    return shifted


def cross_correlation_lag(a, b):
    """Find the optimal lag (shift) of b relative to a via cross-correlation."""
    a_norm = a - a.mean()
    b_norm = b - b.mean()
    corr = np.correlate(a_norm, b_norm, mode="full")
    lags = np.arange(-len(a) + 1, len(a))
    best_lag = lags[np.argmax(corr)]
    return best_lag, lags, corr


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading spectra...")
    fard, wavelengths = load_spectra(args.data_root, 4, "FARD")
    fards, _ = load_spectra(args.data_root, 5, "FARDS")
    far, _ = load_spectra(args.data_root, 3, "FAR")
    print(f"FARD: {fard.shape}, FARDS: {fards.shape}, FAR: {far.shape}")

    mean_fard = fard.mean(axis=0)
    mean_fards = fards.mean(axis=0)
    mean_far = far.mean(axis=0)

    # Cross-correlation analysis
    print("\nCross-correlation analysis:")
    lag_fard, lags_fard, corr_fard = cross_correlation_lag(mean_far, mean_fard)
    lag_fards, lags_fards, corr_fards = cross_correlation_lag(mean_far, mean_fards)
    print(f"  FARD vs FAR optimal lag: {lag_fard}")
    print(f"  FARDS vs FAR optimal lag: {lag_fards}")

    # --- Plot 1: Side-by-side comparison (like fars_vs_fard_shift_comparison) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(wavelengths, mean_fard, label=f"FARD mean (n={len(fard)})", alpha=0.8)
    ax.plot(wavelengths, mean_far, label=f"FAR mean (n={len(far)})", alpha=0.8,
            linestyle="--")
    ax.set_xlabel("Wavenumber (1/cm)")
    ax.set_ylabel("Intensity")
    ax.set_title("FARD vs FAR (unshifted)")
    ax.legend()

    ax = axes[1]
    ax.plot(wavelengths, mean_fards, label=f"FARDS mean (n={len(fards)})", alpha=0.8)
    ax.plot(wavelengths, mean_far, label=f"FAR mean (n={len(far)})", alpha=0.8,
            linestyle="--")
    ax.set_xlabel("Wavenumber (1/cm)")
    ax.set_ylabel("Intensity")
    ax.set_title("FARDS vs FAR (unshifted)")
    ax.legend()

    plt.tight_layout()
    out1 = output_dir / "fard_vs_far_and_fards_vs_far.png"
    plt.savefig(out1, dpi=150)
    print(f"\nSaved: {out1}")

    # --- Plot 2: FARD, FARDS (shifted), FAR overlay ---
    shifted_fards = shift_spectra(fards, args.shift)
    mean_shifted_fards = shifted_fards.mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(wavelengths, mean_fard,
            label=f"FARD mean (n={len(fard)})", alpha=0.8)
    ax.plot(wavelengths, mean_shifted_fards,
            label=f"FARDS mean shifted {args.shift} (n={len(fards)})", alpha=0.8)
    ax.plot(wavelengths, mean_far,
            label=f"FAR mean (n={len(far)})", alpha=0.8, linestyle="--")
    ax.set_xlabel("Wavenumber (1/cm)")
    ax.set_ylabel("Intensity")
    ax.set_title(f"Mean fingerprints: FARD, FARDS (shifted {args.shift}), FAR")
    ax.legend()

    plt.tight_layout()
    out2 = output_dir / "fard_fards_shifted_far.png"
    plt.savefig(out2, dpi=150)
    print(f"Saved: {out2}")

    # --- Plot 3: Cross-correlation ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    mask = (lags_fard >= -30) & (lags_fard <= 30)
    ax.plot(lags_fard[mask], corr_fard[mask])
    ax.axvline(lag_fard, color="r", linestyle="--", label=f"best lag = {lag_fard}")
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Lag (steps)")
    ax.set_ylabel("Cross-correlation")
    ax.set_title("FARD vs FAR cross-correlation")
    ax.legend()

    ax = axes[1]
    mask = (lags_fards >= -30) & (lags_fards <= 30)
    ax.plot(lags_fards[mask], corr_fards[mask])
    ax.axvline(lag_fards, color="r", linestyle="--", label=f"best lag = {lag_fards}")
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Lag (steps)")
    ax.set_ylabel("Cross-correlation")
    ax.set_title("FARDS vs FAR cross-correlation")
    ax.legend()

    plt.tight_layout()
    out3 = output_dir / "cross_correlation.png"
    plt.savefig(out3, dpi=150)
    print(f"Saved: {out3}")

    # --- Plot 4: Shift sweep ---
    print("\nShift sweep for FARDS vs FAR:")
    shifts = list(range(-15, 6))
    correlations_fards = []
    for s in shifts:
        shifted = shift_spectra(fards, s).mean(axis=0)
        trim = max(abs(s), 1)
        corr = np.corrcoef(shifted[trim:-trim], mean_far[trim:-trim])[0, 1]
        correlations_fards.append(corr)
        if s in [0, -4, -6, -8, -10]:
            print(f"  shift={s:3d}: corr={corr:.6f}")

    best_fards_idx = np.argmax(correlations_fards)
    best_fards = shifts[best_fards_idx]
    print(f"  Best shift for FARDS: {best_fards} (corr={correlations_fards[best_fards_idx]:.6f})")

    print("\nShift sweep for FARD vs FAR:")
    correlations_fard = []
    for s in shifts:
        shifted = shift_spectra(fard, s).mean(axis=0)
        trim = max(abs(s), 1)
        corr = np.corrcoef(shifted[trim:-trim], mean_far[trim:-trim])[0, 1]
        correlations_fard.append(corr)

    best_fard_idx = np.argmax(correlations_fard)
    best_fard = shifts[best_fard_idx]
    print(f"  Best shift for FARD: {best_fard} (corr={correlations_fard[best_fard_idx]:.6f})")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(shifts, correlations_fard, "o-", label="FARD vs FAR", markersize=4)
    ax.plot(shifts, correlations_fards, "s-", label="FARDS vs FAR", markersize=4)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Shift (steps)")
    ax.set_ylabel("Pearson correlation with FAR")
    ax.set_title("Shift sweep: correlation with FAR")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out4 = output_dir / "shift_sweep.png"
    plt.savefig(out4, dpi=150)
    print(f"\nSaved: {out4}")

    if not args.no_show:
        plt.show()

    print("\nDone.")


if __name__ == "__main__":
    main()
