"""Compare a dipeptide fingerprint with its constituent single amino acids.

Loads the dipeptide fingerprint and each single amino acid fingerprint,
computes Pearson correlations, and plots them overlaid.

Usage:
    python scripts/compare_dipeptide_vs_amino_acids.py \
        data/OTS/processed/primary_magic \
        AG

    python scripts/compare_dipeptide_vs_amino_acids.py \
        data/custom/processed/primary_magic \
        AS --rep rep2

    # Shift A by 8 steps and G by -3 steps before comparing:
    python scripts/compare_dipeptide_vs_amino_acids.py \
        data/OTS/processed/primary_magic \
        AG --shift-1 8 --shift-2 -3
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from primarymagic.data.spectraio import load_from_npz


def load_fingerprint(npz_path: Path, shift_steps: float = 0):
    """Load wavelengths and intensities from a fingerprint .npz file.

    If shift_steps is non-zero, shift the signal by that many steps.
    """
    collection = load_from_npz(npz_path)
    if shift_steps:
        wl_step = np.mean(np.diff(collection.wavelengths))
        collection = collection.shift(shift_steps * wl_step)
    spectrum = collection[0]
    return spectrum.wavelengths, spectrum.intensities


def find_fingerprint(base_dir: Path, length: str, name: str, rep: str):
    """Find fingerprint.npz, trying the requested rep first, then fallback."""
    primary = base_dir / length / name / rep / "fingerprint.npz"
    if primary.exists():
        return primary
    # Fallback to rep1
    fallback = base_dir / length / name / "rep1" / "fingerprint.npz"
    if fallback.exists():
        print(f"  {length}/{name}/{rep} not found, using rep1")
        return fallback
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Compare dipeptide fingerprint with constituent amino acids."
    )
    parser.add_argument("root", type=Path,
                        help="Root data directory (e.g. data/OTS/processed/primary_magic)")
    parser.add_argument("dipeptide", type=str,
                        help="Dipeptide name (e.g. AG, AS, GA)")
    parser.add_argument("--rep", type=str, default="rep1",
                        help="Replicate to use (default: rep1)")
    parser.add_argument("--shift-1", type=float, default=0,
                        help="Shift first amino acid by N steps (negative=left, positive=right)")
    parser.add_argument("--shift-2", type=float, default=0,
                        help="Shift second amino acid by N steps (negative=left, positive=right)")
    parser.add_argument("--output", type=Path, default=Path("results"),
                        help="Output directory (default: results)")
    args = parser.parse_args()

    root = args.root
    dipeptide = args.dipeptide.upper()
    rep = args.rep

    if len(dipeptide) != 2:
        print(f"ERROR: Expected 2-letter dipeptide, got '{dipeptide}'")
        return

    aa1, aa2 = dipeptide[0], dipeptide[1]

    # Load fingerprints
    dp_path = find_fingerprint(root, "2-letter", dipeptide, rep)
    aa1_path = find_fingerprint(root, "1-letter", aa1, rep)
    aa2_path = find_fingerprint(root, "1-letter", aa2, rep)

    if not dp_path:
        print(f"ERROR: Fingerprint not found for {dipeptide}")
        return
    if not aa1_path:
        print(f"ERROR: Fingerprint not found for {aa1}")
        return
    if not aa2_path:
        print(f"ERROR: Fingerprint not found for {aa2}")
        return

    print(f"Dipeptide: {dp_path}")
    print(f"AA {aa1}:    {aa1_path}")
    print(f"AA {aa2}:    {aa2_path}")

    wl_dp, int_dp = load_fingerprint(dp_path)
    wl_aa1, int_aa1 = load_fingerprint(aa1_path, shift_steps=args.shift_1)
    wl_aa2, int_aa2 = load_fingerprint(aa2_path, shift_steps=args.shift_2)

    # Interpolate to dipeptide wavelength grid
    if not np.array_equal(wl_dp, wl_aa1):
        int_aa1 = np.interp(wl_dp, wl_aa1, int_aa1)
    if not np.array_equal(wl_dp, wl_aa2):
        int_aa2 = np.interp(wl_dp, wl_aa2, int_aa2)

    # Correlations
    r_dp_aa1, _ = pearsonr(int_dp, int_aa1)
    r_dp_aa2, _ = pearsonr(int_dp, int_aa2)
    r_aa1_aa2, _ = pearsonr(int_aa1, int_aa2)

    print(f"\n  {dipeptide} vs {aa1}: r = {r_dp_aa1:.4f}")
    print(f"  {dipeptide} vs {aa2}: r = {r_dp_aa2:.4f}")
    print(f"  {aa1} vs {aa2}:  r = {r_aa1_aa2:.4f}")

    # Plot
    source = root.parts[-3] if len(root.parts) >= 3 else root.name  # OTS or custom
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(wl_dp, int_dp, "k-", lw=1.4, alpha=0.9, label=f"{dipeptide}")
    lbl1 = f"{aa1} (r={r_dp_aa1:.3f})" + (f" [shift {args.shift_1:+g}]" if args.shift_1 else "")
    lbl2 = f"{aa2} (r={r_dp_aa2:.3f})" + (f" [shift {args.shift_2:+g}]" if args.shift_2 else "")
    ax.plot(wl_dp, int_aa1, "b-", lw=1.0, alpha=0.7, label=lbl1)
    ax.plot(wl_dp, int_aa2, "r-", lw=1.0, alpha=0.7, label=lbl2)
    ax.set_xlabel("Wavenumber (cm\u207b\u00b9)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_title(f"{source} — {dipeptide} vs {aa1} & {aa2} ({rep})")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    args.output.mkdir(parents=True, exist_ok=True)
    out_file = args.output / f"{source}_{dipeptide}_vs_{aa1}_{aa2}_{rep}.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to {out_file}")


if __name__ == "__main__":
    main()
