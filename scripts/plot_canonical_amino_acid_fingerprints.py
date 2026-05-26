"""
Plot fingerprints for the 20 canonical (unmodified) amino acids.

Generates two types of output:
1. Individual plots for each amino acid (20 files)
2. Stacked fingerprint plot of all 20 amino acids (1 file)

Usage:
    python scripts/plot_canonical_amino_acid_fingerprints.py
    python scripts/plot_canonical_amino_acid_fingerprints.py \
        --data-dir data/custom/processed/magic/1-letter \
        --output-dir results/magic1/canonical
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Allow importing spectra from the source tree
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from primarymagic.data.spectraio import load_from_npz

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"

_PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = _PROJECT_ROOT / "data" / "processed" / "primary_magic" / "1-letter"
OUTPUT_DIR = _PROJECT_ROOT / "results" / "canonical"

# 20 canonical amino acids (alphabetical by one-letter code)
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# Full names for labels
FULL_NAMES = {
    "A": "Alanine", "C": "Cysteine", "D": "Aspartic acid", "E": "Glutamic acid",
    "F": "Phenylalanine", "G": "Glycine", "H": "Histidine", "I": "Isoleucine",
    "K": "Lysine", "L": "Leucine", "M": "Methionine", "N": "Asparagine",
    "P": "Proline", "Q": "Glutamine", "R": "Arginine", "S": "Serine",
    "T": "Threonine", "V": "Valine", "W": "Tryptophan", "Y": "Tyrosine",
}

# Use tab20 colormap for 20 distinct colors
COLORS = [plt.cm.tab20(i / 20) for i in range(20)]


def load_fingerprint(aa_dir: Path):
    """Load fingerprint from the first rep and return (wavelengths, intensities)."""
    for rep_dir in sorted(aa_dir.glob("rep*")):
        npz_path = rep_dir / "fingerprint.npz"
        if npz_path.exists():
            collection = load_from_npz(npz_path)
            spectrum = collection.spectra[0]
            return collection.wavelengths, spectrum.intensities
    return None, None


def plot_individual(aa_code):
    """Save an individual amino acid fingerprint plot."""
    wl, mean_spec = load_fingerprint(DATA_DIR / aa_code)
    if wl is None:
        print(f"Skipped {aa_code}: no data found")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(wl, mean_spec, linewidth=1.5, color="black")

    full = FULL_NAMES[aa_code]
    ax.set_title(f"{full} ({aa_code})", fontsize=16, fontweight="bold")
    ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=14)
    ax.set_ylabel("Intensity (a.u.)", fontsize=14)
    plt.tight_layout()

    out = OUTPUT_DIR / f"fingerprint_{aa_code}_{full.lower().replace(' ', '_')}.png"
    plt.savefig(out, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out}")


def plot_stacked():
    """Save a stacked fingerprint plot of all 20 canonical amino acids."""
    n = len(AMINO_ACIDS)
    fig, ax = plt.subplots(figsize=(14, n * 1.0))

    wavelengths = None
    names = []
    plotted = 0
    for i, aa_code in enumerate(AMINO_ACIDS):
        wl, mean_spec = load_fingerprint(DATA_DIR / aa_code)
        if wl is None:
            continue
        if wavelengths is None:
            wavelengths = wl

        offset = plotted
        ax.plot(wavelengths, mean_spec + offset, color=COLORS[i], linewidth=1.5)
        names.append(f"{aa_code} — {FULL_NAMES[aa_code]}")
        plotted += 1

    ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=14)
    ax.set_ylabel("")
    ax.set_title("SERS Fingerprints — 20 Canonical Amino Acids", fontsize=16, fontweight="bold")
    ax.set_xlim(wavelengths.min(), wavelengths.max())

    ax.set_yticks(np.arange(plotted))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_ylim(-0.5, plotted)
    ax.grid(True, axis="y", alpha=1.0)

    plt.tight_layout()

    out = OUTPUT_DIR / "canonical_fingerprints_stacked.png"
    plt.savefig(out, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot fingerprints for the 20 canonical amino acids.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DATA_DIR,
        help="Path to 1-letter data directory containing amino acid subdirs",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="Directory to save output plots",
    )
    return parser.parse_args()


def main():
    global DATA_DIR, OUTPUT_DIR
    args = parse_args()
    DATA_DIR = args.data_dir
    OUTPUT_DIR = args.output_dir

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Individual plots
    for aa_code in AMINO_ACIDS:
        plot_individual(aa_code)

    # 2. Stacked fingerprints
    plot_stacked()

    print("\nAll canonical amino acid fingerprint images generated successfully!")


if __name__ == "__main__":
    main()
