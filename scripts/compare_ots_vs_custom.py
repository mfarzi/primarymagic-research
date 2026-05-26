"""Compare fingerprints between OTS and custom processed spectra.

For each sequence that exists in both directories, computes the Pearson
correlation and plots the paired fingerprints side by side. Sequences
present in only one source are skipped.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
from primarymagic.data.spectraio import load_from_npz

OTS_ROOT = Path(
    r"C:\Users\mfarzi\mycodes\visiogen\spectra\data\OTS\processed\primary_magic"
)
CUSTOM_ROOT = Path(
    r"C:\Users\mfarzi\mycodes\visiogen\spectra\data\custom\processed\primary_magic"
)
RESULTS_DIR = Path(
    r"C:\Users\mfarzi\mycodes\visiogen\spectra\results\ots_vs_custom"
)


def load_fingerprint(npz_path: Path):
    """Load a single fingerprint Spectrum from an .npz file."""
    collection = load_from_npz(npz_path)
    spectrum = collection[0]
    return spectrum.wavelengths, spectrum.intensities


def find_common_sequences(ots_root: Path, custom_root: Path):
    """Find sequences that exist in both OTS and custom directories.

    Returns list of (sequence_name, ots_fingerprint_path, custom_fingerprint_path).
    """
    pairs = []
    for length_dir in sorted(ots_root.iterdir()):
        if not length_dir.is_dir():
            continue
        length_name = length_dir.name
        if length_name == "1-letter":
            continue
        custom_length = custom_root / length_name
        if not custom_length.exists():
            continue

        ots_seqs = {d.name for d in length_dir.iterdir() if d.is_dir()}
        custom_seqs = {d.name for d in custom_length.iterdir() if d.is_dir()}
        common = sorted(ots_seqs & custom_seqs)

        for seq in common:
            # Use rep1 for OTS, prefer rep2 for custom (fall back to rep1)
            ots_fp = length_dir / seq / "rep1" / "fingerprint.npz"
            custom_fp = custom_length / seq / "rep2" / "fingerprint.npz"
            if not custom_fp.exists():
                custom_fp = custom_length / seq / "rep1" / "fingerprint.npz"
            if ots_fp.exists() and custom_fp.exists():
                pairs.append((f"{length_name}/{seq}", ots_fp, custom_fp))
            else:
                print(f"  SKIP {length_name}/{seq}: missing fingerprint.npz")

    return pairs


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pairs = find_common_sequences(OTS_ROOT, CUSTOM_ROOT)
    if not pairs:
        print("No common sequences found.")
        return

    print(f"Found {len(pairs)} common sequences\n")

    correlations = []
    labels = []

    # Individual paired plots
    for label, ots_path, custom_path in pairs:
        wl_ots, int_ots = load_fingerprint(ots_path)
        wl_cust, int_cust = load_fingerprint(custom_path)

        # Interpolate to common wavelength grid if needed
        if not np.array_equal(wl_ots, wl_cust):
            wl_common = wl_ots
            int_cust = np.interp(wl_common, wl_cust, int_cust)
        else:
            wl_common = wl_ots

        r, p = pearsonr(int_ots, int_cust)
        correlations.append(r)
        labels.append(label)
        print(f"  {label}: r = {r:.4f}  (p = {p:.2e})")

        # Plot paired fingerprints
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(wl_common, int_cust, "b-", linewidth=1.2, alpha=0.8,
                label="Custom")
        ax.plot(wl_common, int_ots, "r-", linewidth=1.2, alpha=0.8,
                label="OTS")
        ax.set_xlabel("Wavenumber (cm\u207b\u00b9)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title(f"{label}  —  Pearson r = {r:.4f}")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        seq_name = label.replace("/", "_")
        fig.savefig(RESULTS_DIR / f"{seq_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Summary bar chart of correlations
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.8), 5))
    seq_labels = [l.split("/")[-1] for l in labels]
    colors = ["green" if r > 0.9 else "orange" if r > 0.7 else "red"
              for r in correlations]
    bars = ax.bar(seq_labels, correlations, color=colors, edgecolor="black",
                  linewidth=0.5)
    ax.set_ylabel("Pearson Correlation")
    ax.set_xlabel("Sequence")
    ax.set_title("OTS vs Custom Fingerprint Correlation")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.9, color="green", linestyle="--", alpha=0.5, label="r = 0.9")
    ax.axhline(y=0.7, color="orange", linestyle="--", alpha=0.5, label="r = 0.7")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")

    for bar, r in zip(bars, correlations):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{r:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "correlation_summary.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    print(f"\nMean correlation: {np.nanmean(correlations):.4f}")
    print(f"Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
