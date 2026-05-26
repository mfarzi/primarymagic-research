"""
Plot fingerprints for 6 PTM amino acids alongside their unmodified counterparts.

Generates three types of output:
1. Individual plots for each PTM vs its unmodified counterpart (6 files)
2. Combined 3x2 grid of all PTM vs unmodified pairs (1 file)
3. Stacked fingerprint plot of all 6 PTM amino acids (1 file)

Data from data/processed/primary_magic/1-letter/.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"

DATA_DIR = Path(__file__).parent.parent / "data" / "processed" / "primary_magic" / "1-letter"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "PTMs"

# PTM code -> unmodified single-letter code
PTM_PAIRS = {
    "acetyl-K": "K",
    "acetyl-S": "S",
    "hydroxy-P": "P",
    "phos-S": "S",
    "phos-T": "T",
    "phos-Y": "Y",
}

PTM_CODES = list(PTM_PAIRS.keys())

# Colors for stacked plot (last one = phos-Y -> black)
COLORS = list(plt.cm.Set1(np.linspace(0, 1, 9))[:5]) + [(0, 0, 0, 1)]


def load_mean_spectrum(aa_dir: Path):
    """Load all reps from aa_dir and return (wavelengths, mean_intensity)."""
    all_intensities = []
    wavelengths = None
    for rep_dir in sorted(aa_dir.glob("rep*")):
        npz_path = rep_dir / "clean_data.npz"
        if npz_path.exists():
            data = np.load(npz_path)
            all_intensities.append(data["intensities"])
            if wavelengths is None:
                wavelengths = data["wavelengths"]
    if not all_intensities:
        return None, None
    intensities = np.concatenate(all_intensities, axis=0)
    mean_spec = intensities.mean(axis=0)
    return wavelengths, mean_spec


def fix_baseline(wavelengths, spectrum):
    """Remove baseline offset: subtract value at wavenumber 200, clamp before 200."""
    idx_200 = np.argmin(np.abs(wavelengths - 200))
    offset = spectrum[idx_200]
    spectrum = spectrum - offset
    spectrum[:idx_200] = 0.0
    return spectrum


def plot_individual(ptm_code, base_code):
    """Save an individual PTM vs unmodified plot."""
    ptm_wl, ptm_mean = load_mean_spectrum(DATA_DIR / "PTM" / ptm_code)
    base_wl, base_mean = load_mean_spectrum(DATA_DIR / base_code)

    if ptm_code == "phos-Y" and ptm_wl is not None:
        ptm_mean = fix_baseline(ptm_wl, ptm_mean.copy())
    if base_code == "Y" and base_wl is not None:
        base_mean = fix_baseline(base_wl, base_mean.copy())

    fig, ax = plt.subplots(figsize=(12, 5))
    if base_wl is not None:
        ax.plot(base_wl, base_mean, label=base_code, linewidth=1.5, alpha=0.8)
    if ptm_wl is not None:
        ax.plot(ptm_wl, ptm_mean, label=ptm_code, linewidth=1.5, alpha=0.8)

    ax.set_title(f"{ptm_code}  vs  {base_code}", fontsize=16, fontweight="bold")
    ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=14)
    ax.set_ylabel("Intensity (a.u.)", fontsize=14)
    ax.legend(fontsize=12)
    plt.tight_layout()

    out = OUTPUT_DIR / f"ptm_{ptm_code}_vs_{base_code}.png"
    plt.savefig(out, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out}")


def plot_combined_grid():
    """Save the 3x2 grid of all PTM vs unmodified pairs."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    axes = axes.flatten()

    for idx, (ptm_code, base_code) in enumerate(PTM_PAIRS.items()):
        ax = axes[idx]
        ptm_wl, ptm_mean = load_mean_spectrum(DATA_DIR / "PTM" / ptm_code)
        base_wl, base_mean = load_mean_spectrum(DATA_DIR / base_code)

        if ptm_code == "phos-Y" and ptm_wl is not None:
            ptm_mean = fix_baseline(ptm_wl, ptm_mean.copy())
        if base_code == "Y" and base_wl is not None:
            base_mean = fix_baseline(base_wl, base_mean.copy())

        if ptm_wl is not None and base_wl is not None:
            ax.plot(base_wl, base_mean, label=base_code, linewidth=1.5, alpha=0.8)
            ax.plot(ptm_wl, ptm_mean, label=ptm_code, linewidth=1.5, alpha=0.8)

        ax.set_title(f"{ptm_code}  vs  {base_code}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=12)
        ax.set_ylabel("Intensity (a.u.)", fontsize=12)
        ax.legend(fontsize=11)

    plt.suptitle("PTM vs Unmodified Amino Acid Fingerprints", fontsize=18, fontweight="bold", y=1.01)
    plt.tight_layout()

    out = OUTPUT_DIR / "ptm_fingerprints.png"
    plt.savefig(out, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out}")


def plot_stacked():
    """Save a stacked fingerprint plot of all 6 PTM amino acids."""
    n = len(PTM_CODES)
    fig, ax = plt.subplots(figsize=(12, max(6, n * 1.2)))

    wavelengths = None
    names = []
    for i, ptm_code in enumerate(PTM_CODES):
        wl, mean_spec = load_mean_spectrum(DATA_DIR / "PTM" / ptm_code)
        if wl is None:
            continue
        if ptm_code == "phos-Y":
            mean_spec = fix_baseline(wl, mean_spec.copy())
        if wavelengths is None:
            wavelengths = wl

        # Offset each spectrum vertically
        offset = i
        ax.plot(wavelengths, mean_spec + offset, color=COLORS[i], linewidth=2, label=ptm_code)
        names.append(ptm_code.upper())

    ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=14)
    ax.set_ylabel("")
    ax.set_title("SERS Fingerprints — PTM Amino Acids", fontsize=16, fontweight="bold")
    ax.set_xlim(wavelengths.min(), wavelengths.max())

    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(names, fontsize=12)
    ax.set_ylim(-0.5, n)
    ax.grid(True, axis="y", alpha=1.0)

    plt.tight_layout()

    out = OUTPUT_DIR / "ptm_fingerprints_stacked.png"
    plt.savefig(out, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Individual plots
    for ptm_code, base_code in PTM_PAIRS.items():
        plot_individual(ptm_code, base_code)

    # 2. Combined grid
    plot_combined_grid()

    # 3. Stacked PTM fingerprints
    plot_stacked()

    print("\nAll PTM fingerprint images generated successfully!")


if __name__ == "__main__":
    main()
