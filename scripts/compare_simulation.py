"""Compare a simulation spectrum against a lab fingerprint.

Loads both spectra from .npz files, normalises to [0, 1], clips to the
common wavenumber range, and plots a comparison with Pearson correlation
and cosine similarity.

Usage:
    python scripts/compare_simulation.py \
        data/simulation/2-letter/GA/rep1/raman.npz \
        data/custom/processed/primary_magic/2-letter/GA/rep2/fingerprint.npz

    python scripts/compare_simulation.py \
        data/simulation/2-letter/GA/rep1/raman.npz \
        data/custom/processed/primary_magic/2-letter/GA/rep2/fingerprint.npz \
        -o results/GA_rep1_comparison.png

    # Shift the fingerprint by -16 steps before comparing:
    python scripts/compare_simulation.py \
        data/simulation/2-letter/GA/rep1/raman.npz \
        data/custom/processed/primary_magic/2-letter/GA/rep3/fingerprint.npz \
        --shift -16
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from primarymagic.data.spectraio import load_from_npz


def normalise(x: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]."""
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else x * 0.0


def compare(sim_path: Path, fp_path: Path, out_png: Path,
            shift: float = 0) -> None:
    """Load simulation and fingerprint .npz, normalise both, and plot."""
    sim = load_from_npz(sim_path)
    fp = load_from_npz(fp_path)

    if shift:
        wl_step = np.mean(np.diff(fp.wavelengths))
        fp = fp.shift(shift * wl_step)

    fp_wl = fp.wavelengths
    fp_int = normalise(fp[0].intensities)

    sim_wl = sim.wavelengths
    sim_raw = sim[0].intensities

    # Clip simulation to the fingerprint wavenumber range before normalising
    mask = (sim_wl >= fp_wl[0]) & (sim_wl <= fp_wl[-1])
    sim_wl = sim_wl[mask]
    sim_int = normalise(sim_raw[mask])

    # Interpolate simulation onto fingerprint wavelength grid for correlation
    sim_interp = np.interp(fp_wl, sim_wl, sim_int)
    pearson = np.corrcoef(fp_int, sim_interp)[0, 1]
    cosine = np.dot(fp_int, sim_interp) / (
        np.linalg.norm(fp_int) * np.linalg.norm(sim_interp)
    )
    print(f"Pearson correlation: {pearson:.4f}")
    print(f"Cosine similarity:   {cosine:.4f}")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(fp_wl, fp_int, label="Lab fingerprint", alpha=0.8)
    ax.plot(sim_wl, sim_int, label="Simulation", alpha=0.8)
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Normalised Intensity")
    ax.set_title(f"Simulation vs Lab Fingerprint  "
                 f"(Pearson={pearson:.3f}, Cosine={cosine:.3f})")
    ax.set_xlim(fp_wl[0], fp_wl[-1])
    ax.legend()
    plt.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.show()
    print(f"Saved {out_png}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare simulation spectrum against lab fingerprint"
    )
    parser.add_argument("simulation", type=Path,
                        help="Simulation .npz file (e.g. raman.npz)")
    parser.add_argument("fingerprint", type=Path,
                        help="Lab fingerprint .npz file")
    parser.add_argument("--shift", type=float, default=0,
                        help="Shift lab fingerprint by N steps (negative=left, positive=right)")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output PNG path (default: comparison.png next to simulation)")
    args = parser.parse_args()

    out = args.output or args.simulation.parent / "comparison.png"
    compare(args.simulation, args.fingerprint, out, shift=args.shift)


if __name__ == "__main__":
    main()
