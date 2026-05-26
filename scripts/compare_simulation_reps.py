"""Compare all simulation reps against a lab fingerprint.

Loads all simulation reps from a directory, optionally shifts the lab
fingerprint, normalises both to [0, 1], computes Pearson correlation,
and plots comparisons at multiple wavenumber ranges.

Usage:
    python scripts/compare_simulation_reps.py \
        data/simulation/2-letter/GA \
        data/custom/processed/primary_magic/2-letter/GA/rep3/fingerprint.npz

    python scripts/compare_simulation_reps.py \
        data/simulation/2-letter/GA \
        data/custom/processed/primary_magic/2-letter/GA/rep3/fingerprint.npz \
        --shift -16

    python scripts/compare_simulation_reps.py \
        data/simulation/2-letter/GA \
        data/custom/processed/primary_magic/2-letter/GA/rep3/fingerprint.npz \
        --shift -16 --ranges 1000-1500 750-1000

    # Per-rep shifts (applied to simulation reps instead of lab fingerprint):
    python scripts/compare_simulation_reps.py \
        data/simulation/2-letter/SA \
        data/custom/processed/primary_magic/2-letter/SA/rep1/fingerprint.npz \
        --rep-shifts rep1=46 rep2=40 rep3=50
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from primarymagic.data.spectraio import load_from_npz


def normalise(x: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]."""
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else x * 0.0


def parse_range(s: str) -> tuple[float, float]:
    """Parse a 'low-high' string into a tuple."""
    lo, hi = s.split("-")
    return float(lo), float(hi)


def main():
    parser = argparse.ArgumentParser(
        description="Compare all simulation reps against a lab fingerprint"
    )
    parser.add_argument("sim_dir", type=Path,
                        help="Directory containing rep1/, rep2/, ... with raman.npz")
    parser.add_argument("fingerprint", type=Path,
                        help="Lab fingerprint .npz file")
    parser.add_argument("--shift", type=float, default=0,
                        help="Shift lab fingerprint by N steps (negative=left, positive=right)")
    parser.add_argument("--rep-shifts", nargs="*", type=str, default=None,
                        help="Per-rep shifts applied to simulation (e.g. rep1=46 rep2=40 rep3=50)")
    parser.add_argument("--ranges", nargs="*", type=str, default=None,
                        help="Additional wavenumber ranges to plot (e.g. 1000-1500 750-1000)")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output directory for plots (default: sim_dir)")
    args = parser.parse_args()

    # Parse per-rep shifts
    rep_shift_map = {}
    if args.rep_shifts:
        for item in args.rep_shifts:
            name, val = item.split("=")
            rep_shift_map[name] = float(val)

    out_dir = args.output or args.sim_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load and optionally shift lab fingerprint
    cust = load_from_npz(args.fingerprint)
    if args.shift:
        wl_step = np.mean(np.diff(cust.wavelengths))
        cust = cust.shift(args.shift * wl_step)

    cust_wl = cust.wavelengths
    cust_int = normalise(cust[0].intensities)

    # Discover and load simulation reps
    rep_dirs = sorted(args.sim_dir.glob("rep*/raman.npz"))
    if not rep_dirs:
        print(f"No raman.npz found in {args.sim_dir}/rep*/")
        return

    sims = {}
    for npz_path in rep_dirs:
        rep_name = npz_path.parent.name
        c = load_from_npz(npz_path)

        # Apply per-rep shift to simulation (use lab wl_step for consistent step units)
        if rep_name in rep_shift_map:
            lab_wl_step = np.mean(np.diff(cust_wl))
            c = c.shift(rep_shift_map[rep_name] * lab_wl_step)

        wl = c.wavelengths
        mask = (wl >= cust_wl[0]) & (wl <= cust_wl[-1])
        sim_wl = wl[mask]
        sim_int = normalise(c[0].intensities[mask])
        # Compute correlation
        sim_interp = np.interp(cust_wl, sim_wl, sim_int)
        r, _ = pearsonr(cust_int, sim_interp)
        shift_info = f" (shift {rep_shift_map[rep_name]:+g})" if rep_name in rep_shift_map else ""
        print(f"{rep_name}{shift_info}: Pearson r = {r:.4f}")

        sims[rep_name] = (sim_wl, sim_int, r)

    # Build list of wavenumber ranges to plot
    shift_tag = f"_shift{int(args.shift)}" if args.shift else ""
    if rep_shift_map:
        shift_tag = "_repshifts"
    fp_label = "Lab fingerprint" + (f" (shift {int(args.shift)})" if args.shift else "")

    plot_ranges = [(None, "full")]
    if args.ranges:
        for r in args.ranges:
            lo, hi = parse_range(r)
            plot_ranges.append(((lo, hi), f"{int(lo)}_{int(hi)}"))

    for xlim, suffix in plot_ranges:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(cust_wl, cust_int, "k-", lw=1.5, label=fp_label, alpha=0.9)
        for rep_name, (wl, intensities, r) in sims.items():
            ax.plot(wl, intensities, label=f"{rep_name} (r={r:.3f})", alpha=0.7, lw=1.0)
        ax.set_xlabel("Wavenumber (cm⁻¹)")
        ax.set_ylabel("Normalised Intensity")
        if xlim:
            ax.set_xlim(*xlim)
            ax.set_title(f"Simulation vs Lab ({xlim[0]:.0f}-{xlim[1]:.0f} cm⁻¹)")
        else:
            ax.set_title("Simulation vs Lab (full range)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        out_file = out_dir / f"comparison{shift_tag}_{suffix}.png"
        plt.savefig(out_file, dpi=150)
        plt.close(fig)
        print(f"Saved {out_file}")


if __name__ == "__main__":
    main()
