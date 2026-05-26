"""Shift spectra signal by a given number of steps.

Resamples intensities onto the original wavelength grid so that the
signal moves left (negative steps) or right (positive steps) while
keeping wavelength values unchanged.

Backs up the original files as *_noshift.npz before overwriting.

Usage:
    # Shift GA left by 8 steps:
    python scripts/shift_spectra.py \
        data/OTS/processed/primary_magic/2-letter/GA/rep1 \
        --steps -8

    # Shift SF right by 8 steps:
    python scripts/shift_spectra.py \
        data/OTS/processed/primary_magic/2-letter/SF/rep1 \
        --steps 8

    # Shift only fingerprint:
    python scripts/shift_spectra.py \
        data/OTS/processed/primary_magic/2-letter/GA/rep1 \
        --steps -8 --files fingerprint
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
from primarymagic.data.spectraio import load_from_npz, export_to_npz


def main():
    parser = argparse.ArgumentParser(description="Shift spectra signal by N steps.")
    parser.add_argument("rep_dir", type=Path,
                        help="Path to the rep directory containing .npz files")
    parser.add_argument("--steps", type=float, required=True,
                        help="Number of steps to shift (negative=left, positive=right)")
    parser.add_argument("--files", type=str, nargs="+",
                        default=["clean_data", "fingerprint"],
                        help="Which .npz files to shift (default: clean_data fingerprint)")
    parser.add_argument("--no-backup", action="store_true",
                        help="Skip creating *_noshift.npz backups")
    args = parser.parse_args()

    rep_dir = args.rep_dir
    if not rep_dir.exists():
        print(f"ERROR: {rep_dir} does not exist")
        return

    step_cm = None

    for name in args.files:
        src = rep_dir / f"{name}.npz"
        if not src.exists():
            print(f"  {name}.npz not found, skipping")
            continue

        # Backup
        if not args.no_backup:
            backup = rep_dir / f"{name}_noshift.npz"
            if backup.exists():
                # Load from existing backup (original data)
                collection = load_from_npz(backup)
                print(f"  Loaded from {name}_noshift.npz (existing backup)")
            else:
                shutil.copy2(src, backup)
                collection = load_from_npz(src)
                print(f"  Backed up {name}.npz -> {name}_noshift.npz")
        else:
            collection = load_from_npz(src)

        if step_cm is None:
            wl_step = np.mean(np.diff(collection.wavelengths))
            step_cm = args.steps * wl_step
            print(f"  Step size: {wl_step:.2f} cm-1, total shift: {step_cm:.2f} cm-1")

        shifted = collection.shift(step_cm)
        export_to_npz(shifted, src)
        print(f"  {name}.npz shifted by {args.steps} steps ({step_cm:.2f} cm-1)")

    print("Done")


if __name__ == "__main__":
    main()
