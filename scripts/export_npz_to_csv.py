"""Export a spectrum .npz file to CSV format.

Writes wavenumber and intensity columns matching the simulation CSV format.

Usage:
    python scripts/export_npz_to_csv.py \
        data/custom/processed/primary_magic/2-letter/SA/rep1/test_spectrum.npz

    python scripts/export_npz_to_csv.py \
        data/custom/processed/primary_magic/2-letter/SA/rep1/test_spectrum.npz \
        -o output.csv
"""

import argparse
from pathlib import Path

import numpy as np

from primarymagic.data.spectraio import load_from_npz


def main():
    parser = argparse.ArgumentParser(
        description="Export spectrum .npz to CSV"
    )
    parser.add_argument("npz", type=Path, help="Input .npz file")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output CSV path (default: same name with .csv extension)")
    args = parser.parse_args()

    out = args.output or args.npz.with_suffix('.csv')

    collection = load_from_npz(args.npz)
    wl = collection.wavelengths
    intensities = collection[0].intensities

    with open(out, 'w') as f:
        f.write("wavenumber,intensity\n")
        for w, i in zip(wl, intensities):
            f.write(f"{w:.1f},{i:.8f}\n")

    print(f"Wrote {len(wl)} points to {out}")


if __name__ == "__main__":
    main()
