"""Convert simulation CSV to Raman TXT and NPZ formats.

Reads the simulation CSV, normalises intensities to [0, 1],
and writes both a tab-separated raman.txt (spectraio-compatible)
and a raman.npz (SpectraCollection format).

Usage:
    python scripts/convert_simulation.py data/simulation/2-letter/GA/rep1/GA_crystal_sp_raman.csv
    python scripts/convert_simulation.py data/simulation/2-letter/GA/rep1/GA_crystal_sp_raman.csv -o output_dir/
"""

import argparse
from pathlib import Path

import numpy as np

from primarymagic.data.spectraio import export_to_npz, read_spectrum_file


def convert_csv(csv_path: Path, out_dir: Path) -> None:
    """Read simulation CSV and write normalised raman.txt and raman.npz."""
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    wavenumbers = data[:, 0]
    intensities = data[:, 1]

    # Min-max normalise to [0, 1]
    i_min, i_max = intensities.min(), intensities.max()
    if i_max > i_min:
        intensities = (intensities - i_min) / (i_max - i_min)

    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / "raman.txt"
    npz_path = out_dir / "raman.npz"

    # Write TXT in descending wavenumber order (spectraio reverses on read)
    with open(txt_path, "w") as f:
        f.write("#Wave\t\t#Intensity\n")
        for w, i in zip(wavenumbers[::-1], intensities[::-1]):
            f.write(f"{w}\t{i:.8f}\n")

    print(f"Wrote {len(wavenumbers)} points to {txt_path}")

    # Write NPZ via spectraio round-trip
    collection = read_spectrum_file(txt_path)
    export_to_npz(collection, npz_path)
    print(f"Wrote {npz_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert simulation CSV to raman.txt and raman.npz"
    )
    parser.add_argument("csv", type=Path, help="Input simulation CSV file")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output directory (default: same dir as CSV)")
    args = parser.parse_args()

    out_dir = args.output or args.csv.parent
    convert_csv(args.csv, out_dir)


if __name__ == "__main__":
    main()
