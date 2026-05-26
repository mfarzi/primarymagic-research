"""Generate sample folders for sequence-based analysis.

Given a peptide sequence (e.g., "FAR"), creates numbered sample folders
containing randomly selected spectra for each prefix length, drawn from
the corresponding clean_data.npz files in the processed data directory.

Example usage:
    python scripts/generate_sequence_samples.py --sequence FAR --num-samples 3
    python scripts/generate_sequence_samples.py --sequence FA --num-samples 1 --spectra-per-file 5
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

# Add project root to path so we can import spectra package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from primarymagic.data.spectra_dataset import SpectraDataset


def find_next_sample_number(output_dir: Path, sequence: str) -> int:
    """Find the next available sample number for a sequence.

    Scans existing {SEQ}-sample## folders and returns the next number.
    """
    pattern = re.compile(rf"^{re.escape(sequence)}-sample(\d+)$")
    max_num = 0
    if output_dir.exists():
        for folder in output_dir.iterdir():
            if folder.is_dir():
                match = pattern.match(folder.name)
                if match:
                    max_num = max(max_num, int(match.group(1)))
    return max_num + 1


def main():
    parser = argparse.ArgumentParser(
        description="Generate sample folders for sequence-based analysis."
    )
    parser.add_argument(
        "--sequence", type=str, required=True,
        help="Peptide sequence (e.g., 'FAR')"
    )
    parser.add_argument(
        "--num-samples", type=int, default=1,
        help="Number of sample folders to generate (default: 1)"
    )
    parser.add_argument(
        "--spectra-per-file", type=int, default=1,
        help="Number of random spectra per npz file (default: 1)"
    )
    parser.add_argument(
        "--data-root", type=str, default="data/processed/primary_magic",
        help="Path to processed data (default: data/processed/primary_magic)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/sequences",
        help="Output directory (default: data/sequences)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    sequence = args.sequence.upper()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    rng = np.random.default_rng(args.seed)

    # Load data
    print(f"Loading spectra from {data_root} ...")
    dataset = SpectraDataset(data_root, npz_filename="raw_data.npz")
    print(f"  {dataset}")

    # Validate that all required sub-sequences exist
    missing = []
    for k in range(1, len(sequence) + 1):
        prefix = sequence[:k]
        spectra = dataset.get_spectra(prefix)
        if spectra is None:
            missing.append(prefix)
        else:
            print(f"  '{prefix}': {len(spectra)} spectra available")

    if missing:
        print(f"\nError: Missing spectra for sub-sequences: {missing}")
        sys.exit(1)

    # Find next available sample number
    start_num = find_next_sample_number(output_dir, sequence)
    print(f"\nGenerating {args.num_samples} sample(s) starting from {sequence}-sample{start_num:02d}")

    # Generate samples
    for i in range(args.num_samples):
        sample_num = start_num + i
        sample_name = f"{sequence}-sample{sample_num:02d}"
        sample_dir = output_dir / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)

        for k in range(1, len(sequence) + 1):
            prefix = sequence[:k]
            all_spectra = dataset.get_spectra(prefix)

            # Randomly select spectra (with replacement if needed)
            n_available = len(all_spectra)
            replace = args.spectra_per_file > n_available
            indices = rng.choice(n_available, size=args.spectra_per_file, replace=replace)
            selected = all_spectra[indices]

            # Save as k-letter.npz
            out_path = sample_dir / f"{k}-letter.npz"
            np.savez(out_path, wavelengths=dataset.wavelengths, intensities=selected)

        print(f"  Created {sample_dir}")

    # Summary
    print(f"\nDone. Generated {args.num_samples} sample folder(s) in {output_dir}/")
    for k in range(1, len(sequence) + 1):
        prefix = sequence[:k]
        print(f"  {k}-letter.npz <- '{prefix}'")


if __name__ == "__main__":
    main()
