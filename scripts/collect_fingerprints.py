"""
Script to collect all fingerprint.png files from data/processed/oprl
and copy them to a centralized fingerprints folder with descriptive names.
"""

import shutil
from pathlib import Path


def collect_fingerprints(base_dir: Path, output_dir: Path) -> None:
    """
    Find all fingerprint.png files in base_dir and copy them to output_dir
    with the parent folder name (amino acid/peptide sequence) as filename.

    Args:
        base_dir: Root directory to search for fingerprint.png files
        output_dir: Directory to copy fingerprints to
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all fingerprint.png files
    fingerprint_files = list(base_dir.rglob("fingerprint.png"))

    print(f"Found {len(fingerprint_files)} fingerprint files")

    for fp_file in fingerprint_files:
        # Get the parent folder name (amino acid or peptide sequence)
        sequence_name = fp_file.parent.name

        # Create new filename with sequence name
        new_filename = f"{sequence_name}.png"
        dest_path = output_dir / new_filename

        # Copy the file
        shutil.copy2(fp_file, dest_path)
        print(f"Copied: {fp_file.relative_to(base_dir)} -> {new_filename}")

    print(f"\nAll fingerprints copied to: {output_dir}")


if __name__ == "__main__":
    # Define paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    mode='dipeptides'
    base_dir = project_root / "data" / "processed" / "orpl" / mode
    output_dir = base_dir / "fingerprints" / mode

    collect_fingerprints(base_dir, output_dir)
