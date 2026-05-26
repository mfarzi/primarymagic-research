"""Copy peptide data from Google Drive to a local directory.

Organises data into: {n}-letter/{SEQUENCE}/rep{#}/
- Groups by peptide length (2-letter, 3-letter, etc.)
- Within each group, organises by sequence (e.g., AD, ADF, DSAF)
- If the source has rep subdirectories (e.g., "SG rep1", "SG rep2"),
  each is mapped to rep1, rep2, etc.
- If no rep subdirectories exist, all files go into rep1.

Usage:
    # Default (copies all lengths from 4. Peptides -> data/raw):
    python copy_peptide_data.py

    # Custom source/dest, single length:
    python copy_peptide_data.py \
        --src "G:/.../5. OTS peptide data" \
        --dst "data/OTS/processed/primary_magic" \
        --lengths 2
"""

import argparse
import re
import shutil
from pathlib import Path

DEFAULT_SRC = Path(
    r"G:\.shortcut-targets-by-id\1TZnijGfU2d4nK24bamqJIjmEvbfqYao5"
    r"\Spectra Data Files\4. Peptides"
)
DEFAULT_DST = Path(r"C:\Users\mfarzi\mycodes\visiogen\spectra\data\raw")

# Map source folder names to (n-letter, sequence)
ALL_LENGTH_DIRS = {
    2: ("2 letter peptides", "2-letter"),
    3: ("3 letter peptides", "3-letter"),
    4: ("4 letter peptides", "4-letter"),
    5: ("5 letter peptides", "5-letter"),
}

# Regex to extract rep number from directory names like "SG rep1", "ARS rep2"
REP_RE = re.compile(r"^.+\s+rep(\d+)$", re.IGNORECASE)


def clean_sequence_name(dirname: str) -> str:
    """Extract the pure sequence letters from a directory name.

    Handles cases like:
        "RG (1)"                  -> "RG"
        "RGA - could not spec map" -> "RGA"
    """
    # Take only leading uppercase letters
    match = re.match(r"^([A-Z]+)", dirname)
    return match.group(1) if match else dirname


def copy_tree(src_dir: Path, dst_dir: Path) -> tuple[int, int]:
    """Copy new files from src_dir to dst_dir. Returns (copied, skipped)."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0
    for f in src_dir.iterdir():
        if f.is_file():
            dest = dst_dir / f.name
            if dest.exists():
                skipped += 1
                continue
            shutil.copy2(f, dest)
            copied += 1
    return copied, skipped


def parse_args():
    parser = argparse.ArgumentParser(description="Copy peptide data.")
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC,
                        help="Source root directory")
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST,
                        help="Destination root directory")
    parser.add_argument("--lengths", type=int, nargs="+", default=[2, 3, 4, 5],
                        help="Which peptide lengths to copy (default: 2 3 4 5)")
    return parser.parse_args()


def main():
    args = parse_args()
    src_root = args.src
    dst_root = args.dst
    length_dirs = {
        n: ALL_LENGTH_DIRS[n] for n in args.lengths if n in ALL_LENGTH_DIRS
    }

    total_copied = 0
    total_skipped = 0
    total_sequences = 0

    for n, (src_length_name, dst_length_name) in length_dirs.items():
        src_length_dir = src_root / src_length_name
        if not src_length_dir.exists():
            print(f"WARNING: {src_length_dir} does not exist, skipping.")
            continue

        for seq_dir in sorted(src_length_dir.iterdir()):
            if not seq_dir.is_dir():
                continue

            seq_name = clean_sequence_name(seq_dir.name)

            # Check for rep subdirectories
            rep_dirs = [
                d for d in seq_dir.iterdir()
                if d.is_dir() and REP_RE.match(d.name)
            ]

            if rep_dirs:
                # Has explicit rep subdirectories
                for rep_dir in sorted(rep_dirs):
                    m = REP_RE.match(rep_dir.name)
                    rep_num = int(m.group(1))
                    dst = dst_root / dst_length_name / seq_name / f"rep{rep_num}"
                    copied, skipped = copy_tree(rep_dir, dst)
                    total_copied += copied
                    total_skipped += skipped
                    if copied:
                        print(f"  {dst_length_name}/{seq_name}/rep{rep_num}: "
                              f"{copied} new files"
                              + (f" ({skipped} skipped)" if skipped else ""))
                    elif skipped:
                        print(f"  {dst_length_name}/{seq_name}/rep{rep_num}: "
                              f"up to date ({skipped} skipped)")
                total_sequences += 1
            else:
                # No rep directories — check if there are any files
                files = [f for f in seq_dir.iterdir() if f.is_file()]
                if not files:
                    print(f"  SKIP {dst_length_name}/{seq_name}: "
                          f"empty directory")
                    continue
                dst = dst_root / dst_length_name / seq_name / "rep1"
                copied, skipped = copy_tree(seq_dir, dst)
                total_copied += copied
                total_skipped += skipped
                total_sequences += 1
                if copied:
                    print(f"  {dst_length_name}/{seq_name}/rep1: "
                          f"{copied} new files"
                          + (f" ({skipped} skipped)" if skipped else ""))
                elif skipped:
                    print(f"  {dst_length_name}/{seq_name}/rep1: "
                          f"up to date ({skipped} skipped)")

    print(f"\nDone: {total_copied} new files copied, {total_skipped} skipped "
          f"(already existed) across {total_sequences} sequences.")


if __name__ == "__main__":
    main()
