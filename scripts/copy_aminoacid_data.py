"""Copy single amino acid data from Google Drive to local raw2 directory.

Organises data into: data/raw2/1-letter/{X}/rep1/ for unmodified amino acids
and data/raw2/1-letter/PTM/{short-code}/rep1/ for PTM amino acids.

The one-letter code is extracted from the directory name parentheses,
e.g. "L-Alanine (A)" -> "A", "N-acetyl-DL-serine (acetyl-S)" -> "acetyl-S".
"""

import re
import shutil
from pathlib import Path

from primarymagic.models.dataset import AMINO_ACID_CODES

SRC = Path(
    r"G:\.shortcut-targets-by-id\1TZnijGfU2d4nK24bamqJIjmEvbfqYao5"
    r"\Spectra Data Files\2. Single amino acids"
)
DST = Path(r"C:\Users\mfarzi\mycodes\visiogen\spectra\data\raw\1-letter")

# Build reverse lookup: full name (lowercase) -> one-letter code
FULL_NAME_TO_CODE = {name: code for name, code in AMINO_ACID_CODES.items()}

# Regex to extract the short code from parentheses, e.g. "(A)" or "(acetyl-S)"
PARENS_RE = re.compile(r"\(([^)]+)\)$")


def extract_code(dirname: str) -> str | None:
    """Extract the short code from a directory name like 'L-Alanine (A)'."""
    m = PARENS_RE.search(dirname.strip())
    return m.group(1) if m else None


def is_unmodified(code: str) -> bool:
    """Check if a code is a standard unmodified amino acid one-letter code."""
    return code in FULL_NAME_TO_CODE.values()


def copy_files(src_dir: Path, dst_dir: Path) -> tuple[int, int]:
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


def process_category(category_dir: Path, is_ptm: bool) -> tuple[int, int, int]:
    """Process one category (Unmodified or PTM).

    Returns (copied, skipped, sequences).
    """
    total_copied = 0
    total_skipped = 0
    total_sequences = 0

    if not category_dir.exists():
        print(f"WARNING: {category_dir} does not exist, skipping.")
        return 0, 0, 0

    for aa_dir in sorted(category_dir.iterdir()):
        if not aa_dir.is_dir():
            continue

        code = extract_code(aa_dir.name)
        if code is None:
            print(f"  SKIP {aa_dir.name}: cannot extract code from name")
            continue

        # Check for rep subdirectories (pattern: anything ending with "rep<N>")
        rep_re = re.compile(r"^.+\s+rep(\d+)$", re.IGNORECASE)
        rep_dirs = [
            d for d in aa_dir.iterdir()
            if d.is_dir() and rep_re.match(d.name)
        ]

        if is_ptm:
            base_dst = DST / "PTM" / code
        else:
            base_dst = DST / code

        if rep_dirs:
            for rep_dir in sorted(rep_dirs):
                m = rep_re.match(rep_dir.name)
                rep_num = int(m.group(1))
                dst = base_dst / f"rep{rep_num}"
                copied, skipped = copy_files(rep_dir, dst)
                total_copied += copied
                total_skipped += skipped
                label = f"PTM/{code}" if is_ptm else code
                if copied:
                    print(f"  {label}/rep{rep_num}: {copied} new files"
                          + (f" ({skipped} skipped)" if skipped else ""))
                elif skipped:
                    print(f"  {label}/rep{rep_num}: "
                          f"up to date ({skipped} skipped)")
        else:
            # No rep directories — all files go to rep1
            files = [f for f in aa_dir.iterdir() if f.is_file()]
            if not files:
                print(f"  SKIP {code}: empty directory")
                continue
            dst = base_dst / "rep1"
            copied, skipped = copy_files(aa_dir, dst)
            total_copied += copied
            total_skipped += skipped
            label = f"PTM/{code}" if is_ptm else code
            if copied:
                print(f"  {label}/rep1: {copied} new files"
                      + (f" ({skipped} skipped)" if skipped else ""))
            elif skipped:
                print(f"  {label}/rep1: up to date ({skipped} skipped)")

        total_sequences += 1

    return total_copied, total_skipped, total_sequences


def main():
    print("Copying unmodified amino acids...")
    c1, k1, s1 = process_category(
        SRC / "Unmodified Amino Acids", is_ptm=False
    )

    print("\nCopying PTM amino acids...")
    c2, k2, s2 = process_category(
        SRC / "Post-Translationally Modified (PTM) Amino Acids", is_ptm=True
    )

    total_copied = c1 + c2
    total_skipped = k1 + k2
    total_sequences = s1 + s2
    print(f"\nDone: {total_copied} new files copied, {total_skipped} skipped "
          f"(already existed) across {total_sequences} amino acids "
          f"({s1} unmodified, {s2} PTM).")


if __name__ == "__main__":
    main()
