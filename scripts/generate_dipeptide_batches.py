"""Generate dipeptide batches for data collection.

Creates 4 batches of dipeptides from 20 standard amino acids,
excluding dipeptides that have already been collected.

Output: data/dipeptide_batches.xlsx with 4 sheets (batch1..batch4).
"""

import random
from pathlib import Path

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from primarymagic.models.dataset import (
    AMINO_ACID_CODES,
    AMINO_ACID_THREE_LETTER,
    AMINO_ACIDS,
)

# Standard 20 amino acids (exclude pyrrolysine=O and selenocysteine=U)
EXCLUDE_CODES = {"O", "U"}
STANDARD_AA = [
    aa for aa in AMINO_ACIDS
    if AMINO_ACID_CODES[aa] not in EXCLUDE_CODES
]
STANDARD_CODES = [AMINO_ACID_CODES[aa] for aa in STANDARD_AA]

NUM_BATCHES = 4
GROUP_SIZE = len(STANDARD_CODES) // NUM_BATCHES  # 5


def get_existing_dipeptides(data_dir: Path) -> set[str]:
    """Read existing dipeptide folder names from data/custom/raw/2-letter/."""
    two_letter_dir = data_dir / "custom" / "raw" / "2-letter"
    if not two_letter_dir.exists():
        print(f"Warning: {two_letter_dir} not found, assuming no existing dipeptides.")
        return set()
    return {d.name for d in two_letter_dir.iterdir() if d.is_dir() and len(d.name) == 2}


def generate_batches(seed: int = 42) -> dict[int, list[str]]:
    """Generate 4 batches of dipeptides.

    For each of the 20 second amino acids (in fixed order):
      - Shuffle the 20 first amino acids randomly (second letter fixed)
      - Split shuffled first amino acids into 4 groups of 5
      - Group i -> batch i (each group has random first + fixed second)
    """
    rng = random.Random(seed)
    batches: dict[int, list[str]] = {i: [] for i in range(NUM_BATCHES)}

    for second_aa in STANDARD_CODES:
        first_aas = list(STANDARD_CODES)
        rng.shuffle(first_aas)
        for i in range(NUM_BATCHES):
            group = first_aas[i * GROUP_SIZE : (i + 1) * GROUP_SIZE]
            for first_aa in group:
                batches[i].append(first_aa + second_aa)

    return batches


def code_to_name(code: str) -> str:
    """Convert 1-letter code to full amino acid name."""
    code_to_aa = {v: k for k, v in AMINO_ACID_CODES.items()}
    return code_to_aa[code]


def make_dataframe(dipeptides: list[str]) -> pd.DataFrame:
    """Create a DataFrame with 1-letter, 3-letter, and full name columns."""
    rows = []
    for pair in sorted(dipeptides, key=lambda p: (p[1], p[0])):
        c1, c2 = pair[0], pair[1]
        name1 = code_to_name(c1)
        name2 = code_to_name(c2)
        three1 = AMINO_ACID_THREE_LETTER[name1]
        three2 = AMINO_ACID_THREE_LETTER[name2]
        rows.append({
            "1-letter": pair,
            "3-letter": f"{three1}-{three2}",
            "full name": f"{name1}-{name2}",
        })
    return pd.DataFrame(rows)


def main():
    data_dir = PROJECT_ROOT / "data"
    existing = get_existing_dipeptides(data_dir)
    print(f"Existing dipeptides ({len(existing)}): {sorted(existing)}")

    batches = generate_batches()

    # Verify all 400 unique pairs are covered
    all_dipeptides = set()
    for dipeptides in batches.values():
        all_dipeptides.update(dipeptides)
    assert len(all_dipeptides) == 400, f"Expected 400 unique pairs, got {len(all_dipeptides)}"
    print(f"Total unique dipeptides: {len(all_dipeptides)}")

    # Remove existing dipeptides from each batch
    filtered_batches = {}
    for i, dipeptides in batches.items():
        filtered = [d for d in dipeptides if d not in existing]
        filtered_batches[i] = filtered
        removed = len(dipeptides) - len(filtered)
        print(f"Batch {i+1}: {len(dipeptides)} -> {len(filtered)} ({removed} removed)")

    # Verify no existing dipeptide remains
    for i, dipeptides in filtered_batches.items():
        overlap = set(dipeptides) & existing
        assert not overlap, f"Batch {i+1} still contains existing dipeptides: {overlap}"

    # Save to Excel
    output_path = data_dir / "dipeptide_batches.xlsx"
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for i in range(NUM_BATCHES):
            df = make_dataframe(filtered_batches[i])
            df.to_excel(writer, sheet_name=f"batch{i+1}", index=False)

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
