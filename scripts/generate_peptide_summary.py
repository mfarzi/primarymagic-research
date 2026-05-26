"""
Generate peptide summary Excel file from processed data.

Analyzes {data-root}/{n-letter}/{SEQ}/{rep#}/ structure,
counting spectra in data.npz and clean_data.npz for each rep.

Sheets: Amino Acids, PTM Amino Acids, Dipeptides, Tripeptides,
        Tetrapeptides, Pentapeptides.

Each row: Sequence | Rep 1 | Rep 2 | Total
  - Rep columns show "clean / total" counts
  - Total is the sum of clean spectra across reps
  - Color coded by total clean: Red < 100, Yellow 100-199, Green >= 200

Usage:
    python scripts/generate_peptide_summary.py
    python scripts/generate_peptide_summary.py \
        --data-root data/custom/processed/magic \
        --output data/custom/processed/magic/peptide_summary.xlsx
"""

import argparse
import json
from pathlib import Path

import numpy as np
import xlsxwriter


def count_spectra(rep_dir):
    """Return (clean, total) spectra counts for a rep directory.

    Tries data.npz first (primary_magic layout), then falls back to
    metadata.json n_total (magic layout) for the total count.
    """
    total = 0
    clean = 0

    data_path = rep_dir / "data.npz"
    if data_path.exists():
        try:
            data = np.load(data_path, allow_pickle=True)
            if "intensities" in data.files:
                total = data["intensities"].shape[0]
        except Exception as e:
            print(f"Error loading {data_path}: {e}")
    else:
        # Magic pipeline stores total in metadata.json
        meta_path = rep_dir / "metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                total = meta.get("n_total", 0)
            except Exception as e:
                print(f"Error loading {meta_path}: {e}")

    clean_path = rep_dir / "clean_data.npz"
    if clean_path.exists():
        try:
            data = np.load(clean_path, allow_pickle=True)
            if "intensities" in data.files:
                clean = data["intensities"].shape[0]
        except Exception as e:
            print(f"Error loading {clean_path}: {e}")

    return clean, total


def discover_max_reps(base_dir):
    """Find the highest rep number across all sequences in a category."""
    import re
    base_dir = Path(base_dir)
    max_rep = 0
    rep_re = re.compile(r"^rep(\d+)$")
    if not base_dir.exists():
        return 0
    for seq_dir in base_dir.iterdir():
        if not seq_dir.is_dir():
            continue
        for d in seq_dir.iterdir():
            if d.is_dir():
                m = rep_re.match(d.name)
                if m:
                    max_rep = max(max_rep, int(m.group(1)))
    return max_rep


def analyze_category(base_dir):
    """Analyze a category directory (e.g., 2-letter/) for all sequences.

    Dynamically discovers rep directories up to the maximum rep found.

    Returns:
        (max_reps, results) where results is a list of dicts with keys:
        Sequence, rep{N}_clean, rep{N}_total for each rep, and Total.
    """
    base_dir = Path(base_dir)
    max_reps = discover_max_reps(base_dir)
    if max_reps == 0:
        max_reps = 1  # default to at least 1 column
    results = []

    for seq_dir in sorted(base_dir.iterdir()):
        if not seq_dir.is_dir():
            continue

        row = {"Sequence": seq_dir.name}
        total_clean = 0

        for rep_num in range(1, max_reps + 1):
            rep_dir = seq_dir / f"rep{rep_num}"
            if rep_dir.is_dir() and any(rep_dir.glob("*.npz")):
                clean, total = count_spectra(rep_dir)
                row[f"rep{rep_num}_clean"] = clean
                row[f"rep{rep_num}_total"] = total
                total_clean += clean
            else:
                row[f"rep{rep_num}_clean"] = None
                row[f"rep{rep_num}_total"] = None

        row["Total"] = total_clean
        results.append(row)

    # Sort by Total clean ascending
    results.sort(key=lambda r: r["Total"])
    return max_reps, results


def write_sheet(workbook, sheet_name, rows, max_reps, formats):
    """Write rows to an Excel worksheet with conditional formatting."""
    header_fmt = formats["header"]
    red_fmt = formats["red"]
    yellow_fmt = formats["yellow"]
    green_fmt = formats["green"]

    worksheet = workbook.add_worksheet(sheet_name)

    # Column widths
    worksheet.set_column(0, 0, 15)  # Sequence
    for i in range(max_reps):
        worksheet.set_column(1 + i, 1 + i, 18)  # Rep columns
    worksheet.set_column(1 + max_reps, 1 + max_reps, 12)  # Total

    # Headers
    headers = ["Sequence"] + [f"Rep {i}" for i in range(1, max_reps + 1)] + ["Total"]
    for col, h in enumerate(headers):
        worksheet.write(0, col, h, header_fmt)

    # Data rows
    for row_idx, row in enumerate(rows, start=1):
        total_clean = row["Total"]
        if total_clean < 100:
            fmt = red_fmt
        elif total_clean < 200:
            fmt = yellow_fmt
        else:
            fmt = green_fmt

        worksheet.write(row_idx, 0, row["Sequence"], fmt)

        for rep_num in range(1, max_reps + 1):
            col = rep_num  # col 0 is Sequence
            if row[f"rep{rep_num}_clean"] is not None:
                worksheet.write(
                    row_idx, col,
                    f"{row[f'rep{rep_num}_clean']} / {row[f'rep{rep_num}_total']}",
                    fmt,
                )
            else:
                worksheet.write(row_idx, col, "", fmt)

        # Total clean
        worksheet.write(row_idx, 1 + max_reps, total_clean, fmt)

    return worksheet


def parse_args():
    _default_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Generate peptide summary Excel file from processed data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root", type=Path,
        default=_default_root / "data" / "processed" / "primary_magic",
        help="Path to processed data directory (containing 1-letter/, 2-letter/, etc.)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output Excel file path. Default: <data-root>/peptide_summary.xlsx",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    processed = args.data_root
    output_path = args.output if args.output else processed / "peptide_summary.xlsx"

    # Categories: (sheet_name, directory, description)
    categories = [
        ("Amino Acids",     processed / "1-letter",       "amino acids"),
        ("PTM Amino Acids", processed / "1-letter" / "PTM", "PTM amino acids"),
        ("Dipeptides",      processed / "2-letter",       "dipeptides"),
        ("Tripeptides",     processed / "3-letter",       "tripeptides"),
        ("Tetrapeptides",   processed / "4-letter",       "tetrapeptides"),
        ("Pentapeptides",   processed / "5-letter",       "pentapeptides"),
    ]

    # For amino acids, exclude the PTM subdirectory
    all_results = {}  # sheet_name -> (max_reps, rows)
    for sheet_name, cat_dir, desc in categories:
        if not cat_dir.exists():
            print(f"WARNING: {cat_dir} does not exist, skipping {desc}.")
            continue

        print(f"Analyzing {desc}...")
        max_reps, rows = analyze_category(cat_dir)

        # For amino acids sheet, filter out PTM directory
        if sheet_name == "Amino Acids":
            rows = [r for r in rows if r["Sequence"] != "PTM"]

        all_results[sheet_name] = (max_reps, rows)

    # Print summaries
    for sheet_name, (max_reps, rows) in all_results.items():
        total_clean = sum(r["Total"] for r in rows)
        print(f"\n{'=' * 60}")
        print(f"{sheet_name} Summary (sorted by Total clean, low to high)")
        print(f"{'=' * 60}")
        for r in rows:
            parts = []
            for rep_num in range(1, max_reps + 1):
                if r[f"rep{rep_num}_clean"] is not None:
                    parts.append(
                        f"Rep{rep_num}: "
                        f"{r[f'rep{rep_num}_clean']}/{r[f'rep{rep_num}_total']}"
                    )
                else:
                    parts.append(f"Rep{rep_num}: {'-':>10s}")
            reps_str = "  ".join(f"{p:>18s}" for p in parts)
            print(f"  {r['Sequence']:>12s}  {reps_str}  Total: {r['Total']}")
        print(f"\n  Count: {len(rows)}  |  Total clean: {total_clean}")

    # Write Excel
    workbook = xlsxwriter.Workbook(str(output_path))
    formats = {
        "header": workbook.add_format({
            "bold": True,
            "bg_color": "#4472C4",
            "font_color": "white",
            "border": 1,
        }),
        "red":    workbook.add_format({"bg_color": "#FF6B6B", "border": 1}),
        "yellow": workbook.add_format({"bg_color": "#FFD93D", "border": 1}),
        "green":  workbook.add_format({"bg_color": "#6BCB77", "border": 1}),
    }

    for sheet_name, (max_reps, rows) in all_results.items():
        write_sheet(workbook, sheet_name, rows, max_reps, formats)

    workbook.close()
    print(f"\nExcel file saved to: {output_path}")

    # Color category summaries
    for sheet_name, (_, rows) in all_results.items():
        totals = [r["Total"] for r in rows]
        red = sum(1 for t in totals if t < 100)
        yellow = sum(1 for t in totals if 100 <= t < 200)
        green = sum(1 for t in totals if t >= 200)
        print(f"\n{sheet_name} color categories:")
        print(f"  Red (< 100): {red}")
        print(f"  Yellow (100-199): {yellow}")
        print(f"  Green (>= 200): {green}")


if __name__ == "__main__":
    main()
