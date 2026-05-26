"""Apply manual spectral shifts (from config/shifted_sequences.csv) to a target dataset.

The CSV at ``config/shifted_sequences.csv`` lists rep directories that need a
known integer-sample lag applied (typically a calibration drift detected in a
historical processing run; see commit `fa63f70` for how it was captured).

For each row in the CSV, this script:
1. Loads ``<target>/<rep_path>/<stem>.npz`` (default stem: ``clean_data``).
2. If a ``<stem>_noshift.npz`` backup already exists in that directory, the
   shifted version is recomputed from the backup (idempotent — running the
   script twice gives the same result as running it once).
3. Otherwise, copies the current file to ``<stem>_noshift.npz`` as a backup.
4. Shifts the loaded SpectraCollection by ``lag_samples * wl_step`` wavenumber
   units (interpolation onto the original wavelength grid; same logic as
   ``scripts/shift_spectra.py``).
5. Writes the shifted result back to ``<stem>.npz``.

Usage:
    # Dry run (just report what would happen)
    uv run scripts/replay_shifts.py \\
        --target /path/to/processed/magic_bayes \\
        --dry-run

    # Apply shifts to clean_data.npz + fingerprint.npz (defaults)
    uv run scripts/replay_shifts.py \\
        --target /path/to/processed/magic_bayes

    # Custom CSV / custom stems
    uv run scripts/replay_shifts.py \\
        --target /path/to/processed/magic_bayes \\
        --csv config/shifted_sequences.csv \\
        --files clean_data fingerprint raw_data

To produce or refresh ``config/shifted_sequences.csv``, see commit `fa63f70`
which derived it from cross-correlation against historical
``*_noshift.npz`` backups.
"""

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np

from primarymagic.data.spectraio import load_from_npz, export_to_npz


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "config" / "shifted_sequences.csv"


def load_lags_from_csv(csv_path: Path):
    """Read shift records from CSV. Returns list of (rep_path, lag_samples, lag_cm1)."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rep_path = r["rep_path"].replace("\\", "/")
            lag_samples = int(r["lag_samples"])
            lag_cm1 = float(r["lag_cm1"])
            rows.append((rep_path, lag_samples, lag_cm1))
    return rows


def apply_shift(target_rep: Path, stem: str, lag_samples: int) -> str:
    """Apply integer-sample shift to ``<stem>.npz`` in ``target_rep``.

    Idempotent: if a ``<stem>_noshift.npz`` backup already exists, we reload
    from it before applying the shift, so the result is the same regardless of
    how many times this is run.

    Returns one of:
        'applied'        — file shifted and written back.
        'skipped-missing'— ``<stem>.npz`` not present in target_rep.
        'skipped-zero'   — lag is 0; nothing to do.
    """
    src = target_rep / f"{stem}.npz"
    if not src.exists():
        return "skipped-missing"
    if lag_samples == 0:
        return "skipped-zero"
    backup = target_rep / f"{stem}_noshift.npz"
    if backup.exists():
        collection = load_from_npz(backup)
    else:
        shutil.copy2(src, backup)
        collection = load_from_npz(src)
    wl_step = float(np.mean(np.diff(collection.wavelengths)))
    shifted = collection.shift(lag_samples * wl_step)
    export_to_npz(shifted, src)
    return "applied"


def main():
    parser = argparse.ArgumentParser(
        description="Apply manual spectral shifts (from config CSV) to a target dataset.",
    )
    parser.add_argument(
        "--target", type=Path, required=True,
        help="Root directory of processed dataset to apply shifts to "
             "(e.g. /path/to/processed/magic_bayes).",
    )
    parser.add_argument(
        "--csv", type=Path, default=DEFAULT_CSV,
        help=f"Path to the shifts CSV. Default: {DEFAULT_CSV.relative_to(PROJECT_ROOT)}",
    )
    parser.add_argument(
        "--files", nargs="+", default=["clean_data", "fingerprint"],
        help="Which file stems to shift in each target rep. "
             "Default: clean_data fingerprint",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report only; do not modify any files.",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv}")
    if not args.target.exists():
        raise SystemExit(f"Target directory not found: {args.target}")

    shifts = load_lags_from_csv(args.csv)
    if not shifts:
        print(f"No shift records in {args.csv}. Nothing to do.")
        return

    print(f"CSV:    {args.csv}")
    print(f"Target: {args.target}")
    print(f"Stems:  {args.files}")
    print()
    print(f"Loaded {len(shifts)} shift record(s):")
    for rep_path, lag_samples, lag_cm1 in shifts:
        target_rep = args.target / rep_path
        marker = "OK  " if target_rep.exists() else "MISS"
        print(f"  [{marker}] {rep_path:<28}  lag={lag_samples:+d} samples  ({lag_cm1:+.2f} cm-1)")

    if args.dry_run:
        print("\nDry run — no files modified.")
        return

    print()
    counts = {"applied": 0, "skipped-missing": 0, "skipped-zero": 0}
    for rep_path, lag_samples, _ in shifts:
        target_rep = args.target / rep_path
        if not target_rep.exists():
            print(f"  [MISS] {rep_path} not present in target — skipping all stems")
            counts["skipped-missing"] += len(args.files)
            continue
        for stem in args.files:
            status = apply_shift(target_rep, stem, lag_samples)
            counts[status] += 1
            print(f"  [{status:>15}] {rep_path}/{stem}.npz")

    print()
    print(f"Summary: {counts}")


if __name__ == "__main__":
    main()
