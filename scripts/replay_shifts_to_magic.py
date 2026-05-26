"""Replay manual spectral shifts from primary_magic onto the new magic data.

Discovers all ``*_noshift.npz`` backups in ``data/custom/processed/primary_magic/``,
recovers the integer-sample lag via cross-correlation against its backup, and
applies the same shift to the matching rep directory in
``data/custom/processed/magic/`` using the same logic as
``scripts/shift_spectra.py`` (interpolation onto the original wavelength grid,
backup written to ``*_noshift.npz``).

Usage:
    python scripts/replay_shifts_to_magic.py --dry-run
    python scripts/replay_shifts_to_magic.py
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
from scipy.signal import correlate

from primarymagic.data.spectraio import load_from_npz, export_to_npz


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "custom" / "processed" / "primary_magic"
DEFAULT_TARGET = PROJECT_ROOT / "data" / "custom" / "processed" / "magic"


def best_lag(current: np.ndarray, original: np.ndarray) -> int:
    """Integer sample lag s such that shifting `original` by s reproduces `current`."""
    a = (current - current.mean()) / (current.std() + 1e-9)
    b = (original - original.mean()) / (original.std() + 1e-9)
    corr = correlate(a, b, mode="full")
    return int(np.argmax(corr) - (len(b) - 1))


def discover_rep_lags(source: Path):
    """Walk source for clean_data_noshift.npz; return {rep_rel_path: (lag, wl_step)}."""
    lags = {}
    for backup in source.rglob("clean_data_noshift.npz"):
        rep_dir = backup.parent
        shifted = rep_dir / "clean_data.npz"
        if not shifted.exists():
            continue
        orig = np.load(backup, allow_pickle=True)
        cur = np.load(shifted, allow_pickle=True)
        lag = best_lag(cur["intensities"].mean(0), orig["intensities"].mean(0))
        wl_step = float(np.mean(np.diff(orig["wavelengths"])))
        lags[rep_dir.relative_to(source)] = (lag, wl_step)
    return lags


def apply_shift(target_rep: Path, stem: str, lag: int) -> str:
    """Apply integer-sample shift to <stem>.npz in target_rep. Idempotent: if a
    *_noshift.npz backup already exists we re-load from it before applying.

    Returns one of: 'applied', 'skipped-missing', 'skipped-zero'.
    """
    src = target_rep / f"{stem}.npz"
    if not src.exists():
        return "skipped-missing"
    if lag == 0:
        return "skipped-zero"
    backup = target_rep / f"{stem}_noshift.npz"
    if backup.exists():
        collection = load_from_npz(backup)
    else:
        shutil.copy2(src, backup)
        collection = load_from_npz(src)
    wl_step = float(np.mean(np.diff(collection.wavelengths)))
    shifted = collection.shift(lag * wl_step)
    export_to_npz(shifted, src)
    return "applied"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="Directory containing the manually-shifted data and *_noshift.npz backups")
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET,
                        help="Directory to apply the same shifts to")
    parser.add_argument("--files", nargs="+", default=["clean_data", "fingerprint"],
                        help="Which file stems to shift in the target")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report only; do not modify any files")
    args = parser.parse_args()

    print(f"Source: {args.source}")
    print(f"Target: {args.target}")
    print()

    rep_lags = discover_rep_lags(args.source)
    if not rep_lags:
        print("No *_noshift.npz backups found in source. Nothing to do.")
        return

    print(f"Found {len(rep_lags)} rep directories with manual shifts in source:")
    for rep_rel in sorted(rep_lags):
        lag, wl_step = rep_lags[rep_rel]
        target_rep = args.target / rep_rel
        marker = "OK  " if target_rep.exists() else "MISS"
        print(f"  [{marker}] {rep_rel}  lag={lag:+d} samples  ({lag*wl_step:+.2f} cm-1)")

    if args.dry_run:
        print("\nDry run — no files modified.")
        return

    print()
    counts = {"applied": 0, "skipped-missing": 0, "skipped-zero": 0}
    for rep_rel, (lag, _) in sorted(rep_lags.items()):
        target_rep = args.target / rep_rel
        if not target_rep.exists():
            print(f"  [MISS] {rep_rel} not present in target — skipping all stems")
            counts["skipped-missing"] += len(args.files)
            continue
        for stem in args.files:
            status = apply_shift(target_rep, stem, lag)
            counts[status] += 1
            print(f"  [{status:>15}] {rep_rel}/{stem}.npz")

    print()
    print(f"Summary: {counts}")


if __name__ == "__main__":
    main()
