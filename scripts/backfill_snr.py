"""Backfill per-spectrum SNR fields into existing clean_data.npz files.

For each rep directory under the target tree (default: data/custom/processed/magic),
reads stage1_cleaned.npz, stage2_denoised.npz, stage3_baseline_removed.npz, and
clean_data.npz; matches the foreground spectra in clean_data to their
corresponding stage indices (by coordinates, falling back to scale_factors);
computes per-spectrum SNR using ``calculate_snr_from_stages``; updates
clean_data.npz in place by adding ``snr`` and ``noise_std`` arrays.

Usage:
    python scripts/backfill_snr.py --dry-run
    python scripts/backfill_snr.py
    python scripts/backfill_snr.py --target data/custom/processed/magic
"""

import argparse
from pathlib import Path

import numpy as np

from primarymagic.preprocessing.snr import calculate_snr_from_stages


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGET = PROJECT_ROOT / "data" / "custom" / "processed" / "magic"

REQUIRED_FILES = [
    "clean_data.npz",
    "stage1_cleaned.npz",
    "stage2_denoised.npz",
    "stage3_baseline_removed.npz",
]


def _match_indices_by_coords(cd_coords: np.ndarray, stage_coords: np.ndarray) -> np.ndarray:
    """For each row in cd_coords, find the index in stage_coords with the same (x,y).
    Raises ValueError if any clean_data row has no match."""
    # Build a lookup from coord tuple -> first stage index
    lut = {}
    for i, c in enumerate(stage_coords):
        key = (float(c[0]), float(c[1]))
        lut.setdefault(key, i)
    out = np.empty(len(cd_coords), dtype=np.int64)
    for j, c in enumerate(cd_coords):
        key = (float(c[0]), float(c[1]))
        if key not in lut:
            raise ValueError(f"clean_data coord {key} not found in stage spectra")
        out[j] = lut[key]
    return out


def _match_indices_by_scale(cd_intensities: np.ndarray, cd_scales: np.ndarray,
                            stage3_intensities: np.ndarray) -> np.ndarray:
    """Fallback when coordinates are unavailable: clean_data[i] * scales[i]
    should equal stage3[fg_indices[i]] elementwise. We pick the unique stage3
    row whose max equals scales[i] AND whose normalised version matches."""
    n_clean = cd_intensities.shape[0]
    out = np.empty(n_clean, dtype=np.int64)
    used = set()
    stage3_max = stage3_intensities.max(axis=1)
    for i in range(n_clean):
        target = cd_intensities[i] * cd_scales[i]
        candidates = np.where(np.isclose(stage3_max, cd_scales[i], rtol=1e-6))[0]
        candidates = [c for c in candidates if c not in used]
        if not candidates:
            raise ValueError(f"clean_data[{i}] could not be matched by scale")
        # Pick the candidate whose values match target
        diffs = [np.max(np.abs(stage3_intensities[c] - target)) for c in candidates]
        best = candidates[int(np.argmin(diffs))]
        used.add(best)
        out[i] = best
    return out


def update_rep(rep_dir: Path) -> str:
    """Returns one of: updated, skipped-missing-file, skipped-already, error:<reason>."""
    paths = {name: rep_dir / name for name in REQUIRED_FILES}
    for p in paths.values():
        if not p.exists():
            return "skipped-missing-file"

    cd = dict(np.load(paths["clean_data.npz"], allow_pickle=True))
    if "snr" in cd and "noise_std" in cd:
        return "skipped-already"

    s1 = np.load(paths["stage1_cleaned.npz"], allow_pickle=True)
    s2 = np.load(paths["stage2_denoised.npz"], allow_pickle=True)
    s3 = np.load(paths["stage3_baseline_removed.npz"], allow_pickle=True)

    # Match clean_data spectra back to their stage row indices.
    if "coordinates" in cd and "coordinates" in s1.files:
        try:
            idx = _match_indices_by_coords(cd["coordinates"], s1["coordinates"])
        except ValueError as e:
            return f"error:{e}"
    else:
        try:
            idx = _match_indices_by_scale(
                cd["intensities"], cd["scale_factors"], s3["intensities"]
            )
        except ValueError as e:
            return f"error:{e}"

    s1_mat = s1["intensities"][idx]
    s2_mat = s2["intensities"][idx]
    s3_mat = s3["intensities"][idx]

    snr, signal, noise_std = calculate_snr_from_stages(s1_mat, s2_mat, s3_mat)

    # Sanity: the recovered signal should match the stored scale_factors
    # (since scale_factors were computed as max(stage3) per spectrum).
    if not np.allclose(signal, cd["scale_factors"], rtol=1e-6, atol=1e-8):
        return "error:signal-scale-mismatch"

    cd["snr"] = snr.astype(np.float32)
    cd["noise_std"] = noise_std.astype(np.float32)
    np.savez_compressed(paths["clean_data.npz"], **cd)
    return "updated"


def iter_rep_dirs(root: Path):
    """Yield every directory under root that contains all REQUIRED_FILES."""
    for clean_path in root.rglob("clean_data.npz"):
        rep_dir = clean_path.parent
        if all((rep_dir / f).exists() for f in REQUIRED_FILES):
            yield rep_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET,
                        help="Root directory to walk")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report only; do not modify files")
    args = parser.parse_args()

    target = args.target
    print(f"Target: {target}")

    rep_dirs = list(iter_rep_dirs(target))
    print(f"Found {len(rep_dirs)} rep directories with all required stage files")

    if args.dry_run:
        for rd in rep_dirs[:10]:
            print(f"  would update {rd.relative_to(target)}")
        if len(rep_dirs) > 10:
            print(f"  ... and {len(rep_dirs) - 10} more")
        return

    counts = {"updated": 0, "skipped-already": 0, "skipped-missing-file": 0, "error": 0}
    for rd in rep_dirs:
        status = update_rep(rd)
        if status.startswith("error"):
            counts["error"] += 1
            print(f"  [{status}] {rd.relative_to(target)}")
        else:
            counts[status] = counts.get(status, 0) + 1

    print()
    print(f"Summary: {counts}")


if __name__ == "__main__":
    main()
