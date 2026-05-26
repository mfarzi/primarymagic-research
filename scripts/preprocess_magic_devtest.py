"""
Magic Devtest Preprocessing Pipeline
=====================================

Replays the magic cleaning pipeline (stages 1, 2, 3, 6) on the exact
raw spectra population that ``preprocess_spectra.py`` already selected
via SNR/ASSI filtering (input: ``primary_magic/.../raw_data.npz``).

Stages 4 and 5 (AUC-seeded fingerprint and czekanowski classification)
are skipped — every input spectrum is treated as foreground by
definition.

Output: ``data/custom/processed/magic_devtest/{n-letter}/{SEQ}/rep#/``
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Make scripts/ importable so we can reuse preprocess_magic helpers.
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from preprocess_magic import (  # noqa: E402
    run_stage1,
    run_stage3,
    run_stage6,
    plot_stage6_mean_std,
    plot_clean_data_gallery,
)
from primarymagic import PreprocessingPipeline  # noqa: E402
from primarymagic.data.spectraio import load_from_npz, export_to_npz  # noqa: E402
from primarymagic.preprocessing.snr import calculate_snr_from_stages  # noqa: E402


def run_stage2(stage1):
    """Wavelet denoise with sym6 + BayesShrink (level-dependent threshold).

    Overrides preprocess_magic.run_stage2 (which uses VisuShrink). BayesShrink
    preserves more inter-peak detail, which the differential classifier relies on
    for distinguishing tripeptides like ADR / ASD / RFA from neighbours.
    """
    return (
        PreprocessingPipeline(stage1)
        .smooth(method='wavelet', wavelet='sym6', threshold_method='bayes')
        .result()
    )


def load_raw_npz(path):
    """Load primary_magic raw_data.npz. Return None if missing or empty.

    Args:
        path: Path to raw_data.npz file.

    Returns:
        SpectraCollection, or None if the file does not exist or has
        zero spectra.
    """
    path = Path(path)
    if not path.is_file():
        return None
    coll = load_from_npz(path)
    if len(coll) == 0:
        return None
    return coll


def _sample_indices(n_total, n_samples, rng):
    """Return up to n_samples indices drawn without replacement from range(n_total)."""
    n = min(n_samples, n_total)
    if n == 0:
        return np.array([], dtype=int)
    return rng.choice(n_total, size=n, replace=False)


def _plot_samples(intensity_matrix, idx, wavelengths, save_path, title):
    """Single-panel plot: N sample spectra, each with a distinct colour."""
    sample_colors = ['black', 'red', 'green', 'blue', 'goldenrod']
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, gi in enumerate(idx):
        c = sample_colors[i % len(sample_colors)]
        ax.plot(wavelengths, intensity_matrix[gi], color=c, alpha=1.0,
                linewidth=0.8, label=f'#{gi}')
    ax.set_xlabel('Wavenumber (cm⁻¹)')
    ax.set_ylabel('Intensity')
    ax.set_title(f'{title} (n_samples={len(idx)})')
    if len(idx) > 0:
        ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def process_rep(input_dir, save_dir, label, n_samples=5, noise_percentile=None):
    """Run stages 1, 2, 3, 6 on input_dir/raw_data.npz; write outputs to save_dir.

    Args:
        input_dir: Path containing the source raw_data.npz (from primary_magic).
        save_dir: Output directory; created if absent.
        label: Display label, e.g. '2-letter/AD/rep1'.
        n_samples: Number of sample spectra in each diagnostic plot.
        noise_percentile: Per-spectrum percentile clip floor passed to stage 3
            (None → clip to zero).
    """
    input_dir = Path(input_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    diag_dir = save_dir / 'diagnostics'
    stages_dir = diag_dir / 'stages'
    stages_dir.mkdir(parents=True, exist_ok=True)

    # --- Load input ---
    raw_path = input_dir / 'raw_data.npz'
    raw = load_raw_npz(raw_path)
    if raw is None:
        print(f"  [{label}] No usable raw_data.npz at {raw_path} -- skipping.")
        return
    n_total = len(raw)
    wavelengths = raw.wavelengths
    print(f"  [{label}] Loaded {n_total} spectra, {len(wavelengths)} wavenumber points")

    # --- Stage 1 ---
    print(f"  [{label}] Stage 1: edge spike + cosmic ray removal ...")
    stage1 = run_stage1(raw)
    export_to_npz(stage1, save_dir / 'stage1_cleaned.npz')

    # --- Stage 2 ---
    print(f"  [{label}] Stage 2: wavelet denoising (sym6 BayesShrink) ...")
    stage2 = run_stage2(stage1)
    export_to_npz(stage2, save_dir / 'stage2_denoised.npz')

    # --- Stage 3 ---
    clip_desc = f'noise_percentile={noise_percentile}' if noise_percentile is not None else 'zero'
    print(f"  [{label}] Stage 3: BubbleFill baseline + clip ({clip_desc}) ...")
    stage3 = run_stage3(stage2, noise_percentile=noise_percentile)
    export_to_npz(stage3, save_dir / 'stage3_baseline_removed.npz')

    # --- Stage 6: normalise ALL spectra (input is already pre-filtered) ---
    fg_mask = np.ones(len(stage3), dtype=bool)
    print(f"  [{label}] Stage 6: per-spectrum max-normalising {n_total} spectra ...")
    normalised, scales = run_stage6(stage3, fg_mask)

    # Per-spectrum SNR
    s1_mat = stage1.to_intensity_matrix()
    s2_mat = stage2.to_intensity_matrix()
    s3_mat = stage3.to_intensity_matrix()
    snr, _, noise_std = calculate_snr_from_stages(s1_mat, s2_mat, s3_mat)

    # --- Save clean_data.npz ---
    coords = raw.get_coordinates()
    clean_dict = {
        'wavelengths': wavelengths,
        'intensities': normalised.to_intensity_matrix(),
        'scale_factors': scales,
        'snr': snr,
        'noise_std': noise_std,
        'source_file': np.array(stage3.source_file),
    }
    if coords is not None:
        clean_dict['coordinates'] = coords
    np.savez_compressed(save_dir / 'clean_data.npz', **clean_dict)

    # --- Save raw_data.npz (copy of input) ---
    raw_mat = raw.to_intensity_matrix()
    raw_dict = {
        'wavelengths': raw.wavelengths,
        'intensities': raw_mat,
        'source_file': np.array(raw.source_file),
    }
    if coords is not None:
        raw_dict['coordinates'] = coords
    np.savez_compressed(save_dir / 'raw_data.npz', **raw_dict)

    # --- Diagnostic plots ---
    rng = np.random.default_rng(42)
    idx = _sample_indices(n_total, n_samples, rng)

    _plot_samples(raw.to_intensity_matrix(), idx, wavelengths,
                  stages_dir / 'stage0_raw.png', 'Stage 0: Raw Spectra')
    _plot_samples(s1_mat, idx, wavelengths,
                  stages_dir / 'stage1_fix_cosmic_ray.png',
                  'Stage 1: After Edge Spike + Cosmic Ray Removal')
    _plot_samples(s2_mat, idx, wavelengths,
                  stages_dir / 'stage2_denoise.png',
                  'Stage 2: After Wavelet Denoising (sym6 VisuShrink)')
    _plot_samples(s3_mat, idx, wavelengths,
                  stages_dir / 'stage3_baseline_corrected.png',
                  'Stage 3: Baseline-Corrected Spectra')

    plot_stage6_mean_std(normalised, wavelengths, diag_dir)
    plot_clean_data_gallery(normalised, wavelengths, diag_dir)

    # --- Metadata ---
    metadata = {
        'label': label,
        'source': str(raw_path),
        'n_total': int(n_total),
        'n_wavenumber': int(len(wavelengths)),
        'noise_percentile': noise_percentile,
        'snr': {
            'mean': float(np.mean(snr)),
            'median': float(np.median(snr)),
        },
        'stage6': {
            'scale_min': float(scales.min()),
            'scale_max': float(scales.max()),
            'scale_mean': float(scales.mean()),
        },
    }
    with open(save_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  [{label}] Done -> clean_data.npz ({n_total} rows), raw_data.npz, metadata.json")


def _discover_reps(input_root, length=None, sequence=None, rep=None):
    """Yield rep directories under input_root that contain raw_data.npz.

    Layout: input_root/{n-letter}/{SEQ}/rep#/raw_data.npz
    PTM exception: input_root/1-letter/PTM/{code}/rep#/raw_data.npz
    """
    rep_dirs = []
    for length_dir in sorted(input_root.iterdir()):
        if not length_dir.is_dir():
            continue
        if length is not None and length_dir.name != length:
            continue
        for seq_dir in sorted(length_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            if seq_dir.name == 'PTM':
                for ptm_dir in sorted(seq_dir.iterdir()):
                    if not ptm_dir.is_dir():
                        continue
                    if sequence is not None and ptm_dir.name != sequence:
                        continue
                    for rep_dir in sorted(ptm_dir.iterdir()):
                        if not rep_dir.is_dir() or not rep_dir.name.startswith('rep'):
                            continue
                        if rep is not None and rep_dir.name != rep:
                            continue
                        rep_dirs.append(rep_dir)
                continue
            if sequence is not None and seq_dir.name != sequence:
                continue
            for rep_dir in sorted(seq_dir.iterdir()):
                if not rep_dir.is_dir() or not rep_dir.name.startswith('rep'):
                    continue
                if rep is not None and rep_dir.name != rep:
                    continue
                rep_dirs.append(rep_dir)
    return rep_dirs


def main():
    parser = argparse.ArgumentParser(
        description='Magic devtest: replay magic cleaning on primary_magic raw_data.npz.'
    )
    parser.add_argument('--input-root', type=Path,
                        default=Path('data/custom/processed/primary_magic'),
                        help='Root containing primary_magic rep directories')
    parser.add_argument('--output-root', type=Path,
                        default=Path('data/custom/processed/magic_devtest'),
                        help='Root for processed output')
    parser.add_argument('--noise-percentile', type=float, default=None,
                        help='Per-spectrum percentile clip floor for stage 3 '
                             '(None -> clip to zero)')
    parser.add_argument('--n-samples', type=int, default=5,
                        help='Number of sample spectra per diagnostic plot')
    parser.add_argument('--length', type=str, default=None,
                        help='Filter by length dir (e.g. 1-letter)')
    parser.add_argument('--sequence', type=str, default=None,
                        help='Filter by sequence (e.g. AD)')
    parser.add_argument('--rep', type=str, default=None,
                        help='Filter by rep name (e.g. rep1)')
    parser.add_argument('--overwrite-existing', action='store_true',
                        help='Reprocess reps even if metadata.json already exists')
    args = parser.parse_args()

    input_root = args.input_root
    output_root = args.output_root

    print('Magic Devtest Preprocessing Pipeline')
    print('=' * 60)
    print(f'Input root:   {input_root.resolve()}')
    print(f'Output root:  {output_root.resolve()}')
    print(f'Noise pctl:   {args.noise_percentile if args.noise_percentile is not None else "None (clip to zero)"}')
    print(f'Length:       {args.length or "all"}')
    print(f'Sequence:     {args.sequence or "all"}')
    print(f'Rep:          {args.rep or "all"}')
    print('=' * 60)

    if not input_root.is_dir():
        print(f'ERROR: input root does not exist: {input_root}')
        return

    rep_dirs = _discover_reps(input_root, length=args.length,
                              sequence=args.sequence, rep=args.rep)
    rep_dirs = [d for d in rep_dirs if (d / 'raw_data.npz').is_file()]

    if not rep_dirs:
        print('No matching rep directories with raw_data.npz found.')
        return

    print(f'Found {len(rep_dirs)} rep(s) with raw_data.npz.\n')

    for rep_dir in rep_dirs:
        rel = rep_dir.relative_to(input_root)
        label = str(rel).replace('\\', '/')
        save_dir = output_root / rel

        print(f"\n{'='*60}")
        print(f'Processing: {label}')
        print(f'Output:     {save_dir}')
        print('=' * 60)

        if not args.overwrite_existing and (save_dir / 'metadata.json').exists():
            print(f"  [{label}] Skipping -- metadata.json already exists")
            continue

        try:
            process_rep(
                input_dir=rep_dir,
                save_dir=save_dir,
                label=label,
                n_samples=args.n_samples,
                noise_percentile=args.noise_percentile,
            )
        except Exception as e:
            print(f'  ERROR processing {label}: {e}')
            import traceback
            traceback.print_exc()

    print('\nDone.')


if __name__ == '__main__':
    main()
