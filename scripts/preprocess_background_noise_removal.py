"""
Background-Noise Removal Preprocessing Script
=============================================

Variant of preprocess_magic_bayes.py with two key differences:

1. **No cosmic-ray removal in stage 1.** Only edge-spike removal runs.
2. **Stage 3 subtracts the bubblefill baseline (estimated from stage 2,
   the denoised signal) from stage 1 (the still-noisy signal).** This
   preserves the per-pixel noise structure so per-wavenumber population
   statistics can measure it.

    stage3 = stage1 - (stage2 - bubblefill(stage2))
           = stage1 - baseline_estimated_from_stage2

Output is NOT clipped to zero (negative values must survive for unbiased
MAD-sigma estimation downstream).

Stages:
    1. remove_edge_spikes(edge_n=10, factor=5.0)
    2. smooth(wavelet='sym6', threshold_method='bayes')
    3. background-removed signal: stage1 - baseline_from_stage2

No SNR filter, no normalisation, no fingerprint. The pipeline terminates
after stage 3.

See spec: docs/superpowers/specs/2026-05-20-background-noise-removal-population-snr-cr-investigation-design.md
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from primarymagic import (
    Spectrum,
    SpectraCollection,
    PreprocessingPipeline,
    read_spectrum_file,
    export_to_npz,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_txt_filepath(folder):
    """Find the mapped spectra .txt file (contains 'power' in name)."""
    folder = Path(folder)
    for f in folder.glob('*.txt'):
        if 'power' in f.stem.lower():
            return f
    return None


def collection_from_matrix(template, intensity_matrix):
    """Wrap an (N, W) intensity matrix into a SpectraCollection using the
    template's wavelengths and per-spectrum x/y coordinates."""
    wavelengths = template.wavelengths
    new_spectra = []
    for i, s in enumerate(template.spectra):
        new_data = np.column_stack([wavelengths, intensity_matrix[i]])
        new_spectra.append(Spectrum(data=new_data, x=s.x, y=s.y))
    return SpectraCollection(
        spectra=new_spectra,
        source_file=template.source_file,
        wavelengths=wavelengths.copy(),
    )


def sample_indices(n_total, n_samples, rng):
    """Sample up to n_samples row indices from [0, n_total)."""
    n = min(n_samples, n_total)
    if n == 0:
        return np.array([], dtype=int)
    return rng.choice(n_total, size=n, replace=False)


# ---------------------------------------------------------------------------
# Processing stages
# ---------------------------------------------------------------------------

def run_stage1(collection):
    """Stage 1: edge-spike removal only (no cosmic-ray removal)."""
    return (
        PreprocessingPipeline(collection)
        .remove_edge_spikes(edge_n=10, factor=5.0)
        .result()
    )


def run_stage2(stage1):
    """Stage 2: wavelet denoising (sym6, BayesShrink)."""
    return (
        PreprocessingPipeline(stage1)
        .smooth(method='wavelet', wavelet='sym6', threshold_method='bayes')
        .result()
    )


def run_stage3(stage1, stage2):
    """Stage 3: stage3 = stage1 - (stage2 - bubblefill(stage2)).

    Equivalent to subtracting the baseline (estimated from the denoised
    signal) from the original noisy signal. Preserves per-pixel noise.

    Returns
    -------
    stage3 : SpectraCollection
    baseline_mat : ndarray, shape (N, W)
        The bubblefill baseline curves, i.e. s2_mat - nbg_mat.
    """
    denoised_no_bg = (
        PreprocessingPipeline(stage2)
        .subtract_baseline(method='bubblefill', min_bubble_widths=50, fit_order=1)
        .result()
    )
    s1_mat = stage1.to_intensity_matrix()
    s2_mat = stage2.to_intensity_matrix()
    nbg_mat = denoised_no_bg.to_intensity_matrix()
    baseline_mat = s2_mat - nbg_mat
    stage3_mat = s1_mat - baseline_mat
    return collection_from_matrix(stage1, stage3_mat), baseline_mat


# ---------------------------------------------------------------------------
# Diagnostic plots (no pass/fail grouping)
# ---------------------------------------------------------------------------

def _plot_samples(matrix, indices, wavelengths, save_path, title, ylabel='Intensity'):
    colors = ['black', 'red', 'green', 'blue', 'goldenrod', 'purple', 'brown']
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, gi in enumerate(indices):
        c = colors[i % len(colors)]
        ax.plot(wavelengths, matrix[gi], color=c, linewidth=0.8, label=f'#{gi}')
    ax.set_xlabel('Wavenumber (cm-1)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if len(indices) > 0:
        ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_stage0_raw(raw, indices, save_dir):
    _plot_samples(raw.to_intensity_matrix(), indices, raw.wavelengths,
                  save_dir / 'stage0_raw.png', 'Stage 0: Raw Spectra')


def plot_stage1_edge_fixed(stage1, indices, save_dir):
    _plot_samples(stage1.to_intensity_matrix(), indices, stage1.wavelengths,
                  save_dir / 'stage1_edge_fixed.png',
                  'Stage 1: Edge Spike Removal Only')


def plot_stage2_denoised(stage2, indices, save_dir):
    _plot_samples(stage2.to_intensity_matrix(), indices, stage2.wavelengths,
                  save_dir / 'stage2_denoised.png',
                  'Stage 2: Wavelet Denoising (sym6 BayesShrink)')


def plot_stage3_bg_removed(stage3, indices, save_dir):
    _plot_samples(stage3.to_intensity_matrix(), indices, stage3.wavelengths,
                  save_dir / 'stage3_bg_removed.png',
                  'Stage 3: Stage1 - Baseline (background removed, noise preserved)')


def plot_stage3_baseline_overlay(stage2, baseline_mat, indices, save_dir):
    """stage2 vs baseline_mat overlay = denoised signal + estimated baseline."""
    s2 = stage2.to_intensity_matrix()

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, gi in enumerate(indices):
        ax.plot(stage2.wavelengths, s2[gi], color='black',
                linewidth=0.6, alpha=0.8, label='Denoised (stage2)' if i == 0 else None)
        ax.plot(stage2.wavelengths, baseline_mat[gi], color='darkgreen',
                linewidth=1.0, alpha=0.9, label='Baseline' if i == 0 else None)
    ax.set_xlabel('Wavenumber (cm-1)')
    ax.set_ylabel('Intensity')
    ax.set_title('Stage 3: Denoised Spectrum + Estimated Baseline')
    if len(indices) > 0:
        ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_dir / 'stage3_baseline_overlay.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-rep processing
# ---------------------------------------------------------------------------

def process_rep(raw_folder, save_dir, label, n_samples):
    raw_folder = Path(raw_folder)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    stages_dir = save_dir / 'diagnostics' / 'stages'
    stages_dir.mkdir(parents=True, exist_ok=True)

    collection_path = get_txt_filepath(raw_folder)
    if collection_path is None:
        print(f"  [{label}] No 'power' .txt file found - skipping.")
        return

    print(f"  [{label}] Loading: {collection_path.name}")
    raw = read_spectrum_file(collection_path)
    n_total = len(raw)
    wavelengths = raw.wavelengths
    print(f"  [{label}] {n_total} spectra, {len(wavelengths)} wavenumber points")

    print(f"  [{label}] Stage 1: edge spike removal ...")
    stage1 = run_stage1(raw)
    export_to_npz(stage1, save_dir / 'stage1_edge_fixed.npz')

    print(f"  [{label}] Stage 2: wavelet denoising (sym6 BayesShrink) ...")
    stage2 = run_stage2(stage1)
    export_to_npz(stage2, save_dir / 'stage2_denoised.npz')

    print(f"  [{label}] Stage 3: stage1 - baseline (no clipping) ...")
    stage3, baseline_mat = run_stage3(stage1, stage2)
    export_to_npz(stage3, save_dir / 'stage3_bg_removed.npz')

    export_to_npz(raw, save_dir / 'raw_all.npz')

    # Diagnostics
    rng = np.random.default_rng(42)
    sample_idx = sample_indices(n_total, n_samples, rng)
    plot_stage0_raw(raw, sample_idx, stages_dir)
    plot_stage1_edge_fixed(stage1, sample_idx, stages_dir)
    plot_stage2_denoised(stage2, sample_idx, stages_dir)
    plot_stage3_baseline_overlay(stage2, baseline_mat, sample_idx, stages_dir)
    plot_stage3_bg_removed(stage3, sample_idx, stages_dir)

    s3_mat = stage3.to_intensity_matrix()
    metadata = {
        'label': label,
        'source_file': str(collection_path),
        'n_total': int(n_total),
        'n_wavenumber': int(len(wavelengths)),
        'stages_run': [1, 2, 3],
        'parameters': {
            'edge_n': 10,
            'edge_factor': 5.0,
            'wavelet': 'sym6',
            'threshold_method': 'bayes',
            'min_bubble_widths': 50,
            'fit_order': 1,
        },
        'stage3': {
            'intensity_min': float(s3_mat.min()),
            'intensity_max': float(s3_mat.max()),
            'intensity_median': float(np.median(s3_mat)),
        },
    }
    with open(save_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  [{label}] metadata.json saved")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def discover_reps(raw_root, length=None, sequence=None, rep=None):
    rep_dirs = []
    for length_dir in sorted(raw_root.iterdir()):
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
        description='Background-noise removal preprocessing (no CR removal, '
                    'noise preserved through stage 3).'
    )
    parser.add_argument('--raw-root', type=Path, default=Path('data/custom/raw'))
    parser.add_argument('--output-root', type=Path,
                        default=Path('data/custom/processed/magic_explore'))
    parser.add_argument('--length', type=str, default=None)
    parser.add_argument('--sequence', type=str, default=None)
    parser.add_argument('--rep', type=str, default=None)
    parser.add_argument('--n-samples', type=int, default=5)
    parser.add_argument('--overwrite-existing', action='store_true')
    args = parser.parse_args()

    print('Background-Noise Removal Preprocessing')
    print('=' * 60)
    print(f'Raw root:    {args.raw_root.resolve()}')
    print(f'Output root: {args.output_root.resolve()}')
    print(f'Length:      {args.length or "all"}')
    print(f'Sequence:    {args.sequence or "all"}')
    print(f'Rep:         {args.rep or "all"}')
    print('=' * 60)

    rep_dirs = discover_reps(args.raw_root, args.length, args.sequence, args.rep)
    if not rep_dirs:
        print('No matching rep directories found.')
        return
    print(f'Found {len(rep_dirs)} rep(s) to process.\n')

    for rep_dir in rep_dirs:
        rel = rep_dir.relative_to(args.raw_root)
        label = str(rel).replace('\\', '/')
        save_dir = args.output_root / rel

        print(f"\n{'=' * 60}")
        print(f'Processing: {label}')
        print(f'Output:     {save_dir}')
        print(f"{'=' * 60}")

        if not args.overwrite_existing and (save_dir / 'metadata.json').exists():
            print(f"  [{label}] Skipping -- metadata.json already exists")
            continue

        try:
            process_rep(rep_dir, save_dir, label, args.n_samples)
        except Exception as e:
            print(f'  ERROR processing {label}: {e}')
            import traceback
            traceback.print_exc()

    print('\nDone.')


if __name__ == '__main__':
    main()
