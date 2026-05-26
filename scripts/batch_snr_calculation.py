"""
Per-Rep Population SNR Calculation
==================================

For each rep, reads stage3_bg_removed.npz (produced by
preprocess_background_noise_removal.py) and computes per-wavenumber
population statistics across the rep's spectra:

  - signal_med(k)         = median over spectra of intensity at column k
  - signal_mean(k)        = mean over spectra
  - noise_mad_sigma(k)    = 1.4826 * median(|I(k) - median(I(k))|)
  - snr_med(k)            = signal_med(k) / max(noise_mad_sigma(k), eps)
  - snr_mean(k)           = signal_mean(k) / max(noise_mad_sigma(k), eps)

The npz always carries both snr_med and snr_mean. The plot and metadata
report one of them, selected by --snr-method (default: mean).

Outputs analysis/population_snr.{npz,png} per rep.

See spec: docs/superpowers/specs/2026-05-20-background-noise-removal-population-snr-cr-investigation-design.md
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


_MAD_SCALE = 1.4826


def compute_population_stats(intensity_matrix, wavelengths, eps=1e-10):
    """Per-wavenumber median, mean, MAD-sigma, and SNR across the population.

    Args:
        intensity_matrix: shape (N, W).
        wavelengths: shape (W,).
        eps: division-by-zero floor for noise_mad_sigma.

    Returns:
        dict of arrays with keys:
            wavelengths, signal_med, signal_mean, noise_mad_sigma,
            snr_med, snr_mean, n_spectra, eps
    """
    matrix = np.asarray(intensity_matrix, dtype=np.float64)
    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    signal_med = np.median(matrix, axis=0)
    signal_mean = matrix.mean(axis=0)
    noise_mad_sigma = _MAD_SCALE * np.median(
        np.abs(matrix - signal_med[None, :]), axis=0
    )
    safe_sigma = np.maximum(noise_mad_sigma, eps)
    snr_med = signal_med / safe_sigma
    snr_mean = signal_mean / safe_sigma
    return {
        'wavelengths': wavelengths,
        'signal_med': signal_med,
        'signal_mean': signal_mean,
        'noise_mad_sigma': noise_mad_sigma,
        'snr_med': snr_med,
        'snr_mean': snr_mean,
        'n_spectra': np.int64(matrix.shape[0]),
        'eps': np.float64(eps),
    }


def plot_population_snr(stats, save_path, title, snr_method='mean'):
    """Plot the 3-panel population SNR figure.

    snr_method selects which central-tendency statistic drives the top panel's
    centred band and the bottom panel's SNR curve ('mean' or 'median'). The
    other statistic is overlaid in the top panel for comparison.
    """
    wavelengths = stats['wavelengths']
    signal_med = stats['signal_med']
    signal_mean = stats['signal_mean']
    noise = stats['noise_mad_sigma']

    if snr_method == 'mean':
        center = signal_mean
        overlay = signal_med
        center_label = 'mean'
        overlay_label = 'median'
        snr = stats['snr_mean']
        snr_label_long = 'mean / MAD-sigma'
    elif snr_method == 'median':
        center = signal_med
        overlay = signal_mean
        center_label = 'median'
        overlay_label = 'mean'
        snr = stats['snr_med']
        snr_label_long = 'median / MAD-sigma'
    else:
        raise ValueError(f"snr_method must be 'mean' or 'median', got {snr_method!r}")

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    ax = axes[0]
    ax.plot(wavelengths, center, color='black', linewidth=1.0, label=center_label)
    ax.fill_between(wavelengths, center - noise, center + noise,
                    color='steelblue', alpha=0.3, label='+/- MAD-sigma')
    ax.plot(wavelengths, overlay, color='red', linewidth=0.8,
            linestyle='--', alpha=0.7, label=overlay_label)
    ax.set_ylabel('Intensity')
    ax.set_title(f'Population signal ({center_label} +/- MAD-sigma, '
                 f'{overlay_label} overlaid)')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    ax.plot(wavelengths, noise, color='darkorange', linewidth=1.0)
    ax.set_ylabel('MAD-sigma')
    ax.set_title('Per-wavenumber noise (MAD-sigma across spectra)')
    ax.grid(True, alpha=0.2)

    ax = axes[2]
    ax.plot(wavelengths, snr, color='darkgreen', linewidth=1.0)
    ax.axhline(3.0, color='gray', linestyle='--', linewidth=0.8, label='SNR = 3')
    ax.set_xlabel('Wavenumber (cm-1)')
    ax.set_ylabel('SNR')
    ax.set_title(f'Per-wavenumber SNR ({snr_label_long})')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.2)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def process_rep(rep_dir, label, overwrite_existing, snr_method='mean'):
    rep_dir = Path(rep_dir)
    src = rep_dir / 'stage3_bg_removed.npz'
    if not src.exists():
        print(f"  [{label}] Missing {src.name} - skipping.")
        return
    analysis_dir = rep_dir / 'analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)
    out_npz = analysis_dir / 'population_snr.npz'
    out_png = analysis_dir / 'population_snr.png'
    out_meta = analysis_dir / 'population_snr_metadata.json'

    if not overwrite_existing and out_npz.exists():
        print(f"  [{label}] Skipping -- population_snr.npz already exists")
        return

    print(f'  [{label}] Loading {src.name} ...')
    with np.load(src, allow_pickle=True) as f:
        intensities = f['intensities']
        wavelengths = f['wavelengths']

    print(f'  [{label}] Computing population stats ({intensities.shape[0]} spectra) ...')
    stats = compute_population_stats(intensities, wavelengths, eps=1e-10)

    np.savez_compressed(out_npz, **stats)
    plot_population_snr(stats, out_png,
                        title=f'Population SNR ({snr_method}) -- {label}',
                        snr_method=snr_method)

    snr_arr = stats['snr_mean'] if snr_method == 'mean' else stats['snr_med']
    metadata = {
        'label': label,
        'n_spectra': int(stats['n_spectra']),
        'n_wavenumber': int(len(stats['wavelengths'])),
        'eps': float(stats['eps']),
        'snr_method': snr_method,
        'stats': {
            'noise_mad_sigma_min': float(stats['noise_mad_sigma'].min()),
            'noise_mad_sigma_max': float(stats['noise_mad_sigma'].max()),
            'noise_mad_sigma_median': float(np.median(stats['noise_mad_sigma'])),
            'snr_max': float(snr_arr.max()),
            'snr_argmax_wavenumber': float(
                stats['wavelengths'][int(np.argmax(snr_arr))]
            ),
        },
    }
    with open(out_meta, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f'  [{label}] population_snr.{{npz,png}} + metadata saved')


def discover_processed_reps(root, length=None, sequence=None, rep=None):
    """Find rep directories under processed root that contain stage3_bg_removed.npz."""
    rep_dirs = []
    if not root.exists():
        return rep_dirs
    for length_dir in sorted(root.iterdir()):
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
                        if (rep_dir / 'stage3_bg_removed.npz').exists():
                            rep_dirs.append(rep_dir)
                continue
            if sequence is not None and seq_dir.name != sequence:
                continue
            for rep_dir in sorted(seq_dir.iterdir()):
                if not rep_dir.is_dir() or not rep_dir.name.startswith('rep'):
                    continue
                if rep is not None and rep_dir.name != rep:
                    continue
                if (rep_dir / 'stage3_bg_removed.npz').exists():
                    rep_dirs.append(rep_dir)
    return rep_dirs


def main():
    parser = argparse.ArgumentParser(
        description='Per-rep population SNR calculation from stage3_bg_removed.npz'
    )
    parser.add_argument('--processed-root', type=Path,
                        default=Path('data/custom/processed/magic_explore'))
    parser.add_argument('--length', type=str, default=None)
    parser.add_argument('--sequence', type=str, default=None)
    parser.add_argument('--rep', type=str, default=None)
    parser.add_argument('--snr-method', type=str, default='mean',
                        choices=('mean', 'median'),
                        help='Central-tendency statistic for the displayed SNR '
                             '(default: mean). Both snr_med and snr_mean are '
                             'always written to population_snr.npz; this only '
                             'selects what the plot and metadata report.')
    parser.add_argument('--overwrite-existing', action='store_true')
    args = parser.parse_args()

    print('Batch SNR Calculation (per-rep population statistics)')
    print('=' * 60)
    print(f'Processed root: {args.processed_root.resolve()}')
    print(f'SNR method:     {args.snr_method}')
    print('=' * 60)

    rep_dirs = discover_processed_reps(args.processed_root,
                                       args.length, args.sequence, args.rep)
    if not rep_dirs:
        print('No matching processed reps found.')
        return
    print(f'Found {len(rep_dirs)} rep(s) to analyse.\n')

    for rep_dir in rep_dirs:
        rel = rep_dir.relative_to(args.processed_root)
        label = str(rel).replace('\\', '/')
        print(f"\n{'=' * 60}")
        print(f'Analysing: {label}')
        print(f"{'=' * 60}")
        try:
            process_rep(rep_dir, label, args.overwrite_existing,
                        snr_method=args.snr_method)
        except Exception as e:
            print(f'  ERROR analysing {label}: {e}')
            import traceback
            traceback.print_exc()

    print('\nDone.')


if __name__ == '__main__':
    main()
