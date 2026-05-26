"""
Cosmic-Ray Investigation
========================

For each rep, reads stage3_bg_removed.npz + analysis/population_snr.npz
and detects positive outlier events in each spectrum using the
per-wavenumber MAD-sigma as the robust noise floor.

For each event, records: spec_idx, z_thresh, peak_pixel, peak_wavenumber,
amp_sigma, amp_raw, fwhm_pixels, energy_ratio.

Outputs per rep:
  - analysis/cr_events.npz             (flat event table)
  - analysis/cr_amplitude_hist.png
  - analysis/cr_width_hist.png
  - analysis/cr_amp_vs_width.png
  - analysis/cr_position_vs_signal.png
  - analysis/cr_per_spectrum.png
  - analysis/cr_threshold_proposal.json

See spec: docs/superpowers/specs/2026-05-20-background-noise-removal-population-snr-cr-investigation-design.md
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _cluster_runs(mask_1d):
    """Yield (start, end_exclusive) for each run of True in mask_1d."""
    if not mask_1d.any():
        return
    idx = np.where(mask_1d)[0]
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]]))
    for s, e in zip(starts, ends):
        yield int(s), int(e) + 1


def detect_events(intensities, wavelengths, noise_mad_sigma,
                  z_thresholds=(5.0, 8.0, 12.0), eps=1e-10):
    """Detect positive outlier events per spectrum at each z_thresh.

    Args:
        intensities: (N, W) array of stage3 values.
        wavelengths: (W,) array of wavenumbers.
        noise_mad_sigma: (W,) per-wavenumber sigma from Script 2.
        z_thresholds: iterable of sigma-multiples to sweep.
        eps: floor for sigma to avoid division-by-zero.

    Returns:
        List of dict records, one per detected event:
            spec_idx (int), z_thresh (float), peak_pixel (int),
            peak_wavenumber (float), amp_sigma (float), amp_raw (float),
            fwhm_pixels (int), energy_ratio (float).
    """
    intensities = np.asarray(intensities, dtype=np.float64)
    safe_sigma = np.maximum(np.asarray(noise_mad_sigma, dtype=np.float64), eps)
    z = intensities / safe_sigma[None, :]

    events = []
    for z_thresh in z_thresholds:
        z_th = float(z_thresh)
        for spec_idx in range(intensities.shape[0]):
            mask = z[spec_idx] > z_th  # positive excursions only
            for start, end_excl in _cluster_runs(mask):
                window_raw = intensities[spec_idx, start:end_excl]
                window_z = z[spec_idx, start:end_excl]
                peak_offset = int(np.argmax(window_z))
                peak_pixel = start + peak_offset
                amp_raw = float(window_raw[peak_offset])
                amp_sigma = float(window_z[peak_offset])
                half = amp_raw / 2.0
                fwhm_pixels = int((window_raw >= half).sum())
                window_sum = float(window_raw.sum())
                energy_ratio = float(amp_raw / window_sum) if window_sum > 0 else 1.0
                events.append({
                    'spec_idx': int(spec_idx),
                    'z_thresh': z_th,
                    'peak_pixel': peak_pixel,
                    'peak_wavenumber': float(wavelengths[peak_pixel]),
                    'amp_sigma': amp_sigma,
                    'amp_raw': amp_raw,
                    'fwhm_pixels': fwhm_pixels,
                    'energy_ratio': energy_ratio,
                })
    return events


def events_to_record_array(events):
    """Pack events list-of-dict into structured ndarray for npz storage."""
    dtype = [
        ('spec_idx', np.int32),
        ('z_thresh', np.float64),
        ('peak_pixel', np.int32),
        ('peak_wavenumber', np.float64),
        ('amp_sigma', np.float64),
        ('amp_raw', np.float64),
        ('fwhm_pixels', np.int32),
        ('energy_ratio', np.float64),
    ]
    arr = np.zeros(len(events), dtype=dtype)
    for i, e in enumerate(events):
        for name, _ in dtype:
            arr[name][i] = e[name]
    return arr


def plot_amplitude_hist(events_arr, save_path):
    fig, ax = plt.subplots(figsize=(10, 4))
    for z_thresh in sorted(np.unique(events_arr['z_thresh'])):
        sel = events_arr[events_arr['z_thresh'] == z_thresh]
        if len(sel) == 0:
            continue
        ax.hist(sel['amp_sigma'], bins=80, alpha=0.5,
                label=f'z_thresh = {z_thresh}')
    ax.set_xlabel('Amplitude (sigma units)')
    ax.set_ylabel('Count')
    ax.set_yscale('log')
    ax.set_title('Event amplitude distribution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_width_hist(events_arr, save_path):
    fig, ax = plt.subplots(figsize=(10, 4))
    for z_thresh in sorted(np.unique(events_arr['z_thresh'])):
        sel = events_arr[events_arr['z_thresh'] == z_thresh]
        if len(sel) == 0:
            continue
        bins = np.arange(0, max(20, int(sel['fwhm_pixels'].max()) + 2))
        ax.hist(sel['fwhm_pixels'], bins=bins, alpha=0.5,
                label=f'z_thresh = {z_thresh}')
    ax.set_xlabel('FWHM (pixels)')
    ax.set_ylabel('Count')
    ax.set_title('Event FWHM distribution (narrow = CR-like)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_amp_vs_width(events_arr, save_path, z_thresh_used):
    sel = events_arr[events_arr['z_thresh'] == z_thresh_used]
    fig, ax = plt.subplots(figsize=(8, 6))
    if len(sel) > 0:
        ax.scatter(sel['fwhm_pixels'], sel['amp_sigma'],
                   s=8, alpha=0.4, color='steelblue')
    ax.set_xlabel('FWHM (pixels)')
    ax.set_ylabel('Amplitude (sigma units)')
    ax.set_title(f'Amplitude vs Width  (z_thresh = {z_thresh_used})')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_position_vs_signal(events_arr, z_thresh_used, signal_med,
                            wavelengths, save_path):
    sel = events_arr[events_arr['z_thresh'] == z_thresh_used]
    fig, ax1 = plt.subplots(figsize=(11, 4))
    ax1.plot(wavelengths, signal_med, color='black', linewidth=0.8,
             label='signal_med (population)')
    ax1.set_xlabel('Wavenumber (cm-1)')
    ax1.set_ylabel('signal_med', color='black')
    ax1.grid(True, alpha=0.2)

    ax2 = ax1.twinx()
    bins = 60
    counts, edges = np.histogram(sel['peak_wavenumber'],
                                 bins=bins,
                                 range=(float(wavelengths.min()),
                                        float(wavelengths.max())))
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax2.bar(centers, counts, width=(edges[1] - edges[0]),
            color='red', alpha=0.3, label='event count')
    ax2.set_ylabel('event count', color='red')

    ax1.set_title(f'Event positions vs population signal '
                  f'(z_thresh = {z_thresh_used})')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_per_spectrum(events_arr, n_spectra, save_path):
    fig, ax = plt.subplots(figsize=(10, 4))
    for z_thresh in sorted(np.unique(events_arr['z_thresh'])):
        sel = events_arr[events_arr['z_thresh'] == z_thresh]
        counts_per_spec = np.bincount(sel['spec_idx'].astype(np.int64),
                                       minlength=n_spectra)
        bins = np.arange(0, max(counts_per_spec.max() + 2, 5))
        ax.hist(counts_per_spec, bins=bins, alpha=0.5,
                label=f'z_thresh = {z_thresh}')
    ax.set_xlabel('Events per spectrum')
    ax.set_ylabel('Number of spectra')
    ax.set_title('Events per spectrum')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def build_threshold_proposal(events_arr, z_thresh_used, n_spectra):
    sel = events_arr[events_arr['z_thresh'] == z_thresh_used]
    if len(sel) == 0:
        return {
            'z_thresh_used': float(z_thresh_used),
            'n_events_total': 0,
            'events_per_spectrum_mean': 0.0,
            'fwhm_p95': None,
            'energy_ratio_p05': None,
            'proposed': {'cr_fwhm_min': None, 'cr_energy_ratio_max': None},
            'notes': 'No events detected at this z_thresh.',
        }
    # CR candidates: high-amplitude, narrow events at z_thresh_used.
    cr_like = sel[(sel['amp_sigma'] > 8.0)]
    if len(cr_like) == 0:
        cr_like = sel  # fall back to all selected events
    fwhm_p05 = float(np.percentile(cr_like['fwhm_pixels'], 5))
    energy_p95 = float(np.percentile(cr_like['energy_ratio'], 95))
    return {
        'z_thresh_used': float(z_thresh_used),
        'n_events_total': int(len(sel)),
        'events_per_spectrum_mean': float(len(sel)) / max(n_spectra, 1),
        'fwhm_p95': float(np.percentile(sel['fwhm_pixels'], 95)),
        'energy_ratio_p05': float(np.percentile(sel['energy_ratio'], 5)),
        'proposed': {
            'cr_fwhm_min': float(max(2.0, fwhm_p05)),
            'cr_energy_ratio_max': float(energy_p95),
        },
        'notes': (
            'cr_fwhm_min = max(2.0, p05 of FWHM for amp_sigma>8 events). '
            'cr_energy_ratio_max = p95 of energy_ratio for amp_sigma>8 events. '
            'Feed into preprocess_magic_bayes.py --cr-fwhm-min and '
            '--cr-energy-ratio-max, then validate.'
        ),
    }


def process_rep(rep_dir, label, z_thresholds, overwrite_existing):
    rep_dir = Path(rep_dir)
    src = rep_dir / 'stage3_bg_removed.npz'
    pop = rep_dir / 'analysis' / 'population_snr.npz'
    if not src.exists():
        print(f'  [{label}] Missing stage3_bg_removed.npz - skipping.')
        return
    if not pop.exists():
        print(f'  [{label}] Missing analysis/population_snr.npz - '
              'run batch_snr_calculation.py first. Skipping.')
        return
    analysis_dir = rep_dir / 'analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)
    out_events = analysis_dir / 'cr_events.npz'

    if not overwrite_existing and out_events.exists():
        print(f'  [{label}] Skipping -- cr_events.npz already exists')
        return

    with np.load(src, allow_pickle=True) as f:
        intensities = f['intensities']
        wavelengths = f['wavelengths']
    with np.load(pop, allow_pickle=True) as f:
        noise_mad_sigma = f['noise_mad_sigma']
        signal_med = f['signal_med']
        eps_pop = float(f['eps']) if 'eps' in f.files else 1e-10

    n_spectra = int(intensities.shape[0])
    print(f'  [{label}] Detecting events ({n_spectra} spectra, '
          f'z_thresholds={list(z_thresholds)}) ...')
    events = detect_events(
        intensities, wavelengths, noise_mad_sigma,
        z_thresholds=z_thresholds, eps=eps_pop,
    )
    print(f'  [{label}] {len(events)} total events across thresholds')

    events_arr = events_to_record_array(events)
    np.savez_compressed(out_events, events=events_arr,
                        wavelengths=wavelengths,
                        z_thresholds=np.asarray(list(z_thresholds), dtype=np.float64))

    z_thresh_used = float(min(z_thresholds))
    plot_amplitude_hist(events_arr, analysis_dir / 'cr_amplitude_hist.png')
    plot_width_hist(events_arr, analysis_dir / 'cr_width_hist.png')
    plot_amp_vs_width(events_arr, analysis_dir / 'cr_amp_vs_width.png', z_thresh_used)
    plot_position_vs_signal(events_arr, z_thresh_used, signal_med,
                            wavelengths, analysis_dir / 'cr_position_vs_signal.png')
    plot_per_spectrum(events_arr, n_spectra, analysis_dir / 'cr_per_spectrum.png')

    proposal = build_threshold_proposal(events_arr, z_thresh_used, n_spectra)
    with open(analysis_dir / 'cr_threshold_proposal.json', 'w') as f:
        json.dump(proposal, f, indent=2)
    print(f'  [{label}] cr_*.png + cr_events.npz + cr_threshold_proposal.json saved')


def discover_processed_reps(root, length=None, sequence=None, rep=None):
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


def parse_z_thresholds(spec):
    parts = [p.strip() for p in spec.split(',') if p.strip()]
    return tuple(float(p) for p in parts)


def main():
    parser = argparse.ArgumentParser(
        description='Per-rep cosmic-ray event investigation from stage3_bg_removed.npz'
    )
    parser.add_argument('--processed-root', type=Path,
                        default=Path('data/custom/processed/magic_explore'))
    parser.add_argument('--length', type=str, default=None)
    parser.add_argument('--sequence', type=str, default=None)
    parser.add_argument('--rep', type=str, default=None)
    parser.add_argument('--z-thresholds', type=str, default='5,8,12',
                        help='Comma-separated list of sigma thresholds (default: 5,8,12)')
    parser.add_argument('--overwrite-existing', action='store_true')
    args = parser.parse_args()

    z_thresholds = parse_z_thresholds(args.z_thresholds)

    print('Cosmic-Ray Investigation (per-rep)')
    print('=' * 60)
    print(f'Processed root: {args.processed_root.resolve()}')
    print(f'z-thresholds:   {z_thresholds}')
    print('=' * 60)

    rep_dirs = discover_processed_reps(args.processed_root,
                                       args.length, args.sequence, args.rep)
    if not rep_dirs:
        print('No matching processed reps found.')
        return
    print(f'Found {len(rep_dirs)} rep(s) to investigate.\n')

    for rep_dir in rep_dirs:
        rel = rep_dir.relative_to(args.processed_root)
        label = str(rel).replace('\\', '/')
        print(f"\n{'=' * 60}")
        print(f'Investigating: {label}')
        print(f"{'=' * 60}")
        try:
            process_rep(rep_dir, label, z_thresholds, args.overwrite_existing)
        except Exception as e:
            print(f'  ERROR investigating {label}: {e}')
            import traceback
            traceback.print_exc()

    print('\nDone.')


if __name__ == '__main__':
    main()
