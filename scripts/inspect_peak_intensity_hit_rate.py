"""Confirm hit-rate hypothesis: histogram per-spectrum intensity at the
dominant peak position for one or more reps. Dense-hit reps (every spectrum
has the peak) should show a unimodal high distribution; sparse-hit reps
(most spectra empty) should show a peak near zero with a tail of "hits".

Usage:
    python inspect_peak_intensity_hit_rate.py <rep_dir> [<rep_dir> ...]
e.g.
    python inspect_peak_intensity_hit_rate.py \
        data/custom/processed/magic_explore/1-letter/F/rep1 \
        data/custom/processed/magic_explore/3-letter/FDA/rep1
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_rep(rep_dir):
    rep_dir = Path(rep_dir)
    src = rep_dir / 'stage3_bg_removed.npz'
    with np.load(src, allow_pickle=True) as f:
        intensities = f['intensities']
        wavelengths = f['wavelengths']
    return intensities, wavelengths


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    rep_dirs = [Path(p) for p in sys.argv[1:]]

    n_reps = len(rep_dirs)
    fig, axes = plt.subplots(n_reps, 1, figsize=(11, 4 * n_reps), squeeze=False)

    summary = []
    for ax, rep_dir in zip(axes[:, 0], rep_dirs):
        intensities, wavelengths = load_rep(rep_dir)
        N = intensities.shape[0]
        mean_signal = intensities.mean(axis=0)
        peak_idx = int(np.argmax(mean_signal))
        peak_wn = float(wavelengths[peak_idx])
        per_spectrum_at_peak = intensities[:, peak_idx]

        # Hit-rate via bimodality probe: fraction above 3 * MAD-sigma of the
        # full population at this wn (a "hit" stands out from the baseline
        # noise of empty spectra).
        med = float(np.median(per_spectrum_at_peak))
        mad_sigma = 1.4826 * float(np.median(np.abs(per_spectrum_at_peak - med)))
        hit_threshold = max(med + 3 * mad_sigma, 0.0)
        n_hits = int((per_spectrum_at_peak > hit_threshold).sum())
        hit_rate = n_hits / N if N > 0 else 0.0

        label = f'{rep_dir.parent.parent.name}/{rep_dir.parent.name}/{rep_dir.name}'
        ax.hist(per_spectrum_at_peak, bins=80, color='steelblue',
                edgecolor='black', alpha=0.8)
        ax.axvline(hit_threshold, color='red', linestyle='--', linewidth=1.0,
                   label=f'hit threshold = median + 3*MAD = {hit_threshold:.1f}')
        ax.axvline(med, color='black', linestyle=':', linewidth=0.8,
                   label=f'median = {med:.1f}')
        ax.set_xlabel(f'Intensity at peak wn = {peak_wn:.1f} cm-1 (pixel {peak_idx})')
        ax.set_ylabel('Number of spectra')
        ax.set_title(
            f'{label}: per-spectrum intensity at dominant peak  '
            f'(N={N}, hits={n_hits}, hit_rate={hit_rate*100:.1f}%)')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.2)
        ax.set_yscale('log')

        summary.append((label, N, peak_wn, peak_idx, med, mad_sigma,
                        n_hits, hit_rate))

    plt.tight_layout()
    out = Path('peak_intensity_hit_rate.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved: {out.resolve()}')
    print()
    print('Summary:')
    print(f"  {'rep':<40s} {'N':>6s} {'peak_wn':>10s} {'median':>10s} "
          f"{'mad_sigma':>10s} {'hits':>6s} {'hit_rate':>10s}")
    for label, N, peak_wn, peak_idx, med, mad_sigma, n_hits, hit_rate in summary:
        print(f"  {label:<40s} {N:>6d} {peak_wn:>10.2f} {med:>10.2f} "
              f"{mad_sigma:>10.2f} {n_hits:>6d} {hit_rate*100:>9.2f}%")


if __name__ == '__main__':
    main()
