"""One-off inspection: plot N random stage3_bg_removed spectra for one rep,
overlaid with the population mean and median, to check whether the per-wn
MAD-sigma in population_snr.png is dominated by detector noise or by
spectrum-to-spectrum amplitude variability at peak positions.

Usage:
    python inspect_stage3_random_samples.py <rep_dir> [n_samples]
e.g.
    python inspect_stage3_random_samples.py data/custom/processed/magic_explore/1-letter/F/rep1 50
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    rep_dir = Path(sys.argv[1])
    n_samples = int(sys.argv[2]) if len(sys.argv) >= 3 else 50

    src = rep_dir / 'stage3_bg_removed.npz'
    if not src.exists():
        print(f'Missing {src}')
        sys.exit(1)

    with np.load(src, allow_pickle=True) as f:
        intensities = f['intensities']
        wavelengths = f['wavelengths']
    N = intensities.shape[0]
    print(f'Loaded {N} spectra, {len(wavelengths)} wn from {src}')

    rng = np.random.default_rng(42)
    n = min(n_samples, N)
    idx = rng.choice(N, size=n, replace=False)

    mean = intensities.mean(axis=0)
    median = np.median(intensities, axis=0)
    mad_sigma = 1.4826 * np.median(np.abs(intensities - median[None, :]), axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    ax = axes[0]
    for i in idx:
        ax.plot(wavelengths, intensities[i], linewidth=0.4, alpha=0.4, color='steelblue')
    ax.plot(wavelengths, mean, color='black', linewidth=1.4, label='mean')
    ax.plot(wavelengths, median, color='red', linewidth=1.0, linestyle='--', label='median')
    ax.set_ylabel('Intensity (stage3, bg-removed)')
    ax.set_title(f'{rep_dir.name}: {n} random of {N} spectra (faint blue) + mean (black) + median (red dashed)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    ax.plot(wavelengths, mad_sigma, color='darkorange', linewidth=1.0, label='MAD-sigma (across spectra)')
    ax.set_xlabel('Wavenumber (cm-1)')
    ax.set_ylabel('MAD-sigma')
    ax.set_title('Per-wavenumber MAD-sigma  (peaks coincide with signal peaks => amplitude variability, not detector noise)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out = rep_dir / 'analysis' / 'stage3_random_samples.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
