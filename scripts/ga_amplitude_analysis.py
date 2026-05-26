"""
Exploratory analysis: measure natural signal amplitude per GA dipeptide replicate
after denoising and baseline removal, but BEFORE normalization.

SNR is computed AFTER preprocessing (not on raw data) to properly separate
foreground (signal) from background (noise) spectra.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from primarymagic import (
    Spectrum, SpectraCollection, PreprocessingPipeline,
    read_spectrum_file, calculate_snr,
)

DATA_BASE = "data/custom/raw/2-letter/GA"
OUTPUT_DIR = Path("data/synthetic/GA/amplitude_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REP_FILES = {
    "rep1": f"{DATA_BASE}/rep1/GA_crystal_532_.25s_100power_1900spectra.txt",
    "rep2": f"{DATA_BASE}/rep2/GA_crystal_532_.25s_100power_1530spectra_redo.txt",
    "rep3": f"{DATA_BASE}/rep3/GA_crystal_532_.25s_100power_700spectra_rep3.txt",
}

SNR_THRESHOLD = 50
N_PLOT_SAMPLES = 5  # number of foreground/background samples to plot

plt.rcParams['figure.figsize'] = (14, 5)
plt.rcParams['font.size'] = 10

all_peaks = []   # collect peaks across reps for global summary

print("=" * 70)
print("GA Dipeptide — Peak Amplitude Analysis (post-denoise, pre-normalise)")
print("  SNR computed AFTER preprocessing")
print("=" * 70)

for rep_name, filepath in REP_FILES.items():
    print(f"\n{'─' * 50}")
    print(f"  {rep_name}: {filepath}")
    print(f"{'─' * 50}")

    # 1. Load raw spectra
    collection = read_spectrum_file(filepath)
    print(f"  Raw spectra loaded: {len(collection)}")

    # 2. Preprocess ALL spectra first: cosmic ray → SG smooth → BubbleFill baseline
    #    (NO normalisation)
    processed = (
        PreprocessingPipeline(collection)
        .remove_cosmic_rays(width=3, std_factor=5)
        .smooth(window_length=11, polyorder=3)
        .subtract_baseline(method='bubblefill', min_bubble_widths=50, fit_order=1)
        .result()
    )

    # 3. Compute SNR AFTER preprocessing
    snr_values = calculate_snr(processed)
    signal_mask = snr_values > SNR_THRESHOLD
    bg_mask = ~signal_mask

    n_signal = signal_mask.sum()
    n_bg = bg_mask.sum()
    print(f"  After preprocessing — SNR > {SNR_THRESHOLD}:")
    print(f"    Foreground (signal): {n_signal} / {len(collection)}")
    print(f"    Background (noise):  {n_bg} / {len(collection)}")
    print(f"    SNR range: {snr_values.min():.1f} – {snr_values.max():.1f} "
          f"(median {np.median(snr_values):.1f})")

    if n_signal == 0:
        print("  WARNING: No signal spectra found!")
        continue

    # 4. Measure peak intensity for foreground spectra (pre-normalisation)
    fg_spectra = [s for s, keep in zip(processed.spectra, signal_mask) if keep]
    bg_spectra = [s for s, keep in zip(processed.spectra, bg_mask) if keep]
    peak_amplitudes = np.array([np.max(s.intensities) for s in fg_spectra])

    all_peaks.extend(peak_amplitudes.tolist())

    # 5. Per-rep statistics
    print(f"\n  Peak amplitude statistics (foreground, post-preprocessing, pre-normalise):")
    print(f"    Count  : {len(peak_amplitudes)}")
    print(f"    Mean   : {peak_amplitudes.mean():.2f}")
    print(f"    Std    : {peak_amplitudes.std():.2f}")
    print(f"    Min    : {peak_amplitudes.min():.2f}")
    print(f"    Max    : {peak_amplitudes.max():.2f}")
    print(f"    Median : {np.median(peak_amplitudes):.2f}")

    # 6. Plot sample foreground vs background spectra
    wavelengths = processed.wavelengths

    # Pick foreground samples: top SNR (sure foreground)
    fg_snr = snr_values[signal_mask]
    fg_sorted_idx = np.argsort(fg_snr)[::-1]  # highest SNR first
    fg_plot_idx = fg_sorted_idx[:N_PLOT_SAMPLES]

    # Pick background samples: lowest SNR (sure background)
    bg_snr = snr_values[bg_mask]
    if len(bg_snr) > 0:
        bg_sorted_idx = np.argsort(bg_snr)  # lowest SNR first
        bg_plot_idx = bg_sorted_idx[:N_PLOT_SAMPLES]
    else:
        bg_plot_idx = []

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{rep_name} — Foreground vs Background (post-preprocessing, pre-normalise)',
                 fontsize=12)

    # Panel 1: Sure foreground (highest SNR)
    ax = axes[0]
    for i in fg_plot_idx:
        s = fg_spectra[i]
        ax.plot(wavelengths, s.intensities, linewidth=0.8, alpha=0.7,
                label=f'SNR={fg_snr[i]:.0f}')
    ax.set_xlabel('Wavenumber (cm⁻¹)')
    ax.set_ylabel('Intensity (a.u.)')
    ax.set_title(f'Sure foreground (top {N_PLOT_SAMPLES} SNR)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 2: Sure background (lowest SNR)
    ax = axes[1]
    if len(bg_plot_idx) > 0:
        for i in bg_plot_idx:
            s = bg_spectra[i]
            ax.plot(wavelengths, s.intensities, linewidth=0.8, alpha=0.7,
                    label=f'SNR={bg_snr[i]:.0f}')
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, 'No background spectra', transform=ax.transAxes,
                ha='center', va='center')
    ax.set_xlabel('Wavenumber (cm⁻¹)')
    ax.set_ylabel('Intensity (a.u.)')
    ax.set_title(f'Sure background (bottom {N_PLOT_SAMPLES} SNR)')
    ax.grid(True, alpha=0.3)

    # Panel 3: Max amplitude histogram — foreground vs background overlaid
    ax = axes[2]
    bg_peak_amplitudes = np.array([np.max(s.intensities) for s in bg_spectra]) if bg_spectra else np.array([])
    all_amps = np.concatenate([peak_amplitudes, bg_peak_amplitudes]) if len(bg_peak_amplitudes) > 0 else peak_amplitudes
    bin_edges = np.linspace(all_amps.min(), np.percentile(all_amps, 99), 50)
    if len(bg_peak_amplitudes) > 0:
        ax.hist(bg_peak_amplitudes, bins=bin_edges, edgecolor='black', alpha=0.5,
                color='blue', label=f'Background (n={len(bg_spectra)})')
    ax.hist(peak_amplitudes, bins=bin_edges, edgecolor='black', alpha=0.5,
            color='red', label=f'Foreground (n={len(fg_spectra)})')
    if len(peak_amplitudes) > 0:
        ax.axvline(np.median(peak_amplitudes), color='r', linestyle='--',
                   label=f'FG median={np.median(peak_amplitudes):.0f}')
    if len(bg_peak_amplitudes) > 0:
        ax.axvline(np.median(bg_peak_amplitudes), color='b', linestyle='--',
                   label=f'BG median={np.median(bg_peak_amplitudes):.0f}')
    ax.set_xlabel('Max amplitude (per spectrum)')
    ax.set_ylabel('Count')
    ax.set_title('Max amplitude distribution')
    ax.legend(fontsize=7)

    plt.tight_layout()
    save_path = OUTPUT_DIR / f'{rep_name}_amplitude_analysis.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved plot: {save_path}")

# Global summary
print(f"\n{'=' * 70}")
print("  GLOBAL SUMMARY (all reps combined)")
print(f"{'=' * 70}")
if all_peaks:
    g = np.array(all_peaks)
    print(f"    Total signal spectra : {len(g)}")
    print(f"    Mean peak amplitude  : {g.mean():.2f}")
    print(f"    Std peak amplitude   : {g.std():.2f}")
    print(f"    Min peak amplitude   : {g.min():.2f}")
    print(f"    Max peak amplitude   : {g.max():.2f}")
    print(f"    Median peak amplitude: {np.median(g):.2f}")
print("=" * 70)
