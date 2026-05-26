"""Experiment 1 — Wavelet denoising for Raman spectra.

Compares wavelet-based denoising (BayesShrink, VisuShrink) against
Savitzky-Golay smoothing on raw test spectra, followed by BubbleFill
baseline correction.

Usage:
    python scripts/debug_wavelet_denoising.py

    python scripts/debug_wavelet_denoising.py \
        --input data/custom/processed/primary_magic/2-letter/SA/rep1/raw_test_spectrum.npz
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pywt

from primarymagic import PreprocessingPipeline
from primarymagic.data.spectraio import load_from_npz, SpectraCollection, Spectrum


def wavelet_denoise(intensities, wavelet='db4', level=None, mode='soft',
                    threshold_method='bayes'):
    """Denoise a 1D signal using discrete wavelet transform.

    Args:
        intensities: 1D array of intensities.
        wavelet: Wavelet name (e.g. 'db4', 'db6', 'sym6').
        level: Decomposition level (None = max level).
        mode: Thresholding mode ('soft' or 'hard').
        threshold_method: 'bayes' (BayesShrink) or 'visu' (VisuShrink).

    Returns:
        Denoised 1D array.
    """
    n_orig = len(intensities)

    # Pad to next power of 2 by repeating the last value
    n_padded = int(2 ** np.ceil(np.log2(n_orig)))
    if n_padded > n_orig:
        intensities = np.concatenate([intensities,
                                      np.full(n_padded - n_orig, intensities[-1])])

    coeffs = pywt.wavedec(intensities, wavelet, level=level)

    # Estimate noise from finest detail coefficients
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745

    denoised_coeffs = [coeffs[0]]  # keep approximation coefficients
    for i, detail in enumerate(coeffs[1:], 1):
        if threshold_method == 'visu':
            # VisuShrink: universal threshold
            threshold = sigma * np.sqrt(2 * np.log(n_padded))
        elif threshold_method == 'bayes':
            # BayesShrink: level-dependent adaptive threshold
            detail_var = np.mean(detail ** 2)
            signal_var = max(detail_var - sigma ** 2, 0)
            if signal_var == 0:
                threshold = np.max(np.abs(detail))
            else:
                threshold = sigma ** 2 / np.sqrt(signal_var)
        else:
            raise ValueError(f"Unknown threshold method: {threshold_method}")

        denoised_coeffs.append(pywt.threshold(detail, threshold, mode=mode))

    return pywt.waverec(denoised_coeffs, wavelet)[:n_orig]


def apply_wavelet_to_collection(collection, **kwargs):
    """Apply wavelet denoising to a SpectraCollection, return new collection."""
    wl = collection.wavelengths
    new_spectra = []
    for spectrum in collection.spectra:
        denoised = wavelet_denoise(spectrum.intensities, **kwargs)
        new_spectra.append(Spectrum(
            data=np.column_stack([wl, denoised]),
            x=spectrum.x, y=spectrum.y,
        ))
    return SpectraCollection(
        spectra=new_spectra,
        source_file=collection.source_file,
    )


def fix_edges(collection, window=50, threshold=2.0):
    """Automatically detect and fix edge artifacts at both ends of a spectrum."""
    wl = collection.wavelengths
    new_spectra = []
    for spectrum in collection.spectra:
        intensities = spectrum.intensities.copy()

        # Fix start
        edge_mean = np.mean(intensities[:window])
        adj_mean = np.mean(intensities[window:2 * window])
        adj_std = np.std(intensities[window:2 * window])
        if adj_std > 0 and abs(edge_mean - adj_mean) > threshold * adj_std:
            intensities[:window] = adj_mean

        # Fix end
        edge_mean = np.mean(intensities[-window:])
        adj_mean = np.mean(intensities[-2 * window:-window])
        adj_std = np.std(intensities[-2 * window:-window])
        if adj_std > 0 and abs(edge_mean - adj_mean) > threshold * adj_std:
            intensities[-window:] = adj_mean

        new_spectra.append(Spectrum(
            data=np.column_stack([wl, intensities]),
            x=spectrum.x, y=spectrum.y,
        ))
    return SpectraCollection(spectra=new_spectra, source_file=collection.source_file)


def shift_baseline(collection, percentile=5):
    """Subtract low percentile and clip negatives to zero."""
    wl = collection.wavelengths
    new_spectra = []
    for spectrum in collection.spectra:
        intensities = spectrum.intensities.copy()
        intensities -= np.percentile(intensities, percentile)
        intensities = np.clip(intensities, 0, None)
        new_spectra.append(Spectrum(
            data=np.column_stack([wl, intensities]),
            x=spectrum.x, y=spectrum.y,
        ))
    return SpectraCollection(spectra=new_spectra, source_file=collection.source_file)


def main():
    default_input = (Path(__file__).resolve().parent.parent
                     / 'data' / 'custom' / 'processed' / 'primary_magic'
                     / '2-letter' / 'GA')

    parser = argparse.ArgumentParser(
        description="Experiment 1: Wavelet denoising for Raman spectra"
    )
    parser.add_argument("--input", type=Path, default=default_input,
                        help="Path to sequence directory with rep1/rep2/rep3")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory for plots")
    parser.add_argument("--no-display", action="store_true",
                        help="Do not display plots")
    args = parser.parse_args()

    out_dir = args.output or Path(
        r'C:\Users\mfarzi\Documents\obsidian\primarybio\assets\experiments\dev_raman_spectroscopy_preprocessing')
    out_dir.mkdir(parents=True, exist_ok=True)

    seq_name = args.input.name

    # Load all reps and apply edge fix
    reps = ['rep1', 'rep2', 'rep3']
    spectra = {}
    for rep in reps:
        npz = args.input / rep / 'raw_test_spectrum.npz'
        if npz.exists():
            raw = load_from_npz(npz)
            spectra[rep] = fix_edges(raw)
            print(f"Loaded {rep}: {len(spectra[rep].wavelengths)} points")

    # Helper: BubbleFill -> shift baseline -> normalise
    def bubblefill_normalise(s):
        corrected = (PreprocessingPipeline(s)
                     .subtract_baseline(method='bubblefill',
                                        min_bubble_widths=50, fit_order=1)
                     .result())
        shifted = shift_baseline(corrected)
        return PreprocessingPipeline(shifted).normalize().result()

    # Define pipelines to compare
    pipelines = [
        ("SG(11,3) → BubbleFill [DEFAULT]", "sg_default",
         lambda s: bubblefill_normalise(
             PreprocessingPipeline(s)
             .smooth(window_length=11, polyorder=3)
             .result())),
        ("Wavelet db4 BayesShrink → BubbleFill", "wav_db4_bayes",
         lambda s: bubblefill_normalise(
             apply_wavelet_to_collection(s, wavelet='db4',
                                         threshold_method='bayes'))),
        ("Wavelet db6 BayesShrink → BubbleFill", "wav_db6_bayes",
         lambda s: bubblefill_normalise(
             apply_wavelet_to_collection(s, wavelet='db6',
                                         threshold_method='bayes'))),
        ("Wavelet db4 VisuShrink → BubbleFill", "wav_db4_visu",
         lambda s: bubblefill_normalise(
             apply_wavelet_to_collection(s, wavelet='db4',
                                         threshold_method='visu'))),
        ("Wavelet sym6 BayesShrink → BubbleFill", "wav_sym6_bayes",
         lambda s: bubblefill_normalise(
             apply_wavelet_to_collection(s, wavelet='sym6',
                                         threshold_method='bayes'))),
        ("Wavelet sym6 VisuShrink → BubbleFill", "wav_sym6_visu",
         lambda s: bubblefill_normalise(
             apply_wavelet_to_collection(s, wavelet='sym6',
                                         threshold_method='visu'))),
    ]

    n_pipelines = len(pipelines)
    n_reps = len(spectra)

    # One figure per pipeline, showing all reps
    for title, tag, pipeline_fn in pipelines:
        fig, axes = plt.subplots(n_reps, 1, figsize=(14, 3 * n_reps), sharex=True)
        if n_reps == 1:
            axes = [axes]
        for i, (rep, spec) in enumerate(spectra.items()):
            result = pipeline_fn(spec)
            axes[i].plot(result.wavelengths, result[0].intensities, lw=1.0)
            axes[i].set_ylabel('Intensity')
            axes[i].set_title(f'{seq_name} {rep} — {title}')
            axes[i].grid(True, alpha=0.3)
        axes[-1].set_xlabel('Wavenumber (cm⁻¹)')
        plt.tight_layout()
        out_file = out_dir / f'{seq_name}_{tag}.png'
        plt.savefig(out_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {out_file.name}")

    # Summary figure: all pipelines overlaid for each rep
    fig, axes = plt.subplots(n_reps, 1, figsize=(14, 4 * n_reps), sharex=True)
    if n_reps == 1:
        axes = [axes]
    colors = ['k', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    for i, (rep, spec) in enumerate(spectra.items()):
        for j, (title, tag, pipeline_fn) in enumerate(pipelines):
            result = pipeline_fn(spec)
            label = title.split('→')[0].strip()
            axes[i].plot(result.wavelengths, result[0].intensities,
                         color=colors[j], lw=1.0, alpha=0.8, label=label)
        axes[i].set_ylabel('Intensity')
        axes[i].set_title(f'{seq_name} {rep} — All Methods Compared')
        axes[i].legend(fontsize=8, loc='upper right')
        axes[i].grid(True, alpha=0.3)
    axes[-1].set_xlabel('Wavenumber (cm⁻¹)')
    plt.tight_layout()
    out_file = out_dir / f'{seq_name}_wavelet_comparison.png'
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    if not args.no_display:
        plt.show()
    plt.close()
    print(f"Saved {out_file.name}")


if __name__ == "__main__":
    main()
