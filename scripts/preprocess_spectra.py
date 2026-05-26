"""
ORPL-Based Preprocessing Script for Raman Spectra (Clean API)
=============================================================

This script uses the spectra package's native ORPL-adapted algorithms for
baseline correction (BubbleFill), cosmic ray removal, and quality metrics
(ASSI). It produces identical results to preprocess_orpl.py but without
any sys.path hacks or direct ORPL imports.

Directory structure:
    data/raw/{n-letter}/{SEQ}/{rep#}/       (input)
    data/processed/primary_magic/{n-letter}/{SEQ}/{rep#}/  (output)

Key features:
- Cosmic ray removal via remove_cosmic_rays()
- BubbleFill morphological baseline via subtract_baseline(method='bubblefill')
- ASSI quality metric via calculate_assi()
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from primarymagic import (
    SpectraCollection,
    Spectrum,
    read_spectrum_file,
    PreprocessingPipeline,
    normalize_minmax,
    calculate_snr,
    calculate_assi,
    export_to_npz,
)

# Set random seed for reproducible example selection
np.random.seed(42)

# Plot style
plt.rcParams['figure.figsize'] = (12, 4)
plt.rcParams['font.size'] = 10


def get_txt_filepath(folder):
    """Get file paths for spectra data from a rep directory.

    Args:
        folder: Path to the rep directory containing .txt files

    Returns:
        tuple: (collection_path, test_spectrum_path)
    """
    folder = Path(folder)
    filenames = list(folder.glob('*.txt'))

    collection_path = None
    test_spectrum_path = None
    if len(filenames) > 0:
        for filename in filenames:
            if 'test' in filename.stem.lower():
                test_spectrum_path = filename
            if 'power' in filename.stem.lower():
                collection_path = filename
    return collection_path, test_spectrum_path


def get_fingerprint(collection, threshold=50):
    """Generate fingerprint from high-quality spectra.

    Args:
        collection: SpectraCollection object
        threshold: SNR threshold for filtering

    Returns:
        Normalized fingerprint Spectrum
    """
    snr = calculate_snr(collection)
    intensities = collection.to_intensity_matrix()
    mean_intensity = intensities[snr > threshold, :].mean(axis=0)
    fingerprint = Spectrum(np.stack((collection.wavelengths, mean_intensity), axis=1))
    return normalize_minmax(fingerprint)


def write_quality_mask(save_dir, snr, assi_values, signal_mask,
                       snr_threshold, assi_threshold):
    """Write the per-spectrum quality-mask sidecar to ``save_dir / 'mask.npz'``.

    See docs/superpowers/specs/2026-05-10-preprocess-spectra-mask-sidecar-design.md
    for the file contract.

    Args:
        save_dir: Directory in which to write ``mask.npz``.
        snr: Per-spectrum SNR values, length N.
        assi_values: Per-spectrum ASSI values, length N.
        signal_mask: Boolean array-like of length N — pass/fail per spectrum.
        snr_threshold: SNR threshold used for the pass decision.
        assi_threshold: ASSI threshold used for the pass decision.
    """
    save_dir = Path(save_dir)
    passed = np.asarray(signal_mask, dtype=bool)
    n = len(passed)
    clean_index = np.full(n, -1, dtype=np.int32)
    clean_index[passed] = np.arange(int(passed.sum()), dtype=np.int32)
    np.savez_compressed(
        save_dir / 'mask.npz',
        raw_index=np.arange(n, dtype=np.int32),
        clean_index=clean_index,
        passed=passed,
        snr=np.asarray(snr, dtype=np.float64),
        assi=np.asarray(assi_values, dtype=np.float64),
        snr_threshold=np.float64(snr_threshold),
        assi_threshold=np.float64(assi_threshold),
    )


def plot_collection_and_test_spectrum(collection, test_spectrum, title='Raman Spectroscopy Data',
                                       save_path=None, display=False):
    """Visualise collection and test spectrum."""
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 4))

    if collection:
        mean_intensities = collection.to_intensity_matrix().mean(axis=0)
        ax1.plot(collection.wavelengths, mean_intensities,
                 'k-', linewidth=1.5, label='Fingerprint')

    if test_spectrum:
        ax1.plot(test_spectrum.wavelengths, test_spectrum.intensities,
                 'g-', linewidth=1.5, alpha=1, label='Test Spectrum')

    ax1.set_xlabel('Wavenumber (cm\u207b\u00b9)')
    ax1.set_ylabel('Intensity (a.u.)')
    ax1.set_title(title)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot at {save_path}")
    if display:
        plt.show()
    plt.close(fig)


def process_pipeline_orpl(raw_folder, save_dir, label,
                          cosmic_ray_width=3, cosmic_ray_std_factor=5,
                          bubble_min_widths=50, bubble_fit_order=1,
                          savgol_window=11, savgol_polyorder=3,
                          snr_threshold=50, assi_threshold=0.65,
                          save_npz=True, save_png=True, display=False):
    """Process spectra data using ORPL-based pipeline.

    Pipeline steps:
    1. Load raw data
    2. Cosmic ray removal
    3. Baseline correction with BubbleFill
    4. Smoothing with Savitzky-Golay
    5. Normalization
    6. SNR filtering
    7. Export processed data

    Args:
        raw_folder: Path to the raw rep directory (e.g., data/raw/2-letter/AD/rep1)
        save_dir: Path to the output directory (e.g., data/processed/.../2-letter/AD/rep1)
        label: Display label (e.g., '2-letter/AD/rep1')
        cosmic_ray_width: Cosmic ray filter width (default: 3)
        cosmic_ray_std_factor: Cosmic ray detection threshold (default: 5)
        bubble_min_widths: BubbleFill minimum bubble width (default: 50)
        bubble_fit_order: BubbleFill polynomial order (default: 1)
        savgol_window: Savitzky-Golay window length (default: 11)
        savgol_polyorder: Savitzky-Golay polynomial order (default: 3)
        snr_threshold: SNR quality threshold (default: 50)
        assi_threshold: ASSI quality threshold (default: 0.65)
        save_npz: Whether to save data as .npz files
        save_png: Whether to save plots as .png files
        display: Whether to display graphs for visual inspection
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load raw data
    collection_path, test_spectrum_path = get_txt_filepath(raw_folder)
    if collection_path:
        collection = read_spectrum_file(collection_path)
        print(f"{label}: {len(collection)} spectra loaded")
    else:
        print(f"{label}: Missing spectra")
        return

    if test_spectrum_path:
        test_collection = read_spectrum_file(test_spectrum_path)
        print(f"Test spectrum loaded for {label}")
        if save_npz:
            export_to_npz(test_collection, save_dir / 'raw_test_spectrum.npz')
        test_spectrum = test_collection[0]
    else:
        test_spectrum = None
        print(f"Missing test spectrum for {label}")

    # Plot raw data
    plot_collection_and_test_spectrum(collection, test_spectrum,
                                       title=f'{label} Raman Spectroscopy Data (RAW)',
                                       save_path=save_dir / 'raw_data.png' if save_png else None,
                                       display=display)

    # Extract sequence name from label (e.g., '2-letter/DR/rep1' -> 'DR')
    seq_name = Path(label).parent.name

    # Ad hoc fix for specific peptides
    if seq_name in ['DR', 'FA', 'FD', 'SF', 'RAD', 'GDR', 'AG']:
        spectra = []
        length = 50
        for spectrum in collection.spectra:
            intesities = spectrum.intensities
            intesities[:length] = np.mean(intesities[length:2 * length])
            spectrum.data[:, 1] = intesities
            spectra.append(spectrum)
        if test_spectrum:
            intesities = test_spectrum.intensities
            intesities[:length] = np.mean(intesities[length:2 * length])
            test_spectrum.data[:, 1] = intesities

    raw_collection = collection

    # Steps 2-5: Full preprocessing pipeline
    print(f"  Applying cosmic ray removal (width={cosmic_ray_width}, std_factor={cosmic_ray_std_factor})...")
    print(f"  Applying BubbleFill baseline removal (min_bubble_widths={bubble_min_widths}, fit_order={bubble_fit_order})...")
    print(f"  Applying Savitzky-Golay smoothing (window={savgol_window}, polyorder={savgol_polyorder})...")

    collection = (PreprocessingPipeline(collection)
                  .remove_cosmic_rays(width=cosmic_ray_width, std_factor=cosmic_ray_std_factor)
                  .subtract_baseline(method='bubblefill',
                                     min_bubble_widths=bubble_min_widths,
                                     fit_order=bubble_fit_order)
                  .smooth(window_length=savgol_window, polyorder=savgol_polyorder)
                  .normalize()
                  .result())

    if test_spectrum:
        test_spectrum = (PreprocessingPipeline(test_spectrum)
                         .remove_cosmic_rays(width=cosmic_ray_width, std_factor=cosmic_ray_std_factor)
                         .subtract_baseline(method='bubblefill',
                                            min_bubble_widths=bubble_min_widths,
                                            fit_order=bubble_fit_order)
                         .smooth(window_length=savgol_window, polyorder=savgol_polyorder)
                         .normalize()
                         .result())

    # Save preprocessed data
    if save_npz:
        export_to_npz(collection, save_dir / 'data.npz')
        print(f"Saved preprocessed spectra data at {save_dir / 'data.npz'}")

        if test_spectrum:
            test_collection = SpectraCollection([test_spectrum], source_file=test_spectrum_path)
            export_to_npz(test_collection, save_dir / 'test_spectrum.npz')
            print(f"Saved preprocessed test spectrum at {save_dir / 'test_spectrum.npz'}")

    # Plot preprocessed data
    plot_collection_and_test_spectrum(collection, test_spectrum,
                                       title=f'{label} Raman Spectroscopy Data (ORPL preprocessed)',
                                       save_path=save_dir / 'preprocessed_data.png' if save_png else None,
                                       display=display)

    # Step 6: Quality filtering with SNR and ASSI
    snr = calculate_snr(collection)

    # Calculate ASSI for each spectrum
    assi_values = calculate_assi(collection)
    mean_assi = assi_values.mean()
    print(f"  Quality metrics: Mean ASSI = {mean_assi:.4f}")

    # Filter spectra with SNR > threshold AND ASSI > threshold
    signal_mask = [s > snr_threshold and a > assi_threshold for s, a in zip(snr, assi_values)]
    clean_spectra = [sp for sp, keep in zip(collection, signal_mask) if keep]
    raw_signal = [sp for sp, keep in zip(raw_collection, signal_mask) if keep]
    raw_background = [sp for sp, keep in zip(raw_collection, signal_mask) if not keep]

    if save_npz:
        write_quality_mask(save_dir, snr, assi_values, signal_mask,
                           snr_threshold, assi_threshold)
        print(f"Saved quality mask at {save_dir / 'mask.npz'}")

        export_to_npz(raw_collection, save_dir / 'raw_all.npz')
        print(f"Saved full raw collection at {save_dir / 'raw_all.npz'}")

    if clean_spectra:
        clean_collection = SpectraCollection(clean_spectra, source_file=collection_path)
        print(f"{label}: {len(clean_collection)} spectra with SNR > {snr_threshold} and ASSI > {assi_threshold}")
        if save_npz:
            export_to_npz(clean_collection, save_dir / 'clean_data.npz')
            print(f"Saved clean data at {save_dir / 'clean_data.npz'}")

            raw_signal_collection = SpectraCollection(raw_signal, source_file=collection_path)
            export_to_npz(raw_signal_collection, save_dir / 'raw_data.npz')
            print(f"Saved raw signal data at {save_dir / 'raw_data.npz'}")
    else:
        print(f"{label}: No spectra with SNR > {snr_threshold} and ASSI > {assi_threshold}")
        clean_collection = None

    if raw_background and save_npz:
        raw_bg_collection = SpectraCollection(raw_background, source_file=collection_path)
        export_to_npz(raw_bg_collection, save_dir / 'raw_background.npz')
        print(f"Saved raw background data at {save_dir / 'raw_background.npz'}")

    # Step 7: Generate fingerprint
    fingerprint = get_fingerprint(collection, threshold=snr_threshold)
    fingerprint_collection = SpectraCollection([fingerprint], source_file=collection_path)
    if save_npz:
        export_to_npz(fingerprint_collection, save_dir / 'fingerprint.npz')
        print(f"Saved fingerprint data at {save_dir / 'fingerprint.npz'}")

    plot_collection_and_test_spectrum(fingerprint_collection, test_spectrum,
                                       title=f'{label} Raman Spectroscopy Data (SNR>{snr_threshold})',
                                       save_path=save_dir / 'fingerprint.png' if save_png else None,
                                       display=display)

    # Summary
    print(f"\n  {label} Processing Summary:")
    print(f"    - Total spectra: {len(collection)}")
    print(f"    - Clean spectra (SNR > {snr_threshold}, ASSI > {assi_threshold}): {len(clean_spectra) if clean_spectra else 0}")
    print(f"    - Background spectra: {len(raw_background) if raw_background else 0}")
    print(f"    - Mean ASSI: {mean_assi:.4f}")


if __name__ == '__main__':
    default_root = Path(__file__).resolve().parent.parent / 'data'

    parser = argparse.ArgumentParser(description="Preprocess Raman spectra.")
    parser.add_argument("--raw-root", type=Path, default=default_root / 'raw',
                        help="Root directory of raw data")
    parser.add_argument("--processed-root", type=Path,
                        default=default_root / 'processed' / 'primary_magic',
                        help="Root directory for processed output")
    parser.add_argument("--snr-threshold", type=float, default=50,
                        help="SNR quality threshold (default: 50)")
    parser.add_argument("--assi-threshold", type=float, default=0.65,
                        help="ASSI quality threshold (default: 0.65)")
    args = parser.parse_args()

    raw_root = args.raw_root
    processed_root = args.processed_root

    # ORPL preprocessing parameters
    params = {
        'cosmic_ray_width': 3,
        'cosmic_ray_std_factor': 5,
        'bubble_min_widths': 50,
        'bubble_fit_order': 1,
        'savgol_window': 11,
        'savgol_polyorder': 3,
        'snr_threshold': args.snr_threshold,
        'assi_threshold': args.assi_threshold,
    }

    print("ORPL-Based Preprocessing Pipeline")
    print("=" * 50)
    print(f"Raw data:   {raw_root.resolve()}")
    print(f"Output:     {processed_root.resolve()}")
    print(f"Parameters: {params}")
    print("=" * 50)

    # Collect all rep directories (leaf directories containing raw data).
    # Standard: {n-letter}/{SEQ}/rep#
    # PTM:      1-letter/PTM/{code}/rep#
    rep_dirs = []
    for length_dir in sorted(raw_root.iterdir()):
        if not length_dir.is_dir():
            continue
        for seq_dir in sorted(length_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            for child in sorted(seq_dir.iterdir()):
                if not child.is_dir():
                    continue
                if child.name.startswith('rep'):
                    # Standard case: child is a rep directory
                    rep_dirs.append(child)
                else:
                    # PTM case: child is a code directory, look one level deeper
                    for rep_dir in sorted(child.iterdir()):
                        if rep_dir.is_dir() and rep_dir.name.startswith('rep'):
                            rep_dirs.append(rep_dir)

    for rep_dir in rep_dirs:
        rel = rep_dir.relative_to(raw_root)
        label = str(rel).replace('\\', '/')
        save_dir = processed_root / rel

        print(f"\n{'='*50}")
        print(f"Processing {label}")
        print(f"{'='*50}")

        if (save_dir / 'clean_data.npz').is_file():
            print('skipped (already processed)')
        else:
            process_pipeline_orpl(
                rep_dir,
                save_dir,
                label,
                **params,
                display=False
            )
