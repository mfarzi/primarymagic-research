"""
ORPL-Based Preprocessing Script for Raman Spectra
=================================================

This script uses ORPL (Open Raman Processing Library) for improved baseline
correction and cosmic ray removal, while maintaining compatibility with the
existing spectra package workflow.

Key improvements over preprocess.py:
- Cosmic ray removal via crfilter_single() - eliminates ad-hoc spike fixes
- BubbleFill morphological baseline - more robust than ALS
- ASSI quality metric alongside SNR
"""

import sys
from pathlib import Path

# Add ORPL modules to path
_orpl_path = Path(__file__).parent.parent / 'dependencies/orpl/src/orpl'
sys.path.insert(0, str(_orpl_path.parent))
sys.path.insert(0, str(_orpl_path))

import numpy as np
import matplotlib.pyplot as plt
from primarymagic import SpectraCollection, Spectrum
from primarymagic import read_spectrum_file, PreprocessingPipeline, normalize_minmax, calculate_snr, export_to_npz

# ORPL imports - import specific modules directly to avoid file_io dependency (requires sif_parser)
import importlib.util

def _import_orpl_module(module_name):
    """Import ORPL module directly without triggering __init__.py"""
    spec = importlib.util.spec_from_file_location(
        module_name,
        _orpl_path / f"{module_name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"orpl.{module_name}"] = module
    spec.loader.exec_module(module)
    return module

# Import normalization first (dependency for metrics)
normalization = _import_orpl_module("normalization")
sys.modules["orpl.normalization"] = normalization

cosmic_ray = _import_orpl_module("cosmic_ray")
baseline_removal = _import_orpl_module("baseline_removal")
metrics = _import_orpl_module("metrics")

# Set random seed for reproducible example selection
np.random.seed(42)

# Plot style
plt.rcParams['figure.figsize'] = (12, 4)
plt.rcParams['font.size'] = 10


def get_txt_filepath(name, root='./data/raw/dipeptides'):
    """Get file paths for dipeptide spectra data.

    Args:
        name: Dipeptide name (e.g., 'AR', 'GR', 'SA', 'SR')
        root: Root directory for raw dipeptide data

    Returns:
        tuple: (collection_path, test_spectrum_path)
    """
    folder = Path(root) / name
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


def remove_cosmic_rays(spectrum, width=3, std_factor=5):
    """Apply ORPL cosmic ray filter to a single spectrum.

    Args:
        spectrum: Spectrum object with wavelengths and intensities
        width: Detection filter width (default: 3)
        std_factor: Spike detection threshold (default: 5)

    Returns:
        New Spectrum object with cosmic rays removed
    """
    filtered_intensities = cosmic_ray.crfilter_single(
        spectrum.intensities,
        width=width,
        std_factor=std_factor
    )
    new_data = np.stack((spectrum.wavelengths, filtered_intensities), axis=1)
    return Spectrum(new_data, x=spectrum.x, y=spectrum.y)


def remove_baseline_bubblefill(spectrum, min_bubble_widths=50, fit_order=1):
    """Apply ORPL BubbleFill baseline removal.

    Args:
        spectrum: Spectrum object with wavelengths and intensities
        min_bubble_widths: Minimum bubble width (larger = more conservative)
        fit_order: Polynomial order for slope removal (1 = linear)

    Returns:
        New Spectrum object with baseline removed
    """
    raman, baseline = baseline_removal.bubblefill(
        spectrum.intensities,
        min_bubble_widths=min_bubble_widths,
        fit_order=fit_order,
        do_savgol=True
    )
    new_data = np.stack((spectrum.wavelengths, raman), axis=1)
    return Spectrum(new_data, x=spectrum.x, y=spectrum.y)


def remove_baseline_imodpoly(spectrum, poly_order=6, precision=0.005, max_iter=1000):
    """Apply ORPL IModPoly baseline removal (alternative method).

    Args:
        spectrum: Spectrum object with wavelengths and intensities
        poly_order: Polynomial fit order (default: 6)
        precision: Convergence precision (default: 0.005)
        max_iter: Maximum iterations (default: 1000)

    Returns:
        New Spectrum object with baseline removed
    """
    raman, baseline = baseline_removal.imodpoly(
        spectrum.intensities,
        poly_order=poly_order,
        precision=precision,
        max_iter=max_iter,
        imod=True
    )
    new_data = np.stack((spectrum.wavelengths, raman), axis=1)
    return Spectrum(new_data, x=spectrum.x, y=spectrum.y)


def calculate_assi(spectrum):
    """Calculate ORPL ASSI (Average Signed Squared Intensity) quality metric.

    Args:
        spectrum: Spectrum object

    Returns:
        float: ASSI quality factor between -1 and 1
    """
    return metrics.assi(spectrum.intensities)


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


def process_pipeline_orpl(name, raw_data_folder, save_dir,
                          cosmic_ray_width=3, cosmic_ray_std_factor=5,
                          bubble_min_widths=50, bubble_fit_order=1,
                          savgol_window=11, savgol_polyorder=3,
                          snr_threshold=50,
                          save_npz=True, save_png=True, display=False):
    """Process spectra data using ORPL-based pipeline.

    Pipeline steps:
    1. Load raw data (spectra package)
    2. Cosmic ray removal (ORPL)
    3. Baseline correction with BubbleFill (ORPL)
    4. Smoothing with Savitzky-Golay (spectra package)
    5. Normalization (spectra package)
    6. SNR filtering (spectra package)
    7. Export processed data (spectra package)

    Args:
        name: Peptide name (e.g., 'AR', 'GR', 'SA', 'SR')
        raw_data_folder: Raw data folder path
        save_dir: Directory to save processed data
        cosmic_ray_width: Cosmic ray filter width (default: 3)
        cosmic_ray_std_factor: Cosmic ray detection threshold (default: 5)
        bubble_min_widths: BubbleFill minimum bubble width (default: 50)
        bubble_fit_order: BubbleFill polynomial order (default: 1)
        savgol_window: Savitzky-Golay window length (default: 11)
        savgol_polyorder: Savitzky-Golay polynomial order (default: 3)
        snr_threshold: SNR quality threshold (default: 50)
        save_npz: Whether to save data as .npz files
        save_png: Whether to save plots as .png files
        display: Whether to display graphs for visual inspection
    """
    save_dir = save_dir / name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load raw data
    collection_path, test_spectrum_path = get_txt_filepath(name, root=raw_data_folder)
    if collection_path:
        collection = read_spectrum_file(collection_path)
        print(f"{name}: {len(collection)} spectra loaded")
        if save_npz:
            export_to_npz(collection, save_dir / 'raw_data.npz')
            print(f"Saved raw spectra data at {save_dir / 'raw_data.npz'}")
    else:
        print(f"{name}: Missing spectra")
        return

    if test_spectrum_path:
        test_collection = read_spectrum_file(test_spectrum_path)
        print(f"Test spectrum loaded for {name}")
        if save_npz:
            export_to_npz(test_collection, save_dir / 'raw_test_spectrum.npz')
        test_spectrum = test_collection[0]
    else:
        test_spectrum = None
        print(f"Missing test spectrum for {name}")

    # Plot raw data
    plot_collection_and_test_spectrum(collection, test_spectrum,
                                       title=f'{name} Raman Spectroscopy Data (RAW)',
                                       save_path=save_dir / 'raw_data.png' if save_png else None,
                                       display=display)

    # ad hoc solution for DR
    if name in ['DR', 'FA', 'FD', 'SF', 'RAD', 'GDR']:
        spectra = []
        length = 50
        for spectrum in collection.spectra:
            intesities = spectrum.intensities
            intesities[:length] = np.mean(intesities[length:2 * length])
            spectrum.data[:, 1] = intesities
            spectra.append(spectrum)
        intesities = test_spectrum.intensities
        intesities[:length] = np.mean(intesities[length:2 * length])
        test_spectrum.data[:, 1] = intesities

    # Step 2: Cosmic ray removal (ORPL) - replaces ad-hoc spike fixes
    print(f"  Applying cosmic ray removal (width={cosmic_ray_width}, std_factor={cosmic_ray_std_factor})...")
    processed_spectra = []
    for spectrum in collection.spectra:
        processed = remove_cosmic_rays(spectrum, width=cosmic_ray_width, std_factor=cosmic_ray_std_factor)
        processed_spectra.append(processed)
    collection = SpectraCollection(processed_spectra, source_file=collection_path)

    if test_spectrum:
        test_spectrum = remove_cosmic_rays(test_spectrum, width=cosmic_ray_width, std_factor=cosmic_ray_std_factor)

    # Step 3: Baseline removal with BubbleFill (ORPL)
    print(f"  Applying BubbleFill baseline removal (min_bubble_widths={bubble_min_widths}, fit_order={bubble_fit_order})...")
    processed_spectra = []
    for spectrum in collection.spectra:
        processed = remove_baseline_bubblefill(spectrum, min_bubble_widths=bubble_min_widths, fit_order=bubble_fit_order)
        processed_spectra.append(processed)
    collection = SpectraCollection(processed_spectra, source_file=collection_path)

    if test_spectrum:
        test_spectrum = remove_baseline_bubblefill(test_spectrum, min_bubble_widths=bubble_min_widths, fit_order=bubble_fit_order)

    # Step 4 & 5: Smoothing and normalization (spectra package)
    print(f"  Applying Savitzky-Golay smoothing (window={savgol_window}, polyorder={savgol_polyorder})...")
    collection = (PreprocessingPipeline(collection)
                  .smooth(window_length=savgol_window, polyorder=savgol_polyorder)
                  .normalize()  # Min-max normalization to [0,1]
                  .result())

    if test_spectrum:
        test_spectrum = (PreprocessingPipeline(test_spectrum)
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
                                       title=f'{name} Raman Spectroscopy Data (ORPL preprocessed)',
                                       save_path=save_dir / 'preprocessed_data.png' if save_png else None,
                                       display=display)

    # Step 6: Quality filtering with SNR and ASSI
    snr = calculate_snr(collection)

    # Calculate ASSI for each spectrum
    assi_values = np.array([calculate_assi(spectrum) for spectrum in collection.spectra])
    mean_assi = assi_values.mean()
    print(f"  Quality metrics: Mean ASSI = {mean_assi:.4f}")

    # Filter spectra with SNR > threshold
    clean_spectra = [spectrum for spectrum, s in zip(collection, snr) if s > snr_threshold]
    if clean_spectra:
        clean_collection = SpectraCollection(clean_spectra, source_file=collection_path)
        print(f"{name}: {len(clean_collection)} spectra with SNR > {snr_threshold}")
        if save_npz:
            export_to_npz(clean_collection, save_dir / 'clean_data.npz')
            print(f"Saved clean data at {save_dir / 'clean_data.npz'}")
    else:
        print(f"{name}: No spectra with SNR > {snr_threshold}")
        clean_collection = None

    # Step 7: Generate fingerprint
    fingerprint = get_fingerprint(collection, threshold=snr_threshold)
    fingerprint_collection = SpectraCollection([fingerprint], source_file=collection_path)
    if save_npz:
        export_to_npz(fingerprint_collection, save_dir / 'fingerprint.npz')
        print(f"Saved fingerprint data at {save_dir / 'fingerprint.npz'}")

    plot_collection_and_test_spectrum(fingerprint_collection, test_spectrum,
                                       title=f'{name} Raman Spectroscopy Data (SNR>{snr_threshold})',
                                       save_path=save_dir / 'fingerprint.png' if save_png else None,
                                       display=display)

    # Summary
    print(f"\n  {name} Processing Summary:")
    print(f"    - Total spectra: {len(collection)}")
    print(f"    - Clean spectra (SNR > {snr_threshold}): {len(clean_spectra) if clean_spectra else 0}")
    print(f"    - Mean ASSI: {mean_assi:.4f}")


if __name__ == '__main__':
    root = Path('../data/')
    mode = 'dipeptides'  # Change to 'dipeptides' as needed
    raw_data_folder = root / 'raw' / mode
    processed_data_folder = root / 'processed' / 'orpl' / mode

    # ORPL preprocessing parameters
    params = {
        'cosmic_ray_width': 3,
        'cosmic_ray_std_factor': 5,
        'bubble_min_widths': 50,
        'bubble_fit_order': 1,
        'savgol_window': 11,
        'savgol_polyorder': 3,
        'snr_threshold': 50,
    }

    print("ORPL-Based Preprocessing Pipeline")
    print("=" * 50)
    print(f"Parameters: {params}")
    print("=" * 50)

    # Process all subfolders in raw_data_folder
    for subfolder in sorted(raw_data_folder.iterdir()):
        if subfolder.is_dir():
            name = subfolder.name
            print(f"\n{'='*50}")
            print(f"Processing {name}")
            print(f"{'='*50}")
            if (processed_data_folder / name / 'clean_data.npz').is_file():
                print('skipped (already processed)')
            else:
                process_pipeline_orpl(
                    name,
                    raw_data_folder,
                    processed_data_folder,
                    **params,
                    display=False
                )
