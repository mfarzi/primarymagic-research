import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from primarymagic import SpectraCollection, Spectrum
from primarymagic import read_spectrum_file, read_simple_spectrum
from primarymagic import subtract_baseline_arpls, subtract_baseline_als, PreprocessingPipeline, normalize_minmax, \
    calculate_snr, Spectrum
from primarymagic import calculate_snr, export_to_npz

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


def get_fingerprint(collection, threshold=50):
    snr = calculate_snr(collection)
    intensities = collection.to_intensity_matrix()
    mean_intensity = intensities[snr > threshold, :].mean(axis=0)
    fingerprint = Spectrum(np.stack((collection.wavelengths, mean_intensity), axis=1))
    return normalize_minmax(fingerprint)


def plot_fingerprints(*spectra, title='Dipeptide Experiment'):
    """Plot arbitrary number of fingerprints.

    Args:
        *spectra: Variable number of (fingerprint, label) tuples.
        title: Plot title (keyword argument only).

    Example:
        plot_fingerprints((fg0, 'AR'), (fg1, 'alanine'), (fg2, 'arginine'), title='Dipeptide')
    """
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 4))

    colors = plt.cm.tab10.colors
    for i, (fg, label) in enumerate(spectra):
        if fg:
            color = colors[i % len(colors)]
            alpha = 1.0 if i == 0 else 0.6
            ax1.plot(fg.wavelengths, fg.intensities,
                     '-', color=color, linewidth=2, alpha=alpha, label=label)

    ax1.set_xlabel('Wavenumber (cm⁻¹)')
    ax1.set_ylabel('Intensity (a.u.)')
    ax1.set_title(title)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_colletion_and_test_spectrum(collection, test_spectrum, title=' Raman Spectroscopy Data',
                                      save_path=None, display=False):
    # visualise raw data
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


def process_pipeline(name, raw_data_folder, save_dir, save_npz=True, save_png=True, display=False):
    """Process spectra data: baseline removal, denoise, normalisation

    Args:
        name: peptide name (e.g., 'AR', 'GR', 'SA', 'SR')
        raw_data_folder: Raw data folder
        save_dir: Directory to save processed data
        save_npz: Whether to save processed data as .npz files
        display: whether to display graphs for visual inspection
    """
    save_dir = save_dir / name
    save_dir.mkdir(parents=True, exist_ok=True)

    # IO: read raw txt files for Raman Spectroscopy Data
    collection_path, test_spectrum_path = get_txt_filepath(name, root=raw_data_folder)
    if collection_path:
        collection = read_spectrum_file(collection_path)
        print(f"{name}: {len(collection)} spectra loaded")
        if save_npz:
            export_to_npz(collection, save_dir / 'raw_data.npz')
            print(f"Saved spectra data at {save_dir / 'raw_data.npz'}")
    else:
        print(f"{name}: Missing spectra")
        return

    if test_spectrum_path:
        test_collection = read_spectrum_file(test_spectrum_path)
        print(f"Test spectrum is loaded for {name}")
        if save_npz:
            export_to_npz(test_collection, save_dir / 'raw_test_spectrum.npz')
            print(f"Saved test spectrum data at {save_dir / 'raw_test_spectrum.npz'}")
        test_spectrum = test_collection[0]
    else:
        test_spectrum = None
        print(f"Missing test spectrum for {name}")

    # Plot the raw data
    plot_colletion_and_test_spectrum(collection, test_spectrum, title=f'{name} Raman Spectroscopy Data (RAW)',
                                      save_path=save_dir / 'raw_data.png' if save_png else None, display=display)

    # ad hoc solution for DR
    if name in ['DR', 'GA', 'FD', 'AS', 'FA', 'GD', 'SF', 'RAD', 'GDS', 'GDR']:
        spectra = []
        length = 50
        for spectrum in collection.spectra:
            intesities = spectrum.intensities
            intesities[:length] = np.mean(intesities[length:2*length])
            spectrum.data[:,1] = intesities
            spectra.append(spectrum)
        intesities = test_spectrum.intensities
        intesities[:length] = np.mean(intesities[length:2*length])
        test_spectrum.data[:,1] = intesities

    # Preprocessing: Baseline correction, Noise removal and min-max normalisation
    collection = (PreprocessingPipeline(collection)
                  .normalize()  # Min-max normalization to [0,1]
                  .subtract_baseline(lam=1e2, method='als')  # Simple Asymmetric Least Squares baseline correction.
                  .smooth(window_length=11, polyorder=3)  # Savitzky-Golay filter
                  .normalize()  # Min-max normalization to [0,1]
                  .result()
                  )
    if save_npz:
        export_to_npz(collection, save_dir / 'data.npz')
        print(f"Saved preprocessed spectra data at {save_dir / 'data.npz'}")

    if test_spectrum:
        test_spectrum = (PreprocessingPipeline(test_spectrum)
                         .normalize()  # Min-max normalization to [0,1]
                         .subtract_baseline(lam=1e2,
                                            method='als')  # Simple Asymmetric Least Squares baseline correction.
                         .smooth(window_length=11, polyorder=3)  # Savitzky-Golay filter
                         .normalize()  # Min-max normalization to [0,1]
                         .result()
                         )

        if save_npz:
            test_collection = SpectraCollection([test_spectrum], source_file=test_spectrum_path)
            export_to_npz(test_collection, save_dir / 'test_spectrum.npz')
            print(f"Saved preprocessed test spectrum data at {save_dir / 'test_spectrum.npz'}")

    # Plot the preprocessed data
    plot_colletion_and_test_spectrum(collection, test_spectrum, title=f'{name} Raman Spectroscopy Data (preprocessed)',
                                      save_path=save_dir / 'preprocessed_data.png' if save_png else None, display=display)

    # Filter spectra with SNR > 50 and save as clean data
    snr = calculate_snr(collection)
    clean_spectra = [spectrum for spectrum, s in zip(collection, snr) if s > 50]
    if clean_spectra:
        clean_collection = SpectraCollection(clean_spectra, source_file=collection_path)
        print(f"{name}: {len(clean_collection)} spectra with SNR > 50")
        if save_npz:
            export_to_npz(clean_collection, save_dir / 'clean_data.npz')
            print(f"Saved clean data at {save_dir / 'clean_data.npz'}")
    else:
        print(f"{name}: No spectra with SNR > 50")
        clean_collection = None

    # Fingerprint Generation: filter spectra data based on SNR and select top 100 results for averaging
    fingerprint = get_fingerprint(collection, threshold=50)
    fingerprint_collection = SpectraCollection([fingerprint], source_file=collection_path)
    if save_npz:
        export_to_npz(fingerprint_collection, save_dir / 'fingerprint.npz')
        print(f"Saved fingerprint data at {save_dir / 'fingerprint.npz'}")

    plot_colletion_and_test_spectrum(fingerprint_collection, test_spectrum,
                                      title=f'{name} Raman Spectroscopy Data (SNR>50)',
                                      save_path=save_dir / 'fingerprint.png' if save_png else None, display=display)

if __name__ == '__main__':
    root = Path('../data/')
    mode = 'tripeptides'
    raw_data_folder = root / 'raw' / mode
    processed_data_folder = root / 'processed' / 'baseline' / mode

    # Process all subfolders in raw_data_folder
    for subfolder in sorted(raw_data_folder.iterdir()):
        if subfolder.is_dir():
            name = subfolder.name
            print(f"\n{'='*50}")
            print(f"Processing {name}")
            print(f"{'='*50}")
            if (processed_data_folder / name / 'clean_data.npz').is_file():
                print('skipped')
                pass
            else:
                process_pipeline(name,
                                 raw_data_folder,
                                 processed_data_folder,
                                 display=False)