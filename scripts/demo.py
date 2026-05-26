from pathlib import Path

from fontTools.cffLib import privateDictOperators

from primarymagic import read_spectrum_file, subtract_baseline_arpls, subtract_baseline_als, PreprocessingPipeline, \
    normalize_minmax, calculate_snr, Spectrum, read_simple_spectrum
from primarymagic.data.spectraio import export_to_npz, load_from_npz
import numpy as np

amino_acid_files = {
    "Glycine": "data/batch1/Glycine/glycine_powder_532_.25s_100power_2394spectra.txt",
    "L-Alanine": "data/batch1/L-Alanine/alanine_powder_532_.25s_100power_2565spectra.txt",
    "L-Arginine": "data/batch1/L-Arginine/arginine_powder_532_0.25s_100power_2254spectra.txt",
    "L-Asparagine": "data/batch1/L-Asparagine/asparagine_powder_532_.25s_100power_2516spectra.txt",
    "L-Aspartic acid": "data/batch1/L-Aspartic acid/aspartic_acid_powder_532_.25s_100power_2478spectra.txt",
    "L-Cysteine": "data/batch1/L-Cysteine/cysteine_powder_532_.25s_100power_2491spectra.txt",
    "L-Glutamic acid": "data/batch1/L-Glutamic acid/glutamic_acid_powder_532_0.5s_50power_2100spectra.txt",
    "L-Glutamine": "data/batch1/L-Glutamine/glutamine_powder_532_.25s_100power_2420spectra.txt",
    "L-Isoleucine": "data/batch1/L-Isoleucine/isoleucine_powder_532_.5s_100power_2166spectra.txt",
    "L-Leucine": "data/batch1/L-Leucine/leucine_powder_532_.25s_100power_2585spectra.txt",
    "L-Lysine": "data/batch1/L-Lysine/lysine_powder_532_.25s_100power_2478spectra.txt",
    "L-Methionine": "data/batch1/L-Methionine/methionine_powder_532_.25s_100power_2442spectra.txt",
    "L-Phenylalanine": "data/batch1/L-Phenylalanine/phenylalanine_powder_532_.25s_100power_2346spectra.txt",
    "L-Proline": "data/batch1/L-Proline/proline_powder_532_.25s_100power_2196spectra.txt",
    "L-Serine": "data/batch1/L-Serine/serine_powder_532_.25s_100power_2546spectra.txt",
    "L-Threonine": "data/batch1/L-Threonine/threonine_powder_532_.25s_100power_2484spectra.txt",
    "L-Tryptophan": "data/batch1/L-Tryptophan/tryptophan_powder_532_.5s_50power_2150spectra.txt",
    "L-Tyrosine": "data/batch1/L-Tyrosine/tyrosine_powder_532_.25s_100power_2376spectra.txt",
    "L-Valine": "data/batch1/L-Valine/valine_powder_532_.25s_100power_2535spectra.txt",
}

amino_acid_test_files = {
    "Glycine": "data/batch1/Glycine/glycine spectra test.txt",
    "L-Alanine": "data/batch1/L-Alanine/alanine spectra test.txt",
    # "L-Arginine": "data/batch1/L-Arginine/arginine_powder_532_0.25s_100power_2254spectra.txt",
    "L-Asparagine": "data/batch1/L-Asparagine/asparagine spectra test.txt",
    "L-Aspartic acid": "data/batch1/L-Aspartic acid/aspartic acid spectra test.txt",
    # "L-Cysteine": "data/batch1/L-Cysteine/cysteine_powder_532_.25s_100power_2491spectra.txt",
    # "L-Glutamic acid": "data/batch1/L-Glutamic acid/glutamic_acid_powder_532_0.5s_50power_2100spectra.txt",
    # "L-Glutamine": "data/batch1/L-Glutamine/glutamine_powder_532_.25s_100power_2420spectra.txt",
    "L-Isoleucine": "data/batch1/L-Isoleucine/isoleucine spectra test.txt",
    "L-Leucine": "data/batch1/L-Leucine/leucine spectra test.txt",
    "L-Lysine": "data/batch1/L-Lysine/lysine spectra test.txt",
    "L-Methionine": "data/batch1/L-Methionine/methionine spectra test.txt",
    # "L-Phenylalanine": "data/batch1/L-Phenylalanine/phenylalanine_powder_532_.25s_100power_2346spectra.txt",
    # "L-Proline": "data/batch1/L-Proline/proline_powder_532_.25s_100power_2196spectra.txt",
    "L-Serine": "data/batch1/L-Serine/serine spectra test.txt",
    "L-Threonine": "data/batch1/L-Threonine/threonine spectra test.txt",
    # "L-Tryptophan": "data/batch1/L-Tryptophan/tryptophan_powder_532_.5s_50power_2150spectra.txt",
    "L-Tyrosine": "data/batch1/L-Tyrosine/tyrosine spectra test.txt",
    "L-Valine": "data/batch1/L-Valine/valine spectra test.txt",
}

if __name__ == "__main__":
    root = Path("../data").resolve()
    name = "L-Tyrosine"
    collection = read_spectrum_file(Path('..')/amino_acid_files[name])
    test_spectrum = read_spectrum_file(Path('..')/amino_acid_test_files[name])

    # # Save collection
    raw_data_path = root / 'monopeptide' / name / 'raw_data.npz'
    # export_to_npz(collection, raw_data_path)
    #
    # # Save fingerprint separately
    test_path = root / 'monopeptide' / name / 'test_data.npz'
    # export_to_npz(test_spectrum, test_path)

    collection = load_from_npz(raw_data_path)
    test_spectrum = load_from_npz(test_path)

