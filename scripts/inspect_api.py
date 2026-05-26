from primarymagic import SpectraCollection, read_spectrum_file, calculate_snr
import inspect

print("SpectraCollection.__init__ signature:")
print(inspect.signature(SpectraCollection.__init__))
print()

# Check what read_spectrum_file returns
sc = read_spectrum_file('data/custom/raw/2-letter/GA/rep1/GA_crystal_532_.25s_100power_1900spectra.txt')
print("Type returned by read_spectrum_file:", type(sc))
print()

# Check the first spectrum
sp0 = sc[0]
print("Type of spectrum[0]:", type(sp0))
print("Spectrum attributes:", [a for a in dir(sp0) if not a.startswith('_')])
print()

# Check SpectraCollection attributes
print("SpectraCollection attributes:", [a for a in dir(sc) if not a.startswith('_')])
print()

# Try to understand how to subset
print("len(sc):", len(sc))
print()

# Check calculate_snr signature
print("calculate_snr signature:", inspect.signature(calculate_snr))
