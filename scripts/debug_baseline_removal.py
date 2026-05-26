"""Debug and compare baseline removal techniques on a test spectrum.

Loads a raw test spectrum and applies all four available baseline removal
methods, plotting the results side by side for visual comparison.

Usage:
    python scripts/debug_baseline_removal.py

    python scripts/debug_baseline_removal.py \
        --input data/custom/processed/primary_magic/2-letter/GA/rep3/raw_test_spectrum.npz

    python scripts/debug_baseline_removal.py \
        --input data/custom/processed/primary_magic/2-letter/SA/rep1/raw_test_spectrum.npz \
        --output results/baseline_debug_SA.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from primarymagic import PreprocessingPipeline
from primarymagic.data.spectraio import load_from_npz


def run_baseline_comparison(spectrum, methods, output_path=None, display=True):
    """Apply each baseline method and plot results."""
    n = len(methods)
    fig, axes = plt.subplots(n + 1, 1, figsize=(14, 3 * (n + 1)), sharex=True)

    wl = spectrum.wavelengths
    raw_int = spectrum[0].intensities

    # Plot raw spectrum
    axes[0].plot(wl, raw_int, 'k-', lw=1.0)
    axes[0].set_ylabel('Intensity')
    axes[0].set_title('Raw Spectrum')
    axes[0].grid(True, alpha=0.3)

    for i, (name, method, kwargs) in enumerate(methods):
        corrected = (PreprocessingPipeline(spectrum)
                     .subtract_baseline(method=method, **kwargs)
                     .normalize()
                     .result())

        axes[i + 1].plot(wl, corrected[0].intensities, lw=1.0)
        axes[i + 1].set_ylabel('Intensity')
        axes[i + 1].set_title(f'{name}')
        axes[i + 1].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Wavenumber (cm⁻¹)')
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved {output_path}")

    if display:
        plt.show()
    plt.close(fig)


def main():
    default_input = (Path(__file__).resolve().parent.parent
                     / 'data' / 'custom' / 'processed' / 'primary_magic'
                     / '2-letter' / 'SA' / 'rep1' / 'raw_test_spectrum.npz')

    parser = argparse.ArgumentParser(
        description="Debug and compare baseline removal techniques"
    )
    parser.add_argument("--input", type=Path, default=default_input,
                        help="Path to raw_test_spectrum.npz")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output PNG path (default: results/baseline_debug.png)")
    parser.add_argument("--no-display", action="store_true",
                        help="Do not display the plot")
    args = parser.parse_args()

    output_path = args.output or Path("results") / "baseline_debug.png"

    print(f"Loading: {args.input}")
    spectrum = load_from_npz(args.input)
    print(f"Loaded {len(spectrum)} spectrum, {len(spectrum.wavelengths)} points")

    methods = [
        ("BubbleFill (min_widths=50, fit_order=1)", "bubblefill",
         {"min_bubble_widths": 50, "fit_order": 1}),
        ("arPLS (lam=1e5)", "arpls",
         {"lam": 1e5}),
        ("arPLS (lam=1e6)", "arpls",
         {"lam": 1e6}),
        ("ALS (lam=1e5, p=0.01)", "als",
         {"lam": 1e5, "p": 0.01}),
        ("IModPoly (order=6)", "imodpoly",
         {"poly_order": 6}),
    ]

    run_baseline_comparison(spectrum, methods, output_path,
                            display=not args.no_display)


if __name__ == "__main__":
    main()
