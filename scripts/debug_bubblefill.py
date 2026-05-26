"""Debug BubbleFill baseline removal with various parameter sets.

Loads a raw test spectrum and applies BubbleFill with different
combinations of min_bubble_widths and fit_order to explore the
effect on baseline estimation.

Usage:
    python scripts/debug_bubblefill.py

    python scripts/debug_bubblefill.py \
        --input data/custom/processed/primary_magic/2-letter/GA/rep3/raw_test_spectrum.npz
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from primarymagic import PreprocessingPipeline
from primarymagic.data.spectraio import load_from_npz


def main():
    default_input = (Path(__file__).resolve().parent.parent
                     / 'data' / 'custom' / 'processed' / 'primary_magic'
                     / '2-letter' / 'SA' / 'rep1' / 'raw_test_spectrum.npz')

    parser = argparse.ArgumentParser(
        description="Debug BubbleFill baseline removal with various parameters"
    )
    parser.add_argument("--input", type=Path, default=default_input,
                        help="Path to raw_test_spectrum.npz")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output PNG path (default: results/debug_bubblefill.png)")
    parser.add_argument("--no-display", action="store_true",
                        help="Do not display the plot")
    args = parser.parse_args()

    output_path = args.output or Path("results") / "debug_bubblefill.png"

    print(f"Loading: {args.input}")
    spectrum = load_from_npz(args.input)
    print(f"Loaded {len(spectrum)} spectrum, {len(spectrum.wavelengths)} points")

    wl = spectrum.wavelengths
    raw_int = spectrum[0].intensities

    # Pipeline 1: Single BubbleFill pass (current default)
    single_pass = (PreprocessingPipeline(spectrum)
                   .subtract_baseline(method='bubblefill',
                                      min_bubble_widths=50, fit_order=1)
                   .normalize()
                   .result())

    # Pipeline 2: Smooth -> BubbleFill
    smooth_first = (PreprocessingPipeline(spectrum)
                    .smooth(window_length=7, polyorder=2)
                    .subtract_baseline(method='bubblefill',
                                       min_bubble_widths=50, fit_order=1)
                    .normalize()
                    .result())

    # Pipeline 3: BubbleFill -> Smooth -> BubbleFill
    double_pass = (PreprocessingPipeline(spectrum)
                   .subtract_baseline(method='bubblefill',
                                      min_bubble_widths=50, fit_order=1)
                   .smooth(window_length=7, polyorder=2)
                   .subtract_baseline(method='bubblefill',
                                      min_bubble_widths=50, fit_order=1)
                   .normalize()
                   .result())

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # Plot raw spectrum
    axes[0].plot(wl, raw_int, 'k-', lw=1.0)
    axes[0].set_ylabel('Intensity')
    axes[0].set_title('Raw Spectrum')
    axes[0].grid(True, alpha=0.3)

    # Single pass
    axes[1].plot(wl, single_pass[0].intensities, lw=1.0)
    axes[1].set_ylabel('Intensity')
    axes[1].set_title('BubbleFill (single pass) [DEFAULT]')
    axes[1].grid(True, alpha=0.3)

    # Smooth first
    axes[2].plot(wl, smooth_first[0].intensities, lw=1.0)
    axes[2].set_ylabel('Intensity')
    axes[2].set_title('Smooth → BubbleFill')
    axes[2].grid(True, alpha=0.3)

    # Double pass
    axes[3].plot(wl, double_pass[0].intensities, lw=1.0)
    axes[3].set_ylabel('Intensity')
    axes[3].set_title('BubbleFill → Smooth → BubbleFill (double pass)')
    axes[3].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Wavenumber (cm⁻¹)')
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved {output_path}")

    if not args.no_display:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
