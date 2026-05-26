"""Test whether clip-to-zero in stage 3 is responsible for the ADR accuracy drop.

For AD, ADR, ASD, RFA: regenerate stage 3 from stage2_denoised.npz with NO clip,
then renormalize per-spectrum max (stage 6). Plot mean spectra:
    primary_magic vs magic_devtest_shifted (clipped) vs magic_devtest_noclip.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from primarymagic.data.spectraio import load_from_npz
from preprocess_magic import run_stage3  # noqa: E402

# Sequence directories under both pipelines
SEQUENCES = {
    "AD":  ("2-letter", "AD"),
    "ADR": ("3-letter", "ADR"),
    "ASD": ("3-letter", "ASD"),
    "RFA": ("3-letter", "RFA"),
}

PRIMARY = "data/custom/processed/primary_magic"
SHIFTED = "data/custom/processed/magic_devtest_shifted"


def regen_stage3_no_clip(stage2_path):
    """Run BubbleFill baseline removal but skip the clip-to-zero. Then per-spectrum max-normalize."""
    stage2 = load_from_npz(stage2_path)
    # run_stage3 with noise_percentile None defaults to clip-to-zero. We need to bypass clip.
    # Easiest: do BubbleFill manually using the same params and skip clip_to_zero.
    from primarymagic import PreprocessingPipeline
    stage3 = (
        PreprocessingPipeline(stage2)
        .subtract_baseline(method="bubblefill", min_bubble_widths=50, fit_order=1)
        .result()
    )
    mat = stage3.to_intensity_matrix()
    scales = mat.max(axis=1)
    safe = np.where(scales > 0, scales, 1.0)
    normed = mat / safe[:, None]
    return stage3.wavelengths, normed


def mean_norm(intensities):
    mean = intensities.mean(0)
    return (mean - mean.min()) / (mean.max() - mean.min())


def main():
    fig, axes = plt.subplots(len(SEQUENCES), 1, figsize=(13, 3.2 * len(SEQUENCES)), sharex=True)
    if len(SEQUENCES) == 1:
        axes = [axes]

    colors = {
        "primary_magic":              "tab:blue",
        "magic_devtest_shifted":      "tab:red",
        "magic_devtest_noclip":       "tab:green",
    }

    for ax, (seq, (length_dir, code)) in zip(axes, SEQUENCES.items()):
        # primary_magic clean_data
        d_prim = np.load(f"{PRIMARY}/{length_dir}/{code}/rep1/clean_data.npz", allow_pickle=True)
        ax.plot(d_prim["wavelengths"], mean_norm(d_prim["intensities"]),
                color=colors["primary_magic"], linewidth=1.4, alpha=0.85,
                label=f"primary_magic (n={d_prim['intensities'].shape[0]})")

        # magic_devtest_shifted clean_data (with clip-to-zero already)
        d_clipped = np.load(f"{SHIFTED}/{length_dir}/{code}/rep1/clean_data.npz", allow_pickle=True)
        ax.plot(d_clipped["wavelengths"], mean_norm(d_clipped["intensities"]),
                color=colors["magic_devtest_shifted"], linewidth=1.4, alpha=0.85,
                label=f"magic_devtest_shifted clipped (n={d_clipped['intensities'].shape[0]})")

        # Regenerate stage3 with NO clip, then renormalize
        stage2_path = Path(SHIFTED) / length_dir / code / "rep1" / "stage2_denoised.npz"
        wl_noclip, normed_noclip = regen_stage3_no_clip(stage2_path)
        ax.plot(wl_noclip, mean_norm(normed_noclip),
                color=colors["magic_devtest_noclip"], linewidth=1.4, alpha=0.85, linestyle="--",
                label=f"magic_devtest NO clip (n={normed_noclip.shape[0]})")

        ax.set_title(f"{code}/rep1 — mean spectrum (min-max normalized)")
        ax.set_ylabel("Normalized intensity")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel("Wavenumber (cm-1)")
    plt.tight_layout()
    out = "noclip_hypothesis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
