"""Test whether BayesShrink (instead of VisuShrink) preserves more inter-peak detail.

For AD, ADR, ASD, RFA: regenerate stage 2 from stage1_cleaned.npz with
threshold_method='bayes', then stage 3 (BubbleFill + clip-to-zero, matching
default), then per-spectrum max-normalize. Plot mean spectra:
    primary_magic vs magic_devtest_shifted (visu+clip) vs magic_devtest_bayes (bayes+clip).
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent / "src"))

from primarymagic import PreprocessingPipeline
from primarymagic.data.spectraio import load_from_npz

SEQUENCES = {
    "AD":  ("2-letter", "AD"),
    "ADR": ("3-letter", "ADR"),
    "ASD": ("3-letter", "ASD"),
    "RFA": ("3-letter", "RFA"),
}

PRIMARY = "data/custom/processed/primary_magic"
SHIFTED = "data/custom/processed/magic_devtest_shifted"


def regen_bayes(stage1_path):
    """Stage1 -> wavelet denoise (sym6, bayes) -> BubbleFill -> clip-to-zero -> max-norm."""
    stage1 = load_from_npz(stage1_path)
    stage2 = (
        PreprocessingPipeline(stage1)
        .smooth(method="wavelet", wavelet="sym6", threshold_method="bayes")
        .result()
    )
    stage3 = (
        PreprocessingPipeline(stage2)
        .subtract_baseline(method="bubblefill", min_bubble_widths=50, fit_order=1)
        .result()
    )
    mat = stage3.to_intensity_matrix()
    mat = np.maximum(mat, 0.0)  # clip-to-zero (matches default magic stage 3)
    scales = mat.max(axis=1)
    safe = np.where(scales > 0, scales, 1.0)
    normed = mat / safe[:, None]
    return stage3.wavelengths, normed


def mean_norm(intensities):
    mean = intensities.mean(0)
    return (mean - mean.min()) / (mean.max() - mean.min())


def main():
    fig, axes = plt.subplots(len(SEQUENCES), 1, figsize=(13, 3.2 * len(SEQUENCES)), sharex=True)

    for ax, (seq, (length_dir, code)) in zip(axes, SEQUENCES.items()):
        # primary_magic
        d_prim = np.load(f"{PRIMARY}/{length_dir}/{code}/rep1/clean_data.npz", allow_pickle=True)
        ax.plot(d_prim["wavelengths"], mean_norm(d_prim["intensities"]),
                color="tab:blue", linewidth=1.4, alpha=0.85,
                label=f"primary_magic (n={d_prim['intensities'].shape[0]})")

        # magic_devtest_shifted (visu)
        d_visu = np.load(f"{SHIFTED}/{length_dir}/{code}/rep1/clean_data.npz", allow_pickle=True)
        ax.plot(d_visu["wavelengths"], mean_norm(d_visu["intensities"]),
                color="tab:red", linewidth=1.4, alpha=0.85,
                label=f"magic_devtest_shifted visu+clip (n={d_visu['intensities'].shape[0]})")

        # magic_devtest_bayes (regenerated)
        stage1_path = Path(SHIFTED) / length_dir / code / "rep1" / "stage1_cleaned.npz"
        wl_b, normed_b = regen_bayes(stage1_path)
        ax.plot(wl_b, mean_norm(normed_b),
                color="tab:green", linewidth=1.4, alpha=0.85, linestyle="--",
                label=f"magic_devtest bayes+clip (n={normed_b.shape[0]})")

        ax.set_title(f"{code}/rep1 — mean spectrum (min-max normalized)")
        ax.set_ylabel("Normalized intensity")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel("Wavenumber (cm-1)")
    plt.tight_layout()
    out = "bayes_clip_hypothesis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
