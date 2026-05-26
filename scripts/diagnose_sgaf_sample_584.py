"""
Diagnostic graphs for SGAF rep2 sample 584.

Runs the same pipeline stages as preprocess_magic_bayes.py
(edge_spike -> cosmic_ray -> wavelet/bayes -> bubblefill) and produces
detailed per-sample plots focused on cosmic-ray removal quality.

Outputs go to ``analysis_outputs/sgaf_rep2_sample_584/``.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation

from primarymagic import (
    PreprocessingPipeline,
    SpectraCollection,
    Spectrum,
    read_spectrum_file,
)
from primarymagic.preprocessing.snr import calculate_snr_from_stages

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RAW_FILE = Path(
    "data/custom/raw/4-letter/SGAF/rep2/SGAF_crystal_532_.25s_100power_696spectra_rep2.txt"
)
MASK_FILE = Path(
    "data/custom/processed/magic_bayes/4-letter/SGAF/rep2/mask.npz"
)
OUT_DIR = Path("analysis_outputs/sgaf_rep2_sample_584")
SAMPLE_IDX = 584

# Stage-1 parameters (must mirror preprocess_magic_bayes.run_stage1)
EDGE_N = 10
EDGE_FACTOR = 5.0
CR_WIDTH = 3
CR_STD_FACTOR = 5.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_cosmic_rays_for_inspection(signal: np.ndarray, width: int,
                                      std_factor: float):
    """Re-implement the detection logic so we can plot diff2 + mask + threshold."""
    origin = signal.min()
    scale = signal.max()
    sig = (signal - origin) / scale

    diff2 = np.diff(sig, n=2) ** 2
    threshold = diff2.mean() + std_factor * diff2.std()

    mask = np.zeros(len(sig), dtype=bool)
    mask[1:-1] = np.abs(diff2) > threshold
    mask_dilated = binary_dilation(mask, structure=[1] * width)
    return diff2, threshold, mask, mask_dilated


def clip_to_zero(coll):
    new_spectra = []
    for s in coll.spectra:
        d = s.data.copy()
        d[:, 1] = np.where(d[:, 1] >= 0, d[:, 1], 0.0)
        new_spectra.append(Spectrum(data=d, x=s.x, y=s.y))
    return SpectraCollection(
        spectra=new_spectra,
        source_file=coll.source_file,
        wavelengths=coll.wavelengths.copy(),
    )


# ---------------------------------------------------------------------------
# Plot routines
# ---------------------------------------------------------------------------

def plot_raw(wn, raw_intensities, idx, out):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(wn, raw_intensities, color="black", linewidth=0.7)
    peak_i = int(np.argmax(raw_intensities))
    ax.axvline(wn[peak_i], color="red", linestyle=":", linewidth=0.8,
               label=f"global max @ wn={wn[peak_i]:.1f}, "
                     f"I={raw_intensities[peak_i]:.0f}")
    ax.set_xlabel("Wavenumber (cm-1)")
    ax.set_ylabel("Raw intensity (counts)")
    ax.set_title(f"Sample #{idx} - raw spectrum")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_stage_overlay(wn, raw, after_edge, after_cr, idx, out):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax = axes[0]
    ax.plot(wn, raw, color="black", linewidth=0.7, label="raw")
    ax.plot(wn, after_edge, color="orange", linewidth=0.7, alpha=0.8,
            label="after edge-spike")
    ax.plot(wn, after_cr, color="red", linewidth=0.7, alpha=0.8,
            label="after cosmic-ray")
    ax.set_ylabel("Intensity (counts)")
    ax.set_title(f"Sample #{idx} - stage 1 overlay")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    ax.plot(wn, after_edge - raw, color="orange", linewidth=0.7,
            label="edge-spike delta")
    ax.plot(wn, after_cr - after_edge, color="red", linewidth=0.7,
            label="cosmic-ray delta")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Wavenumber (cm-1)")
    ax.set_ylabel("Delta")
    ax.set_title("What stage 1 changed (per substep)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_cr_detection(wn, raw, diff2, threshold, mask, mask_dilated, idx, out):
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    ax = axes[0]
    ax.plot(wn, raw, color="black", linewidth=0.7)
    detected = np.where(mask_dilated)[0]
    if len(detected):
        ax.scatter(wn[detected], raw[detected], color="red", s=18,
                   zorder=5, label=f"flagged points (n={len(detected)})")
    ax.set_ylabel("Raw intensity")
    ax.set_title(f"Sample #{idx} - cosmic-ray detection on raw spectrum")
    if len(detected):
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    diff2_x = wn[1:-1]
    ax.plot(diff2_x, diff2, color="navy", linewidth=0.6, label="diff2 = (d2/dx2 normed)^2")
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1.0,
               label=f"threshold = mean + {CR_STD_FACTOR}*std = {threshold:.2e}")
    ax.set_yscale("log")
    ax.set_ylabel("diff2 (log scale)")
    ax.set_title("Squared second derivative of normalised signal")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[2]
    ax.plot(wn, mask.astype(int), color="navy", linewidth=0.8,
            label="raw mask (pre-dilation)")
    ax.plot(wn, mask_dilated.astype(int) + 1.2, color="red", linewidth=0.8,
            label="dilated mask (used)")
    ax.set_yticks([0, 1, 1.2, 2.2])
    ax.set_yticklabels(["0", "1 (raw)", "0", "1 (dil)"])
    ax.set_xlabel("Wavenumber (cm-1)")
    ax.set_ylabel("flag")
    ax.set_title("Detection mask before/after dilation")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_zoom_panels(wn, raw, after_cr, mask_dilated, idx, out, half=20):
    """Zoom in on each flagged region and on residual extremes after CR removal."""
    flagged = np.where(mask_dilated)[0]
    # Cluster contiguous flags into regions
    regions = []
    if len(flagged):
        cur = [flagged[0]]
        for k in flagged[1:]:
            if k - cur[-1] <= 2:
                cur.append(k)
            else:
                regions.append((cur[0], cur[-1]))
                cur = [k]
        regions.append((cur[0], cur[-1]))

    # Also include the top-3 abs residuals (raw - cr) NOT covered by mask
    residual = raw - after_cr
    abs_res = np.abs(residual.copy())
    abs_res[mask_dilated] = 0.0
    top_unflagged = np.argsort(abs_res)[-3:][::-1]
    for j in top_unflagged:
        if abs_res[j] > 0:
            regions.append((j, j))

    if not regions:
        return None

    n = len(regions)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.0 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax, (lo, hi) in zip(axes, regions):
        a = max(0, lo - half)
        b = min(len(wn), hi + half + 1)
        ax.plot(wn[a:b], raw[a:b], color="black", linewidth=0.9, label="raw")
        ax.plot(wn[a:b], after_cr[a:b], color="red", linewidth=0.9,
                label="after CR")
        in_region = mask_dilated[a:b]
        if in_region.any():
            ax.scatter(wn[a:b][in_region], raw[a:b][in_region],
                       color="red", s=12, zorder=5)
        ax.set_title(f"wn={wn[lo]:.1f}-{wn[hi]:.1f}", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle(f"Sample #{idx} - zoomed view of flagged regions "
                 f"+ top-3 unflagged residuals", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return regions


def plot_full_pipeline(wn, raw, s1, s2, s3, idx, out):
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    panels = [
        (raw, "Stage 0: raw", "black"),
        (s1, "Stage 1: after edge-spike + cosmic-ray", "tab:blue"),
    ]
    for ax, (y, title, c) in zip(axes, panels):
        ax.plot(wn, y, color=c, linewidth=0.7)
        ax.set_title(f"Sample #{idx} - {title}")
        ax.set_ylabel("Intensity")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Wavenumber (cm-1)")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_noise_residual(wn, s1, s2, idx, out):
    residual = s1 - s2
    mad = np.median(np.abs(residual - np.median(residual)))
    sigma = 1.4826 * mad
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(wn, residual, color="black", linewidth=0.6)
    ax.axhline(0, color="black", linewidth=0.5)
    for k in (1, 3, 5):
        ax.axhline(k * sigma, color="red", linewidth=0.5, linestyle=":")
        ax.axhline(-k * sigma, color="red", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Wavenumber (cm-1)")
    ax.set_ylabel("Stage1 - Stage2")
    ax.set_title(f"Sample #{idx} - noise residual "
                 f"(MAD-sigma={sigma:.2f}); horizontal lines = +/- 1,3,5 sigma")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading raw: {RAW_FILE}")
    raw_coll = read_spectrum_file(RAW_FILE)
    wn = raw_coll.wavelengths
    n_total = len(raw_coll)
    print(f"  {n_total} spectra, {len(wn)} wavenumber points")
    assert SAMPLE_IDX < n_total

    # Pull the single raw spectrum (before any processing)
    raw_mat = raw_coll.to_intensity_matrix()
    raw_584 = raw_mat[SAMPLE_IDX].copy()

    # Run stage 1 as TWO substeps so we can plot the intermediate
    after_edge_coll = (
        PreprocessingPipeline(raw_coll)
        .remove_edge_spikes(edge_n=EDGE_N, factor=EDGE_FACTOR)
        .result()
    )
    after_edge_584 = after_edge_coll.to_intensity_matrix()[SAMPLE_IDX]

    stage1_coll = (
        PreprocessingPipeline(after_edge_coll)
        .remove_cosmic_rays(width=CR_WIDTH, std_factor=CR_STD_FACTOR)
        .result()
    )
    s1_584 = stage1_coll.to_intensity_matrix()[SAMPLE_IDX]

    # Stages 2 and 3 — done on full collection for parity with main pipeline
    stage2_coll = (
        PreprocessingPipeline(stage1_coll)
        .smooth(method="wavelet", wavelet="sym6", threshold_method="bayes")
        .result()
    )
    s2_584 = stage2_coll.to_intensity_matrix()[SAMPLE_IDX]

    stage3_coll = clip_to_zero(
        PreprocessingPipeline(stage2_coll)
        .subtract_baseline(method="bubblefill", min_bubble_widths=50, fit_order=1)
        .result()
    )
    s3_584 = stage3_coll.to_intensity_matrix()[SAMPLE_IDX]

    # Detection diagnostics — run on the input the algorithm actually sees:
    # remove_cosmic_rays is called with the edge-spike-cleaned signal.
    diff2, threshold, mask, mask_dilated = detect_cosmic_rays_for_inspection(
        after_edge_584, width=CR_WIDTH, std_factor=CR_STD_FACTOR
    )

    # Mask info
    mask_npz = np.load(MASK_FILE)
    passed = bool(mask_npz["passed"][SAMPLE_IDX])
    snr_584 = float(mask_npz["snr"][SAMPLE_IDX])
    noise_584 = float(mask_npz["noise_std"][SAMPLE_IDX])
    signal_584 = float(mask_npz["signal"][SAMPLE_IDX])
    thr = float(mask_npz["snr_threshold"])
    print(f"  sample {SAMPLE_IDX}: passed={passed}, snr={snr_584:.2f} "
          f"(threshold={thr}), signal={signal_584:.1f}, noise_std={noise_584:.2f}")
    print(f"  cosmic-ray detection on edge-cleaned signal: "
          f"{int(mask_dilated.sum())} points flagged (dilated), "
          f"{int(mask.sum())} pre-dilation; threshold={threshold:.3e}")
    print(f"  raw max: {raw_584.max():.1f}, after edge: {after_edge_584.max():.1f}, "
          f"after CR: {s1_584.max():.1f}")

    # ---- Plots ----
    plot_raw(wn, raw_584, SAMPLE_IDX, OUT_DIR / "01_raw.png")
    plot_stage_overlay(wn, raw_584, after_edge_584, s1_584, SAMPLE_IDX,
                       OUT_DIR / "02_stage1_overlay.png")
    plot_cr_detection(wn, after_edge_584, diff2, threshold, mask, mask_dilated,
                      SAMPLE_IDX, OUT_DIR / "03_cr_detection.png")
    regions = plot_zoom_panels(wn, raw_584, s1_584, mask_dilated, SAMPLE_IDX,
                               OUT_DIR / "04_zoom.png")
    plot_noise_residual(wn, s1_584, s2_584, SAMPLE_IDX,
                        OUT_DIR / "05_noise_residual.png")
    plot_full_pipeline(wn, raw_584, s1_584, s2_584, s3_584, SAMPLE_IDX,
                       OUT_DIR / "06_full_pipeline.png")

    # Cross-check the noise that drove the SNR fail
    s1_mat = stage1_coll.to_intensity_matrix()
    s2_mat = stage2_coll.to_intensity_matrix()
    s3_mat = stage3_coll.to_intensity_matrix()
    snr_calc, sig_calc, noise_calc = calculate_snr_from_stages(s1_mat, s2_mat, s3_mat)
    print(f"  recomputed for #{SAMPLE_IDX}: snr={snr_calc[SAMPLE_IDX]:.2f}, "
          f"signal={sig_calc[SAMPLE_IDX]:.1f}, noise={noise_calc[SAMPLE_IDX]:.2f}")

    print(f"\nAll plots written to {OUT_DIR.resolve()}")
    if regions:
        print(f"  Flagged/inspected regions: {regions}")


if __name__ == "__main__":
    main()
