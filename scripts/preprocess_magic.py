"""
Staged Raman Spectra Preprocessing Script (Magic Pipeline)
===========================================================

Stages:
    1. remove_edge_spikes + cosmic_ray removal
    2. Wavelet denoising (sym6, VisuShrink)
    3. BubbleFill baseline removal + clip to zero
    4. Fingerprint computation (top 5% by AUC → max-normalise → median → min-max normalise)
    5. Similarity classification (peak_energy_ratio vs fingerprint → fg / borderline / bg)
    6. Normalise and store foreground signals (raw + normalised processed)

Diagnostic plots are generated retroactively after stage 5, using the
classification masks so every plot can colour-code groups.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from primarymagic import (
    Spectrum,
    SpectraCollection,
    PreprocessingPipeline,
    read_spectrum_file,
    export_to_npz,
    spectral_energy,
    spectral_auc,
    peak_energy_ratio,
    czekanowski,
    normalize_minmax,
)
from primarymagic.data.spectraio import load_from_npz
from primarymagic.preprocessing.snr import calculate_snr_from_stages

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_txt_filepath(folder):
    """Find the mapped spectra .txt file (contains 'power' in name)."""
    folder = Path(folder)
    for f in folder.glob('*.txt'):
        if 'power' in f.stem.lower():
            return f
    return None


def clip_to_zero(data, noise_percentile=None):
    """Clip intensities below a floor. Works on Spectrum or SpectraCollection.

    Args:
        data: Spectrum or SpectraCollection.
        noise_percentile: If None, clip to 0 (default). If a number (e.g. 5),
            the floor is computed per spectrum as np.percentile(intensities,
            noise_percentile). Values below the floor are set to zero.
    """
    if isinstance(data, Spectrum):
        new_data = data.data.copy()
        intensities = new_data[:, 1]
        floor = 0.0
        if noise_percentile is not None:
            floor = np.percentile(intensities, noise_percentile)
        new_data[:, 1] = np.where(intensities >= floor, intensities, 0.0)
        return Spectrum(data=new_data, x=data.x, y=data.y)

    new_spectra = []
    for s in data.spectra:
        new_data = s.data.copy()
        intensities = new_data[:, 1]
        floor = 0.0
        if noise_percentile is not None:
            floor = np.percentile(intensities, noise_percentile)
        new_data[:, 1] = np.where(intensities >= floor, intensities, 0.0)
        new_spectra.append(Spectrum(data=new_data, x=s.x, y=s.y))
    return SpectraCollection(
        spectra=new_spectra,
        source_file=data.source_file,
        wavelengths=data.wavelengths.copy(),
    )


def sample_indices(mask, n, rng):
    """Sample up to n indices from boolean mask."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return np.array([], dtype=int)
    n = min(n, len(idx))
    return rng.choice(idx, size=n, replace=False)


# ---------------------------------------------------------------------------
# Processing stages
# ---------------------------------------------------------------------------

def run_stage1(collection):
    """remove_edge_spikes(edge_n=10, factor=5.0) → remove_cosmic_rays(width=3, std_factor=5)"""
    return (
        PreprocessingPipeline(collection)
        .remove_edge_spikes(edge_n=10, factor=5.0)
        .remove_cosmic_rays(width=3, std_factor=5)
        .result()
    )


def run_stage2(stage1):
    """smooth(method='wavelet', wavelet='sym6', threshold_method='visu')"""
    return (
        PreprocessingPipeline(stage1)
        .smooth(method='wavelet', wavelet='sym6', threshold_method='visu')
        .result()
    )


def run_stage3(stage2, noise_percentile=None):
    """subtract_baseline(method='bubblefill', min_bubble_widths=50, fit_order=1) → clip_to_zero"""
    result = (
        PreprocessingPipeline(stage2)
        .subtract_baseline(method='bubblefill', min_bubble_widths=50, fit_order=1)
        .result()
    )
    return clip_to_zero(result, noise_percentile=noise_percentile)


def _max_normalise(intensities):
    """Max-normalise a single intensity array to [0, 1]."""
    scale = np.max(intensities)
    if scale > 0:
        return intensities / scale
    return intensities


def _minmax_normalise(intensities):
    """Min-max normalise so min=0 and max=1."""
    lo = np.min(intensities)
    hi = np.max(intensities)
    if hi == lo:
        return np.zeros_like(intensities)
    return (intensities - lo) / (hi - lo)


def _compute_initial_fingerprint(stage3, percentile=5):
    """Phase 1: AUC-based seed selection + element-wise median.

    Selects the top ``percentile``% of spectra by AUC (area under curve
    without squaring), max-normalises each, computes element-wise median,
    then min-max normalises the result to [0, 1].
    """
    auc = spectral_auc(stage3)
    threshold = np.percentile(auc, 100 - percentile)
    seed_mask = auc >= threshold

    normalised_seeds = np.array([
        _max_normalise(s.intensities)
        for s, keep in zip(stage3.spectra, seed_mask) if keep
    ])
    median_intensities = np.median(normalised_seeds, axis=0)
    fp_intensities = _minmax_normalise(median_intensities)
    fingerprint = Spectrum(
        data=np.column_stack([stage3.wavelengths, fp_intensities]))
    return auc, fingerprint


def run_stage4(stage3, seed_pool_percentile=5):
    """Stage 4: Fingerprint computation via AUC seed selection + median.

    Select top seed_pool_percentile% by AUC, max-normalise each,
    element-wise median, then min-max normalise to [0, 1].

    Returns: (auc_scores, fingerprint)
    """
    auc_scores, fingerprint = _compute_initial_fingerprint(
        stage3, percentile=seed_pool_percentile)

    return auc_scores, fingerprint


def run_stage5(stage3, fingerprint, sim_high, sim_low):
    """Stage 5: Similarity classification.

    Compute czekanowski (continuous Dice) for all spectra vs fingerprint,
    then classify: fg (sim > sim_high), bg (sim < sim_low), borderline.

    Returns: (similarity, fg_mask, bg_mask, bl_mask)
    """
    similarity = np.array([
        czekanowski(s, fingerprint)
        for s in stage3.spectra
    ])

    fg_mask = similarity > sim_high
    bg_mask = similarity < sim_low
    bl_mask = ~fg_mask & ~bg_mask

    return similarity, fg_mask, bg_mask, bl_mask


def run_stage6(stage3, fg_mask):
    """Stage 6: Normalise foreground spectra.

    scale = max(intensities) per spectrum, normalised = spectrum / scale.

    Returns: (normalised_collection, scales_array)
    """
    mat = stage3.to_intensity_matrix()
    fg_mat = mat[fg_mask]
    scales = fg_mat.max(axis=1)          # shape (n_fg,)
    safe_scales = np.where(scales > 0, scales, 1.0)
    normed_mat = fg_mat / safe_scales[:, None]

    wavelengths = stage3.wavelengths
    new_spectra = []
    fg_indices = np.where(fg_mask)[0]
    src = stage3.source_file
    for i, gi in enumerate(fg_indices):
        s = stage3.spectra[gi]
        new_data = np.column_stack([wavelengths, normed_mat[i]])
        new_spectra.append(Spectrum(data=new_data, x=s.x, y=s.y))

    normalised = SpectraCollection(
        spectra=new_spectra,
        source_file=src,
        wavelengths=wavelengths.copy(),
    )
    return normalised, scales


def compute_fingerprint(normalised):
    """Mean of normalised foreground spectra."""
    mat = normalised.to_intensity_matrix()
    mean_int = mat.mean(axis=0)
    wavelengths = normalised.wavelengths
    fp_data = np.column_stack([wavelengths, mean_int])
    return Spectrum(data=fp_data)


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

GROUP_COLORS = {
    'fg': 'red',
    'bl': 'grey',
    'bg': 'blue',
}
GROUP_LABELS = {
    'fg': 'Foreground',
    'bl': 'Borderline',
    'bg': 'Background',
}


def _get_matrix(collection):
    return collection.to_intensity_matrix()


def _plot_samples_3x1(data_mat, fg_idx, bl_idx, bg_idx, wavelengths, save_dir,
                      filename, title):
    """Generic 3x1 plot with distinct-coloured samples per group.

    Used for stage0_raw, stage1_fix_cosmic_ray, stage2_denoise, stage3_baseline_corrected.
    """
    groups = [('fg', fg_idx), ('bl', bl_idx), ('bg', bg_idx)]
    sample_colors = ['black', 'red', 'green', 'blue', 'goldenrod']

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for ax, (gname, idx) in zip(axes, groups):
        label = GROUP_LABELS[gname]
        for i, gi in enumerate(idx):
            c = sample_colors[i % len(sample_colors)]
            ax.plot(wavelengths, data_mat[gi], color=c, alpha=1.0,
                    linewidth=0.8, label=f'#{gi}')
        ax.set_title(f'{label} (n={len(idx)})')
        ax.set_ylabel('Intensity')
        if len(idx) > 0:
            ax.legend(fontsize=6, loc='upper right')
        ax.grid(True, alpha=0.2)
    axes[-1].set_xlabel('Wavenumber (cm⁻¹)')
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    path = save_dir / filename
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_stage0_raw(raw, fg_idx, bl_idx, bg_idx, wavelengths, save_dir):
    """Stage 0: raw spectra samples per group."""
    return _plot_samples_3x1(
        _get_matrix(raw), fg_idx, bl_idx, bg_idx, wavelengths, save_dir,
        'stage0_raw.png', 'Stage 0: Raw Spectra')


def plot_stage1_cleaned(stage1, fg_idx, bl_idx, bg_idx, wavelengths, save_dir):
    """Stage 1: cleaned spectra (after edge spike removal + cosmic_ray) samples per group."""
    return _plot_samples_3x1(
        _get_matrix(stage1), fg_idx, bl_idx, bg_idx, wavelengths, save_dir,
        'stage1_fix_cosmic_ray.png', 'Stage 1: After Edge Spike Removal + Cosmic Ray Removal')


def plot_stage2_denoised(stage2, fg_idx, bl_idx, bg_idx, wavelengths, save_dir):
    """Stage 2: denoised spectra samples per group."""
    return _plot_samples_3x1(
        _get_matrix(stage2), fg_idx, bl_idx, bg_idx, wavelengths, save_dir,
        'stage2_denoise.png', 'Stage 2: After Wavelet Denoising (sym6 VisuShrink)')


def plot_stage2_residual(stage1, stage2, fg_idx, bl_idx, bg_idx, wavelengths, save_dir):
    """Noise residual (stage1 - stage2). One subplot per group. Horizontal line at 0."""
    groups = [('fg', fg_idx), ('bl', bl_idx), ('bg', bg_idx)]
    s1_mat = _get_matrix(stage1)
    s2_mat = _get_matrix(stage2)
    residual = s1_mat - s2_mat

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for ax, (gname, idx) in zip(axes, groups):
        label = GROUP_LABELS[gname]
        for i, gi in enumerate(idx):
            ax.plot(wavelengths, residual[gi], color='black', alpha=0.6,
                    linewidth=0.5, label='Residual' if i == 0 else None)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_title(f'{label} residuals (n={len(idx)})')
        ax.set_ylabel('Stage1 - Stage2')
        if len(idx) > 0:
            ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.2)
    axes[-1].set_xlabel('Wavenumber (cm⁻¹)')
    fig.suptitle('Stage 2: Noise Residual (Stage1 - Stage2)', fontsize=12)
    plt.tight_layout()
    path = save_dir / 'stage2_residual.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_stage3_baseline(stage2, stage3, fg_idx, bl_idx, bg_idx, wavelengths, save_dir):
    """Stage2 spectrum + estimated baseline (stage2 - stage3) overlay. One subplot per group."""
    groups = [('fg', fg_idx), ('bl', bl_idx), ('bg', bg_idx)]
    s2_mat = _get_matrix(stage2)
    s3_mat = _get_matrix(stage3)
    baseline_mat = s2_mat - s3_mat  # estimated baseline

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for ax, (gname, idx) in zip(axes, groups):
        label = GROUP_LABELS[gname]
        for i, gi in enumerate(idx):
            ax.plot(wavelengths, s2_mat[gi], color='black', alpha=1.0,
                    linewidth=0.5, label='Denoised' if i == 0 else None)
            ax.plot(wavelengths, baseline_mat[gi], color='darkgreen', alpha=1.0,
                    linewidth=0.8, label='Baseline' if i == 0 else None)
        ax.set_title(f'{label} (n={len(idx)})')
        ax.set_ylabel('Intensity')
        if len(idx) > 0:
            ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.2)
    axes[-1].set_xlabel('Wavenumber (cm⁻¹)')
    fig.suptitle('Stage 3: Denoised Spectrum + Estimated Baseline', fontsize=12)
    plt.tight_layout()
    path = save_dir / 'stage3_baseline_overlay.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_stage3_corrected(stage3, fg_idx, bl_idx, bg_idx, wavelengths, save_dir):
    """Corrected spectra per group. Distinct colours per sample, matching baseline gallery style."""
    groups = [('fg', fg_idx), ('bl', bl_idx), ('bg', bg_idx)]
    s3_mat = _get_matrix(stage3)
    sample_colors = ['black', 'red', 'green', 'blue', 'goldenrod']

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for ax, (gname, idx) in zip(axes, groups):
        label = GROUP_LABELS[gname]
        for i, gi in enumerate(idx):
            c = sample_colors[i % len(sample_colors)]
            ax.plot(wavelengths, s3_mat[gi], color=c, alpha=1.0,
                    linewidth=0.8, label=f'#{gi}')
        ax.set_title(f'{label} corrected (n={len(idx)})')
        ax.set_ylabel('Intensity')
        if len(idx) > 0:
            ax.legend(fontsize=6, loc='upper right')
        ax.grid(True, alpha=0.2)
    axes[-1].set_xlabel('Wavenumber (cm⁻¹)')
    fig.suptitle('Stage 3: Baseline-Corrected Spectra', fontsize=12)
    plt.tight_layout()
    path = save_dir / 'stage3_baseline_corrected.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_fingerprint(fingerprint, wavelengths, save_dir):
    """Plot the final refined fingerprint spectrum."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(wavelengths, fingerprint.intensities, color='black', linewidth=1.0)
    ax.set_xlabel('Wavenumber (cm⁻¹)')
    ax.set_ylabel('Intensity')
    ax.set_title('Seed Fingerprint (top 5% AUC, median, min-max normalised)')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    path = save_dir / 'fingerprint.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_stage6_mean_std(normalised, wavelengths, save_dir):
    """Mean +/- std shading plot for normalised foreground spectra."""
    mat = normalised.to_intensity_matrix()
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(wavelengths, mean, color='black', linewidth=1.2, label='Mean')
    ax.fill_between(wavelengths, mean - std, mean + std,
                    color='steelblue', alpha=0.3, label='± 1 std')
    ax.set_xlabel('Wavenumber (cm⁻¹)')
    ax.set_ylabel('Normalised Intensity')
    ax.set_title(f'Stage 6: Foreground Mean ± Std (n={len(mat)})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    path = save_dir / 'clean_data_mean_std.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_clean_data_gallery(normalised, wavelengths, save_dir, n_samples=16):
    """4x4 gallery of random normalised foreground spectra."""
    mat = normalised.to_intensity_matrix()
    rng = np.random.default_rng(42)
    n = min(n_samples, len(mat))
    chosen = rng.choice(len(mat), size=n, replace=False)

    ncols = 4
    nrows = 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, gi in enumerate(chosen):
        ax = axes[i]
        ax.plot(wavelengths, mat[gi], color='black', linewidth=0.6)
        ax.set_title(f'#{gi}', fontsize=7)
        ax.tick_params(labelsize=5)
        ax.grid(True, alpha=0.2)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'Clean Data Gallery (showing {n} of {len(mat)})', fontsize=11)
    fig.supxlabel('Wavenumber (cm⁻¹)', fontsize=9)
    fig.supylabel('Normalised Intensity', fontsize=9)
    plt.tight_layout()
    path = save_dir / 'clean_data_gallery.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return path




# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_group(collection, mask, wavelengths, filepath):
    """Save subset of collection filtered by mask as npz."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        # Save empty arrays so file exists
        np.savez_compressed(
            filepath,
            wavelengths=wavelengths,
            intensities=np.zeros((0, len(wavelengths)), dtype=np.float32),
            source_file=np.array(collection.source_file),
        )
        return

    mat = collection.to_intensity_matrix()
    sub_mat = mat[idx]
    coords = collection.get_coordinates()
    data_dict = {
        'wavelengths': wavelengths,
        'intensities': sub_mat,
        'source_file': np.array(collection.source_file),
    }
    if coords is not None:
        data_dict['coordinates'] = coords[idx]
    np.savez_compressed(filepath, **data_dict)


# ---------------------------------------------------------------------------
# Main processing function
# ---------------------------------------------------------------------------

def process_rep(raw_folder, save_dir, label, stage,
                sim_high, sim_low, n_samples, noise_percentile=None,
                seed_pool=5):
    """Process a single rep through all stages."""
    raw_folder = Path(raw_folder)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    diag_dir = save_dir / 'diagnostics'
    diag_dir.mkdir(parents=True, exist_ok=True)

    # --- Load raw ---
    collection_path = get_txt_filepath(raw_folder)
    if collection_path is None:
        print(f"  [{label}] No 'power' .txt file found — skipping.")
        return

    print(f"  [{label}] Loading: {collection_path.name}")
    raw = read_spectrum_file(collection_path)
    n_total = len(raw)
    wavelengths = raw.wavelengths
    print(f"  [{label}] {n_total} spectra, {len(wavelengths)} wavenumber points")

    # --- Stage 1 ---
    print(f"  [{label}] Stage 1: edge spike removal + cosmic_ray removal …")
    stage1 = run_stage1(raw)
    export_to_npz(stage1, save_dir / 'stage1_cleaned.npz')
    print(f"  [{label}] Stage 1 done → stage1_cleaned.npz")

    # --- Stage 2 ---
    print(f"  [{label}] Stage 2: wavelet denoising (sym6, VisuShrink) …")
    stage2 = run_stage2(stage1)
    export_to_npz(stage2, save_dir / 'stage2_denoised.npz')
    print(f"  [{label}] Stage 2 done → stage2_denoised.npz")

    # --- Stage 3 ---
    clip_desc = f'noise_percentile={noise_percentile}' if noise_percentile else 'zero'
    print(f"  [{label}] Stage 3: BubbleFill baseline removal + clip ({clip_desc}) …")
    stage3 = run_stage3(stage2, noise_percentile=noise_percentile)
    export_to_npz(stage3, save_dir / 'stage3_baseline_removed.npz')
    print(f"  [{label}] Stage 3 done → stage3_baseline_removed.npz")

    # --- Stage 4: fingerprint computation ---
    print(f"  [{label}] Stage 4: AUC seed selection (top {seed_pool}%) → median fingerprint …")
    auc_scores, fingerprint = run_stage4(stage3, seed_pool_percentile=seed_pool)
    print(f"  [{label}] Stage 4 done (seeds: top {seed_pool}% by AUC, median + min-max norm)")
    fp_collection = SpectraCollection(
        spectra=[fingerprint],
        source_file=stage3.source_file,
        wavelengths=wavelengths.copy(),
    )
    export_to_npz(fp_collection, save_dir / 'fingerprint.npz')

    # --- Stage 5: similarity classification ---
    print(f"  [{label}] Stage 5: similarity classification (sim_high={sim_high}, sim_low={sim_low}) …")
    similarity, fg_mask, bg_mask, bl_mask = run_stage5(stage3, fingerprint, sim_high, sim_low)

    n_fg = int(fg_mask.sum())
    n_bg = int(bg_mask.sum())
    n_bl = int(bl_mask.sum())
    print(f"  [{label}] Classification: fg={n_fg}, borderline={n_bl}, bg={n_bg}")

    save_group(stage3, fg_mask, wavelengths, save_dir / 'stage5_foreground.npz')
    save_group(stage3, bg_mask, wavelengths, save_dir / 'stage5_background.npz')
    save_group(stage3, bl_mask, wavelengths, save_dir / 'stage5_borderline.npz')
    print(f"  [{label}] Saved stage5 group npz files")

    # --- Diagnostic plots (retroactive, using masks) ---
    rng = np.random.default_rng(42)
    fg_idx = sample_indices(fg_mask, n_samples, rng)
    bl_idx = sample_indices(bl_mask, n_samples, rng)
    bg_idx = sample_indices(bg_mask, n_samples, rng)

    print(f"  [{label}] Generating diagnostic plots …")
    stages_dir = diag_dir / 'stages'
    stages_dir.mkdir(parents=True, exist_ok=True)

    # Stage sample plots → diagnostics/stages/
    plot_stage0_raw(raw, fg_idx, bl_idx, bg_idx, wavelengths, stages_dir)
    plot_stage1_cleaned(stage1, fg_idx, bl_idx, bg_idx, wavelengths, stages_dir)
    plot_stage2_denoised(stage2, fg_idx, bl_idx, bg_idx, wavelengths, stages_dir)
    plot_stage3_corrected(stage3, fg_idx, bl_idx, bg_idx, wavelengths, stages_dir)

    # Overlay / residual plots → diagnostics/
    plot_stage2_residual(stage1, stage2, fg_idx, bl_idx, bg_idx, wavelengths, stages_dir)
    plot_stage3_baseline(stage2, stage3, fg_idx, bl_idx, bg_idx, wavelengths, stages_dir)
    plot_fingerprint(fingerprint, wavelengths, diag_dir)
    print(f"  [{label}] Diagnostic plots saved in diagnostics/")

    # --- Stage 6 (optional): normalise + store ---
    normalised = None
    fg_fingerprint = None
    scales = None

    if stage >= 6:
        if n_fg == 0:
            print(f"  [{label}] Stage 6: no foreground spectra — skipping normalisation")
        else:
            print(f"  [{label}] Stage 6: normalising {n_fg} foreground spectra …")
            normalised, scales = run_stage6(stage3, fg_mask)
            fg_fingerprint = compute_fingerprint(normalised)

            # Save normalised foreground as clean_data.npz
            fg_indices = np.where(fg_mask)[0]
            coords = raw.get_coordinates()

            # Per-spectrum SNR: max(stage3) / MAD-σ of (stage1 - stage2),
            # restricted to foreground spectra and computed over the whole
            # wavelength axis. See spectra.preprocessing.snr.
            s1_fg = stage1.to_intensity_matrix()[fg_indices]
            s2_fg = stage2.to_intensity_matrix()[fg_indices]
            s3_fg = stage3.to_intensity_matrix()[fg_indices]
            snr, _, noise_std = calculate_snr_from_stages(s1_fg, s2_fg, s3_fg)

            clean_dict = {
                'wavelengths': wavelengths,
                'intensities': normalised.to_intensity_matrix(),
                'scale_factors': scales,
                'snr': snr,
                'noise_std': noise_std,
                'source_file': np.array(stage3.source_file),
            }
            if coords is not None:
                clean_dict['coordinates'] = coords[fg_indices]
            np.savez_compressed(save_dir / 'clean_data.npz', **clean_dict)

            # Save corresponding raw spectra as raw_data.npz
            raw_mat = raw.to_intensity_matrix()
            raw_dict = {
                'wavelengths': raw.wavelengths,
                'intensities': raw_mat[fg_indices],
                'source_file': np.array(raw.source_file),
            }
            if coords is not None:
                raw_dict['coordinates'] = coords[fg_indices]
            np.savez_compressed(save_dir / 'raw_data.npz', **raw_dict)

            print(f"  [{label}] Stage 6 done → clean_data.npz ({n_fg}), raw_data.npz ({n_fg})")

            plot_stage6_mean_std(normalised, wavelengths, diag_dir)
            plot_clean_data_gallery(normalised, wavelengths, diag_dir)

    # --- Metadata ---
    metadata = {
        'label': label,
        'source_file': str(collection_path),
        'n_total': n_total,
        'n_wavenumber': int(len(wavelengths)),
        'stage_run': stage,
        'thresholds': {
            'sim_high': sim_high,
            'sim_low': sim_low,
            'seed_pool_percentile': seed_pool,
        },
        'stage4': {
            'auc_mean': float(auc_scores.mean()),
            'auc_median': float(np.median(auc_scores)),
        },
        'stage5': {
            'n_foreground': n_fg,
            'n_borderline': n_bl,
            'n_background': n_bg,
            'similarity_mean': float(similarity.mean()),
            'similarity_median': float(np.median(similarity)),
            'similarity_fg_mean': float(similarity[fg_mask].mean()) if n_fg > 0 else None,
        },
        'stage6': {
            'n_normalised': n_fg if normalised is not None else 0,
            'scale_min': float(scales.min()) if scales is not None else None,
            'scale_max': float(scales.max()) if scales is not None else None,
            'scale_mean': float(scales.mean()) if scales is not None else None,
        } if stage >= 6 else None,
    }
    with open(save_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  [{label}] metadata.json saved")


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Magic staged preprocessing pipeline for Raman spectra.'
    )
    parser.add_argument('--raw-root', type=Path,
                        default=Path('data/custom/raw'),
                        help='Root directory containing raw rep directories')
    parser.add_argument('--output-root', type=Path,
                        default=Path('data/custom/processed/magic'),
                        help='Root directory for processed output')
    parser.add_argument('--stage', type=int, default=6, choices=[1, 2, 3, 4, 5, 6],
                        help='Maximum stage to run (stages 1-5 always execute; '
                             'stage 6 only when --stage 6)')
    parser.add_argument('--sim-high', type=float, default=0.8,
                        help='Similarity threshold for foreground (default: 0.7)')
    parser.add_argument('--sim-low', type=float, default=0.4,
                        help='Similarity threshold for background (default: 0.4)')
    parser.add_argument('--length', type=str, default=None,
                        help='Filter by length dir (e.g. 1-letter, 2-letter); default: all')
    parser.add_argument('--sequence', type=str, default=None,
                        help='Filter by sequence name (e.g. GA); default: all')
    parser.add_argument('--rep', type=str, default=None,
                        help='Filter by rep name (e.g. rep2); default: all')
    parser.add_argument('--noise-percentile', type=float, default=None,
                        help='Clip floor as per-spectrum percentile (e.g. 5). '
                             'Default: None (clip to zero)')
    parser.add_argument('--n-samples', type=int, default=5,
                        help='Number of sample spectra per group in diagnostic plots (default: 5)')
    parser.add_argument('--seed-pool', type=int, default=5,
                        help='Percentile of spectra (by AUC) used as seed pool (default: 5)')
    parser.add_argument('--overwrite-existing', action='store_true',
                        help='Reprocess reps even if metadata.json already exists in output dir')
    args = parser.parse_args()

    raw_root = args.raw_root
    output_root = args.output_root

    print("Magic Preprocessing Pipeline")
    print("=" * 60)
    print(f"Raw root:     {raw_root.resolve()}")
    print(f"Output root:  {output_root.resolve()}")
    print(f"Stage:        {args.stage}")
    print(f"Similarity:   [{args.sim_low}, {args.sim_high}]")
    print(f"Seed pool:    top {args.seed_pool}% by AUC")
    print(f"Noise pctl:   {args.noise_percentile or 'None (clip to zero)'}")
    print(f"Length:       {args.length or 'all'}")
    print(f"Sequence:     {args.sequence or 'all'}")
    print(f"Rep:          {args.rep or 'all'}")
    print("=" * 60)

    # Discover rep directories: raw_root/{n-letter}/{SEQ}/rep*/
    # PTM amino acids live one level deeper: raw_root/1-letter/PTM/{ptm-name}/rep*/
    rep_dirs = []
    for length_dir in sorted(raw_root.iterdir()):
        if not length_dir.is_dir():
            continue
        # Filter by length
        if args.length is not None and length_dir.name != args.length:
            continue
        for seq_dir in sorted(length_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            # Handle PTM subdirectory (1-letter/PTM/{ptm-name}/rep*/)
            if seq_dir.name == "PTM":
                for ptm_dir in sorted(seq_dir.iterdir()):
                    if not ptm_dir.is_dir():
                        continue
                    if args.sequence is not None and ptm_dir.name != args.sequence:
                        continue
                    for rep_dir in sorted(ptm_dir.iterdir()):
                        if not rep_dir.is_dir():
                            continue
                        if not rep_dir.name.startswith('rep'):
                            continue
                        if args.rep is not None and rep_dir.name != args.rep:
                            continue
                        rep_dirs.append(rep_dir)
                continue
            # Filter by sequence
            if args.sequence is not None and seq_dir.name != args.sequence:
                continue
            for rep_dir in sorted(seq_dir.iterdir()):
                if not rep_dir.is_dir():
                    continue
                if not rep_dir.name.startswith('rep'):
                    continue
                # Filter by rep
                if args.rep is not None and rep_dir.name != args.rep:
                    continue
                rep_dirs.append(rep_dir)

    if not rep_dirs:
        print("No matching rep directories found.")
        return

    print(f"Found {len(rep_dirs)} rep(s) to process.\n")

    for rep_dir in rep_dirs:
        rel = rep_dir.relative_to(raw_root)
        label = str(rel).replace('\\', '/')
        save_dir = output_root / rel

        print(f"\n{'='*60}")
        print(f"Processing: {label}")
        print(f"Output:     {save_dir}")
        print(f"{'='*60}")

        if not args.overwrite_existing and (save_dir / 'metadata.json').exists():
            print(f"  [{label}] Skipping — metadata.json already exists")
            continue

        try:
            process_rep(
                raw_folder=rep_dir,
                save_dir=save_dir,
                label=label,
                stage=args.stage,
                sim_high=args.sim_high,
                sim_low=args.sim_low,
                n_samples=args.n_samples,
                noise_percentile=args.noise_percentile,
                seed_pool=args.seed_pool,
            )
        except Exception as e:
            print(f"  ERROR processing {label}: {e}")
            import traceback
            traceback.print_exc()

    print("\nDone.")


if __name__ == '__main__':
    main()
