"""
Staged Raman Spectra Preprocessing Script (Magic-Bayes Pipeline)
=================================================================

Variant of ``preprocess_magic.py`` with two changes:

1. **Stage 2 uses BayesShrink** (level-dependent threshold) instead of
   VisuShrink. Preserves more inter-peak detail that the differential
   classifier relies on for distinguishing similar peptides.

2. **SNR-threshold filtering** replaces the AUC fingerprint + czekanowski
   similarity classification. Per-spectrum SNR is computed via
   ``calculate_snr_from_stages`` (noise = MAD-σ of stage1−stage2 residual,
   signal = max(stage3)). Spectra with SNR > threshold pass; the rest are
   dropped.

Stages:
    1. remove_edge_spikes + cosmic_ray removal
    2. Wavelet denoising (sym6, BayesShrink)
    3. BubbleFill baseline removal + clip to zero
    4. SNR computation + threshold filter (replaces old stages 4-5)
    5. Normalise and store passing signals (raw + normalised processed)

Diagnostic plots are generated retroactively after the SNR filter, using
two groups (high-SNR / low-SNR) for colour coding.
"""

import argparse
import json
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
)
from primarymagic.preprocessing.snr import calculate_snr_from_stages
from primarymagic.preprocessing.contamination import detect_cr_contamination

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


def write_quality_mask_bayes(save_dir, snr, signal, noise_std, pass_mask,
                             snr_threshold, snr_pass=None,
                             cr_contaminated=None, cr_fwhm_min=None,
                             cr_energy_ratio_max=None):
    """Write the per-spectrum quality-mask sidecar to ``save_dir / 'mask.npz'``.

    v2 schema additions: ``snr_pass``, ``cr_contaminated``, ``cr_fwhm_min``,
    ``cr_energy_ratio_max``. Default `snr_pass = pass_mask` and
    `cr_contaminated = zeros` when v2 args are omitted (back-compat).

    See docs/superpowers/specs/2026-05-12-robust-cosmic-ray-removal-design.md
    for the v2 schema.

    Args:
        save_dir: Directory in which to write ``mask.npz``.
        snr: Per-spectrum SNR values, length N.
        signal: Per-spectrum max(stage3) values, length N.
        noise_std: Per-spectrum MAD-sigma noise estimate, length N.
        pass_mask: Final pass decision per spectrum (snr_pass & ~cr_contaminated).
        snr_threshold: SNR threshold used.
        snr_pass: Raw SNR-only decision. Defaults to `pass_mask`.
        cr_contaminated: Contamination filter output. Defaults to all-False.
        cr_fwhm_min: Threshold used by contamination Test 1. Defaults to 0.0.
        cr_energy_ratio_max: Threshold used by contamination Test 2. Defaults to 1.0.
    """
    save_dir = Path(save_dir)
    passed = np.asarray(pass_mask, dtype=bool)
    n = len(passed)
    clean_index = np.full(n, -1, dtype=np.int32)
    clean_index[passed] = np.arange(int(passed.sum()), dtype=np.int32)

    if snr_pass is None:
        snr_pass = passed
    if cr_contaminated is None:
        cr_contaminated = np.zeros(n, dtype=bool)
    if cr_fwhm_min is None:
        cr_fwhm_min = 0.0
    if cr_energy_ratio_max is None:
        cr_energy_ratio_max = 1.0

    np.savez_compressed(
        save_dir / 'mask.npz',
        raw_index=np.arange(n, dtype=np.int32),
        clean_index=clean_index,
        passed=passed,
        snr=np.asarray(snr, dtype=np.float64),
        signal=np.asarray(signal, dtype=np.float64),
        noise_std=np.asarray(noise_std, dtype=np.float64),
        snr_threshold=np.float64(snr_threshold),
        snr_pass=np.asarray(snr_pass, dtype=bool),
        cr_contaminated=np.asarray(cr_contaminated, dtype=bool),
        cr_fwhm_min=np.float64(cr_fwhm_min),
        cr_energy_ratio_max=np.float64(cr_energy_ratio_max),
    )


# ---------------------------------------------------------------------------
# Processing stages
# ---------------------------------------------------------------------------

def run_stage1(collection):
    """remove_edge_spikes(edge_n=10, factor=5.0) → remove_cosmic_rays_robust(k=4.5, max_iter=5, width_max=8)"""
    return (
        PreprocessingPipeline(collection)
        .remove_edge_spikes(edge_n=10, factor=5.0)
        .remove_cosmic_rays_robust(k=4.5, max_iter=5, width_max=8)
        .result()
    )


def run_stage2(stage1):
    """smooth(method='wavelet', wavelet='sym6', threshold_method='bayes')"""
    return (
        PreprocessingPipeline(stage1)
        .smooth(method='wavelet', wavelet='sym6', threshold_method='bayes')
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


def run_stage4_snr_filter(stage1, stage2, stage3, snr_threshold,
                          fwhm_min, energy_ratio_max):
    """Stage 4: SNR + cosmic-ray contamination filter.

    Compute per-spectrum SNR via calculate_snr_from_stages, then AND with the
    contamination filter (FWHM + single-pixel-energy two-test OR).

    Returns:
        snr, signal, noise_std, pass_mask, fail_mask, snr_pass, cr_contaminated
    """
    s1_mat = stage1.to_intensity_matrix()
    s2_mat = stage2.to_intensity_matrix()
    s3_mat = stage3.to_intensity_matrix()
    snr, signal, noise_std = calculate_snr_from_stages(s1_mat, s2_mat, s3_mat)

    snr_pass = snr > snr_threshold
    cr_contaminated = detect_cr_contamination(
        stage3, fwhm_min=fwhm_min, energy_ratio_max=energy_ratio_max
    )
    pass_mask = snr_pass & ~cr_contaminated
    fail_mask = ~pass_mask
    return snr, signal, noise_std, pass_mask, fail_mask, snr_pass, cr_contaminated


def run_stage6(stage3, pass_mask):
    """Stage 6: Normalise passing spectra.

    scale = max(intensities) per spectrum, normalised = spectrum / scale.

    Returns: (normalised_collection, scales_array)
    """
    mat = stage3.to_intensity_matrix()
    sub_mat = mat[pass_mask]
    scales = sub_mat.max(axis=1)          # shape (n_pass,)
    safe_scales = np.where(scales > 0, scales, 1.0)
    normed_mat = sub_mat / safe_scales[:, None]

    wavelengths = stage3.wavelengths
    new_spectra = []
    pass_indices = np.where(pass_mask)[0]
    src = stage3.source_file
    for i, gi in enumerate(pass_indices):
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
    """Mean of normalised passing spectra."""
    mat = normalised.to_intensity_matrix()
    mean_int = mat.mean(axis=0)
    wavelengths = normalised.wavelengths
    fp_data = np.column_stack([wavelengths, mean_int])
    return Spectrum(data=fp_data)


# ---------------------------------------------------------------------------
# Diagnostic plots (2-group: pass / fail)
# ---------------------------------------------------------------------------

GROUP_COLORS = {
    'pass': 'red',
    'fail': 'blue',
}
GROUP_LABELS = {
    'pass': 'Passed (SNR > threshold)',
    'fail': 'Rejected (SNR <= threshold)',
}


def _get_matrix(collection):
    return collection.to_intensity_matrix()


def _plot_samples_2x1(data_mat, pass_idx, fail_idx, wavelengths, save_dir,
                      filename, title):
    """Generic 2x1 plot with distinct-coloured samples per group."""
    groups = [('pass', pass_idx), ('fail', fail_idx)]
    sample_colors = ['black', 'red', 'green', 'blue', 'goldenrod']

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
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
    axes[-1].set_xlabel('Wavenumber (cm-1)')
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    path = save_dir / filename
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_stage0_raw(raw, pass_idx, fail_idx, wavelengths, save_dir):
    """Stage 0: raw spectra samples per group."""
    return _plot_samples_2x1(
        _get_matrix(raw), pass_idx, fail_idx, wavelengths, save_dir,
        'stage0_raw.png', 'Stage 0: Raw Spectra')


def plot_stage1_cleaned(stage1, pass_idx, fail_idx, wavelengths, save_dir):
    """Stage 1: cleaned spectra (after edge spike removal + cosmic_ray) samples per group."""
    return _plot_samples_2x1(
        _get_matrix(stage1), pass_idx, fail_idx, wavelengths, save_dir,
        'stage1_fix_cosmic_ray.png', 'Stage 1: After Edge Spike Removal + Cosmic Ray Removal')


def plot_stage2_denoised(stage2, pass_idx, fail_idx, wavelengths, save_dir):
    """Stage 2: denoised spectra samples per group."""
    return _plot_samples_2x1(
        _get_matrix(stage2), pass_idx, fail_idx, wavelengths, save_dir,
        'stage2_denoise.png', 'Stage 2: After Wavelet Denoising (sym6 BayesShrink)')


def plot_stage2_residual(stage1, stage2, pass_idx, fail_idx, wavelengths, save_dir):
    """Noise residual (stage1 - stage2). One subplot per group."""
    groups = [('pass', pass_idx), ('fail', fail_idx)]
    s1_mat = _get_matrix(stage1)
    s2_mat = _get_matrix(stage2)
    residual = s1_mat - s2_mat

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
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
    axes[-1].set_xlabel('Wavenumber (cm-1)')
    fig.suptitle('Stage 2: Noise Residual (Stage1 - Stage2)', fontsize=12)
    plt.tight_layout()
    path = save_dir / 'stage2_residual.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_stage3_baseline(stage2, stage3, pass_idx, fail_idx, wavelengths, save_dir):
    """Stage2 spectrum + estimated baseline (stage2 - stage3) overlay. One subplot per group."""
    groups = [('pass', pass_idx), ('fail', fail_idx)]
    s2_mat = _get_matrix(stage2)
    s3_mat = _get_matrix(stage3)
    baseline_mat = s2_mat - s3_mat  # estimated baseline

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
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
    axes[-1].set_xlabel('Wavenumber (cm-1)')
    fig.suptitle('Stage 3: Denoised Spectrum + Estimated Baseline', fontsize=12)
    plt.tight_layout()
    path = save_dir / 'stage3_baseline_overlay.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_stage3_corrected(stage3, pass_idx, fail_idx, wavelengths, save_dir):
    """Corrected spectra per group."""
    groups = [('pass', pass_idx), ('fail', fail_idx)]
    s3_mat = _get_matrix(stage3)
    sample_colors = ['black', 'red', 'green', 'blue', 'goldenrod']

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
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
    axes[-1].set_xlabel('Wavenumber (cm-1)')
    fig.suptitle('Stage 3: Baseline-Corrected Spectra', fontsize=12)
    plt.tight_layout()
    path = save_dir / 'stage3_baseline_corrected.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_snr_histogram(snr, snr_threshold, save_dir):
    """Histogram of per-spectrum SNR with threshold marker."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(snr, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(snr_threshold, color='red', linewidth=1.5, linestyle='--',
               label=f'threshold = {snr_threshold}')
    ax.set_xlabel('SNR')
    ax.set_ylabel('Count')
    n_pass = int((snr > snr_threshold).sum())
    n_fail = int((snr <= snr_threshold).sum())
    ax.set_title(f'Per-spectrum SNR distribution  (pass={n_pass}, fail={n_fail})')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    path = save_dir / 'snr_histogram.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_stage6_mean_std(normalised, wavelengths, save_dir):
    """Mean ± std shading plot for normalised passing spectra."""
    mat = normalised.to_intensity_matrix()
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(wavelengths, mean, color='black', linewidth=1.2, label='Mean')
    ax.fill_between(wavelengths, mean - std, mean + std,
                    color='steelblue', alpha=0.3, label='+/- 1 std')
    ax.set_xlabel('Wavenumber (cm-1)')
    ax.set_ylabel('Normalised Intensity')
    ax.set_title(f'Stage 6: Passing Mean +/- Std (n={len(mat)})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    path = save_dir / 'clean_data_mean_std.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_clean_data_gallery(normalised, wavelengths, save_dir, n_samples=16):
    """4x4 gallery of random normalised passing spectra."""
    mat = normalised.to_intensity_matrix()
    rng = np.random.default_rng(42)
    n = min(n_samples, len(mat))
    if n == 0:
        return None
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
    fig.supxlabel('Wavenumber (cm-1)', fontsize=9)
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
                snr_threshold, fwhm_min, energy_ratio_max,
                n_samples, noise_percentile=None):
    """Process a single rep through all stages."""
    raw_folder = Path(raw_folder)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    diag_dir = save_dir / 'diagnostics'
    diag_dir.mkdir(parents=True, exist_ok=True)

    # --- Load raw ---
    collection_path = get_txt_filepath(raw_folder)
    if collection_path is None:
        print(f"  [{label}] No 'power' .txt file found - skipping.")
        return

    print(f"  [{label}] Loading: {collection_path.name}")
    raw = read_spectrum_file(collection_path)
    n_total = len(raw)
    wavelengths = raw.wavelengths
    print(f"  [{label}] {n_total} spectra, {len(wavelengths)} wavenumber points")

    # --- Stage 1 ---
    print(f"  [{label}] Stage 1: edge spike removal + cosmic_ray removal ...")
    stage1 = run_stage1(raw)
    export_to_npz(stage1, save_dir / 'stage1_cleaned.npz')
    print(f"  [{label}] Stage 1 done -> stage1_cleaned.npz")

    # --- Stage 2 ---
    print(f"  [{label}] Stage 2: wavelet denoising (sym6, BayesShrink) ...")
    stage2 = run_stage2(stage1)
    export_to_npz(stage2, save_dir / 'stage2_denoised.npz')
    print(f"  [{label}] Stage 2 done -> stage2_denoised.npz")

    # --- Stage 3 ---
    clip_desc = (f'noise_percentile={noise_percentile}'
                 if noise_percentile is not None else 'zero')
    print(f"  [{label}] Stage 3: BubbleFill baseline removal + clip ({clip_desc}) ...")
    stage3 = run_stage3(stage2, noise_percentile=noise_percentile)
    export_to_npz(stage3, save_dir / 'stage3_baseline_removed.npz')
    print(f"  [{label}] Stage 3 done -> stage3_baseline_removed.npz")

    # --- Stage 4: SNR threshold + CR contamination filter ---
    print(f"  [{label}] Stage 4: SNR + CR contamination filter "
          f"(snr_threshold={snr_threshold}, fwhm_min={fwhm_min}, "
          f"energy_ratio_max={energy_ratio_max}) ...")
    (snr, signal, noise_std, pass_mask, fail_mask,
     snr_pass, cr_contaminated) = run_stage4_snr_filter(
        stage1, stage2, stage3, snr_threshold, fwhm_min, energy_ratio_max,
    )

    n_pass = int(pass_mask.sum())
    n_fail = int(fail_mask.sum())
    n_snr_fail = int((~snr_pass).sum())
    n_cr_fail = int(cr_contaminated.sum())
    print(f"  [{label}] Filter results: pass={n_pass}, fail={n_fail} "
          f"(snr_fail={n_snr_fail}, cr_contaminated={n_cr_fail})")
    print(f"  [{label}] SNR stats: min={snr.min():.2f}, "
          f"median={np.median(snr):.2f}, max={snr.max():.2f}")

    save_group(stage3, pass_mask, wavelengths, save_dir / 'stage4_passed.npz')
    save_group(stage3, fail_mask, wavelengths, save_dir / 'stage4_rejected.npz')
    print(f"  [{label}] Saved stage4 group npz files")

    write_quality_mask_bayes(
        save_dir, snr, signal, noise_std, pass_mask, snr_threshold,
        snr_pass=snr_pass,
        cr_contaminated=cr_contaminated,
        cr_fwhm_min=fwhm_min,
        cr_energy_ratio_max=energy_ratio_max,
    )
    print(f"  [{label}] Saved quality mask -> mask.npz")

    export_to_npz(raw, save_dir / 'raw_all.npz')
    print(f"  [{label}] Saved full raw collection -> raw_all.npz")

    # --- Diagnostic plots (retroactive, using masks) ---
    rng = np.random.default_rng(42)
    pass_idx = sample_indices(pass_mask, n_samples, rng)
    fail_idx = sample_indices(fail_mask, n_samples, rng)

    print(f"  [{label}] Generating diagnostic plots ...")
    stages_dir = diag_dir / 'stages'
    stages_dir.mkdir(parents=True, exist_ok=True)

    plot_stage0_raw(raw, pass_idx, fail_idx, wavelengths, stages_dir)
    plot_stage1_cleaned(stage1, pass_idx, fail_idx, wavelengths, stages_dir)
    plot_stage2_denoised(stage2, pass_idx, fail_idx, wavelengths, stages_dir)
    plot_stage3_corrected(stage3, pass_idx, fail_idx, wavelengths, stages_dir)
    plot_stage2_residual(stage1, stage2, pass_idx, fail_idx, wavelengths, stages_dir)
    plot_stage3_baseline(stage2, stage3, pass_idx, fail_idx, wavelengths, stages_dir)
    plot_snr_histogram(snr, snr_threshold, diag_dir)
    print(f"  [{label}] Diagnostic plots saved in diagnostics/")

    # --- Stage 6 (optional): normalise + store ---
    normalised = None
    fingerprint = None
    scales = None

    if stage >= 6:
        if n_pass == 0:
            print(f"  [{label}] Stage 6: no passing spectra - skipping normalisation")
        else:
            print(f"  [{label}] Stage 6: normalising {n_pass} passing spectra ...")
            normalised, scales = run_stage6(stage3, pass_mask)
            fingerprint = compute_fingerprint(normalised)

            # Save fingerprint
            fp_collection = SpectraCollection(
                spectra=[fingerprint],
                source_file=stage3.source_file,
                wavelengths=wavelengths.copy(),
            )
            export_to_npz(fp_collection, save_dir / 'fingerprint.npz')

            # Save normalised passing as clean_data.npz
            pass_indices = np.where(pass_mask)[0]
            coords = raw.get_coordinates()

            clean_dict = {
                'wavelengths': wavelengths,
                'intensities': normalised.to_intensity_matrix(),
                'scale_factors': scales,
                'snr': snr[pass_mask],
                'noise_std': noise_std[pass_mask],
                'source_file': np.array(stage3.source_file),
            }
            if coords is not None:
                clean_dict['coordinates'] = coords[pass_indices]
            np.savez_compressed(save_dir / 'clean_data.npz', **clean_dict)

            # Save corresponding raw spectra as raw_data.npz
            raw_mat = raw.to_intensity_matrix()
            raw_dict = {
                'wavelengths': raw.wavelengths,
                'intensities': raw_mat[pass_indices],
                'source_file': np.array(raw.source_file),
            }
            if coords is not None:
                raw_dict['coordinates'] = coords[pass_indices]
            np.savez_compressed(save_dir / 'raw_data.npz', **raw_dict)

            print(f"  [{label}] Stage 6 done -> clean_data.npz ({n_pass}), "
                  f"raw_data.npz ({n_pass}), fingerprint.npz")

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
            'snr_threshold': snr_threshold,
            'noise_percentile': noise_percentile,
            'cr_fwhm_min': fwhm_min,
            'cr_energy_ratio_max': energy_ratio_max,
        },
        'stage4': {
            'n_passed': n_pass,
            'n_rejected': n_fail,
            'n_snr_fail': n_snr_fail,
            'n_cr_contaminated': n_cr_fail,
            'snr_min': float(snr.min()),
            'snr_median': float(np.median(snr)),
            'snr_max': float(snr.max()),
            'snr_mean': float(snr.mean()),
        },
        'stage6': {
            'n_normalised': n_pass if normalised is not None else 0,
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
        description='Magic-Bayes preprocessing pipeline: BayesShrink denoising + '
                    'SNR-threshold filtering. Variant of preprocess_magic.py.'
    )
    parser.add_argument('--raw-root', type=Path,
                        default=Path('data/custom/raw'),
                        help='Root directory containing raw rep directories')
    parser.add_argument('--output-root', type=Path,
                        default=Path('data/custom/processed/magic_bayes_cr'),
                        help='Root directory for processed output')
    parser.add_argument('--stage', type=int, default=6, choices=[1, 2, 3, 4, 5, 6],
                        help='Maximum stage to run (stages 1-4 always execute; '
                             'stage 6 only when --stage 6)')
    parser.add_argument('--snr-threshold', type=float, default=50.0,
                        help='Per-spectrum SNR threshold (default: 50)')
    parser.add_argument('--length', type=str, default=None,
                        help='Filter by length dir (e.g. 1-letter, 2-letter); default: all')
    parser.add_argument('--sequence', type=str, default=None,
                        help='Filter by sequence name (e.g. GA); default: all')
    parser.add_argument('--rep', type=str, default=None,
                        help='Filter by rep name (e.g. rep2); default: all')
    parser.add_argument('--noise-percentile', type=float, default=None,
                        help='Clip floor as per-spectrum percentile (e.g. 5). '
                             'Default: None (clip to zero)')
    parser.add_argument('--cr-fwhm-min', type=float, default=3.0,
                        help='Minimum FWHM (pixels) of the dominant stage-3 peak '
                             'below which a spectrum is flagged as CR-contaminated. '
                             'Default: 3.0')
    parser.add_argument('--cr-energy-ratio-max', type=float, default=0.15,
                        help='Maximum single-pixel/total-energy ratio at the '
                             'dominant stage-3 peak. Default: 0.15')
    parser.add_argument('--n-samples', type=int, default=5,
                        help='Number of sample spectra per group in diagnostic plots (default: 5)')
    parser.add_argument('--overwrite-existing', action='store_true',
                        help='Reprocess reps even if metadata.json already exists in output dir')
    args = parser.parse_args()

    raw_root = args.raw_root
    output_root = args.output_root

    print("Magic-Bayes Preprocessing Pipeline")
    print("=" * 60)
    print(f"Raw root:      {raw_root.resolve()}")
    print(f"Output root:   {output_root.resolve()}")
    print(f"Stage:         {args.stage}")
    print(f"SNR threshold: {args.snr_threshold}")
    print(f"Noise pctl:    {args.noise_percentile if args.noise_percentile is not None else 'None (clip to zero)'}")
    print(f"CR fwhm min:   {args.cr_fwhm_min}")
    print(f"CR energy max: {args.cr_energy_ratio_max}")
    print(f"Length:        {args.length or 'all'}")
    print(f"Sequence:      {args.sequence or 'all'}")
    print(f"Rep:           {args.rep or 'all'}")
    print("=" * 60)

    # Discover rep directories: raw_root/{n-letter}/{SEQ}/rep*/
    # PTM amino acids live one level deeper: raw_root/1-letter/PTM/{ptm-name}/rep*/
    rep_dirs = []
    for length_dir in sorted(raw_root.iterdir()):
        if not length_dir.is_dir():
            continue
        if args.length is not None and length_dir.name != args.length:
            continue
        for seq_dir in sorted(length_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
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
            if args.sequence is not None and seq_dir.name != args.sequence:
                continue
            for rep_dir in sorted(seq_dir.iterdir()):
                if not rep_dir.is_dir():
                    continue
                if not rep_dir.name.startswith('rep'):
                    continue
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
            print(f"  [{label}] Skipping -- metadata.json already exists")
            continue

        try:
            process_rep(
                raw_folder=rep_dir,
                save_dir=save_dir,
                label=label,
                stage=args.stage,
                snr_threshold=args.snr_threshold,
                fwhm_min=args.cr_fwhm_min,
                energy_ratio_max=args.cr_energy_ratio_max,
                n_samples=args.n_samples,
                noise_percentile=args.noise_percentile,
            )
        except Exception as e:
            print(f"  ERROR processing {label}: {e}")
            import traceback
            traceback.print_exc()

    print("\nDone.")


if __name__ == '__main__':
    main()
