"""
Generate Synthetic Noisy Spectra
=================================

Builds a diverse pool of noise baselines from real background spectra, then
combines DFT simulation spectra with realistic noise at controlled SNR levels.

Steps:
  [1/6] Load background spectra from GA rep1/rep2/rep3
  [2/6] Rank by similarity to fingerprint → select 100 purest noise spectra
  [3/6] Extract low-freq baselines (SG 201,3), build 50-baseline pool via PCA+KMeans
  [4/6] Validate noise (background residuals vs signal residuals)
  [5/6] Interpolate DFT simulation, scale, combine with noise at SNR 10/25/50/100
  [6/6] Save output

Usage:
  python scripts/generate_synthetic_noisy_data.py \\
      --sim-path data/simulation/2-letter/GA/rep3/raman.txt \\
      --background-dir data/custom/processed/primary_magic/2-letter/GA \\
      --output-dir data/synthetic/GA \\
      --validate
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Package imports
# ---------------------------------------------------------------------------
from primarymagic import (
    PreprocessingPipeline,
    Spectrum,
    SpectraCollection,
    cosine_similarity,
    export_to_npz,
    peak_energy_ratio,
    spectral_angle,
)
from primarymagic.data.spectraio import load_from_npz


# ===========================================================================
# SECTION 1 – Background loading & selection
# ===========================================================================


def load_background_spectra(background_dir: Path) -> SpectraCollection:
    """Load and pool raw_background.npz from all rep*/ directories.

    Returns SpectraCollection of all pooled background spectra.
    """
    background_dir = Path(background_dir)
    rep_dirs = sorted(d for d in background_dir.iterdir() if d.is_dir() and d.name.startswith("rep"))

    all_spectra: list[Spectrum] = []
    wavelengths = None

    for rep in rep_dirs:
        npz_path = rep / "raw_background.npz"
        if not npz_path.exists():
            print(f"  [load_background_spectra] Skipping {rep.name}: no raw_background.npz")
            continue
        col = load_from_npz(npz_path)
        if wavelengths is None:
            wavelengths = col.wavelengths.copy()
        print(f"  [load_background_spectra] {rep.name}: {len(col)} spectra loaded")
        all_spectra.extend(col.spectra)

    if not all_spectra:
        raise FileNotFoundError(f"No raw_background.npz files found under {background_dir}")

    print(f"  [load_background_spectra] Total pooled: {len(all_spectra)} spectra")
    return SpectraCollection(
        spectra=all_spectra,
        source_file=str(background_dir),
        wavelengths=wavelengths,
    )


def load_fingerprint(background_dir: Path) -> Spectrum:
    """Load fingerprint.npz from first available rep directory.

    Returns single Spectrum object.
    """
    background_dir = Path(background_dir)
    rep_dirs = sorted(d for d in background_dir.iterdir() if d.is_dir() and d.name.startswith("rep"))

    for rep in rep_dirs:
        fp_path = rep / "fingerprint.npz"
        if fp_path.exists():
            col = load_from_npz(fp_path)
            print(f"  [load_fingerprint] Loaded from {rep.name}/fingerprint.npz")
            return col.spectra[0]

    raise FileNotFoundError(f"No fingerprint.npz found under {background_dir}")


def preprocess_spectrum(spectrum: Spectrum) -> Spectrum:
    """Apply standard pipeline: cosmic ray removal -> BubbleFill baseline -> SG smoothing -> normalise.

    Params: width=3, std_factor=5, min_bubble_widths=50, fit_order=1, window=11, polyorder=3.
    Returns preprocessed Spectrum.
    """
    result = (
        PreprocessingPipeline(spectrum)
        .remove_cosmic_rays(width=3, std_factor=5)
        .subtract_baseline(method="bubblefill", min_bubble_widths=50, fit_order=1)
        .smooth(window_length=11, polyorder=3)
        .normalize()
        .result()
    )
    return result


def rank_background_by_similarity(
    background: SpectraCollection,
    fingerprint: Spectrum,
    n_select: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess each background spectrum, compute 3 similarity metrics vs fingerprint.

    Ranks by peak_energy_ratio (primary, lowest = most noise-like).

    Returns:
        selected_indices: 1D array of selected indices (length n_select).
        metrics: 2D array of shape (n_spectra, 3) — [cosine, spectral_angle, peak_energy_ratio].
    """
    fp_intensities = fingerprint.intensities
    n_spectra = len(background)
    metrics = np.zeros((n_spectra, 3), dtype=float)

    print(f"  [rank_background_by_similarity] Processing {n_spectra} spectra ...")
    for i, spectrum in enumerate(background.spectra):
        if i % 200 == 0:
            print(f"    {i}/{n_spectra}", end="\r", flush=True)
        try:
            preprocessed = preprocess_spectrum(spectrum)
            p_int = preprocessed.intensities
        except Exception:
            # If preprocessing fails, keep zeros (will rank low)
            p_int = np.zeros_like(fp_intensities)

        metrics[i, 0] = cosine_similarity(p_int, fp_intensities)
        metrics[i, 1] = spectral_angle(p_int, fp_intensities)
        metrics[i, 2] = peak_energy_ratio(p_int, fp_intensities)

    print(f"    {n_spectra}/{n_spectra} done")

    # Sort ascending by peak_energy_ratio (lowest = most noise-like)
    sorted_idx = np.argsort(metrics[:, 2])
    selected_indices = sorted_idx[:n_select]

    print(f"  [rank_background_by_similarity] Selected {n_select} spectra")
    print(
        f"    peak_energy_ratio: min={metrics[selected_indices, 2].min():.4f}, "
        f"max={metrics[selected_indices, 2].max():.4f}"
    )
    return selected_indices, metrics


def plot_selection_review(
    background: SpectraCollection,
    fingerprint: Spectrum,
    selected_indices: np.ndarray,
    metrics: np.ndarray,
    save_dir: Path,
) -> None:
    """Plot top-10 worst and top-10 borderline candidates as 2x5 subplot grids.

    Saves review_worst_10.png and review_borderline_10.png.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    wl = background.wavelengths
    fp_int = fingerprint.intensities

    # Worst 10: lowest peak_energy_ratio (most noise-like)
    worst_10 = selected_indices[:10]
    # Borderline 10: indices 90-100 of selected (highest per among selected)
    borderline_10 = selected_indices[max(0, len(selected_indices) - 10) :]

    for label, indices, fname in [
        ("Top-10 Most Noise-like", worst_10, "review_worst_10.png"),
        ("Top-10 Borderline", borderline_10, "review_borderline_10.png"),
    ]:
        fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=False)
        axes = axes.ravel()
        for ax_i, idx in enumerate(indices[:10]):
            spec = background.spectra[idx]
            try:
                preprocessed = preprocess_spectrum(spec)
                p_int = preprocessed.intensities
            except Exception:
                p_int = np.zeros_like(fp_int)

            axes[ax_i].plot(wl, p_int, "b-", linewidth=0.8, alpha=0.8, label="Background")
            axes[ax_i].plot(wl, fp_int, "r-", linewidth=0.8, alpha=0.6, label="Fingerprint")
            per = metrics[idx, 2]
            cos = metrics[idx, 0]
            axes[ax_i].set_title(f"idx={idx}\nPER={per:.3f} cos={cos:.3f}", fontsize=8)
            axes[ax_i].set_xlabel("Wavenumber (cm⁻¹)", fontsize=7)
            axes[ax_i].tick_params(labelsize=7)

        axes[0].legend(fontsize=7)
        fig.suptitle(label, fontsize=12)
        plt.tight_layout()
        out_path = save_dir / fname
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  [plot_selection_review] Saved {out_path}")


# ===========================================================================
# SECTION 2 – Baseline extraction & pooling
# ===========================================================================


def extract_baseline(intensities: np.ndarray, window_length: int = 201, polyorder: int = 3) -> np.ndarray:
    """Extract low-freq baseline using large-window Savitzky-Golay filter.

    Returns 1D baseline array.
    """
    return savgol_filter(intensities, window_length=window_length, polyorder=polyorder)


def extract_baselines_and_residuals(
    background: SpectraCollection,
    selected_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """For each selected spectrum: baseline = SG(201,3), residual = spectrum - baseline.

    Returns:
        baselines: shape (n_selected, n_points)
        residuals: shape (n_selected, n_points)
    """
    n = len(selected_indices)
    n_points = background.wavelengths.shape[0]
    baselines = np.zeros((n, n_points), dtype=float)
    residuals = np.zeros((n, n_points), dtype=float)

    for i, idx in enumerate(selected_indices):
        raw_int = background.spectra[idx].intensities.astype(float)
        bl = extract_baseline(raw_int)
        baselines[i] = bl
        residuals[i] = raw_int - bl

    print(f"  [extract_baselines_and_residuals] Extracted {n} baselines/residuals")
    return baselines, residuals


def build_baseline_pool(
    baselines: np.ndarray,
    n_baselines: int = 50,
    n_clusters: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """PCA (5 components) -> KMeans (10 clusters) -> 5 per cluster (closest to centre + 4 random).

    Returns:
        pool_indices: 1D array of indices into baselines array, length n_baselines.
        cluster_labels: 1D array of cluster assignments, length len(baselines).
    """
    rng = np.random.default_rng(42)

    # PCA
    n_components = min(5, baselines.shape[0], baselines.shape[1])
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(baselines)

    # KMeans
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(coords)

    per_cluster = max(1, n_baselines // n_clusters)
    pool_idx: list[int] = []

    for c in range(n_clusters):
        cluster_mask = np.where(labels == c)[0]
        if len(cluster_mask) == 0:
            continue

        # Closest to centroid
        centre = km.cluster_centers_[c]
        dists = np.linalg.norm(coords[cluster_mask] - centre, axis=1)
        closest = cluster_mask[np.argmin(dists)]
        pool_idx.append(int(closest))

        # Random from remaining (up to per_cluster-1)
        remaining = cluster_mask[cluster_mask != closest]
        n_rand = min(per_cluster - 1, len(remaining))
        if n_rand > 0:
            chosen = rng.choice(remaining, size=n_rand, replace=False)
            pool_idx.extend(int(x) for x in chosen)

    # Trim or pad to exactly n_baselines
    pool_idx = list(dict.fromkeys(pool_idx))  # deduplicate, preserve order
    if len(pool_idx) > n_baselines:
        pool_idx = pool_idx[:n_baselines]
    elif len(pool_idx) < n_baselines:
        # Fill from remaining indices
        all_idx = set(range(len(baselines)))
        extras = sorted(all_idx - set(pool_idx))
        pool_idx.extend(extras[: n_baselines - len(pool_idx)])

    print(f"  [build_baseline_pool] Pool size: {len(pool_idx)} baselines across {n_clusters} clusters")
    return np.array(pool_idx, dtype=int), labels


def plot_baseline_pool(
    baselines: np.ndarray,
    pool_indices: np.ndarray,
    labels: np.ndarray,
    wavelengths: np.ndarray,
    save_dir: Path,
) -> None:
    """Plot selected baselines grouped by cluster as 2x5 subplot grid.

    Saves baseline_pool_clusters.png.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    n_clusters = len(np.unique(labels))
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=False)
    axes = axes.ravel()

    for c in range(min(10, n_clusters)):
        ax = axes[c]
        # Find pool indices belonging to this cluster
        for pi in pool_indices:
            if labels[pi] == c:
                ax.plot(wavelengths, baselines[pi], linewidth=0.7, alpha=0.7)
        ax.set_title(f"Cluster {c}", fontsize=9)
        ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=7)
        ax.tick_params(labelsize=7)

    # Hide unused axes
    for c in range(n_clusters, 10):
        axes[c].set_visible(False)

    fig.suptitle("Baseline Pool by Cluster", fontsize=12)
    plt.tight_layout()
    out_path = save_dir / "baseline_pool_clusters.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot_baseline_pool] Saved {out_path}")


# ===========================================================================
# SECTION 3 – Noise validation
# ===========================================================================


def extract_signal_residuals(
    background_dir: Path,
    n_samples: int = 50,
) -> np.ndarray | None:
    """Approach A: load raw_data.npz (signal spectra), apply SG(201,3), compute residuals.

    Returns array (n_samples, n_points) or None if no raw_data.npz found.
    """
    background_dir = Path(background_dir)
    rep_dirs = sorted(d for d in background_dir.iterdir() if d.is_dir() and d.name.startswith("rep"))

    all_residuals: list[np.ndarray] = []

    for rep in rep_dirs:
        npz_path = rep / "raw_data.npz"
        if not npz_path.exists():
            continue
        col = load_from_npz(npz_path)
        intensities = col.to_intensity_matrix()
        for raw in intensities:
            bl = extract_baseline(raw.astype(float))
            all_residuals.append(raw - bl)
        print(f"  [extract_signal_residuals] {rep.name}: {len(col)} signal spectra")

    if not all_residuals:
        print("  [extract_signal_residuals] No raw_data.npz files found")
        return None

    residuals = np.array(all_residuals)
    if len(residuals) > n_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(residuals), size=n_samples, replace=False)
        residuals = residuals[idx]

    print(f"  [extract_signal_residuals] Total: {len(residuals)} residuals")
    return residuals


def validate_noise(
    residuals_c: np.ndarray,
    residuals_a: np.ndarray | None,
    wavelengths: np.ndarray,
    save_dir: Path,
) -> None:
    """Compare noise from approach C (background) vs A (signal residuals).

    Plots: overlay noise spectra, per-wavenumber std, PSD, autocorrelation.
    Saves: noise_validation.png and noise_validation.json.
    Prints correlation coefficients and pass/fail assessment (>0.8 = pass).
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    std_c = residuals_c.std(axis=0)
    has_a = residuals_a is not None

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Overlay noise spectra
    ax = axes[0, 0]
    for r in residuals_c[:20]:
        ax.plot(wavelengths, r, "b-", linewidth=0.4, alpha=0.4)
    if has_a:
        for r in residuals_a[:20]:
            ax.plot(wavelengths, r, "r-", linewidth=0.4, alpha=0.4)
    ax.set_title("Noise Spectra Overlay (blue=C, red=A)")
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Residual Intensity")

    # 2. Per-wavenumber std
    ax = axes[0, 1]
    ax.plot(wavelengths, std_c, "b-", linewidth=1.0, label="Approach C (background)")
    if has_a:
        std_a = residuals_a.std(axis=0)
        ax.plot(wavelengths, std_a, "r-", linewidth=1.0, label="Approach A (signal)")
    ax.set_title("Per-wavenumber Std Dev")
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Std Dev")
    ax.legend()

    # 3. Power spectral density
    ax = axes[1, 0]
    mean_psd_c = np.mean(np.abs(np.fft.rfft(residuals_c, axis=1)) ** 2, axis=0)
    freqs = np.fft.rfftfreq(residuals_c.shape[1])
    ax.semilogy(freqs, mean_psd_c, "b-", linewidth=1.0, label="Approach C")
    if has_a:
        mean_psd_a = np.mean(np.abs(np.fft.rfft(residuals_a, axis=1)) ** 2, axis=0)
        ax.semilogy(freqs, mean_psd_a, "r-", linewidth=1.0, label="Approach A")
    ax.set_title("Power Spectral Density")
    ax.set_xlabel("Frequency (normalised)")
    ax.set_ylabel("Power")
    ax.legend()

    # 4. Autocorrelation of mean residual
    ax = axes[1, 1]
    mean_r_c = residuals_c.mean(axis=0)
    ac_c = np.correlate(mean_r_c - mean_r_c.mean(), mean_r_c - mean_r_c.mean(), mode="full")
    ac_c = ac_c / ac_c[len(ac_c) // 2]
    lags = np.arange(-(len(ac_c) // 2), len(ac_c) // 2 + 1)
    ax.plot(lags, ac_c, "b-", linewidth=0.8, label="Approach C")
    if has_a:
        mean_r_a = residuals_a.mean(axis=0)
        ac_a = np.correlate(mean_r_a - mean_r_a.mean(), mean_r_a - mean_r_a.mean(), mode="full")
        ac_a = ac_a / ac_a[len(ac_a) // 2]
        ax.plot(lags, ac_a, "r-", linewidth=0.8, label="Approach A")
    ax.set_xlim(-100, 100)
    ax.set_title("Autocorrelation (mean residual)")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Normalised AC")
    ax.legend()

    plt.tight_layout()
    out_path = save_dir / "noise_validation.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [validate_noise] Saved {out_path}")

    # --- Statistics ---
    results: dict = {
        "approach_c_n_spectra": int(residuals_c.shape[0]),
        "approach_c_mean_std": float(std_c.mean()),
        "approach_c_max_std": float(std_c.max()),
    }

    if has_a:
        std_a = residuals_a.std(axis=0)
        corr = float(np.corrcoef(std_c, std_a)[0, 1])
        results["approach_a_n_spectra"] = int(residuals_a.shape[0])
        results["approach_a_mean_std"] = float(std_a.mean())
        results["std_correlation"] = corr
        psd_c = mean_psd_c
        psd_a = mean_psd_a  # noqa
        psd_corr = float(np.corrcoef(np.log(psd_c + 1e-30), np.log(mean_psd_a + 1e-30))[0, 1])
        results["psd_log_correlation"] = psd_corr
        results["pass"] = corr > 0.8
        print(f"  [validate_noise] Std correlation (C vs A): {corr:.4f}  -> {'PASS' if corr > 0.8 else 'FAIL'}")
        print(f"  [validate_noise] PSD log-correlation:       {psd_corr:.4f}")
    else:
        print("  [validate_noise] No approach-A residuals; skipping correlation analysis")
        results["pass"] = None

    json_path = save_dir / "noise_validation.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [validate_noise] Saved {json_path}")


# ===========================================================================
# SECTION 4 – Synthetic data generation
# ===========================================================================


def read_simulation_spectrum(sim_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read raman.txt (2-column: Wave, Intensity). Ensure ascending order.

    Returns (wavelengths, intensities) tuple.
    """
    sim_path = Path(sim_path)
    data = np.loadtxt(sim_path, skiprows=1)  # skip header row (#Wave\t#Intensity)
    # Sort ascending by wavenumber
    order = np.argsort(data[:, 0])
    wavelengths = data[order, 0]
    intensities = data[order, 1]
    return wavelengths, intensities


def interpolate_simulation(
    sim_wavelengths: np.ndarray,
    sim_intensities: np.ndarray,
    target_wavelengths: np.ndarray,
) -> np.ndarray:
    """Interpolate onto target grid. Pad with 0 outside sim range.

    Returns interpolated intensities array.
    """
    interp = np.interp(
        target_wavelengths,
        sim_wavelengths,
        sim_intensities,
        left=0.0,
        right=0.0,
    )
    return interp


def compute_scale_factor(fingerprint: Spectrum, sim_intensities: np.ndarray) -> float:
    """alpha = max(fingerprint.intensities) / max(sim_intensities).

    Returns float.
    """
    max_fp = float(np.max(fingerprint.intensities))
    max_sim = float(np.max(sim_intensities))
    if max_sim == 0.0:
        raise ValueError("Simulation spectrum has all-zero intensities")
    return max_fp / max_sim


def generate_synthetic_spectra(
    sim_intensities: np.ndarray,
    alpha: float,
    baselines: np.ndarray,
    pool_indices: np.ndarray,
    residuals: np.ndarray,
    wavelengths: np.ndarray,
    snr_levels: list[int],
    samples_per_level: int,
    seed: int = 42,
) -> dict:
    """For each sample: noisy = alpha*sim + baseline + beta*noise_residual.

    beta = (alpha * max(sim)) / (target_SNR * std(noise_residual))

    Returns dict mapping SNR -> {noisy, ground_truth, metadata}.
    """
    rng = np.random.default_rng(seed)
    pool_baselines = baselines[pool_indices]
    pool_residuals = residuals[pool_indices]
    ground_truth = alpha * sim_intensities  # shape (n_points,)

    results: dict = {}
    for snr in snr_levels:
        noisy_list: list[np.ndarray] = []
        metadata_list: list[dict] = []

        for s in range(samples_per_level):
            # Pick a random baseline and residual from pool
            bl_idx = int(rng.integers(0, len(pool_baselines)))
            baseline = pool_baselines[bl_idx]
            noise = pool_residuals[bl_idx]

            noise_std = float(np.std(noise))
            if noise_std < 1e-12:
                noise_std = 1e-12

            beta = (alpha * float(np.max(sim_intensities))) / (snr * noise_std)
            noisy = ground_truth + baseline + beta * noise
            noisy_list.append(noisy)
            metadata_list.append(
                {
                    "sample_index": s,
                    "snr_target": snr,
                    "baseline_pool_index": bl_idx,
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "noise_std": float(noise_std),
                }
            )

        results[snr] = {
            "noisy": np.array(noisy_list),          # (samples_per_level, n_points)
            "ground_truth": ground_truth.copy(),     # (n_points,)
            "metadata": metadata_list,
        }
        print(f"  [generate_synthetic_spectra] SNR={snr}: {samples_per_level} samples generated")

    return results


def save_synthetic_data(
    results: dict,
    wavelengths: np.ndarray,
    alpha: float,
    output_dir: Path,
) -> None:
    """Save per SNR level: synthetic_noisy.npz, synthetic_ground_truth.npz, metadata.json.

    Plus synthetic_overview.png (one sample per SNR overlaid with ground truth).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    snr_levels = sorted(results.keys())

    for snr in snr_levels:
        level_dir = output_dir / f"snr_{snr}"
        level_dir.mkdir(exist_ok=True)

        data = results[snr]
        noisy = data["noisy"]        # (n_samples, n_points)
        gt = data["ground_truth"]    # (n_points,)

        # Save noisy spectra as SpectraCollection
        noisy_spectra = [
            Spectrum(data=np.column_stack([wavelengths, noisy[i]]))
            for i in range(len(noisy))
        ]
        noisy_col = SpectraCollection(noisy_spectra, source_file=f"synthetic_SNR{snr}", wavelengths=wavelengths)
        export_to_npz(noisy_col, level_dir / "synthetic_noisy.npz")

        # Save ground truth
        gt_spectrum = Spectrum(data=np.column_stack([wavelengths, gt]))
        gt_col = SpectraCollection([gt_spectrum], source_file=f"synthetic_GT_SNR{snr}", wavelengths=wavelengths)
        export_to_npz(gt_col, level_dir / "synthetic_ground_truth.npz")

        # Save metadata
        meta = {"snr": snr, "alpha": float(alpha), "n_samples": len(noisy), "samples": data["metadata"]}
        with open(level_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  [save_synthetic_data] SNR={snr} saved to {level_dir}")

    # Overview plot: one sample per SNR overlaid with ground truth
    fig, axes = plt.subplots(1, len(snr_levels), figsize=(5 * len(snr_levels), 4), sharey=False)
    if len(snr_levels) == 1:
        axes = [axes]

    for ax, snr in zip(axes, snr_levels):
        noisy = results[snr]["noisy"][0]
        gt = results[snr]["ground_truth"]
        ax.plot(wavelengths, noisy, "b-", linewidth=0.8, alpha=0.8, label="Noisy")
        ax.plot(wavelengths, gt, "r-", linewidth=1.0, alpha=0.9, label="Ground Truth")
        ax.set_title(f"SNR = {snr}", fontsize=10)
        ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=8)
        ax.legend(fontsize=7)

    plt.suptitle("Synthetic Spectra Overview", fontsize=12)
    plt.tight_layout()
    out_path = output_dir / "synthetic_overview.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [save_synthetic_data] Overview plot saved to {out_path}")


# ===========================================================================
# MAIN / CLI
# ===========================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic noisy Raman spectra from DFT simulation + real noise."
    )
    parser.add_argument(
        "--sim-path",
        type=Path,
        default=Path("data/simulation/2-letter/GA/rep3/raman.txt"),
        help="Path to simulation raman.txt file",
    )
    parser.add_argument(
        "--background-dir",
        type=Path,
        default=Path("data/custom/processed/primary_magic/2-letter/GA"),
        help="Directory containing rep*/raw_background.npz files",
    )
    parser.add_argument(
        "--ref-file",
        type=Path,
        default=Path("data/custom/raw/2-letter/GA/rep1/GA_crystal_532_.25s_100power_1900spectra.txt"),
        help="Reference raw data file (unused in generation, kept for metadata)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/synthetic/GA"),
        help="Output directory for synthetic data",
    )
    parser.add_argument(
        "--snr-levels",
        type=int,
        nargs="+",
        default=[10, 25, 50, 100],
        help="SNR levels to generate (default: 10 25 50 100)",
    )
    parser.add_argument(
        "--samples-per-level",
        type=int,
        default=50,
        help="Number of synthetic spectra per SNR level (default: 50)",
    )
    parser.add_argument(
        "--n-select",
        type=int,
        default=100,
        help="Number of background spectra to select as noise (default: 100)",
    )
    parser.add_argument(
        "--n-baselines",
        type=int,
        default=50,
        help="Number of baselines in pool (default: 50)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run noise validation (approach C vs A comparison)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    # Resolve paths relative to working directory
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print("Generate Synthetic Noisy Raman Spectra")
    print("=" * 60)
    print(f"  background_dir    : {args.background_dir}")
    print(f"  sim_path          : {args.sim_path}")
    print(f"  output_dir        : {output_dir}")
    print(f"  snr_levels        : {args.snr_levels}")
    print(f"  samples_per_level : {args.samples_per_level}")
    print(f"  n_select          : {args.n_select}")
    print(f"  n_baselines       : {args.n_baselines}")
    print(f"  validate          : {args.validate}")
    print(f"  seed              : {args.seed}")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # [1/6] Load background spectra
    # -----------------------------------------------------------------------
    print("\n[1/6] Loading background spectra ...")
    background = load_background_spectra(args.background_dir)
    wavelengths = background.wavelengths

    # -----------------------------------------------------------------------
    # [2/6] Select purest noise spectra
    # -----------------------------------------------------------------------
    print("\n[2/6] Selecting purest noise spectra ...")
    fingerprint = load_fingerprint(args.background_dir)
    selected_indices, metrics = rank_background_by_similarity(
        background, fingerprint, n_select=args.n_select
    )
    plot_selection_review(background, fingerprint, selected_indices, metrics, save_dir=plots_dir)

    # -----------------------------------------------------------------------
    # [3/6] Extract baselines, build pool
    # -----------------------------------------------------------------------
    print("\n[3/6] Extracting baselines and building baseline pool ...")
    baselines, residuals = extract_baselines_and_residuals(background, selected_indices)
    pool_indices, cluster_labels = build_baseline_pool(baselines, n_baselines=args.n_baselines)
    plot_baseline_pool(baselines, pool_indices, cluster_labels, wavelengths, save_dir=plots_dir)

    # -----------------------------------------------------------------------
    # [4/6] Validate noise
    # -----------------------------------------------------------------------
    if args.validate:
        print("\n[4/6] Validating noise (approach C vs A) ...")
        residuals_a = extract_signal_residuals(args.background_dir, n_samples=50)
        validate_noise(residuals, residuals_a, wavelengths, save_dir=plots_dir)
    else:
        print("\n[4/6] Noise validation skipped (use --validate to enable)")

    # -----------------------------------------------------------------------
    # [5/6] Generate synthetic spectra
    # -----------------------------------------------------------------------
    print("\n[5/6] Generating synthetic spectra ...")
    sim_wl, sim_int = read_simulation_spectrum(args.sim_path)
    sim_interp = interpolate_simulation(sim_wl, sim_int, wavelengths)
    alpha = compute_scale_factor(fingerprint, sim_interp)
    print(f"  alpha (scale factor) = {alpha:.6f}")

    results = generate_synthetic_spectra(
        sim_intensities=sim_interp,
        alpha=alpha,
        baselines=baselines,
        pool_indices=pool_indices,
        residuals=residuals,
        wavelengths=wavelengths,
        snr_levels=args.snr_levels,
        samples_per_level=args.samples_per_level,
        seed=args.seed,
    )

    # -----------------------------------------------------------------------
    # [6/6] Save output
    # -----------------------------------------------------------------------
    print("\n[6/6] Saving output ...")
    save_synthetic_data(results, wavelengths, alpha, output_dir)

    print("\n" + "=" * 60)
    print(f"Done. Output saved to: {output_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
