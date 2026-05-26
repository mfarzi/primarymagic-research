"""
SGAF (4-letter) — old vs new pipeline pass/fail confusion analysis.

For SGAF across all reps:
  1) Combined pass/fail confusion matrix (magic_old × magic_bayes).
  2) One example raw spectrum per confusion-matrix cell (with both pipelines'
     processed output overlaid so the divergence in the pass decision is
     visible).
  3) Mean ± std shaded fingerprint band for clean spectra in each pipeline.

Outputs PNGs into analysis_outputs/sgaf/.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED = PROJECT_ROOT / "data" / "custom" / "processed"
MAGIC_OLD = PROCESSED / "magic_old" / "4-letter" / "SGAF"
MAGIC_NEW = PROCESSED / "magic_bayes" / "4-letter" / "SGAF"
OUT_DIR = PROJECT_ROOT / "analysis_outputs" / "sgaf"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _rep_dirs(base):
    return sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("rep")])


# ---------------------------------------------------------------------------
# Load mask + spectra across reps
# ---------------------------------------------------------------------------
def load_combined():
    """Return per-rep dict with raw spectra, wavelengths, mask info for both
    pipelines, plus globally concatenated old/new pass arrays."""
    reps_new = {r.name: r for r in _rep_dirs(MAGIC_NEW)}
    reps_old = {r.name: r for r in _rep_dirs(MAGIC_OLD)}
    common = sorted(set(reps_new) & set(reps_old))

    per_rep = {}
    for rep in common:
        m_new = np.load(reps_new[rep] / "mask.npz")
        m_old = np.load(reps_old[rep] / "mask.npz")
        assert np.array_equal(m_new["raw_index"], m_old["raw_index"]), \
            f"raw_index mismatch in {rep}"

        raw = np.load(reps_new[rep] / "raw_all.npz")
        wavelengths = raw["wavelengths"]
        raw_int = raw["intensities"]

        # Build per-raw-index processed spectra on a 0..1 scale.
        # magic_old: data.npz holds the old pipeline's final normalised output
        # for every raw spectrum (already in [0, 1]).
        old_proc = np.load(reps_old[rep] / "data.npz")["intensities"]

        # magic_bayes: passed spectra are read directly from clean_data.npz
        # (already in [0, 1]); failed spectra are normalised on the fly from
        # the post-stage-3 (clip-to-zero) intensities the same way clean_data
        # was produced — i.e. divide by the per-sample max.
        clean_new = np.load(reps_new[rep] / "clean_data.npz")["intensities"]
        stage3 = np.load(reps_new[rep] / "stage3_baseline_removed.npz")["intensities"]
        stage3 = np.where(stage3 >= 0, stage3, 0.0)
        clean_idx_new = m_new["clean_index"]
        new_proc = np.empty_like(stage3)
        for i in range(stage3.shape[0]):
            ci = int(clean_idx_new[i])
            if ci >= 0:
                new_proc[i] = clean_new[ci]
            else:
                s = stage3[i]
                mx = s.max()
                new_proc[i] = s / mx if mx > 0 else s

        per_rep[rep] = {
            "wavelengths": wavelengths,
            "raw": raw_int,
            "old_proc": old_proc,
            "new_proc": new_proc,
            "old_pass": m_old["passed"].astype(bool),
            "new_pass": m_new["passed"].astype(bool),
            "old_snr": m_old["snr"],
            "new_snr": m_new["snr"],
        }

    return per_rep


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
def confusion_matrix(per_rep):
    """Return (cm, cell_indices) with cm[i,j] giving the count for
    cm[old_pass_bool, new_pass_bool].

    Layout:
        cm[1,1] = old PASS & new PASS
        cm[1,0] = old PASS & new FAIL
        cm[0,1] = old FAIL & new PASS
        cm[0,0] = old FAIL & new FAIL

    cell_indices maps (old_bool, new_bool) -> list of (rep, idx_in_rep).
    """
    cm = np.zeros((2, 2), dtype=int)
    cell_indices = {(0, 0): [], (0, 1): [], (1, 0): [], (1, 1): []}
    for rep, d in per_rep.items():
        for i, (op, np_) in enumerate(zip(d["old_pass"], d["new_pass"])):
            cm[int(op), int(np_)] += 1
            cell_indices[(int(op), int(np_))].append((rep, i))
    return cm, cell_indices


# ---------------------------------------------------------------------------
# Plot 1: confusion matrix heatmap
# ---------------------------------------------------------------------------
def plot_confusion_matrix(cm, output_path):
    total = cm.sum()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    # display order: rows old fail, old pass; cols new fail, new pass
    display = np.array([[cm[0, 0], cm[0, 1]],
                        [cm[1, 0], cm[1, 1]]])
    im = ax.imshow(display, cmap="Blues", vmin=0)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["new FAIL", "new PASS"], fontsize=12)
    ax.set_yticklabels(["old FAIL", "old PASS"], fontsize=12)
    ax.set_xlabel("magic_bayes  (SNR ≥ 30)", fontsize=12)
    ax.set_ylabel("magic_old  (SNR ≥ 50  &  ASSI ≥ 0.65)", fontsize=12)
    ax.set_title(f"SGAF — pass/fail confusion (combined over all reps, n={total})",
                 fontsize=12)
    for i in range(2):
        for j in range(2):
            val = display[i, j]
            pct = 100.0 * val / total if total else 0.0
            ax.text(j, i, f"{val}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=14,
                    color="white" if val > display.max() * 0.5 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    return output_path


# ---------------------------------------------------------------------------
# Plot 2: example spectra per outcome (3 rows × 2 cols)
#   rows: both-accept (top), one-accept-one-fail (middle), both-fail (bottom)
#   cols: raw signal (left), processed old vs new (right)
# ---------------------------------------------------------------------------
def plot_examples_per_cell(per_rep, cell_indices, output_path, rng_seed=0):
    rng = np.random.default_rng(rng_seed)

    # outcome rows: (label, (old_pass_bool, new_pass_bool), optional filter)
    # For the "both reject" row, require a genuinely low-SNR sample
    # (magic_bayes SNR < 10) so the rejection is unambiguous.
    rows = [
        ("Both accept  —  old PASS  &  new PASS",  (1, 1), None),
        ("Disagree     —  old FAIL  &  new PASS",  (0, 1), None),
        ("Both reject  —  old FAIL  &  new FAIL",  (0, 0),
            lambda rep, i, d: d["new_snr"][i] < 10),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
    for r, (label, key, filt) in enumerate(rows):
        members = cell_indices[key]
        if filt is not None:
            members = [(rep, i) for rep, i in members if filt(rep, i, per_rep[rep])]
        ax_raw = axes[r, 0]
        ax_proc = axes[r, 1]

        if not members:
            for ax in (ax_raw, ax_proc):
                ax.text(0.5, 0.5, "(empty)", ha="center", va="center",
                        fontsize=14, transform=ax.transAxes)
                ax.set_title(f"{label}  —  n=0", fontsize=11)
            continue

        rep, idx = members[rng.integers(0, len(members))]
        d = per_rep[rep]
        wl = d["wavelengths"]

        # Left column: raw spectrum
        ax_raw.plot(wl, d["raw"][idx], color="#9CA3AF", linewidth=0.9)
        ax_raw.set_title(f"{label}  —  n={len(members)}  ({rep} idx {idx})   |   raw",
                         fontsize=11)
        ax_raw.grid(alpha=0.3)
        ax_raw.set_ylabel("Intensity (a.u.)")

        # Right column: processed old vs new — both on 0..1 scale
        ax_proc.plot(wl, d["old_proc"][idx], color="#1E40AF", linewidth=1.1,
                     label=f"magic_old (SNR={d['old_snr'][idx]:.1f})")
        ax_proc.plot(wl, d["new_proc"][idx], color="#15803D", linewidth=1.1,
                     label=f"magic_bayes (SNR={d['new_snr'][idx]:.1f})")
        ax_proc.set_title("processed (old vs new, normalised)", fontsize=11)
        ax_proc.set_ylim(-0.05, 1.05)
        ax_proc.grid(alpha=0.3)
        ax_proc.legend(loc="upper right", fontsize=9)

        if r == 2:
            ax_raw.set_xlabel("Wavenumber (cm⁻¹)")
            ax_proc.set_xlabel("Wavenumber (cm⁻¹)")

    fig.suptitle("SGAF — example spectra per pass/fail outcome",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    return output_path


# ---------------------------------------------------------------------------
# Plot 3: mean ± std shaded fingerprint band
# ---------------------------------------------------------------------------
def _collect_clean(base):
    """Concatenate clean_data.npz over reps. Each clean_data is max-normalised
    inside that pipeline already; we re-normalise sample-by-sample to be safe."""
    wl = None
    rows = []
    for rep in _rep_dirs(base):
        f = rep / "clean_data.npz"
        if not f.exists():
            continue
        d = np.load(f)
        x = d["intensities"]
        if x.shape[0] == 0:
            continue
        if wl is None:
            wl = d["wavelengths"]
        # normalise sample-by-sample so band is on the same scale
        mx = x.max(axis=1, keepdims=True)
        mx[mx == 0] = 1.0
        rows.append(x / mx)
    if not rows:
        return wl, np.empty((0, 0))
    return wl, np.concatenate(rows, axis=0)


def plot_mean_std_band(output_path):
    wl_old, X_old = _collect_clean(MAGIC_OLD)
    wl_new, X_new = _collect_clean(MAGIC_NEW)

    # Shared y-range so the two panels are directly comparable.
    y_max = 0.0
    for X in (X_old, X_new):
        if X.size:
            mu, sd = X.mean(axis=0), X.std(axis=0)
            y_max = max(y_max, float((mu + sd).max()))
    y_lim = (-0.05, y_max * 1.05 if y_max > 0 else 1.05)

    fig, (ax_new, ax_old) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    def _panel(ax, wl, X, color, name):
        if not X.size:
            ax.text(0.5, 0.5, "(no clean spectra)", ha="center", va="center",
                    fontsize=12, transform=ax.transAxes)
            ax.set_title(f"{name} — n = 0", fontsize=12)
            return
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        ax.fill_between(wl, mu - sd, mu + sd, color=color, alpha=0.25,
                        label="±1σ")
        ax.plot(wl, mu, color=color, linewidth=1.5, label="mean")
        ax.set_title(f"{name}  (n = {X.shape[0]})", fontsize=12)
        ax.set_ylabel("Normalised intensity")
        ax.set_ylim(y_lim)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=10)

    _panel(ax_new, wl_new, X_new, "#15803D", "magic_bayes (new)")
    _panel(ax_old, wl_old, X_old, "#1E40AF", "magic_old (old)")
    ax_old.set_xlabel("Wavenumber (cm⁻¹)")

    fig.suptitle("SGAF — clean-spectrum fingerprint  (mean ± 1σ, combined reps)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    per_rep = load_combined()
    print("Reps loaded:", list(per_rep.keys()))
    cm, cell_idx = confusion_matrix(per_rep)
    print(f"Confusion matrix (rows=old, cols=new; [fail,pass] x [fail,pass]):")
    print(cm)
    print(f"  old PASS & new PASS: {cm[1,1]}")
    print(f"  old PASS & new FAIL: {cm[1,0]}")
    print(f"  old FAIL & new PASS: {cm[0,1]}")
    print(f"  old FAIL & new FAIL: {cm[0,0]}")

    p1 = plot_confusion_matrix(cm, OUT_DIR / "confusion_matrix.png")
    p2 = plot_examples_per_cell(per_rep, cell_idx, OUT_DIR / "cell_examples.png")
    p3 = plot_mean_std_band(OUT_DIR / "mean_std_band.png")
    print(f"\nWrote:\n  {p1}\n  {p2}\n  {p3}")


if __name__ == "__main__":
    main()
