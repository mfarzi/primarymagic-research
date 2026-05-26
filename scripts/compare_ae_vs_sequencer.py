#!/usr/bin/env python
"""Cross-component comparison: align autoencoder fingerprints with sequencer
fingerprints from cached attributions.npz / activations.npz files.

Produces:
  1. ae_vs_seq_overlay_grid.png       — per (class, baseline): AE attr vs sequencer attr_n overlaid
  2. symmetry_grid.png                — per class: sequencer attr_n vs −attr_{n-1} overlaid
  3. prototype_alignment_grid.png     — per class: 4-curve alignment of AE prototype, AE mean,
                                         sequencer synthetic decoded, sequencer raw empirical
  4. similarity.md                    — cosine similarities across all comparisons
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_ae_attributions(npz_path: Path) -> Tuple[np.ndarray, Dict[Tuple[str, str], np.ndarray]]:
    """Load autoencoder attributions.npz: wavenumbers + dict (cls, baseline) -> (N, seq) attribution."""
    data = np.load(npz_path)
    wavenumbers = data["wavenumbers"]
    means: Dict[Tuple[str, str], np.ndarray] = {}
    for key in data.files:
        if key == "wavenumbers":
            continue
        cls, baseline = key.split("__")
        means[(cls, baseline)] = data[key].mean(axis=0)
    return wavenumbers, means


_SEQ_SIDE_MAP = {"attr_n": "n", "attr_n_minus_1": "n_minus_1"}


def load_seq_attributions(npz_path: Path) -> Tuple[np.ndarray, Dict[Tuple[str, str], Dict[str, np.ndarray]]]:
    """Load sequencer attributions.npz: keys like '<cls>__<baseline>__attr_n' / 'attr_n_minus_1'."""
    data = np.load(npz_path)
    wavenumbers = data["wavenumbers"]
    means: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    for key in data.files:
        if key == "wavenumbers":
            continue
        cls, baseline, side = key.split("__")
        if side not in _SEQ_SIDE_MAP:
            raise ValueError(
                f"Unknown side {side!r} in sequencer attributions.npz key {key!r}; "
                f"expected one of {list(_SEQ_SIDE_MAP)}"
            )
        means.setdefault((cls, baseline), {})[_SEQ_SIDE_MAP[side]] = data[key].mean(axis=0)
    return wavenumbers, means


def load_ae_activations(npz_path: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Load AE activations.npz: keys 'A__noise', 'A__mean_init', etc."""
    data = np.load(npz_path)
    wavenumbers = data["wavenumbers"]
    arrays = {k: data[k] for k in data.files if k != "wavenumbers"}
    return wavenumbers, arrays


def load_seq_activations(npz_path: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Load sequencer activations.npz: keys 'synthetic__A__noise', 'decoded_empirical__A',
    'raw_empirical_mean__A', etc."""
    data = np.load(npz_path)
    wavenumbers = data["wavenumbers"]
    arrays = {k: data[k] for k in data.files if k != "wavenumbers"}
    return wavenumbers, arrays


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def plot_ae_vs_seq_overlay_grid(
    wavenumbers: np.ndarray,
    ae_means: Dict[Tuple[str, str], np.ndarray],
    seq_means: Dict[Tuple[str, str], Dict[str, np.ndarray]],
    classes: List[str],
    baselines: List[str],
) -> plt.Figure:
    n_rows, n_cols = len(classes), len(baselines)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 1.7 * n_rows),
                             sharex=True, squeeze=False)
    for r, cls in enumerate(classes):
        for c, b in enumerate(baselines):
            ax = axes[r, c]
            ae = ae_means.get((cls, b))
            seq_n = seq_means.get((cls, b), {}).get("n")
            if ae is not None:
                ax.plot(wavenumbers, ae, label="autoencoder", color="#1f77b4",
                        linewidth=1.0, alpha=0.85)
            if seq_n is not None:
                ax.plot(wavenumbers, seq_n, label="sequencer attr_n", color="#d62728",
                        linewidth=1.0, alpha=0.85)
            ax.axhline(0, color="black", linewidth=0.3)
            ax.set_title(f"{cls}  ·  baseline={b}", fontsize=9)
            ax.tick_params(labelsize=7)
            if r == 0 and c == 0:
                ax.legend(fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=9)
    fig.tight_layout()
    return fig


def plot_symmetry_grid(
    wavenumbers: np.ndarray,
    seq_means: Dict[Tuple[str, str], Dict[str, np.ndarray]],
    classes: List[str],
    baseline: str = "contrastive",
) -> plt.Figure:
    n = len(classes)
    n_cols = int(np.ceil(np.sqrt(n)))
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, 1.8 * n_rows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, cls in zip(axes, classes):
        sides = seq_means.get((cls, baseline), {})
        attr_n = sides.get("n")
        attr_n1 = sides.get("n_minus_1")
        if attr_n is not None:
            ax.plot(wavenumbers, attr_n, color="#d62728", label="attr_n", linewidth=1.0)
        if attr_n1 is not None:
            ax.plot(wavenumbers, -attr_n1, color="#1f77b4", label="−attr_{n-1}",
                    linewidth=1.0, alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.3)
        ax.set_title(f"{cls} removed", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
    for ax in axes[len(classes):]:
        ax.set_visible(False)
    fig.suptitle(f"Symmetry check: sequencer attr_n vs −attr_{{n-1}} (baseline={baseline})",
                 fontsize=11)
    fig.text(0.5, 0.02, "Wavenumber (cm⁻¹)", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    return fig


def plot_prototype_alignment_grid(
    wavenumbers: np.ndarray,
    ae_acts: Dict[str, np.ndarray],
    ae_means: Dict[Tuple[str, str], np.ndarray],  # used for "AE mean" curve via _zero baseline placeholder
    seq_acts: Dict[str, np.ndarray],
    classes: List[str],
    init_name: str = "mean_init",
) -> plt.Figure:
    n = len(classes)
    n_cols = int(np.ceil(np.sqrt(n)))
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 2.0 * n_rows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, cls in zip(axes, classes):
        ae_proto = ae_acts.get(f"{cls}__{init_name}")
        seq_synth = seq_acts.get(f"synthetic__{cls}__{init_name}")
        seq_decoded = seq_acts.get(f"decoded_empirical__{cls}")
        seq_raw = seq_acts.get(f"raw_empirical_mean__{cls}")
        if ae_proto is not None:
            ax.plot(wavenumbers, ae_proto, color="#9467bd",
                    label="AE prototype", linewidth=1.0, linestyle=":")
        if seq_synth is not None:
            ax.plot(wavenumbers, seq_synth, color="#d62728",
                    label="seq synthetic", linewidth=1.2)
        if seq_decoded is not None:
            ax.plot(wavenumbers, seq_decoded, color="#1f77b4",
                    label="seq decoded empirical", linewidth=1.0, linestyle="--")
        if seq_raw is not None:
            ax.plot(wavenumbers, seq_raw, color="black",
                    label="seq raw empirical mean", linewidth=1.0, alpha=0.7)
        ax.axhline(0, color="black", linewidth=0.3)
        ax.set_title(f"{cls}", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="best")
    for ax in axes[len(classes):]:
        ax.set_visible(False)
    fig.suptitle(f"Prototype alignment: AE vs sequencer (init={init_name})", fontsize=11)
    fig.text(0.5, 0.02, "Wavenumber (cm⁻¹)", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    return fig


def render_similarity_md(
    similarities: Dict[str, Dict[str, float]],
    classes: List[str],
    column_order: List[str],
) -> str:
    lines = [
        "# AE vs Sequencer Attribution Similarity",
        "",
        "Cosine similarity between mean attribution vectors. "
        "1.0 = identical pattern; 0.0 = orthogonal; negative = anti-correlated.",
        "",
    ]
    header = "| Class | " + " | ".join(column_order) + " |"
    sep = "|---:|" + "---:|" * len(column_order)
    lines.extend([header, sep])
    for cls in classes:
        row = [cls]
        for col in column_order:
            sim = similarities.get(col, {}).get(cls)
            row.append("—" if sim is None else f"{sim:.3f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare autoencoder vs sequencer attribution fingerprints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--autoencoder-explanations", type=str, required=True,
                   help="Path to AE explanations/ dir (must contain attributions.npz, activations.npz).")
    p.add_argument("--sequencer-explanations", type=str, required=True,
                   help="Path to sequencer explanations/ dir.")
    p.add_argument("--classes", type=str, nargs="+", required=True)
    p.add_argument("--baselines", type=str, nargs="+",
                   default=["zero", "contrastive"], choices=["zero", "contrastive"])
    p.add_argument("--prototype-init", type=str, default="mean_init",
                   choices=["noise", "mean_init"],
                   help="Which init's prototype to use in the alignment plot.")
    p.add_argument("--output-dir", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ae_dir = Path(args.autoencoder_explanations)
    seq_dir = Path(args.sequencer_explanations)

    print(f"Loading AE attributions: {ae_dir / 'attributions.npz'}")
    wavenumbers_a, ae_means = load_ae_attributions(ae_dir / "attributions.npz")
    print(f"Loading sequencer attributions: {seq_dir / 'attributions.npz'}")
    wavenumbers_s, seq_means = load_seq_attributions(seq_dir / "attributions.npz")

    if not np.allclose(wavenumbers_a, wavenumbers_s):
        raise ValueError("Wavenumber axes differ between AE and sequencer attribution files.")
    wavenumbers = wavenumbers_a

    # --- Comparison 1: AE vs sequencer-n-side -----------------------------------
    sims_ae_vs_seq_zero: Dict[str, float] = {}
    sims_ae_vs_seq_contrastive: Dict[str, float] = {}
    for cls in args.classes:
        for b, bucket in [("zero", sims_ae_vs_seq_zero),
                          ("contrastive", sims_ae_vs_seq_contrastive)]:
            ae = ae_means.get((cls, b))
            sq = seq_means.get((cls, b), {}).get("n")
            if ae is None or sq is None:
                continue
            bucket[cls] = cosine_similarity(ae, sq)

    fig = plot_ae_vs_seq_overlay_grid(wavenumbers, ae_means, seq_means,
                                      args.classes, args.baselines)
    fig.savefig(out_dir / "ae_vs_seq_overlay_grid.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_dir / 'ae_vs_seq_overlay_grid.png'}")

    # --- Comparison 2: symmetry attr_n vs -attr_{n-1} ---------------------------
    sims_symmetry: Dict[str, float] = {}
    for cls in args.classes:
        sides = seq_means.get((cls, "contrastive"), {})
        attr_n = sides.get("n")
        attr_n1 = sides.get("n_minus_1")
        if attr_n is None or attr_n1 is None:
            continue
        sims_symmetry[cls] = cosine_similarity(attr_n, -attr_n1)

    fig = plot_symmetry_grid(wavenumbers, seq_means, args.classes, baseline="contrastive")
    fig.savefig(out_dir / "symmetry_grid.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_dir / 'symmetry_grid.png'}")

    # --- Comparison 3: prototype alignment --------------------------------------
    sims_proto: Dict[str, float] = {}
    if (ae_dir / "activations.npz").exists() and (seq_dir / "activations.npz").exists():
        wn_a, ae_acts = load_ae_activations(ae_dir / "activations.npz")
        wn_s, seq_acts = load_seq_activations(seq_dir / "activations.npz")
        if not np.allclose(wn_a, wn_s):
            raise ValueError("Wavenumber axes differ between AE and sequencer activation files.")
        for cls in args.classes:
            ae_proto = ae_acts.get(f"{cls}__{args.prototype_init}")
            seq_synth = seq_acts.get(f"synthetic__{cls}__{args.prototype_init}")
            if ae_proto is None or seq_synth is None:
                continue
            sims_proto[cls] = cosine_similarity(ae_proto, seq_synth)
        fig = plot_prototype_alignment_grid(wavenumbers, ae_acts, ae_means, seq_acts,
                                            args.classes, init_name=args.prototype_init)
        fig.savefig(out_dir / "prototype_alignment_grid.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Wrote {out_dir / 'prototype_alignment_grid.png'}")
    else:
        print("  [SKIP] activations.npz missing in one of the dirs; skipping prototype alignment plot.")

    # --- Similarity table -------------------------------------------------------
    md = render_similarity_md(
        similarities={
            "AE vs Seq-n (zero)":          sims_ae_vs_seq_zero,
            "AE vs Seq-n (contrastive)":   sims_ae_vs_seq_contrastive,
            "Seq-n vs −Seq-(n-1)":         sims_symmetry,
            "AE proto vs Seq proto":       sims_proto,
        },
        classes=args.classes,
        column_order=[
            "AE vs Seq-n (zero)",
            "AE vs Seq-n (contrastive)",
            "Seq-n vs −Seq-(n-1)",
            "AE proto vs Seq proto",
        ],
    )
    (out_dir / "similarity.md").write_text(md, encoding="utf-8")
    print(f"  Wrote {out_dir / 'similarity.md'}")
    print("Done.")


if __name__ == "__main__":
    main()
