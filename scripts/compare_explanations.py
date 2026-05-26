#!/usr/bin/env python
"""Compare classifier-attribution fingerprints across two checkpoints.

Reads ``attributions.npz`` from two ``explanations/`` directories produced by
``explain_classifier.py`` and produces:

  1. A single overlay figure (rows=classes, cols=baselines), each panel
     showing both checkpoints' mean IG attribution as overlaid line plots.
  2. A Markdown table of cosine similarity per (class, baseline) between the
     two checkpoints' mean attributions.

Usage:
    python scripts/compare_explanations.py \\
        --checkpoint-a-explanations checkpoints/.../step01.../explanations \\
        --checkpoint-b-explanations checkpoints/.../step26.../explanations \\
        --label-a step01_aa_only --label-b step26_tp095 \\
        --classes A G S D R F \\
        --output-dir checkpoints/.../comparisons_step01_vs_step26
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_mean_attributions(
    npz_path: Path,
) -> Tuple[np.ndarray, Dict[Tuple[str, str], np.ndarray]]:
    """Load attributions.npz and return (wavenumbers, mean_per_class_baseline)."""
    data = np.load(npz_path)
    wavenumbers = data["wavenumbers"]
    means: Dict[Tuple[str, str], np.ndarray] = {}
    for key in data.files:
        if key == "wavenumbers":
            continue
        cls, baseline = key.split("__")
        means[(cls, baseline)] = data[key].mean(axis=0)
    return wavenumbers, means


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D arrays."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def plot_overlay_grid(
    wavenumbers: np.ndarray,
    means_a: Dict[Tuple[str, str], np.ndarray],
    means_b: Dict[Tuple[str, str], np.ndarray],
    classes: List[str],
    baselines: List[str],
    label_a: str,
    label_b: str,
) -> plt.Figure:
    """Grid (rows=classes, cols=baselines) of overlaid attribution lines."""
    n_rows, n_cols = len(classes), len(baselines)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.5 * n_cols, 1.6 * n_rows),
        sharex=True, squeeze=False,
    )
    for r, cls in enumerate(classes):
        for c, baseline in enumerate(baselines):
            ax = axes[r, c]
            attr_a = means_a.get((cls, baseline))
            attr_b = means_b.get((cls, baseline))
            if attr_a is not None:
                ax.plot(wavenumbers, attr_a, label=label_a, color="#1f77b4",
                        linewidth=1.0, alpha=0.85)
            if attr_b is not None:
                ax.plot(wavenumbers, attr_b, label=label_b, color="#d62728",
                        linewidth=1.0, alpha=0.85)
            ax.axhline(0, color="black", linewidth=0.3)
            ax.set_title(f"{cls}  ·  baseline={baseline}", fontsize=9)
            ax.tick_params(labelsize=7)
            if r == 0 and c == 0:
                ax.legend(fontsize=8, loc="best")
    for ax in axes[-1, :]:
        ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=9)
    fig.tight_layout()
    return fig


def render_similarity_md(
    similarities: Dict[Tuple[str, str], float],
    classes: List[str],
    baselines: List[str],
    label_a: str,
    label_b: str,
) -> str:
    """Markdown table of cosine similarity per (class, baseline)."""
    lines = [
        f"# Attribution similarity: `{label_a}` vs `{label_b}`",
        "",
        "Cosine similarity between mean IG attribution vectors. "
        "1.0 = identical pattern; 0.0 = orthogonal; negative = anti-correlated.",
        "",
    ]
    header = "| Class | " + " | ".join(f"{b}" for b in baselines) + " |"
    sep = "|---:|" + "---:|" * len(baselines)
    lines.extend([header, sep])
    for cls in classes:
        row = [cls]
        for b in baselines:
            sim = similarities.get((cls, b))
            row.append("—" if sim is None else f"{sim:.3f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare attribution fingerprints between two checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint-a-explanations", type=str, required=True,
                   help="Path to explanations/ dir for checkpoint A.")
    p.add_argument("--checkpoint-b-explanations", type=str, required=True,
                   help="Path to explanations/ dir for checkpoint B.")
    p.add_argument("--label-a", type=str, default="A")
    p.add_argument("--label-b", type=str, default="B")
    p.add_argument("--classes", type=str, nargs="+", required=True,
                   help="Classes to compare (must exist in both attributions.npz files).")
    p.add_argument("--baselines", type=str, nargs="+",
                   default=["zero", "contrastive"],
                   choices=["zero", "contrastive"])
    p.add_argument("--output-dir", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_a = Path(args.checkpoint_a_explanations) / "attributions.npz"
    npz_b = Path(args.checkpoint_b_explanations) / "attributions.npz"

    print(f"Loading {npz_a}")
    wavenumbers_a, means_a = load_mean_attributions(npz_a)
    print(f"Loading {npz_b}")
    wavenumbers_b, means_b = load_mean_attributions(npz_b)

    if wavenumbers_a.shape != wavenumbers_b.shape or not np.allclose(wavenumbers_a, wavenumbers_b):
        raise ValueError("Wavenumber axes differ between the two checkpoints.")

    similarities: Dict[Tuple[str, str], float] = {}
    for cls in args.classes:
        for b in args.baselines:
            a = means_a.get((cls, b))
            bb = means_b.get((cls, b))
            if a is None or bb is None:
                print(f"  [WARN] missing ({cls}, {b}) in one of the two files")
                continue
            similarities[(cls, b)] = cosine_similarity(a, bb)

    md = render_similarity_md(similarities, args.classes, args.baselines,
                              args.label_a, args.label_b)
    (out_dir / "similarity.md").write_text(md, encoding="utf-8")
    print(f"  Wrote similarity table to {out_dir / 'similarity.md'}")

    fig = plot_overlay_grid(
        wavenumbers_a, means_a, means_b, args.classes, args.baselines,
        args.label_a, args.label_b,
    )
    out_path = out_dir / "overlay_grid.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote overlay grid to {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
