#!/usr/bin/env python
"""Plot accuracy results from the decoupled-v1 experiment.

Reads results/decoupled_v1/summary.json and plots mean accuracy (+/- std)
for dipeptide, tripeptide, tetrapeptide, and pentapeptide evaluation
versus increasing encoder training diversity (AA -> DP -> TP -> 4P -> 5P steps).

Usage:
    python scripts/plot_decoupled_v1.py
    python scripts/plot_decoupled_v1.py --results-dir results/decoupled_v1
    python scripts/plot_decoupled_v1.py --no-show  # save only, don't display
"""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot decoupled-v1 accuracy results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/decoupled_v1",
        help="Path to results directory containing summary.json",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display the plot, only save to file",
    )
    return parser.parse_args()


def build_x_label(name):
    """Convert step name to a readable x-axis label.

    Examples:
        step01_aa_only -> AA
        step02_dp05    -> DP5
        step07_dp30    -> DP30
        step08_tp005   -> TP5
        step26_tp095   -> TP95
        step27_4p05    -> 4P5
        step35_5p10    -> 5P10
    """
    if "aa_only" in name:
        return "AA"
    # Extract dp/tp/4p/5p and count
    parts = name.split("_", 1)[1]  # drop stepNN_
    for prefix, label in [("dp", "DP"), ("tp", "TP"), ("4p", "4P"), ("5p", "5P")]:
        if parts.startswith(prefix):
            count = int(parts[len(prefix):])
            return f"{label}{count}"
    return name


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    summary_path = results_dir / "summary.json"

    if not summary_path.exists():
        print(f"Error: {summary_path} not found")
        return

    with open(summary_path) as f:
        summary = json.load(f)

    steps = summary["steps"]

    # Build data arrays
    x_labels = [build_x_label(s["name"]) for s in steps]
    x = np.arange(len(steps))

    # Derive sequence counts from step names (last dp/tp/4p/5p step)
    counts = {"dp": 0, "tp": 0, "4p": 0, "5p": 0}
    for s in steps:
        for prefix in counts:
            if prefix in s["name"]:
                label = build_x_label(s["name"])
                num = int(label[len(prefix.upper()):] if label.startswith(prefix.upper())
                          else label[2:])
                counts[prefix] = max(counts[prefix], num)

    eval_types = [
        ("dipeptide", f"Dipeptide ({counts['dp']})"),
        ("tripeptide", f"Tripeptide ({counts['tp']})"),
        ("tetrapeptide", f"Tetrapeptide ({counts['4p']})"),
        ("pentapeptide", f"Pentapeptide ({counts['5p']})"),
    ]

    # Collect means and stds
    data = {}
    for key, label in eval_types:
        means = []
        stds = []
        for s in steps:
            m = s.get(f"{key}_accuracy_mean")
            sd = s.get(f"{key}_accuracy_std", 0.0)
            means.append(m)
            stds.append(sd if sd is not None else 0.0)
        data[key] = {
            "label": label,
            "means": means,
            "stds": stds,
        }

    # --- Plot ---
    if args.no_show:
        matplotlib.use("Agg")

    fig, ax = plt.subplots(figsize=(14, 6))

    colors = {
        "dipeptide": "#1f77b4",
        "tripeptide": "#2ca02c",
        "tetrapeptide": "#d62728",
        "pentapeptide": "#9467bd",
    }
    markers = {
        "dipeptide": "o",
        "tripeptide": "s",
        "tetrapeptide": "^",
        "pentapeptide": "D",
    }

    for key, label in eval_types:
        d = data[key]
        means = np.array(d["means"], dtype=float)
        stds = np.array(d["stds"], dtype=float)

        # Only plot where data exists
        mask = ~np.isnan(means)
        if not mask.any():
            continue

        ax.plot(
            x[mask], means[mask],
            color=colors[key], marker=markers[key], markersize=5,
            linewidth=1.5, label=d["label"],
        )
        ax.fill_between(
            x[mask], means[mask] - stds[mask], means[mask] + stds[mask],
            color=colors[key], alpha=0.15,
        )

    # Mark boundaries between phase transitions
    for prefix, text in [("TP", "TP steps begin"), ("4P", "4P steps begin"), ("5P", "5P steps begin")]:
        for i, lbl in enumerate(x_labels):
            if lbl.startswith(prefix):
                ax.axvline(x=i - 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
                ax.text(i - 0.5, ax.get_ylim()[0] + 1, f" {text}",
                        fontsize=8, color="gray", va="bottom")
                break

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Encoder Training Data")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Sequencer Accuracy vs Encoder Training Diversity")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 105)

    plt.tight_layout()

    # Save
    output_path = results_dir / "accuracy_vs_diversity.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {output_path}")

    output_pdf = results_dir / "accuracy_vs_diversity.pdf"
    fig.savefig(output_pdf, bbox_inches="tight")
    print(f"Saved: {output_pdf}")

    if not args.no_show:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
