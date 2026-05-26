"""
Count spectra accepted by magic_old but rejected by magic_bayes, aggregated
per peptide-length category (1-letter through 5-letter).

The new method is meant to be a strict superset of the old — this script
verifies that claim across the full dataset, not just the SGAF case study.

Output:
  - Console table
  - PNG: analysis_outputs/old_only_pass_table.png
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED = PROJECT_ROOT / "data" / "custom" / "processed"
MAGIC_OLD = PROCESSED / "magic_old"
MAGIC_NEW = PROCESSED / "magic_bayes"
OUT_DIR = PROJECT_ROOT / "analysis_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


CATEGORIES = [
    ("Amino Acids",     "1-letter"),
    ("Dipeptides",      "2-letter"),
    ("Tripeptides",     "3-letter"),
    ("Tetrapeptides",   "4-letter"),
    ("Pentapeptides",   "5-letter"),
]


def _iter_seq_rep(root, subdir):
    """Yield (seq, rep, rep_dir) for every sequence/rep under root/subdir.

    Skips PTM under 1-letter so categories align with the yield tables.
    """
    d = root / subdir
    if not d.exists():
        return
    for seq_dir in sorted(d.iterdir()):
        if not seq_dir.is_dir():
            continue
        if subdir == "1-letter" and seq_dir.name == "PTM":
            continue
        for rep_dir in sorted(seq_dir.iterdir()):
            if rep_dir.is_dir() and rep_dir.name.startswith("rep"):
                yield seq_dir.name, rep_dir.name, rep_dir


def count_category(subdir):
    """For one category, walk every (seq, rep) and count spectra where
    old.passed & ~new.passed.  Also accumulate magic_bayes SNR per cell so
    we can report a mean SNR (new method) for each column."""
    old_only = 0
    new_only = 0
    both_pass = 0
    both_fail = 0
    total_spectra = 0
    n_reps = 0
    mismatches = []

    snr_old_only = []
    snr_new_only = []
    snr_both_pass = []

    seen_seqs = set()
    for seq, rep, old_rep_dir in _iter_seq_rep(MAGIC_OLD, subdir):
        new_rep_dir = MAGIC_NEW / subdir / seq / rep
        if not (new_rep_dir / "mask.npz").exists():
            continue
        if not (old_rep_dir / "mask.npz").exists():
            continue

        m_old = np.load(old_rep_dir / "mask.npz")
        m_new = np.load(new_rep_dir / "mask.npz")
        if not np.array_equal(m_old["raw_index"], m_new["raw_index"]):
            mismatches.append(f"{seq}/{rep}")
            continue

        op = m_old["passed"].astype(bool)
        npass = m_new["passed"].astype(bool)
        new_snr = m_new["snr"]

        mask_old_only = op & ~npass
        mask_new_only = ~op & npass
        mask_both_pass = op & npass

        old_only += int(mask_old_only.sum())
        new_only += int(mask_new_only.sum())
        both_pass += int(mask_both_pass.sum())
        both_fail += int((~op & ~npass).sum())
        total_spectra += len(op)
        n_reps += 1
        seen_seqs.add(seq)

        snr_old_only.extend(new_snr[mask_old_only].tolist())
        snr_new_only.extend(new_snr[mask_new_only].tolist())
        snr_both_pass.extend(new_snr[mask_both_pass].tolist())

    def _mean(xs):
        return float(np.nanmean(xs)) if xs else float("nan")

    return {
        "n_seqs": len(seen_seqs),
        "n_reps": n_reps,
        "total_spectra": total_spectra,
        "old_only_pass": old_only,
        "new_only_pass": new_only,
        "both_pass": both_pass,
        "both_fail": both_fail,
        "snr_old_only_mean": _mean(snr_old_only),
        "snr_new_only_mean": _mean(snr_new_only),
        "snr_both_pass_mean": _mean(snr_both_pass),
        "snr_old_only_vals": snr_old_only,
        "snr_new_only_vals": snr_new_only,
        "snr_both_pass_vals": snr_both_pass,
        "mismatches": mismatches,
    }


def main():
    rows = []
    for label, subdir in CATEGORIES:
        s = count_category(subdir)
        rows.append((label, s))
        print(f"{label} ({subdir}): seqs={s['n_seqs']}  reps={s['n_reps']}  "
              f"old_only={s['old_only_pass']}  new_only={s['new_only_pass']}  "
              f"both_pass={s['both_pass']}  both_fail={s['both_fail']}  "
              f"total={s['total_spectra']}")
        if s['mismatches']:
            print(f"  raw-index mismatches skipped: {s['mismatches']}")

    # ---- PNG table -------------------------------------------------------
    # Each pass/fail cell now shows count + (mean new-method SNR)
    col_labels = [
        "Category", "#Seq", "Total\nSpectra",
        "Old PASS & New FAIL\ncount  (mean new SNR)",
        "Old PASS & New PASS\ncount  (mean new SNR)",
        "Old FAIL & New PASS\ncount  (mean new SNR)",
    ]
    cell = []
    total_old_only = 0
    total_both_pass = 0
    total_new_only = 0
    total_all = 0
    total_seqs = 0
    all_snr_old_only = []
    all_snr_both_pass = []
    all_snr_new_only = []

    def _fmt(count, mean_snr):
        if count == 0:
            return "0"
        return f"{count:,}   ({mean_snr:.1f})"

    for label, s in rows:
        cell.append([
            label,
            str(s["n_seqs"]),
            f"{s['total_spectra']:,}",
            _fmt(s["old_only_pass"], s["snr_old_only_mean"]),
            _fmt(s["both_pass"],     s["snr_both_pass_mean"]),
            _fmt(s["new_only_pass"], s["snr_new_only_mean"]),
        ])
        total_old_only += s["old_only_pass"]
        total_both_pass += s["both_pass"]
        total_new_only += s["new_only_pass"]
        total_all += s["total_spectra"]
        total_seqs += s["n_seqs"]
        all_snr_old_only.extend(s["snr_old_only_vals"])
        all_snr_both_pass.extend(s["snr_both_pass_vals"])
        all_snr_new_only.extend(s["snr_new_only_vals"])

    def _grand_mean(xs):
        return float(np.nanmean(xs)) if xs else float("nan")

    cell.append([
        "Total", str(total_seqs), f"{total_all:,}",
        _fmt(total_old_only, _grand_mean(all_snr_old_only)),
        _fmt(total_both_pass, _grand_mean(all_snr_both_pass)),
        _fmt(total_new_only, _grand_mean(all_snr_new_only)),
    ])

    fig, ax = plt.subplots(figsize=(17, 4.6))
    ax.axis("off")
    # Wider columns for the count+SNR cells so values don't get clipped
    col_widths = [0.13, 0.07, 0.10, 0.23, 0.23, 0.23]
    table = ax.table(cellText=cell, colLabels=col_labels,
                     colWidths=col_widths,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.9)
    for j in range(len(col_labels)):
        h = table[0, j]
        h.set_facecolor("#2E4057")
        h.set_text_props(color="white", fontweight="bold")
        h.set_height(0.22)
    for i in range(len(cell)):
        is_total = i == len(cell) - 1
        color = "#E8EEF4" if i % 2 == 0 else "white"
        if is_total:
            color = "#D6E4F0"
        for j in range(len(col_labels)):
            c = table[i + 1, j]
            c.set_facecolor(color)
            if is_total:
                c.set_text_props(fontweight="bold")
            # Highlight the column under test (old-only-pass)
            if j == 3 and not is_total:
                if cell[i][3].startswith("0"):
                    c.set_text_props(color="#15803D", fontweight="bold")
                else:
                    c.set_text_props(color="#C2410C", fontweight="bold")
    ax.set_title("magic_old PASS  ∧  magic_bayes FAIL  —  per-length count "
                 "(inclusiveness audit)", fontsize=12, pad=10)
    out = OUT_DIR / "old_only_pass_table.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()
