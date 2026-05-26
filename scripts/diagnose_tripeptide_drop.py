"""Per-tripeptide accuracy comparison between primary_magic and magic_devtest_shifted.

Loads the seq_step07_dp30 sequencer for each pipeline, runs evaluation on its
own data root, and reports per-tripeptide accuracy drops.
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from primarymagic.data import SpectraDataset, SequenceDataset
from evaluate_sequencer import load_model, collate_fn  # noqa: E402

MIN_SPECTRA = 40
SAMPLES_PER_PAIR = 100
SEED = 42
AAS = ["A", "D", "F", "G", "R", "S"]

PIPELINES = {
    "primary_magic":          ("checkpoints/primary_magic/seq_step07_dp30/differential_classifier.pt",
                               "data/custom/processed/primary_magic"),
    "magic_devtest_shifted":  ("checkpoints/magic_devtest_shifted/seq_step07_dp30/differential_classifier.pt",
                               "data/custom/processed/magic_devtest_shifted"),
    "magic_devtest_bayes":    ("checkpoints/magic_devtest_bayes/seq_step07_dp30/differential_classifier.pt",
                               "data/custom/processed/magic_devtest_bayes"),
}


def per_sequence_accuracy(ckpt_path, data_root):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config = load_model(ckpt_path, device)
    model = model.to(device)
    aa_codes = config["amino_acid_codes"]

    spectra_data = SpectraDataset(data_root, min_spectra=MIN_SPECTRA)
    ds = SequenceDataset(
        spectra_data,
        include_dipeptide_pairs=False,
        include_tripeptide_pairs=True,
        include_tetrapeptide_pairs=False,
        include_pentapeptide_pairs=False,
        samples_per_pair=SAMPLES_PER_PAIR,
        amino_acid_codes=aa_codes,
        seed=SEED,
    )
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    all_targets, all_preds = [], []
    model.eval()
    with torch.no_grad():
        for s_xy, s_x, targets in loader:
            outputs = model(s_xy.to(device), s_x.to(device))
            _, preds = outputs.max(1)
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    targets = np.array(all_targets)
    preds = np.array(all_preds)
    codes = ds.sample_codes  # parallel list to dataset samples

    # Per-sequence accuracy
    per_seq = defaultdict(lambda: {"correct": 0, "total": 0})
    for code, t, p in zip(codes, targets, preds):
        per_seq[code]["total"] += 1
        if t == p:
            per_seq[code]["correct"] += 1

    return {code: stats["correct"] / stats["total"] for code, stats in per_seq.items() if stats["total"] > 0}


def main():
    accs = {}
    for name, (ckpt, root) in PIPELINES.items():
        print(f"=== {name} ===")
        accs[name] = per_sequence_accuracy(ckpt, root)

    common = sorted(set(accs["primary_magic"]) &
                    set(accs["magic_devtest_shifted"]) &
                    set(accs["magic_devtest_bayes"]))

    rows = []
    for code in common:
        prim = accs["primary_magic"][code]
        visu = accs["magic_devtest_shifted"][code]
        bayes = accs["magic_devtest_bayes"][code]
        rows.append((code, prim, visu, bayes, bayes - prim, bayes - visu))

    # Sort by visu->primary regression (most-broken first under visu)
    rows.sort(key=lambda r: r[2] - r[1])

    print()
    print(f"{'Code':<6} {'primary':>9} {'visu':>9} {'bayes':>9} "
          f"{'bayes-prim':>12} {'bayes-visu':>12}")
    print("-" * 64)
    for code, prim, visu, bayes, dp, dv in rows:
        recovered = " <-- recovered" if (dv > 0.20 and bayes > 0.5) else ""
        print(f"{code:<6} {prim:>9.4f} {visu:>9.4f} {bayes:>9.4f} {dp:>+12.4f} {dv:>+12.4f}{recovered}")

    print()
    n_recovered = sum(1 for _,_,_,_,_,dv in rows if dv > 0.20)
    n_still_broken = sum(1 for _,p,_,b,_,_ in rows if (p - b) > 0.20)
    n_improved_overall = sum(1 for _,p,_,b,_,_ in rows if b > p)
    print(f"Sequences where bayes recovered > 20pp vs visu: {n_recovered}")
    print(f"Sequences still > 20pp below primary_magic with bayes: {n_still_broken}")
    print(f"Sequences where bayes >= primary_magic: {n_improved_overall}")
    print()
    print(f"Mean accuracy across all {len(rows)} tripeptides:")
    print(f"  primary_magic:          {sum(r[1] for r in rows)/len(rows):.4f}")
    print(f"  magic_devtest_shifted:  {sum(r[2] for r in rows)/len(rows):.4f}")
    print(f"  magic_devtest_bayes:    {sum(r[3] for r in rows)/len(rows):.4f}")


if __name__ == "__main__":
    main()
