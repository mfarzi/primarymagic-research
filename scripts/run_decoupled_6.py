#!/usr/bin/env python
"""Decoupled-6 Experiment: encoder diversity vs sequencer performance.

Master orchestration script that runs 27 encoder training steps with
progressive data diversity (AA -> DP -> TP) and evaluates sequencer
performance for each encoder checkpoint.

Phases:
    Phase 1: Encoder training + MSE/PCA evaluation
    Phase 2: Sequencer (classifier) training
    Phase 3: Sequencer evaluation on DP, TP, tetrapeptides, pentapeptides

See docs/experiments_decoupled_6.md for full experiment documentation.

Usage:
    python scripts/run_decoupled_6.py                          # Run everything
    python scripts/run_decoupled_6.py --phase 1                # Encoder only
    python scripts/run_decoupled_6.py --phase 2                # Sequencer training
    python scripts/run_decoupled_6.py --phase 3                # Sequencer evaluation
    python scripts/run_decoupled_6.py --start-step 5           # Resume from step 5
    python scripts/run_decoupled_6.py --start-step 5 --end-step 5  # Single step
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AMINO_ACIDS = ["A", "D", "F", "G", "R", "S"]
DATA_ROOT = "data/processed/orpl"

CHECKPOINT_BASE = Path("checkpoints/decoupled_6")
RESULTS_BASE = Path("results/decoupled_6")

# All 30 dipeptides from the 6 amino acids
ALL_DIPEPTIDES = sorted(
    [f"{a}{b}" for a in AMINO_ACIDS for b in AMINO_ACIDS if a != b]
)
assert len(ALL_DIPEPTIDES) == 30

# Reproducible random permutations
_rng = np.random.default_rng(42)
SHUFFLED_DIPEPTIDES = _rng.permutation(ALL_DIPEPTIDES).tolist()


# ---------------------------------------------------------------------------
# Step definitions
# ---------------------------------------------------------------------------
def _build_steps():
    """Build the list of 27 experiment step definitions.

    Returns:
        List of dicts, each with keys:
            step_num, name, include_aa, include_dp, include_tp,
            dipeptide_codes, tripeptide_codes, init_checkpoint
    """
    steps = []

    # Step 1: AA only
    steps.append({
        "step_num": 1,
        "name": "step01_aa_only",
        "include_aa": True,
        "include_dp": False,
        "include_tp": False,
        "dipeptide_codes": None,
        "tripeptide_codes": None,
        "init_checkpoint": None,
    })

    # Steps 2-7: cumulative dipeptides (5, 10, 15, 20, 25, 30)
    step1_ckpt = str(CHECKPOINT_BASE / "step01_aa_only" / "full_model.pt")
    for i, count in enumerate([5, 10, 15, 20, 25, 30], start=2):
        steps.append({
            "step_num": i,
            "name": f"step{i:02d}_dp{count:02d}",
            "include_aa": True,
            "include_dp": True,
            "include_tp": False,
            "dipeptide_codes": SHUFFLED_DIPEPTIDES[:count],
            "tripeptide_codes": None,
            "init_checkpoint": step1_ckpt,
        })

    # Steps 8-27: cumulative tripeptides (5, 10, ..., 100)
    # We need the shuffled tripeptide list; defer to runtime since we need
    # the actual codes from the dataset. We'll generate with seed=42.
    step7_ckpt = str(CHECKPOINT_BASE / "step07_dp30" / "full_model.pt")
    for i, count in enumerate(range(5, 105, 5), start=8):
        steps.append({
            "step_num": i,
            "name": f"step{i:02d}_tp{count:03d}",
            "include_aa": True,
            "include_dp": True,
            "include_tp": True,
            "dipeptide_codes": ALL_DIPEPTIDES,  # all 30
            "tripeptide_codes_count": count,     # resolved at runtime
            "tripeptide_codes": None,            # filled by resolve_tripeptides()
            "init_checkpoint": step7_ckpt,
        })

    return steps


def resolve_tripeptides(steps):
    """Load dataset and fill in tripeptide_codes for steps 8-27."""
    from primarymagic.data import SpectraDataset

    spectra_data = SpectraDataset(DATA_ROOT)
    all_tp_codes = sorted(spectra_data.tripeptides.keys())

    # Filter to tripeptides composed only of our 6 amino acids
    valid_tp = [
        code for code in all_tp_codes
        if all(c in AMINO_ACIDS for c in code)
    ]

    rng = np.random.default_rng(42)
    shuffled_tp = rng.permutation(valid_tp).tolist()

    for step in steps:
        count = step.pop("tripeptide_codes_count", None)
        if count is not None:
            step["tripeptide_codes"] = shuffled_tp[:count]

    return shuffled_tp


# ---------------------------------------------------------------------------
# Phase 1: Encoder training
# ---------------------------------------------------------------------------
def run_encoder_training(step):
    """Train encoder for one step via subprocess call to train_autoencoder.py."""
    name = step["name"]
    ckpt_dir = CHECKPOINT_BASE / name
    result_dir = RESULTS_BASE / name

    # Skip if already completed
    if (ckpt_dir / "full_model.pt").exists():
        print(f"  [SKIP] {name}: checkpoint already exists")
        return

    cmd = [
        sys.executable, "scripts/train_autoencoder.py",
        "--data-root", DATA_ROOT,
        "--amino-acids", *AMINO_ACIDS,
        "--latent-dim", "32",
        "--encoder-dims", "512", "256",
        "--dropout", "0.3",
        "--activation", "relu",
        "--epochs", "200",
        "--batch-size", "64",
        "--learning-rate", "0.001",
        "--weight-decay", "1e-4",
        "--early-stopping-patience", "20",
        "--samples-per-sequence", "300",
        "--output-dir", str(CHECKPOINT_BASE),
        "--experiment-name", name,
        "--seed", "42",
    ]

    # Include/exclude flags
    if step["include_aa"]:
        cmd.append("--include-aminoacids")
    else:
        cmd.append("--no-include-aminoacids")

    if step["include_dp"]:
        cmd.append("--include-dipeptides")
    else:
        cmd.append("--no-include-dipeptides")

    if step["include_tp"]:
        cmd.append("--include-tripeptides")
    else:
        cmd.append("--no-include-tripeptides")

    # Dipeptide code filtering
    if step["dipeptide_codes"] is not None:
        cmd.extend(["--dipeptide-codes", *step["dipeptide_codes"]])

    # Tripeptide code filtering
    if step["tripeptide_codes"] is not None:
        cmd.extend(["--tripeptide-codes", *step["tripeptide_codes"]])

    # Init checkpoint
    if step["init_checkpoint"] is not None:
        cmd.extend(["--init-checkpoint", step["init_checkpoint"]])

    print(f"  Running: {' '.join(cmd[:6])}...")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  [FAIL] {name} failed with return code {result.returncode}")
        sys.exit(1)

    print(f"  [DONE] {name} ({elapsed:.1f}s)")


# ---------------------------------------------------------------------------
# Phase 1: Post-training evaluation (MSE + PCA/t-SNE)
# ---------------------------------------------------------------------------
def compute_subset_mse(model, spectra_data, device):
    """Compute reconstruction MSE for AA, DP, TP subsets.

    Returns:
        Dict with per-code and aggregate MSE for each category.
    """
    import torch

    model.eval()
    results = {}

    for category, data_dict in [
        ("aminoacids", spectra_data.aminoacids),
        ("dipeptides", spectra_data.dipeptides),
        ("tripeptides", spectra_data.tripeptides),
    ]:
        per_code = {}
        all_mse = []

        for code, spectra in sorted(data_dict.items()):
            # Filter to codes composed of our amino acids
            if not all(c in AMINO_ACIDS for c in code):
                continue

            with torch.no_grad():
                x = torch.tensor(spectra, dtype=torch.float32).to(device)
                z = model.encode(x)
                x_recon = model.decode(z)
                mse = ((x - x_recon) ** 2).mean(dim=1)  # per-sample MSE
                mean_mse = mse.mean().item()

            per_code[code] = mean_mse
            all_mse.extend(mse.cpu().numpy().tolist())

        aggregate = float(np.mean(all_mse)) if all_mse else None
        results[category] = {
            "per_code": per_code,
            "aggregate_mse": aggregate,
            "num_codes": len(per_code),
        }

    return results


def plot_latent_pca_tsne(model, spectra_data, output_dir, device, samples_per_code=100):
    """Generate PCA and t-SNE plots of latent embeddings.

    Colours: AA=blue, DP=green, TP=red.
    Markers: distinct for each of the 6 amino acids; generic for DP/TP.
    """
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    model.eval()

    # Marker map for single amino acids
    aa_markers = {"A": "o", "D": "s", "F": "^", "G": "D", "R": "v", "S": "P"}

    all_z = []
    all_labels = []      # code strings
    all_types = []       # 'AA', 'DP', 'TP'

    rng = np.random.default_rng(42)

    for category, data_dict, seq_type in [
        ("aminoacids", spectra_data.aminoacids, "AA"),
        ("dipeptides", spectra_data.dipeptides, "DP"),
        ("tripeptides", spectra_data.tripeptides, "TP"),
    ]:
        for code, spectra in sorted(data_dict.items()):
            if not all(c in AMINO_ACIDS for c in code):
                continue

            n = min(samples_per_code, len(spectra))
            idx = rng.choice(len(spectra), size=n, replace=False)
            subset = spectra[idx]

            with torch.no_grad():
                x = torch.tensor(subset, dtype=torch.float32).to(device)
                z = model.encode(x).cpu().numpy()

            all_z.append(z)
            all_labels.extend([code] * n)
            all_types.extend([seq_type] * n)

    if not all_z:
        return

    Z = np.concatenate(all_z, axis=0)
    labels = np.array(all_labels)
    types = np.array(all_types)

    type_colors = {"AA": "#1f77b4", "DP": "#2ca02c", "TP": "#d62728"}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- PCA ---
    pca = PCA(n_components=2)
    Z_pca = pca.fit_transform(Z)

    fig, ax = plt.subplots(figsize=(10, 8))
    _scatter_latent(ax, Z_pca, labels, types, type_colors, aa_markers)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("Latent Space PCA")
    ax.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    fig.savefig(output_dir / "pca_2d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- t-SNE ---
    perplexity = min(30, len(Z) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    Z_tsne = tsne.fit_transform(Z)

    fig, ax = plt.subplots(figsize=(10, 8))
    _scatter_latent(ax, Z_tsne, labels, types, type_colors, aa_markers)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("Latent Space t-SNE")
    ax.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    fig.savefig(output_dir / "tsne_2d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _scatter_latent(ax, Z2d, labels, types, type_colors, aa_markers):
    """Helper to scatter plot 2D embeddings with colour/marker coding."""
    plotted = set()
    for seq_type in ["AA", "DP", "TP"]:
        mask = types == seq_type
        if not mask.any():
            continue
        color = type_colors[seq_type]

        if seq_type == "AA":
            # Distinct marker per amino acid
            for aa in AMINO_ACIDS:
                aa_mask = mask & (labels == aa)
                if not aa_mask.any():
                    continue
                lbl = f"AA:{aa}" if f"AA:{aa}" not in plotted else None
                plotted.add(f"AA:{aa}")
                ax.scatter(
                    Z2d[aa_mask, 0], Z2d[aa_mask, 1],
                    c=color, marker=aa_markers.get(aa, "o"),
                    s=20, alpha=0.5, label=lbl,
                )
        else:
            lbl = seq_type if seq_type not in plotted else None
            plotted.add(seq_type)
            marker = "x" if seq_type == "DP" else "+"
            ax.scatter(
                Z2d[mask, 0], Z2d[mask, 1],
                c=color, marker=marker,
                s=15, alpha=0.3, label=lbl,
            )


def run_encoder_evaluation(step):
    """Evaluate encoder: MSE report + PCA/t-SNE plots."""
    import torch
    from primarymagic.data import SpectraDataset
    from primarymagic.models import MultiLabelAutoencoderConfig, MultiLabelRegularizedAutoencoder

    name = step["name"]
    ckpt_dir = CHECKPOINT_BASE / name
    result_dir = RESULTS_BASE / name

    # Skip if already evaluated
    if (result_dir / "mse_report.json").exists() and (result_dir / "pca_2d.png").exists():
        print(f"  [SKIP] {name} evaluation: results already exist")
        return

    ckpt_path = ckpt_dir / "full_model.pt"
    if not ckpt_path.exists():
        print(f"  [SKIP] {name} evaluation: no checkpoint found")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    checkpoint = torch.load(ckpt_path, map_location=device)
    config = MultiLabelAutoencoderConfig.from_dict(checkpoint["model_config"])
    model = MultiLabelRegularizedAutoencoder(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load full spectra data
    spectra_data = SpectraDataset(DATA_ROOT)

    # MSE report
    result_dir.mkdir(parents=True, exist_ok=True)
    mse_report = compute_subset_mse(model, spectra_data, device)
    with open(result_dir / "mse_report.json", "w") as f:
        json.dump(mse_report, f, indent=2)
    print(f"  MSE: AA={mse_report['aminoacids']['aggregate_mse']:.6f}, "
          f"DP={mse_report['dipeptides']['aggregate_mse']:.6f}, "
          f"TP={mse_report['tripeptides']['aggregate_mse']:.6f}")

    # PCA + t-SNE
    plot_latent_pca_tsne(model, spectra_data, result_dir, device)
    print(f"  Saved PCA/t-SNE plots to {result_dir}")


# ---------------------------------------------------------------------------
# Phase 2: Sequencer training
# ---------------------------------------------------------------------------
def run_sequencer_training(step):
    """Train sequencer for one encoder checkpoint via subprocess."""
    name = step["name"]
    seq_name = f"seq_{name}"
    ckpt_dir = CHECKPOINT_BASE / seq_name
    encoder_ckpt = CHECKPOINT_BASE / name / "full_model.pt"

    if not encoder_ckpt.exists():
        print(f"  [SKIP] {seq_name}: encoder checkpoint not found")
        return

    if (ckpt_dir / "differential_classifier.pt").exists():
        print(f"  [SKIP] {seq_name}: checkpoint already exists")
        return

    cmd = [
        sys.executable, "scripts/train_sequencer.py",
        "--encoder-checkpoint", str(encoder_ckpt),
        "--data-root", DATA_ROOT,
        "--include-dipeptide-pairs",
        "--no-include-tripeptide-pairs",
        "--epochs", "200",
        "--batch-size", "64",
        "--learning-rate", "0.001",
        "--weight-decay", "1e-4",
        "--early-stopping-patience", "20",
        "--samples-per-pair", "100",
        "--output-dir", str(CHECKPOINT_BASE),
        "--experiment-name", seq_name,
        "--seed", "42",
    ]

    print(f"  Running sequencer training for {name}...")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  [FAIL] {seq_name} failed with return code {result.returncode}")
        sys.exit(1)

    print(f"  [DONE] {seq_name} ({elapsed:.1f}s)")


# ---------------------------------------------------------------------------
# Phase 3: Sequencer evaluation
# ---------------------------------------------------------------------------
def run_sequencer_evaluation(step):
    """Evaluate sequencer on dipeptide, tripeptide, tetrapeptide, and pentapeptide data."""
    name = step["name"]
    seq_name = f"seq_{name}"
    model_ckpt = CHECKPOINT_BASE / seq_name / "differential_classifier.pt"

    if not model_ckpt.exists():
        print(f"  [SKIP] {seq_name} evaluation: no checkpoint found")
        return

    eval_configs = [
        ("dipeptide_eval",     {"include_dp": True}),
        ("tripeptide_eval",    {"include_tp": True}),
        ("tetrapeptide_eval",  {"include_tetra": True}),
        ("pentapeptide_eval",  {"include_penta": True}),
    ]

    for eval_name, flags in eval_configs:
        label = eval_name.replace("_eval", "")
        result_dir = RESULTS_BASE / seq_name / eval_name
        if (result_dir / "metrics.json").exists():
            print(f"  [SKIP] {seq_name} {label} eval: results already exist")
        else:
            _run_eval_subprocess(model_ckpt, result_dir, label=label, **flags)


def _run_eval_subprocess(
    model_ckpt, output_dir, label,
    include_dp=False, include_tp=False,
    include_tetra=False, include_penta=False,
):
    """Run evaluate_sequencer.py as a subprocess."""
    cmd = [
        sys.executable, "scripts/evaluate_sequencer.py",
        "--model-checkpoint", str(model_ckpt),
        "--data-root", DATA_ROOT,
        "--output-dir", str(output_dir),
        "--batch-size", "64",
        "--samples-per-pair", "100",
        "--seed", "42",
        "--include-dipeptide-pairs" if include_dp else "--no-include-dipeptide-pairs",
        "--include-tripeptide-pairs" if include_tp else "--no-include-tripeptide-pairs",
        "--include-tetrapeptide-pairs" if include_tetra else "--no-include-tetrapeptide-pairs",
        "--include-pentapeptide-pairs" if include_penta else "--no-include-pentapeptide-pairs",
    ]

    print(f"  Evaluating on {label} data...")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  [FAIL] {label} eval failed with return code {result.returncode}")
        sys.exit(1)

    print(f"  [DONE] {label} eval ({elapsed:.1f}s)")


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------
def generate_summary(steps):
    """Aggregate MSE and accuracy across all steps into summary.json."""
    summary = {"steps": []}

    for step in steps:
        name = step["name"]
        seq_name = f"seq_{name}"
        entry = {
            "step_num": step["step_num"],
            "name": name,
        }

        # MSE report
        mse_path = RESULTS_BASE / name / "mse_report.json"
        if mse_path.exists():
            with open(mse_path) as f:
                mse = json.load(f)
            entry["mse_aminoacids"] = mse["aminoacids"]["aggregate_mse"]
            entry["mse_dipeptides"] = mse["dipeptides"]["aggregate_mse"]
            entry["mse_tripeptides"] = mse["tripeptides"]["aggregate_mse"]

        # Dipeptide eval
        dp_metrics_path = RESULTS_BASE / seq_name / "dipeptide_eval" / "metrics.json"
        if dp_metrics_path.exists():
            with open(dp_metrics_path) as f:
                dp = json.load(f)
            entry["dipeptide_accuracy"] = dp["metrics"]["accuracy"]
            entry["dipeptide_macro_f1"] = dp["metrics"]["macro_avg"]["f1"]

        # Tripeptide eval
        tp_metrics_path = RESULTS_BASE / seq_name / "tripeptide_eval" / "metrics.json"
        if tp_metrics_path.exists():
            with open(tp_metrics_path) as f:
                tp = json.load(f)
            entry["tripeptide_accuracy"] = tp["metrics"]["accuracy"]
            entry["tripeptide_macro_f1"] = tp["metrics"]["macro_avg"]["f1"]

        # Tetrapeptide eval
        tetra_metrics_path = RESULTS_BASE / seq_name / "tetrapeptide_eval" / "metrics.json"
        if tetra_metrics_path.exists():
            with open(tetra_metrics_path) as f:
                tetra = json.load(f)
            entry["tetrapeptide_accuracy"] = tetra["metrics"]["accuracy"]
            entry["tetrapeptide_macro_f1"] = tetra["metrics"]["macro_avg"]["f1"]

        # Pentapeptide eval
        penta_metrics_path = RESULTS_BASE / seq_name / "pentapeptide_eval" / "metrics.json"
        if penta_metrics_path.exists():
            with open(penta_metrics_path) as f:
                penta = json.load(f)
            entry["pentapeptide_accuracy"] = penta["metrics"]["accuracy"]
            entry["pentapeptide_macro_f1"] = penta["metrics"]["macro_avg"]["f1"]

        summary["steps"].append(entry)

    summary_path = RESULTS_BASE / "summary.json"
    RESULTS_BASE.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # Print summary table
    print("\n" + "=" * 110)
    print(f"{'Step':<20} {'MSE(AA)':>10} {'MSE(DP)':>10} {'MSE(TP)':>10} "
          f"{'DP Acc':>8} {'TP Acc':>8} {'4P Acc':>8} {'5P Acc':>8}")
    print("-" * 110)
    for e in summary["steps"]:
        def _fmt(key, fmt_str):
            v = e.get(key)
            return format(v, fmt_str) if v is not None else "-".rjust(len(format(0.0, fmt_str)))

        print(f"{e['name']:<20} "
              f"{_fmt('mse_aminoacids', '10.6f')} "
              f"{_fmt('mse_dipeptides', '10.6f')} "
              f"{_fmt('mse_tripeptides', '10.6f')} "
              f"{_fmt('dipeptide_accuracy', '8.2f')} "
              f"{_fmt('tripeptide_accuracy', '8.2f')} "
              f"{_fmt('tetrapeptide_accuracy', '8.2f')} "
              f"{_fmt('pentapeptide_accuracy', '8.2f')}")
    print("=" * 110)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Decoupled-6 experiment pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Run only phase 1 (encoder), 2 (sequencer training), "
             "or 3 (sequencer evaluation). Default: all.",
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=1,
        help="Start from this step number (1-27)",
    )
    parser.add_argument(
        "--end-step",
        type=int,
        default=27,
        help="End at this step number (1-27, inclusive)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("Decoupled-6 Experiment Pipeline")
    print("=" * 70)
    print(f"Amino acids: {AMINO_ACIDS}")
    print(f"Dipeptides: {len(ALL_DIPEPTIDES)} total")
    print(f"Phase: {args.phase or 'both'}")
    print(f"Steps: {args.start_step} - {args.end_step}")
    print()

    # Build step definitions
    steps = _build_steps()
    resolve_tripeptides(steps)

    # Filter steps by range
    active_steps = [s for s in steps if args.start_step <= s["step_num"] <= args.end_step]

    if not active_steps:
        print("No steps to run in the specified range.")
        return

    run_phase1 = args.phase is None or args.phase == 1
    run_phase2 = args.phase is None or args.phase == 2
    run_phase3 = args.phase is None or args.phase == 3

    total_t0 = time.time()

    # Phase 1: Encoder training + evaluation
    if run_phase1:
        print("\n" + "=" * 70)
        print("PHASE 1: Encoder Training")
        print("=" * 70)

        for step in active_steps:
            print(f"\n--- Step {step['step_num']}: {step['name']} ---")
            run_encoder_training(step)
            run_encoder_evaluation(step)

    # Phase 2: Sequencer training
    if run_phase2:
        print("\n" + "=" * 70)
        print("PHASE 2: Sequencer Training")
        print("=" * 70)

        for step in active_steps:
            print(f"\n--- Step {step['step_num']}: seq_{step['name']} ---")
            run_sequencer_training(step)

    # Phase 3: Sequencer evaluation (DP, TP, tetrapeptide, pentapeptide)
    if run_phase3:
        print("\n" + "=" * 70)
        print("PHASE 3: Sequencer Evaluation")
        print("=" * 70)

        for step in active_steps:
            print(f"\n--- Step {step['step_num']}: seq_{step['name']} ---")
            run_sequencer_evaluation(step)

    # Generate summary
    print("\n" + "=" * 70)
    print("Generating Summary")
    print("=" * 70)
    generate_summary(steps)

    total_elapsed = time.time() - total_t0
    print(f"\nTotal elapsed time: {total_elapsed:.1f}s ({total_elapsed/3600:.2f}h)")
    print("Experiment complete!")


if __name__ == "__main__":
    main()
