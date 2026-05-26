#!/usr/bin/env python
"""Magic Experiment: 26-class encoder with 6-class sequencer on custom data.

Master orchestration script that trains the encoder on all 26 amino acid
classes (20 standard + 6 PTM) for better representation learning, while
training/evaluating the sequencer on the 6 amino acids (A, D, F, G, R, S).

Uses custom magic-preprocessed data with multi-rep support and min_spectra=40
filtering. Each random-selection step is repeated 5 times for statistical
reliability.

Phases:
    Phase 1: Encoder training + MSE/PCA evaluation
    Phase 2: Sequencer (classifier) training
    Phase 3: Sequencer evaluation on DP, TP, tetrapeptides, pentapeptides

Usage:
    python scripts/run_magic.py --name magic1                   # Run everything
    python scripts/run_magic.py --name magic1 --phase 1         # Encoder only
    python scripts/run_magic.py --name magic1 --phase 2         # Sequencer training
    python scripts/run_magic.py --name magic1 --phase 3         # Sequencer evaluation
    python scripts/run_magic.py --name magic1 --start-step 2 --end-step 2 --run 1
"""

import argparse
import datetime
import json
import re
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
AMINO_ACIDS_6 = ["A", "D", "F", "G", "R", "S"]
STANDARD_AA = list("ACDEFGHIKLMNPQRSTVWY")  # 20 standard
PTM_AA = ["acetyl-K", "acetyl-S", "hydroxy-P", "phos-S", "phos-T", "phos-Y"]
ALL_AA_26 = STANDARD_AA + PTM_AA

DATA_ROOT = "data/custom/processed/magic"  # default; overridable via --data-root
MIN_SPECTRA = 40
NUM_RUNS = 5
STEP_SIZE = 5

# Set from CLI flags in main()
CHECKPOINT_BASE = None
RESULTS_BASE = None
LOG_DIR = None
SEED = 42
OVERWRITE_SEQ = False


# ---------------------------------------------------------------------------
# Logging: tee stdout/stderr to a log file
# ---------------------------------------------------------------------------
class _Tee:
    """Stream wrapper that writes to multiple streams simultaneously."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

    # So subprocess can inherit the underlying file descriptor
    def fileno(self):
        return self.streams[0].fileno()


_log_file = None  # set in setup_logging()


def setup_logging():
    """Redirect stdout and stderr to both terminal and a timestamped log file."""
    global _log_file
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"run_{timestamp}.log"
    _log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, _log_file)
    sys.stderr = _Tee(sys.__stderr__, _log_file)
    print(f"Logging to: {log_path}")


def teardown_logging():
    """Restore stdout/stderr and close the log file."""
    global _log_file
    if _log_file is not None:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        _log_file.close()
        _log_file = None


def run_logged_subprocess(cmd):
    """Run a subprocess, streaming its output through our tee'd stdout.

    Returns:
        subprocess.CompletedProcess with returncode.
    """
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    for line in proc.stdout:
        sys.stdout.write(line)
    proc.wait()
    return proc


# ---------------------------------------------------------------------------
# Runtime sequence discovery
# ---------------------------------------------------------------------------
def discover_sequences():
    """Load dataset and discover valid dipeptides/tripeptides/tetrapeptides/pentapeptides.

    Returns:
        (all_dp, all_tp, all_4p, all_5p): Sorted lists of valid sequence codes
            composed of the 6 amino acids and meeting min_spectra threshold.
    """
    from primarymagic.data import SpectraDataset

    spectra_data = SpectraDataset(DATA_ROOT, min_spectra=MIN_SPECTRA)
    print(f"Loaded dataset: {spectra_data}")
    print(f"Amino acids: {sorted(spectra_data.aminoacids.keys())}")

    # Dipeptides composed of the 6 amino acids
    all_dp = sorted(
        code for code in spectra_data.dipeptides.keys()
        if all(c in AMINO_ACIDS_6 for c in code)
    )

    # Tripeptides composed of the 6 amino acids
    all_tp = sorted(
        code for code in spectra_data.tripeptides.keys()
        if all(c in AMINO_ACIDS_6 for c in code)
    )

    # Tetrapeptides composed of the 6 amino acids
    all_4p = sorted(
        code for code in spectra_data.tetrapeptides.keys()
        if all(c in AMINO_ACIDS_6 for c in code)
    )

    # Pentapeptides composed of the 6 amino acids
    all_5p = sorted(
        code for code in spectra_data.pentapeptides.keys()
        if all(c in AMINO_ACIDS_6 for c in code)
    )

    print(f"Valid dipeptides (6-AA, min_spectra={MIN_SPECTRA}): {len(all_dp)}")
    print(f"Valid tripeptides (6-AA, min_spectra={MIN_SPECTRA}): {len(all_tp)}")
    print(f"Valid tetrapeptides (6-AA, min_spectra={MIN_SPECTRA}): {len(all_4p)}")
    print(f"Valid pentapeptides (6-AA, min_spectra={MIN_SPECTRA}): {len(all_5p)}")

    return all_dp, all_tp, all_4p, all_5p


# ---------------------------------------------------------------------------
# Step definitions
# ---------------------------------------------------------------------------
def _build_steps(all_dp, all_tp, all_4p, all_5p):
    """Build the list of experiment step definitions.

    Args:
        all_dp: Sorted list of valid dipeptide codes.
        all_tp: Sorted list of valid tripeptide codes.
        all_4p: Sorted list of valid tetrapeptide codes.
        all_5p: Sorted list of valid pentapeptide codes.

    Returns:
        List of dicts, each with keys:
            step_num, name, include_aa, include_dp, include_tp,
            include_4p, include_5p, dipeptide_codes, tripeptide_codes,
            tetrapeptide_codes, pentapeptide_codes, init_checkpoint, run
    """
    steps = []
    step_num = 1

    # Step 1: AA only (single run)
    steps.append({
        "step_num": step_num,
        "name": "step01_aa_only",
        "include_aa": True,
        "include_dp": False,
        "include_tp": False,
        "dipeptide_codes": None,
        "tripeptide_codes": None,
        "init_checkpoint": None,
        "run": None,  # single run, no suffix
    })
    step_num += 1

    step1_ckpt = str(CHECKPOINT_BASE / "step01_aa_only" / "full_model.pt")

    # Dipeptide steps: cumulative counts 5, 10, 15, ..., all_dp
    dp_counts = list(range(STEP_SIZE, len(all_dp), STEP_SIZE))
    if not dp_counts or dp_counts[-1] != len(all_dp):
        dp_counts.append(len(all_dp))

    for count in dp_counts:
        is_final = (count == len(all_dp))
        num_runs = 1 if is_final else NUM_RUNS

        for run in range(1, num_runs + 1):
            base_name = f"step{step_num:02d}_dp{count:02d}"
            name = base_name if num_runs == 1 else f"{base_name}_run{run}"

            # Select dipeptides for this run
            rng = np.random.default_rng(42 + run)
            shuffled = rng.permutation(all_dp).tolist()
            dp_codes = shuffled[:count]

            steps.append({
                "step_num": step_num,
                "name": name,
                "include_aa": True,
                "include_dp": True,
                "include_tp": False,
                "dipeptide_codes": dp_codes,
                "tripeptide_codes": None,
                "init_checkpoint": step1_ckpt,
                "run": run if num_runs > 1 else None,
            })

        step_num += 1

    # Identify the checkpoint from the final dp step (all dipeptides)
    # Find the last dp step name (single run for all-dp)
    last_dp_name = f"step{step_num - 1:02d}_dp{len(all_dp):02d}"
    last_dp_ckpt = str(CHECKPOINT_BASE / last_dp_name / "full_model.pt")

    # Tripeptide steps: cumulative counts 5, 10, 15, ..., all_tp
    tp_counts = list(range(STEP_SIZE, len(all_tp), STEP_SIZE))
    if not tp_counts or tp_counts[-1] != len(all_tp):
        tp_counts.append(len(all_tp))

    for count in tp_counts:
        is_final = (count == len(all_tp))
        num_runs = 1 if is_final else NUM_RUNS

        for run in range(1, num_runs + 1):
            base_name = f"step{step_num:02d}_tp{count:03d}"
            name = base_name if num_runs == 1 else f"{base_name}_run{run}"

            # Select tripeptides for this run
            rng = np.random.default_rng(42 + run)
            shuffled = rng.permutation(all_tp).tolist()
            tp_codes = shuffled[:count]

            steps.append({
                "step_num": step_num,
                "name": name,
                "include_aa": True,
                "include_dp": True,
                "include_tp": True,
                "dipeptide_codes": all_dp,  # all dipeptides
                "tripeptide_codes": tp_codes,
                "init_checkpoint": last_dp_ckpt,
                "run": run if num_runs > 1 else None,
            })

        step_num += 1

    # Identify the checkpoint from the final TP step (all tripeptides)
    last_tp_name = f"step{step_num - 1:02d}_tp{len(all_tp):03d}"
    last_tp_ckpt = str(CHECKPOINT_BASE / last_tp_name / "full_model.pt")

    # Tetrapeptide steps: encoder trains on AA + all DP + all TP + selected 4P
    fp_counts = list(range(STEP_SIZE, len(all_4p), STEP_SIZE))
    if not fp_counts or fp_counts[-1] != len(all_4p):
        fp_counts.append(len(all_4p))

    for count in fp_counts:
        is_final = (count == len(all_4p))
        num_runs = 1 if is_final else NUM_RUNS

        for run in range(1, num_runs + 1):
            base_name = f"step{step_num:02d}_4p{count:02d}"
            name = base_name if num_runs == 1 else f"{base_name}_run{run}"

            # Select tetrapeptides for this run
            rng = np.random.default_rng(42 + run)
            shuffled = rng.permutation(all_4p).tolist()
            tetrapeptide_codes = shuffled[:count]

            steps.append({
                "step_num": step_num,
                "name": name,
                "include_aa": True,
                "include_dp": True,
                "include_tp": True,
                "include_4p": True,
                "include_5p": False,
                "dipeptide_codes": all_dp,
                "tripeptide_codes": all_tp,
                "tetrapeptide_codes": tetrapeptide_codes,
                "pentapeptide_codes": None,
                "init_checkpoint": last_tp_ckpt,
                "run": run if num_runs > 1 else None,
            })

        step_num += 1

    # Identify the checkpoint from the final 4P step (all tetrapeptides)
    last_4p_name = f"step{step_num - 1:02d}_4p{len(all_4p):02d}"
    last_4p_ckpt = str(CHECKPOINT_BASE / last_4p_name / "full_model.pt")

    # Pentapeptide steps: encoder trains on AA + all DP + all TP + all 4P + selected 5P
    pp_counts = list(range(STEP_SIZE, len(all_5p), STEP_SIZE))
    if not pp_counts or pp_counts[-1] != len(all_5p):
        pp_counts.append(len(all_5p))

    for count in pp_counts:
        is_final = (count == len(all_5p))
        num_runs = 1 if is_final else NUM_RUNS

        for run in range(1, num_runs + 1):
            base_name = f"step{step_num:02d}_5p{count:02d}"
            name = base_name if num_runs == 1 else f"{base_name}_run{run}"

            # Select pentapeptides for this run
            rng = np.random.default_rng(42 + run)
            shuffled = rng.permutation(all_5p).tolist()
            pentapeptide_codes = shuffled[:count]

            steps.append({
                "step_num": step_num,
                "name": name,
                "include_aa": True,
                "include_dp": True,
                "include_tp": True,
                "include_4p": True,
                "include_5p": True,
                "dipeptide_codes": all_dp,
                "tripeptide_codes": all_tp,
                "tetrapeptide_codes": all_4p,
                "pentapeptide_codes": pentapeptide_codes,
                "init_checkpoint": last_4p_ckpt,
                "run": run if num_runs > 1 else None,
            })

        step_num += 1

    return steps


# ---------------------------------------------------------------------------
# Phase 1: Encoder training
# ---------------------------------------------------------------------------
def run_encoder_training(step):
    """Train encoder for one step via subprocess call to train_autoencoder.py."""
    name = step["name"]

    ckpt_dir = CHECKPOINT_BASE / name

    # Skip if already completed
    if (ckpt_dir / "full_model.pt").exists():
        print(f"  [SKIP] {name}: checkpoint already exists")
        return

    cmd = [
        sys.executable, "scripts/train_autoencoder.py",
        "--data-root", DATA_ROOT,
        "--amino-acids", *ALL_AA_26,
        "--latent-dim", "32",
        "--encoder-dims", "512", "256",
        "--dropout", "0.3",
        "--activation", "relu",
        "--epochs", "200",
        "--batch-size", "64",
        "--learning-rate", "0.001",
        "--weight-decay", "1e-4",
        "--early-stopping-patience", "20",
        "--samples-per-sequence", "100",
        "--min-spectra", str(MIN_SPECTRA),
        "--output-dir", str(CHECKPOINT_BASE),
        "--experiment-name", name,
        "--seed", str(SEED),
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

    # Tetrapeptide include/exclude
    if step.get("include_4p"):
        cmd.append("--include-tetrapeptides")
    else:
        cmd.append("--no-include-tetrapeptides")

    if step.get("include_5p"):
        cmd.append("--include-pentapeptides")
    else:
        cmd.append("--no-include-pentapeptides")

    # Tetrapeptide code filtering
    if step.get("tetrapeptide_codes") is not None:
        cmd.extend(["--tetrapeptide-codes", *step["tetrapeptide_codes"]])

    # Pentapeptide code filtering
    if step.get("pentapeptide_codes") is not None:
        cmd.extend(["--pentapeptide-codes", *step["pentapeptide_codes"]])

    # Init checkpoint
    if step["init_checkpoint"] is not None:
        cmd.extend(["--init-checkpoint", step["init_checkpoint"]])

    print(f"  Running: {' '.join(cmd[:6])}...")
    t0 = time.time()
    result = run_logged_subprocess(cmd)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  [FAIL] {name} failed with return code {result.returncode}")
        sys.exit(1)

    print(f"  [DONE] {name} ({elapsed:.1f}s)")


# ---------------------------------------------------------------------------
# Phase 1: Post-training evaluation (MSE + PCA/t-SNE)
# ---------------------------------------------------------------------------
def compute_subset_mse(model, spectra_data, device):
    """Compute reconstruction MSE for AA, DP, TP, 4P, 5P subsets.

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
        ("tetrapeptides", spectra_data.tetrapeptides),
        ("pentapeptides", spectra_data.pentapeptides),
    ]:
        per_code = {}
        all_mse = []

        for code, spectra in sorted(data_dict.items()):
            # Filter to codes composed of our 6 amino acids (or PTM in aminoacids)
            if category == "aminoacids":
                if code not in ALL_AA_26:
                    continue
            else:
                if not all(c in AMINO_ACIDS_6 for c in code):
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

    Colours: AA=blue, DP=green, TP=red, 4P=purple, 5P=orange.
    Markers: distinct for each of the 6 amino acids; generic for DP/TP/4P/5P.
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
    all_types = []       # 'AA', 'DP', 'TP', '4P', '5P'

    rng = np.random.default_rng(42)

    for category, data_dict, seq_type in [
        ("aminoacids", spectra_data.aminoacids, "AA"),
        ("dipeptides", spectra_data.dipeptides, "DP"),
        ("tripeptides", spectra_data.tripeptides, "TP"),
        ("tetrapeptides", spectra_data.tetrapeptides, "4P"),
        ("pentapeptides", spectra_data.pentapeptides, "5P"),
    ]:
        for code, spectra in sorted(data_dict.items()):
            if category == "aminoacids":
                if code not in ALL_AA_26:
                    continue
            else:
                if not all(c in AMINO_ACIDS_6 for c in code):
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

    type_colors = {"AA": "#1f77b4", "DP": "#2ca02c", "TP": "#d62728", "4P": "#9467bd", "5P": "#ff7f0e"}
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
    for seq_type in ["AA", "DP", "TP", "4P", "5P"]:
        mask = types == seq_type
        if not mask.any():
            continue
        color = type_colors[seq_type]

        if seq_type == "AA":
            # Distinct marker per amino acid (only for AMINO_ACIDS_6)
            for aa in AMINO_ACIDS_6:
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
            # Plot remaining AAs (standard + PTM not in AMINO_ACIDS_6) with generic marker
            other_mask = mask & ~np.isin(labels, AMINO_ACIDS_6)
            if other_mask.any():
                lbl = "AA:other" if "AA:other" not in plotted else None
                plotted.add("AA:other")
                ax.scatter(
                    Z2d[other_mask, 0], Z2d[other_mask, 1],
                    c=color, marker="*",
                    s=15, alpha=0.3, label=lbl,
                )
        else:
            lbl = seq_type if seq_type not in plotted else None
            plotted.add(seq_type)
            marker_map = {"DP": "x", "TP": "+", "4P": "1", "5P": "2"}
            marker = marker_map.get(seq_type, ".")
            ax.scatter(
                Z2d[mask, 0], Z2d[mask, 1],
                c=color, marker=marker,
                s=15, alpha=0.3, label=lbl,
            )


def run_encoder_evaluation(step):
    """Evaluate encoder: MSE report + PCA/t-SNE plots."""
    name = step["name"]

    import torch
    from primarymagic.data import SpectraDataset
    from primarymagic.models import MultiLabelAutoencoderConfig, MultiLabelRegularizedAutoencoder
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
    spectra_data = SpectraDataset(DATA_ROOT, min_spectra=MIN_SPECTRA)

    # MSE report
    result_dir.mkdir(parents=True, exist_ok=True)
    mse_report = compute_subset_mse(model, spectra_data, device)
    with open(result_dir / "mse_report.json", "w") as f:
        json.dump(mse_report, f, indent=2)

    aa_mse = mse_report['aminoacids']['aggregate_mse']
    dp_mse = mse_report['dipeptides']['aggregate_mse']
    tp_mse = mse_report['tripeptides']['aggregate_mse']
    fp_mse = mse_report['tetrapeptides']['aggregate_mse']
    pp_mse = mse_report['pentapeptides']['aggregate_mse']
    print(f"  MSE: AA={aa_mse:.6f}, "
          f"DP={f'{dp_mse:.6f}' if dp_mse else '-'}, "
          f"TP={f'{tp_mse:.6f}' if tp_mse else '-'}, "
          f"4P={f'{fp_mse:.6f}' if fp_mse else '-'}, "
          f"5P={f'{pp_mse:.6f}' if pp_mse else '-'}")

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
        print(f"  [SKIP] {seq_name}: encoder checkpoint not found ({encoder_ckpt})")
        return

    if (ckpt_dir / "differential_classifier.pt").exists():
        if OVERWRITE_SEQ:
            print(f"  [OVERWRITE] {seq_name}: removing existing checkpoint")
            (ckpt_dir / "differential_classifier.pt").unlink()
        else:
            print(f"  [SKIP] {seq_name}: checkpoint already exists")
            return

    cmd = [
        sys.executable, "scripts/train_sequencer.py",
        "--encoder-checkpoint", str(encoder_ckpt),
        "--data-root", DATA_ROOT,
        "--amino-acids", *AMINO_ACIDS_6,
        "--include-dipeptide-pairs",
        "--no-include-tripeptide-pairs",
        "--epochs", "200",
        "--batch-size", "64",
        "--learning-rate", "0.001",
        "--weight-decay", "1e-4",
        "--early-stopping-patience", "20",
        "--samples-per-pair", "100",
        "--min-spectra", str(MIN_SPECTRA),
        "--output-dir", str(CHECKPOINT_BASE),
        "--experiment-name", seq_name,
        "--seed", str(SEED),
    ]

    print(f"  Running sequencer training for {name}...")
    t0 = time.time()
    result = run_logged_subprocess(cmd)
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
        if (result_dir / "metrics.json").exists() and not OVERWRITE_SEQ:
            print(f"  [SKIP] {seq_name} {label} eval: results already exist")
        else:
            if OVERWRITE_SEQ and (result_dir / "metrics.json").exists():
                print(f"  [OVERWRITE] {seq_name} {label}: removing existing metrics")
                (result_dir / "metrics.json").unlink()
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
        "--min-spectra", str(MIN_SPECTRA),
        "--seed", str(SEED),
        "--include-dipeptide-pairs" if include_dp else "--no-include-dipeptide-pairs",
        "--include-tripeptide-pairs" if include_tp else "--no-include-tripeptide-pairs",
        "--include-tetrapeptide-pairs" if include_tetra else "--no-include-tetrapeptide-pairs",
        "--include-pentapeptide-pairs" if include_penta else "--no-include-pentapeptide-pairs",
    ]

    print(f"  Evaluating on {label} data...")
    t0 = time.time()
    result = run_logged_subprocess(cmd)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  [FAIL] {label} eval failed with return code {result.returncode}")
        sys.exit(1)

    print(f"  [DONE] {label} eval ({elapsed:.1f}s)")


# ---------------------------------------------------------------------------
# Summary generation with mean/std across runs
# ---------------------------------------------------------------------------
def _strip_run_suffix(name):
    """Strip _runN suffix to get the base step name."""
    return re.sub(r"_run\d+$", "", name)


def generate_summary(steps):
    """Aggregate MSE and accuracy across all steps, with mean/std for multi-run steps."""
    # Collect raw results per step
    raw_entries = []
    for step in steps:
        name = step["name"]
        seq_name = f"seq_{name}"
        entry = {
            "step_num": step["step_num"],
            "name": name,
            "base_name": _strip_run_suffix(name),
        }

        # MSE report
        mse_path = RESULTS_BASE / name / "mse_report.json"
        if mse_path.exists():
            with open(mse_path) as f:
                mse = json.load(f)
            entry["mse_aminoacids"] = mse["aminoacids"]["aggregate_mse"]
            entry["mse_dipeptides"] = mse["dipeptides"]["aggregate_mse"]
            entry["mse_tripeptides"] = mse["tripeptides"]["aggregate_mse"]
            if "tetrapeptides" in mse:
                entry["mse_tetrapeptides"] = mse["tetrapeptides"]["aggregate_mse"]
            if "pentapeptides" in mse:
                entry["mse_pentapeptides"] = mse["pentapeptides"]["aggregate_mse"]

        # Eval metrics
        for eval_key, eval_dir in [
            ("dipeptide", "dipeptide_eval"),
            ("tripeptide", "tripeptide_eval"),
            ("tetrapeptide", "tetrapeptide_eval"),
            ("pentapeptide", "pentapeptide_eval"),
        ]:
            metrics_path = RESULTS_BASE / seq_name / eval_dir / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    m = json.load(f)
                entry[f"{eval_key}_accuracy"] = m["metrics"]["accuracy"]
                entry[f"{eval_key}_macro_f1"] = m["metrics"]["macro_avg"]["f1"]

        raw_entries.append(entry)

    # Group by base_name for mean/std computation
    from collections import OrderedDict
    groups = OrderedDict()
    for entry in raw_entries:
        base = entry["base_name"]
        if base not in groups:
            groups[base] = []
        groups[base].append(entry)

    metric_keys = [
        "mse_aminoacids", "mse_dipeptides", "mse_tripeptides",
        "mse_tetrapeptides", "mse_pentapeptides",
        "dipeptide_accuracy", "dipeptide_macro_f1",
        "tripeptide_accuracy", "tripeptide_macro_f1",
        "tetrapeptide_accuracy", "tetrapeptide_macro_f1",
        "pentapeptide_accuracy", "pentapeptide_macro_f1",
    ]

    summary_entries = []
    for base_name, group in groups.items():
        agg = {
            "step_num": group[0]["step_num"],
            "name": base_name,
            "num_runs": len(group),
        }

        for key in metric_keys:
            values = [e[key] for e in group if key in e and e[key] is not None]
            if values:
                agg[f"{key}_mean"] = float(np.mean(values))
                agg[f"{key}_std"] = float(np.std(values))

        summary_entries.append(agg)

    summary = {"steps": summary_entries, "raw": raw_entries}

    summary_path = RESULTS_BASE / "summary.json"
    RESULTS_BASE.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # Print summary table
    print("\n" + "=" * 160)
    print(f"{'Step':<28} {'Runs':>4} {'MSE(AA)':>14} {'MSE(DP)':>14} {'MSE(TP)':>14} {'MSE(4P)':>14} {'MSE(5P)':>14} "
          f"{'DP Acc':>14} {'TP Acc':>14} {'4P Acc':>14} {'5P Acc':>14}")
    print("-" * 160)

    for e in summary_entries:
        def _fmt(key):
            mean = e.get(f"{key}_mean")
            std = e.get(f"{key}_std")
            if mean is None:
                return "-".rjust(14)
            if e["num_runs"] == 1 or std is None or std == 0.0:
                return f"{mean:.4f}".rjust(14)
            return f"{mean:.4f}+/-{std:.4f}".rjust(14)

        print(f"{e['name']:<28} {e['num_runs']:>4} "
              f"{_fmt('mse_aminoacids')} "
              f"{_fmt('mse_dipeptides')} "
              f"{_fmt('mse_tripeptides')} "
              f"{_fmt('mse_tetrapeptides')} "
              f"{_fmt('mse_pentapeptides')} "
              f"{_fmt('dipeptide_accuracy')} "
              f"{_fmt('tripeptide_accuracy')} "
              f"{_fmt('tetrapeptide_accuracy')} "
              f"{_fmt('pentapeptide_accuracy')}")
    print("=" * 160)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run magic experiment pipeline (26-class encoder, 6-class sequencer)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Experiment name (e.g., magic1). Sets checkpoints/<name> and results/<name>.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=DATA_ROOT,
        help="Root directory of preprocessed data (default: data/custom/processed/magic).",
    )
    parser.add_argument(
        "--checkpoints-root",
        type=str,
        default="checkpoints",
        help="Root directory where checkpoints/<name>/ lives. Default: checkpoints (cwd-relative).",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="results",
        help="Root directory where results/<name>/ + log files go. Default: results (cwd-relative).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed passed to encoder + sequencer training (default: 42).",
    )
    parser.add_argument(
        "--overwrite-seq",
        action="store_true",
        help="Force-retrain sequencer + redo eval for active steps even if checkpoints exist. "
             "Encoder checkpoints are still skipped if present.",
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
        help="Start from this step number",
    )
    parser.add_argument(
        "--end-step",
        type=int,
        default=None,
        help="End at this step number (inclusive). Default: all steps.",
    )
    parser.add_argument(
        "--run",
        type=int,
        default=None,
        help="Run only this run number (1-5). Enables parallelism across runs.",
    )
    return parser.parse_args()


def main():
    global CHECKPOINT_BASE, RESULTS_BASE, LOG_DIR, DATA_ROOT, SEED, OVERWRITE_SEQ

    args = parse_args()

    # Derive output paths from experiment name; override data root if given
    CHECKPOINT_BASE = Path(args.checkpoints_root) / args.name
    RESULTS_BASE = Path(args.results_root) / args.name
    LOG_DIR = RESULTS_BASE
    DATA_ROOT = args.data_root
    SEED = args.seed
    OVERWRITE_SEQ = args.overwrite_seq

    setup_logging()

    print("=" * 70)
    print(f"Magic Experiment: {args.name}")
    print("=" * 70)
    print(f"Encoder amino acids: {len(ALL_AA_26)} (20 standard + 6 PTM)")
    print(f"Sequencer amino acids: {AMINO_ACIDS_6}")
    print(f"Data root: {DATA_ROOT}")
    print(f"Checkpoints: {CHECKPOINT_BASE}")
    print(f"Results: {RESULTS_BASE}")
    print(f"Min spectra: {MIN_SPECTRA}")
    print(f"Runs per step: {NUM_RUNS}")
    print(f"Phase: {args.phase or 'all'}")
    print(f"Steps: {args.start_step} - {args.end_step or 'end'}")
    if args.run:
        print(f"Run filter: {args.run}")
    print()

    # Discover valid sequences from data
    all_dp, all_tp, all_4p, all_5p = discover_sequences()

    # Build step definitions
    steps = _build_steps(all_dp, all_tp, all_4p, all_5p)

    max_step = max(s["step_num"] for s in steps)
    end_step = args.end_step if args.end_step is not None else max_step

    # Filter steps by range and run
    active_steps = [
        s for s in steps
        if args.start_step <= s["step_num"] <= end_step
    ]

    if args.run is not None:
        active_steps = [
            s for s in active_steps
            if s["run"] is None or s["run"] == args.run
        ]

    if not active_steps:
        print("No steps to run in the specified range.")
        return

    print(f"Active steps: {len(active_steps)}")
    for s in active_steps:
        print(f"  Step {s['step_num']}: {s['name']}")
    print()

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

    # Generate summary (always use all steps)
    print("\n" + "=" * 70)
    print("Generating Summary")
    print("=" * 70)
    generate_summary(steps)

    total_elapsed = time.time() - total_t0
    print(f"\nTotal elapsed time: {total_elapsed:.1f}s ({total_elapsed/3600:.2f}h)")
    print("Experiment complete!")
    teardown_logging()


if __name__ == "__main__":
    main()
