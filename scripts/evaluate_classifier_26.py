#!/usr/bin/env python
"""Evaluate the 26-class single amino acid classifier (20 canonical + 6 PTM).

Loads the MultiLabelRegularizedAutoencoder checkpoint, draws 100 random samples
per class from SpectraDataset, runs inference, and produces an Excel report
with per-class precision, recall, F1, and a confusion matrix sheet.

Usage:
    python scripts/evaluate_classifier_26.py
    python scripts/evaluate_classifier_26.py --checkpoint path/to/best_multilabel_autoencoder.pt
    python scripts/evaluate_classifier_26.py --samples-per-class 200
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

# Allow importing spectra from the source tree
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from primarymagic.data import SpectraDataset
from primarymagic.models.autoencoder import (
    MultiLabelAutoencoderConfig,
    MultiLabelRegularizedAutoencoder,
)

STANDARD_AA = list("ACDEFGHIKLMNPQRSTVWY")
PTM_AA = ["acetyl-K", "acetyl-S", "hydroxy-P", "phos-S", "phos-T", "phos-Y"]
ALL_AA_26 = STANDARD_AA + PTM_AA

DATA_ROOT = "data/processed/primary_magic"
MIN_SPECTRA = 40
DEFAULT_CHECKPOINT = (
    "checkpoints/decoupled_v1/step31_5p14/best_multilabel_autoencoder.pt"
)
DEFAULT_OUTPUT = "results/classifier_26/classification_report.xlsx"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate 26-class amino acid classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
        help="Path to best_multilabel_autoencoder.pt",
    )
    parser.add_argument(
        "--data-root", type=str, default=DATA_ROOT,
        help="Path to processed spectra data",
    )
    parser.add_argument(
        "--samples-per-class", type=int, default=100,
        help="Number of random samples to draw per class",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT,
        help="Output Excel file path",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # ── Load model ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    config = MultiLabelAutoencoderConfig.from_dict(checkpoint["model_config"])
    model = MultiLabelRegularizedAutoencoder(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    class_names = (
        checkpoint.get("amino_acid_codes")
        or checkpoint["model_config"].get("amino_acid_codes")
        or ALL_AA_26
    )
    print(f"Model loaded: {len(class_names)} classes on {device}")
    print(f"Classes: {class_names}")

    # ── Load data ──
    spectra_data = SpectraDataset(args.data_root, min_spectra=MIN_SPECTRA)
    aa_dict = spectra_data.aminoacids  # code -> (n, 1023) ndarray

    # ── Build evaluation set ──
    all_spectra = []
    all_true = []
    skipped = []

    for code in class_names:
        if code not in aa_dict:
            skipped.append(code)
            continue

        spectra = aa_dict[code]
        n = min(args.samples_per_class, len(spectra))
        idx = rng.choice(len(spectra), size=n, replace=False)
        all_spectra.append(spectra[idx])
        all_true.extend([code] * n)

    if skipped:
        print(f"WARNING: Skipped classes not in dataset: {skipped}")

    X = np.concatenate(all_spectra, axis=0)
    y_true = np.array(all_true)
    print(f"Evaluation set: {len(X)} samples across {len(class_names) - len(skipped)} classes")

    # ── Inference ──
    y_pred = []
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i : i + batch_size], dtype=torch.float32).to(device)
            outputs = model(batch)
            probs = torch.sigmoid(outputs["logits"])
            pred_idx = probs.argmax(dim=1).cpu().numpy()
            y_pred.extend(class_names[j] for j in pred_idx)

    y_pred = np.array(y_pred)

    # ── Metrics ──
    present_classes = [c for c in class_names if c not in skipped]
    report = classification_report(
        y_true, y_pred, labels=present_classes, output_dict=True, zero_division=0,
    )

    accuracy = (y_true == y_pred).mean()
    print(f"\nOverall accuracy: {accuracy:.4f}")

    # ── Write Excel ──
    import pandas as pd

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sheet 1: Per-class metrics
    rows = []
    for code in present_classes:
        m = report[code]
        rows.append({
            "Amino Acid": code,
            "Precision": round(m["precision"], 4),
            "Recall": round(m["recall"], 4),
            "F1-Score": round(m["f1-score"], 4),
            "Support": int(m["support"]),
        })

    # Macro / weighted averages
    for avg_key, label in [("macro avg", "Macro Avg"), ("weighted avg", "Weighted Avg")]:
        m = report[avg_key]
        rows.append({
            "Amino Acid": label,
            "Precision": round(m["precision"], 4),
            "Recall": round(m["recall"], 4),
            "F1-Score": round(m["f1-score"], 4),
            "Support": int(m["support"]),
        })
    rows.append({
        "Amino Acid": "Accuracy",
        "Precision": "",
        "Recall": "",
        "F1-Score": round(accuracy, 4),
        "Support": len(y_true),
    })

    df_metrics = pd.DataFrame(rows)

    # Sheet 2: Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=present_classes)
    df_cm = pd.DataFrame(cm, index=present_classes, columns=present_classes)
    df_cm.index.name = "True \\ Predicted"

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_metrics.to_excel(writer, sheet_name="Classification Report", index=False)
        df_cm.to_excel(writer, sheet_name="Confusion Matrix")

    print(f"\nResults saved to {output_path}")

    # Print summary table
    print(f"\n{'Class':<14} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>8}")
    print("-" * 56)
    for r in rows:
        p = r["Precision"] if r["Precision"] != "" else ""
        rc = r["Recall"] if r["Recall"] != "" else ""
        f = r["F1-Score"]
        s = r["Support"]
        print(f"{r['Amino Acid']:<14} {str(p):>10} {str(rc):>10} {str(f):>10} {s:>8}")


if __name__ == "__main__":
    main()
