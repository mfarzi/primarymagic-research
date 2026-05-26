#!/usr/bin/env python
"""Evaluate a trained DifferentialClassifierWithPretrainedEncoder on test data.

Evaluates a trained sequencer model and reports confusion matrix,
classification metrics, and per-class performance.

Example (evaluate on tripeptides):
    python scripts/evaluate_sequencer.py \
        --model-checkpoint checkpoints/sequencer/sequencer_dipeptides/differential_classifier.pt \
        --data-root data/processed/orpl \
        --no-include-dipeptide-pairs \
        --include-tripeptide-pairs \
        --output-dir results/tripeptide_eval

Example (evaluate on dipeptides):
    python scripts/evaluate_sequencer.py \
        --model-checkpoint checkpoints/sequencer/sequencer_dipeptides/differential_classifier.pt \
        --data-root data/processed/orpl \
        --include-dipeptide-pairs \
        --no-include-tripeptide-pairs \
        --max-samples-per-class 1000
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from primarymagic.data import SpectraDataset, SequenceDataset
from primarymagic.models.autoencoder import (
    DifferentialClassifierWithPretrainedEncoder,
    SpectralEncoder,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate DifferentialClassifierWithPretrainedEncoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (differential_classifier.pt)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Path to processed data directory (e.g., data/processed/orpl)",
    )
    parser.add_argument(
        "--min-spectra",
        type=int,
        default=1,
        help="Minimum number of spectra required to include a sample",
    )

    # Data filtering arguments
    parser.add_argument(
        "--include-dipeptide-pairs",
        action="store_true",
        default=False,
        help="Include (XY, X) -> Y dipeptide pairs",
    )
    parser.add_argument(
        "--no-include-dipeptide-pairs",
        action="store_false",
        dest="include_dipeptide_pairs",
        help="Exclude dipeptide pairs",
    )
    parser.add_argument(
        "--include-tripeptide-pairs",
        action="store_true",
        default=True,
        help="Include (XYZ, XY) -> Z tripeptide pairs",
    )
    parser.add_argument(
        "--no-include-tripeptide-pairs",
        action="store_false",
        dest="include_tripeptide_pairs",
        help="Exclude tripeptide pairs",
    )
    parser.add_argument(
        "--include-tetrapeptide-pairs",
        action="store_true",
        default=False,
        help="Include (XYZW, XYZ) -> W tetrapeptide pairs",
    )
    parser.add_argument(
        "--no-include-tetrapeptide-pairs",
        action="store_false",
        dest="include_tetrapeptide_pairs",
        help="Exclude tetrapeptide pairs",
    )
    parser.add_argument(
        "--include-pentapeptide-pairs",
        action="store_true",
        default=False,
        help="Include (XYZWV, XYZW) -> V pentapeptide pairs",
    )
    parser.add_argument(
        "--no-include-pentapeptide-pairs",
        action="store_false",
        dest="include_pentapeptide_pairs",
        help="Exclude pentapeptide pairs",
    )
    parser.add_argument(
        "--include-prefixes",
        type=str,
        nargs="*",
        default=None,
        help="Only include pairs with these prefixes (e.g., A D)",
    )
    parser.add_argument(
        "--exclude-prefixes",
        type=str,
        nargs="*",
        default=None,
        help="Exclude pairs with these prefixes",
    )
    parser.add_argument(
        "--tripeptide-codes",
        type=str,
        nargs="*",
        default=None,
        help="Only include tripeptide pairs with these exact codes (e.g., AFD GDA). If not specified, all tripeptides are included.",
    )
    parser.add_argument(
        "--tetrapeptide-codes",
        type=str,
        nargs="*",
        default=None,
        help="Only include tetrapeptide pairs with these exact codes (e.g., DSAF FARD). If not specified, all tetrapeptides are included.",
    )
    parser.add_argument(
        "--pentapeptide-codes",
        type=str,
        nargs="*",
        default=None,
        help="Only include pentapeptide pairs with these exact codes (e.g., DSARF FARDG). If not specified, all pentapeptides are included.",
    )
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=None,
        help="Maximum samples per target class",
    )
    parser.add_argument(
        "--samples-per-pair",
        type=int,
        default=None,
        help="Number of random spectra pairs per sequence pair (None = all)",
    )

    # Evaluation arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results (optional)",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        default=False,
        help="Save all predictions to file",
    )

    return parser.parse_args()


def load_model(checkpoint_path: str, device: str = "cpu") -> Tuple[DifferentialClassifierWithPretrainedEncoder, Dict]:
    """Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint.
        device: Device to load model on.

    Returns:
        Tuple of (model, config_dict).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get configs
    encoder_config = checkpoint.get("encoder_config", {})
    classifier_config = checkpoint.get("classifier_config", {})
    amino_acid_codes = checkpoint.get("amino_acid_codes", ["A", "D", "F", "G", "R", "S"])

    # Create encoder
    encoder = SpectralEncoder(
        input_dim=encoder_config.get("input_dim", 1023),
        hidden_dims=tuple(encoder_config.get("hidden_dims", (512, 256))),
        latent_dim=encoder_config.get("latent_dim", 48),
        dropout=encoder_config.get("dropout", 0.3),
        activation=encoder_config.get("activation", "relu"),
    )

    # Create model
    model = DifferentialClassifierWithPretrainedEncoder(
        encoder=encoder,
        latent_dim=classifier_config.get("latent_dim", encoder_config.get("latent_dim", 48)),
        num_classes=classifier_config.get("num_classes", len(amino_acid_codes)),
        dropout=classifier_config.get("dropout", 0.3),
        activation=classifier_config.get("activation", "relu"),
    )

    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])

    config = {
        "encoder_config": encoder_config,
        "classifier_config": classifier_config,
        "amino_acid_codes": amino_acid_codes,
    }

    return model, config


def collate_fn(batch):
    """Collate function for DataLoader."""
    s_xy = torch.stack([b[0] for b in batch])
    s_x = torch.stack([b[1] for b in batch])
    labels = torch.stack([b[2] for b in batch])
    return s_xy, s_x, labels


def evaluate_model(
    model: DifferentialClassifierWithPretrainedEncoder,
    dataloader: DataLoader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run model evaluation and collect predictions.

    Args:
        model: Trained model.
        dataloader: DataLoader for evaluation data.
        device: Device to run on.

    Returns:
        Tuple of (all_targets, all_predictions) as numpy arrays.
    """
    model.eval()
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch_idx, (s_xy, s_x, targets) in enumerate(dataloader):
            s_xy = s_xy.to(device)
            s_x = s_x.to(device)

            outputs = model(s_xy, s_x)
            _, predicted = outputs.max(1)

            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            # Progress indicator
            if (batch_idx + 1) % 100 == 0:
                print(f"  Processed {(batch_idx + 1) * dataloader.batch_size} samples...")

    return np.array(all_targets), np.array(all_preds)


def compute_confusion_matrix(
    targets: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Compute confusion matrix.

    Args:
        targets: Ground truth labels.
        predictions: Model predictions.
        num_classes: Number of classes.

    Returns:
        Confusion matrix of shape (num_classes, num_classes).
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets, predictions):
        cm[t, p] += 1
    return cm


def compute_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    class_names: List[str],
) -> Dict:
    """Compute classification metrics.

    Args:
        targets: Ground truth labels.
        predictions: Model predictions.
        class_names: List of class names.

    Returns:
        Dictionary with metrics.
    """
    num_classes = len(class_names)
    cm = compute_confusion_matrix(targets, predictions, num_classes)

    # Overall accuracy
    accuracy = np.sum(targets == predictions) / len(targets) * 100

    # Per-class metrics
    per_class = {}
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = cm[i, :].sum()

        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(support),
        }

    # Macro averages
    macro_precision = np.mean([m["precision"] for m in per_class.values()])
    macro_recall = np.mean([m["recall"] for m in per_class.values()])
    macro_f1 = np.mean([m["f1"] for m in per_class.values()])

    # Weighted averages
    total_support = sum(m["support"] for m in per_class.values())
    weighted_precision = sum(m["precision"] * m["support"] for m in per_class.values()) / total_support
    weighted_recall = sum(m["recall"] * m["support"] for m in per_class.values()) / total_support
    weighted_f1 = sum(m["f1"] * m["support"] for m in per_class.values()) / total_support

    return {
        "accuracy": accuracy,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "macro_avg": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1,
        },
        "weighted_avg": {
            "precision": weighted_precision,
            "recall": weighted_recall,
            "f1": weighted_f1,
        },
    }


def print_confusion_matrix(cm: np.ndarray, class_names: List[str]) -> None:
    """Print formatted confusion matrix.

    Args:
        cm: Confusion matrix.
        class_names: List of class names.
    """
    # Header
    header = "      " + "  ".join(f"{name:>6}" for name in class_names)
    print(header)
    print("      " + "-" * (len(class_names) * 8))

    # Rows
    for i, name in enumerate(class_names):
        row = f"{name:>5} |" + "  ".join(f"{cm[i, j]:>6}" for j in range(len(class_names)))
        print(row)


def print_classification_report(metrics: Dict, class_names: List[str]) -> None:
    """Print formatted classification report.

    Args:
        metrics: Metrics dictionary from compute_metrics.
        class_names: List of class names.
    """
    print(f"\n{'Class':<10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 52)

    for name in class_names:
        m = metrics["per_class"][name]
        print(f"{name:<10} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['support']:>10}")

    print("-" * 52)
    m = metrics["macro_avg"]
    print(f"{'macro avg':<10} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")
    m = metrics["weighted_avg"]
    total = sum(metrics["per_class"][n]["support"] for n in class_names)
    print(f"{'weighted':<10} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {total:>10}")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Path,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = "Blues",
    accuracy: Optional[float] = None,
) -> None:
    """Plot and save confusion matrix with both percentages and absolute numbers.

    Args:
        cm: Confusion matrix of shape (num_classes, num_classes).
        class_names: List of class names.
        output_path: Path to save the figure.
        title: Title for the plot.
        figsize: Figure size as (width, height).
        cmap: Colormap name.
        accuracy: Overall accuracy to display in title (optional).
    """
    # Compute percentages (row-normalized for recall/per-class accuracy)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_percent = np.zeros_like(cm, dtype=float)
    nonzero_rows = row_sums.flatten() > 0
    cm_percent[nonzero_rows] = cm[nonzero_rows] / row_sums[nonzero_rows] * 100

    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap using percentages for color intensity
    im = ax.imshow(cm_percent, interpolation='nearest', cmap=cmap, vmin=0, vmax=100)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Percentage (%)', rotation=-90, va="bottom", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=16)
    ax.set_yticklabels(class_names, fontsize=16)

    # Rotate x labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations with both percentage and count
    thresh = 50  # Threshold for text color
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            pct = cm_percent[i, j]
            count = cm[i, j]

            # Choose text color based on background
            color = "white" if pct > thresh else "black"

            # Format text: percentage on top, count below
            text = f"{pct:.1f}%\n({count:,})"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=16, fontweight='bold')

    # Labels and title
    ax.set_xlabel('Predicted Label', fontsize=16)
    ax.set_ylabel('True Label', fontsize=16)

    if accuracy is not None:
        title = f"{title}\nOverall Accuracy: {accuracy:.2f}%"
    ax.set_title(title, fontsize=18, fontweight='bold')

    # Adjust layout
    plt.tight_layout()

    # Save figure
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_normalized_confusion_matrices(
    cm: np.ndarray,
    class_names: List[str],
    output_dir: Path,
    accuracy: Optional[float] = None,
) -> None:
    """Plot and save both row-normalized and column-normalized confusion matrices.

    Args:
        cm: Confusion matrix of shape (num_classes, num_classes).
        class_names: List of class names.
        output_dir: Directory to save figures.
        accuracy: Overall accuracy to display in title (optional).
    """
    # Plot 1: Row-normalized (shows recall/sensitivity per class)
    plot_confusion_matrix(
        cm=cm,
        class_names=class_names,
        output_path=output_dir / "confusion_matrix_recall.png",
        title="Confusion Matrix (Row-Normalized: Recall)",
        accuracy=accuracy,
    )

    # Plot 2: Column-normalized (shows precision per class)
    col_sums = cm.sum(axis=0, keepdims=True)
    cm_col_percent = np.zeros_like(cm, dtype=float)
    nonzero_cols = col_sums.flatten() > 0
    cm_col_percent[:, nonzero_cols] = cm[:, nonzero_cols] / col_sums[:, nonzero_cols] * 100

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_col_percent, interpolation='nearest', cmap='Blues', vmin=0, vmax=100)

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Percentage (%)', rotation=-90, va="bottom", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=16)
    ax.set_yticklabels(class_names, fontsize=16)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = 50
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            pct = cm_col_percent[i, j]
            count = cm[i, j]
            color = "white" if pct > thresh else "black"
            text = f"{pct:.1f}%\n({count:,})"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=16, fontweight='bold')

    ax.set_xlabel('Predicted Label', fontsize=16)
    ax.set_ylabel('True Label', fontsize=16)
    title = "Confusion Matrix (Column-Normalized: Precision)"
    if accuracy is not None:
        title = f"{title}\nOverall Accuracy: {accuracy:.2f}%"
    ax.set_title(title, fontsize=18, fontweight='bold')

    plt.tight_layout()
    fig.savefig(output_dir / "confusion_matrix_precision.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Plot 3: Absolute counts only (no normalization)
    fig, ax = plt.subplots(figsize=(12, 10))

    # Use log scale for color if counts vary widely
    cm_display = cm.astype(float)
    im = ax.imshow(cm_display, interpolation='nearest', cmap='Blues')

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Count', rotation=-90, va="bottom", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=16)
    ax.set_yticklabels(class_names, fontsize=16)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Determine threshold based on data range
    thresh = cm.max() / 2
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            count = cm[i, j]
            color = "white" if count > thresh else "black"
            ax.text(j, i, f"{count:,}", ha="center", va="center", color=color, fontsize=16, fontweight='bold')

    ax.set_xlabel('Predicted Label', fontsize=16)
    ax.set_ylabel('True Label', fontsize=16)
    title = "Confusion Matrix (Absolute Counts)"
    if accuracy is not None:
        title = f"{title}\nOverall Accuracy: {accuracy:.2f}%"
    ax.set_title(title, fontsize=18, fontweight='bold')

    plt.tight_layout()
    fig.savefig(output_dir / "confusion_matrix_counts.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    """Main evaluation function."""
    args = parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 70)
    print("Evaluating DifferentialClassifierWithPretrainedEncoder")
    print("=" * 70)

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Load model
    print(f"\nLoading model from: {args.model_checkpoint}")
    model, config = load_model(args.model_checkpoint, device)
    model = model.to(device)
    model.eval()

    amino_acid_codes = config["amino_acid_codes"]
    print(f"Amino acid codes: {amino_acid_codes}")

    # Load spectra data
    print(f"\nLoading data from: {args.data_root}")
    spectra_data = SpectraDataset(args.data_root, min_spectra=args.min_spectra)
    print(f"Loaded: {spectra_data}")

    # Create evaluation dataset
    print(f"\nCreating evaluation dataset...")
    print(f"  Include dipeptide pairs: {args.include_dipeptide_pairs}")
    print(f"  Include tripeptide pairs: {args.include_tripeptide_pairs}")
    print(f"  Include tetrapeptide pairs: {args.include_tetrapeptide_pairs}")
    print(f"  Include pentapeptide pairs: {args.include_pentapeptide_pairs}")

    eval_dataset = SequenceDataset(
        spectra_data,
        include_dipeptide_pairs=args.include_dipeptide_pairs,
        include_tripeptide_pairs=args.include_tripeptide_pairs,
        include_tetrapeptide_pairs=args.include_tetrapeptide_pairs,
        include_pentapeptide_pairs=args.include_pentapeptide_pairs,
        include_prefixes=args.include_prefixes,
        exclude_prefixes=args.exclude_prefixes,
        include_tripeptide_codes=args.tripeptide_codes,
        include_tetrapeptide_codes=args.tetrapeptide_codes,
        include_pentapeptide_codes=args.pentapeptide_codes,
        max_samples_per_class=args.max_samples_per_class,
        samples_per_pair=args.samples_per_pair,
        amino_acid_codes=amino_acid_codes,
        seed=args.seed,
    )

    print(f"\nEvaluation dataset:")
    print(f"  Total samples: {len(eval_dataset)}")
    print(f"  Class distribution: {eval_dataset.get_class_distribution()}")

    if len(eval_dataset) == 0:
        print("\nError: No samples found. Check your data filtering options.")
        sys.exit(1)

    # Create DataLoader
    dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device == "cuda" else False,
        collate_fn=collate_fn,
    )

    # Run evaluation
    print("\n" + "-" * 70)
    print("Running evaluation...")
    print("-" * 70)

    targets, predictions = evaluate_model(model, dataloader, device)

    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(targets, predictions, amino_acid_codes)

    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print(f"\nOverall Accuracy: {metrics['accuracy']:.2f}%")

    print("\n" + "-" * 70)
    print("CONFUSION MATRIX")
    print("-" * 70)
    print("\nPredicted:")
    cm = np.array(metrics["confusion_matrix"])
    print_confusion_matrix(cm, amino_acid_codes)

    print("\n" + "-" * 70)
    print("CLASSIFICATION REPORT")
    print("-" * 70)
    print_classification_report(metrics, amino_acid_codes)

    # Save results if output directory specified
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics JSON
        metrics_path = output_dir / "metrics.json"
        results = {
            "model_checkpoint": args.model_checkpoint,
            "data_root": args.data_root,
            "include_dipeptide_pairs": args.include_dipeptide_pairs,
            "include_tripeptide_pairs": args.include_tripeptide_pairs,
            "include_tetrapeptide_pairs": args.include_tetrapeptide_pairs,
            "include_pentapeptide_pairs": args.include_pentapeptide_pairs,
            "include_prefixes": args.include_prefixes,
            "exclude_prefixes": args.exclude_prefixes,
            "max_samples_per_class": args.max_samples_per_class,
            "samples_per_pair": args.samples_per_pair,
            "total_samples": len(eval_dataset),
            "class_distribution": eval_dataset.get_class_distribution(),
            "metrics": metrics,
        }
        with open(metrics_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved metrics to: {metrics_path}")

        # Save predictions if requested
        if args.save_predictions:
            preds_path = output_dir / "predictions.npz"
            np.savez(
                preds_path,
                targets=targets,
                predictions=predictions,
                class_names=amino_acid_codes,
            )
            print(f"Saved predictions to: {preds_path}")

        # Save confusion matrix as CSV
        cm_path = output_dir / "confusion_matrix.csv"
        with open(cm_path, "w") as f:
            f.write("," + ",".join(amino_acid_codes) + "\n")
            for i, name in enumerate(amino_acid_codes):
                f.write(name + "," + ",".join(str(cm[i, j]) for j in range(len(amino_acid_codes))) + "\n")
        print(f"Saved confusion matrix to: {cm_path}")

        # Plot and save confusion matrix figures
        print("\nGenerating confusion matrix figures...")
        plot_normalized_confusion_matrices(
            cm=cm,
            class_names=amino_acid_codes,
            output_dir=output_dir,
            accuracy=metrics["accuracy"],
        )
        print(f"Saved confusion_matrix_recall.png (row-normalized)")
        print(f"Saved confusion_matrix_precision.png (column-normalized)")
        print(f"Saved confusion_matrix_counts.png (absolute counts)")

    print("\n" + "=" * 70)
    print("Evaluation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
