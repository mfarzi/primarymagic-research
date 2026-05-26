"""Tripeptide Sequence Selection Script.

This script implements iterative tripeptide sequence selection for training,
starting from a dipeptide-only baseline. It identifies which tripeptides are
most valuable for improving model generalization.

Algorithm:
1. Train baseline model on dipeptides only
2. Evaluate baseline on all tripeptides
3. Iteratively add the tripeptide with lowest accuracy to training
4. Fine-tune from baseline and evaluate after each addition

Output:
- Baseline and iteration models saved to checkpoints
- Results CSV with accuracy trends
- Confusion matrices for each iteration
- Tripeptide selection order
"""

import argparse
import copy
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from primarymagic.models import (
    AutoencoderModelConfig,
    AutoencoderDiffClassifier,
    TrainingConfig,
    train_autoencoder_model,
)
from primarymagic.data import SpectraDataset, SequenceDataset


# Constants
CODE_TO_NAME = {
    'A': 'Alanine',
    'D': 'Aspartic acid',
    'F': 'Phenylalanine',
    'G': 'Glycine',
    'R': 'Arginine',
    'S': 'Serine'
}


def collate_fn(batch):
    """Custom collate function for paired data."""
    s_xy = torch.stack([b[0] for b in batch])
    s_x = torch.stack([b[1] for b in batch])
    labels = torch.stack([b[2] for b in batch])
    return s_xy, s_x, labels


def evaluate_model(
    model: AutoencoderDiffClassifier,
    data_loader: DataLoader,
    device: str,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Evaluate model and return accuracy and predictions.

    Args:
        model: The trained model.
        data_loader: DataLoader for evaluation data.
        device: Device to run inference on.

    Returns:
        Tuple of (accuracy, predictions, labels).
    """
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for s_xy, s_x, labels in data_loader:
            s_xy, s_x = s_xy.to(device), s_x.to(device)
            outputs = model.predict(s_xy, s_x)
            _, predicted = outputs.max(1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy = 100.0 * np.mean(np.array(all_predictions) == np.array(all_labels))
    return accuracy, np.array(all_predictions), np.array(all_labels)


def evaluate_per_tripeptide(
    model: AutoencoderDiffClassifier,
    dataset: SequenceDataset,
    device: str,
    batch_size: int = 32,
) -> Dict[str, float]:
    """Evaluate model accuracy for each tripeptide in the dataset.

    Args:
        model: The trained model.
        dataset: SequenceDataset with tripeptide data.
        device: Device to run inference on.
        batch_size: Batch size for evaluation.

    Returns:
        Dictionary mapping tripeptide codes to accuracy percentages.
    """
    model.eval()

    # Group samples by tripeptide code
    tripeptide_results: Dict[str, Dict[str, int]] = {}

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    sample_idx = 0
    with torch.no_grad():
        for s_xy, s_x, labels in data_loader:
            s_xy, s_x = s_xy.to(device), s_x.to(device)
            outputs = model.predict(s_xy, s_x)
            _, predicted = outputs.max(1)

            for i, (pred, label) in enumerate(zip(predicted.cpu().numpy(), labels.numpy())):
                # Get tripeptide name from sample_codes
                tripeptide_name = dataset.sample_codes[sample_idx]

                if tripeptide_name not in tripeptide_results:
                    tripeptide_results[tripeptide_name] = {'correct': 0, 'total': 0}

                tripeptide_results[tripeptide_name]['total'] += 1
                if pred == label:
                    tripeptide_results[tripeptide_name]['correct'] += 1

                sample_idx += 1

    # Calculate accuracy for each tripeptide
    per_tripeptide_acc = {}
    for tripeptide, results in tripeptide_results.items():
        if results['total'] > 0:
            per_tripeptide_acc[tripeptide] = 100.0 * results['correct'] / results['total']
        else:
            per_tripeptide_acc[tripeptide] = 0.0

    return per_tripeptide_acc


def _plot_single_confusion_matrix(
    cm: np.ndarray,
    cm_pct: np.ndarray,
    class_names: List[str],
    title: str,
    output_path: Path,
) -> None:
    """Plot a single confusion matrix with percentage and count annotations."""
    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(cm_pct, interpolation='nearest', cmap='Blues', vmin=0, vmax=100)

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
            pct = cm_pct[i, j]
            count = cm[i, j]
            color = "white" if pct > thresh else "black"
            text = f"{pct:.1f}%\n({count:,})"
            ax.text(j, i, text, ha="center", va="center", color=color,
                    fontsize=16, fontweight='bold')

    ax.set_xlabel('Predicted Label', fontsize=16)
    ax.set_ylabel('True Label', fontsize=16)
    ax.set_title(title, fontsize=18, fontweight='bold')

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_confusion_matrices(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: List[str],
    title: str,
    output_dir: Path,
    prefix: str,
) -> None:
    """Plot and save recall and precision confusion matrices.

    Args:
        labels: True labels.
        predictions: Predicted labels.
        class_names: List of class names.
        title: Base plot title.
        output_dir: Directory to save plots.
        prefix: Filename prefix for saved plots.
    """
    cm = confusion_matrix(labels, predictions)
    accuracy = 100.0 * np.sum(labels == predictions) / len(labels)

    # Row-normalized (recall)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_row_pct = np.zeros_like(cm, dtype=float)
    nonzero = row_sums.flatten() > 0
    cm_row_pct[nonzero] = cm[nonzero] / row_sums[nonzero] * 100

    _plot_single_confusion_matrix(
        cm, cm_row_pct, class_names,
        f"Confusion Matrix (Row-Normalized: Recall)\n{title}\nOverall Accuracy: {accuracy:.1f}%",
        output_dir / f"{prefix}_recall.png",
    )

    # Column-normalized (precision)
    col_sums = cm.sum(axis=0, keepdims=True)
    cm_col_pct = np.zeros_like(cm, dtype=float)
    nonzero = col_sums.flatten() > 0
    cm_col_pct[:, nonzero] = cm[:, nonzero] / col_sums[:, nonzero] * 100

    _plot_single_confusion_matrix(
        cm, cm_col_pct, class_names,
        f"Confusion Matrix (Column-Normalized: Precision)\n{title}\nOverall Accuracy: {accuracy:.1f}%",
        output_dir / f"{prefix}_precision.png",
    )


def train_baseline(
    spectra_data: SpectraDataset,
    model_config: AutoencoderModelConfig,
    training_config: TrainingConfig,
    samples_per_pair: int = 100,
    verbose: bool = True,
) -> Tuple[AutoencoderDiffClassifier, Dict]:
    """Train baseline model on dipeptides only.

    Args:
        spectra_data: Loaded spectra dataset.
        model_config: Model configuration.
        training_config: Training configuration.
        samples_per_pair: Number of samples per pair.
        verbose: Print progress.

    Returns:
        Tuple of (trained model, training history).
    """
    if verbose:
        print("=" * 70)
        print("Training Baseline Model (Dipeptides Only)")
        print("=" * 70)

    # Create dipeptide-only dataset
    dipeptide_dataset = SequenceDataset(
        spectra_dataset=spectra_data,
        include_dipeptide_pairs=True,
        include_tripeptide_pairs=False,
        samples_per_pair=samples_per_pair,
        seed=42,
    )

    if verbose:
        print(f"Dipeptide-only samples: {len(dipeptide_dataset)}")
        print(f"Class distribution: {dipeptide_dataset.get_class_distribution()}")

    # Train/val split
    labels = [sample[2] for sample in dipeptide_dataset.samples]
    indices = list(range(len(dipeptide_dataset)))
    train_indices, val_indices = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=42
    )

    train_subset = Subset(dipeptide_dataset, train_indices)
    val_subset = Subset(dipeptide_dataset, val_indices)

    # Create and train model
    model = AutoencoderDiffClassifier(model_config)
    if verbose:
        print(f"Model parameters: {model.count_parameters():,}")

    model, history = train_autoencoder_model(
        model=model,
        train_dataset=train_subset,
        val_dataset=val_subset,
        config=training_config,
        reconstruction_weight=100.0,
        classification_weight=1.0,
        pretrain_epochs=0,
        verbose=verbose,
    )

    return model, history


def main():
    """Main function for iterative tripeptide selection."""
    parser = argparse.ArgumentParser(description="Tripeptide sequence selection for training")
    parser.add_argument("--data-root", type=str, default="data/processed/orpl",
                        help="Path to processed data directory")
    parser.add_argument("--output-dir", type=str, default="checkpoints/tripeptide_selection_orpl100",
                        help="Output directory for models and results")
    parser.add_argument("--baseline-epochs", type=int, default=50,
                        help="Number of epochs for baseline training")
    parser.add_argument("--finetune-epochs", type=int, default=5,
                        help="Number of epochs for fine-tuning iterations")
    parser.add_argument("--samples-per-pair", type=int, default=100,
                        help="Number of samples per pair")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.0001,
                        help="Learning rate")
    parser.add_argument("--latent-dim", type=int, default=64,
                        help="Latent dimension for autoencoder")
    parser.add_argument("--max-iterations", type=int, default=30,
                        help="Maximum number of tripeptide addition iterations")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/cpu)")
    args = parser.parse_args()

    # Set up paths
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device configuration
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")

    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)

    # Load data
    print("\nLoading spectra data...")
    spectra_data = SpectraDataset(data_root, min_spectra=1)
    print(f"  Amino acids: {len(spectra_data.aminoacids)}")
    print(f"  Dipeptides: {len(spectra_data.dipeptides)}")
    print(f"  Tripeptides: {len(spectra_data.tripeptides)}")

    # Get all tripeptides
    all_tripeptides = sorted(spectra_data.tripeptides.keys())
    print(f"\nAll tripeptides ({len(all_tripeptides)}): {all_tripeptides}")
    print(f"Initially all {len(all_tripeptides)} tripeptides are holdout (evaluation only)")

    # Model configuration
    model_config = AutoencoderModelConfig(
        seq_length=1023,
        encoder_dims=(512, 256),
        latent_dim=args.latent_dim,
        dropout=0.3,
        num_classes=6,
        activation='relu',
        reconstruction_weight=10.0,
        classification_weight=1.0,
    )

    # Training configuration for baseline
    baseline_training_config = TrainingConfig(
        learning_rate=args.learning_rate,
        weight_decay=1e-4,
        epochs=args.baseline_epochs,
        batch_size=args.batch_size,
        early_stopping_patience=100,  # Disable early stopping
        device=device,
    )

    # Training configuration for fine-tuning
    finetune_training_config = TrainingConfig(
        learning_rate=args.learning_rate,
        weight_decay=1e-4,
        epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        early_stopping_patience=100,  # Disable early stopping
        device=device,
    )

    # Step 1: Train baseline model
    print("\n" + "=" * 70)
    print("STEP 1: TRAINING BASELINE MODEL (DIPEPTIDES ONLY)")
    print("=" * 70)

    baseline_model, baseline_history = train_baseline(
        spectra_data=spectra_data,
        model_config=model_config,
        training_config=baseline_training_config,
        samples_per_pair=args.samples_per_pair,
        verbose=True,
    )

    # Save baseline model state (deep copy for reuse)
    baseline_state = copy.deepcopy(baseline_model.state_dict())
    torch.save({
        'model_state_dict': baseline_state,
        'model_config': model_config.to_dict(),
        'history': baseline_history,
    }, output_dir / "baseline_dipeptides.pt")
    print(f"\nBaseline model saved to {output_dir / 'baseline_dipeptides.pt'}")

    # Create evaluation datasets
    print("\nCreating evaluation datasets...")

    # All tripeptides dataset (for evaluation)
    all_tripeptides_dataset = SequenceDataset(
        spectra_dataset=spectra_data,
        include_dipeptide_pairs=False,
        include_tripeptide_pairs=True,
        samples_per_pair=args.samples_per_pair,
        seed=42,
    )

    # Dipeptide-only dataset for evaluation
    dipeptide_dataset = SequenceDataset(
        spectra_dataset=spectra_data,
        include_dipeptide_pairs=True,
        include_tripeptide_pairs=False,
        samples_per_pair=args.samples_per_pair,
        seed=42,
    )

    print(f"  All tripeptides samples: {len(all_tripeptides_dataset)}")
    print(f"  Dipeptide samples: {len(dipeptide_dataset)}")

    # Create data loaders
    all_tri_loader = DataLoader(all_tripeptides_dataset, batch_size=args.batch_size,
                                shuffle=False, collate_fn=collate_fn)
    dipeptide_loader = DataLoader(dipeptide_dataset, batch_size=args.batch_size,
                                  shuffle=False, collate_fn=collate_fn)

    # Get class names
    amino_acid_codes = all_tripeptides_dataset.get_label_names()

    # Evaluate baseline
    print("\n" + "-" * 50)
    print("Evaluating Baseline Model")
    print("-" * 50)

    dip_acc, dip_preds, dip_labels = evaluate_model(baseline_model, dipeptide_loader, device)
    all_tri_acc, all_tri_preds, all_tri_labels = evaluate_model(baseline_model, all_tri_loader, device)

    print(f"  Dipeptide accuracy: {dip_acc:.2f}%")
    print(f"  All tripeptides accuracy (holdout): {all_tri_acc:.2f}%")

    # Plot baseline confusion matrices
    plot_confusion_matrices(
        all_tri_labels, all_tri_preds, amino_acid_codes,
        "Baseline (Dipeptides Only) - All Tripeptides",
        output_dir, "confusion_matrix_baseline"
    )

    # Get per-tripeptide accuracy for baseline
    per_tripeptide_acc = evaluate_per_tripeptide(baseline_model, all_tripeptides_dataset, device, args.batch_size)

    print("\nPer-tripeptide accuracy (baseline - all holdout):")
    for tripeptide in sorted(per_tripeptide_acc.keys()):
        acc = per_tripeptide_acc[tripeptide]
        print(f"  {tripeptide}: {acc:.1f}%")

    # Initialize results tracking
    results = [{
        'iteration': 0,
        'tripeptide_added': 'baseline',
        'train_acc': dip_acc,
        'all_tripeptide_acc': all_tri_acc,
        'holdout_acc': all_tri_acc,  # Initially all tripeptides are holdout
        'num_training_tripeptides': 0,
        'num_holdout_tripeptides': len(all_tripeptides),
        'selected_tripeptides': '',
    }]

    # Find first tripeptide with lowest accuracy
    first_tripeptide = min(all_tripeptides, key=lambda t: per_tripeptide_acc.get(t, 0))
    selected_tripeptides = [first_tripeptide]

    print(f"\nFirst tripeptide selected (lowest accuracy): {first_tripeptide} ({per_tripeptide_acc.get(first_tripeptide, 0):.1f}%)")

    # Step 2: Iterative tripeptide addition
    print("\n" + "=" * 70)
    print("STEP 2: ITERATIVE TRIPEPTIDE ADDITION")
    print("=" * 70)

    max_iterations = min(args.max_iterations, len(all_tripeptides))
    for iteration in range(1, max_iterations + 1):
        # Calculate holdout tripeptides (those not yet selected)
        holdout_tripeptides = [t for t in all_tripeptides if t not in selected_tripeptides]

        print(f"\n{'='*70}")
        print(f"ITERATION {iteration}: Training tripeptides: {selected_tripeptides}")
        print(f"Holdout tripeptides ({len(holdout_tripeptides)}): {holdout_tripeptides}")
        print(f"{'='*70}")

        # Load fresh baseline weights
        model = AutoencoderDiffClassifier(model_config)
        model.load_state_dict(baseline_state)
        model = model.to(device)

        # Create dataset with dipeptides + selected tripeptides
        combined_dataset = SequenceDataset(
            spectra_dataset=spectra_data,
            include_dipeptide_pairs=True,
            include_tripeptide_pairs=True,
            include_tripeptide_codes=selected_tripeptides,
            samples_per_pair=args.samples_per_pair,
            seed=42,
        )

        print(f"Training samples: {len(combined_dataset)}")
        print(f"  Dipeptides: included")
        print(f"  Tripeptides in training: {selected_tripeptides}")

        # Train/val split
        labels = [sample[2] for sample in combined_dataset.samples]
        indices = list(range(len(combined_dataset)))
        train_indices, val_indices = train_test_split(
            indices, test_size=0.2, stratify=labels, random_state=42
        )

        train_subset = Subset(combined_dataset, train_indices)
        val_subset = Subset(combined_dataset, val_indices)

        # Fine-tune
        model, history = train_autoencoder_model(
            model=model,
            train_dataset=train_subset,
            val_dataset=val_subset,
            config=finetune_training_config,
            reconstruction_weight=100.0,
            classification_weight=1.0,
            pretrain_epochs=0,
            verbose=True,
        )

        # Evaluate on training data
        train_loader = DataLoader(combined_dataset, batch_size=args.batch_size,
                                  shuffle=False, collate_fn=collate_fn)
        train_acc, _, _ = evaluate_model(model, train_loader, device)

        # Evaluate on all tripeptides
        all_tri_acc, all_tri_preds, all_tri_labels = evaluate_model(model, all_tri_loader, device)

        # Evaluate on holdout tripeptides (dynamically computed)
        if holdout_tripeptides:
            holdout_dataset = SequenceDataset(
                spectra_dataset=spectra_data,
                include_dipeptide_pairs=False,
                include_tripeptide_pairs=True,
                include_tripeptide_codes=holdout_tripeptides,
                samples_per_pair=args.samples_per_pair,
                seed=42,
            )
            holdout_loader = DataLoader(holdout_dataset, batch_size=args.batch_size,
                                        shuffle=False, collate_fn=collate_fn)
            holdout_acc, _, _ = evaluate_model(model, holdout_loader, device)
        else:
            holdout_acc = 0.0  # No holdout left

        print(f"\nIteration {iteration} Results:")
        print(f"  Training accuracy: {train_acc:.2f}%")
        print(f"  All tripeptides accuracy: {all_tri_acc:.2f}%")
        print(f"  Holdout tripeptides accuracy ({len(holdout_tripeptides)} remaining): {holdout_acc:.2f}%")

        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': model_config.to_dict(),
            'selected_tripeptides': selected_tripeptides.copy(),
            'holdout_tripeptides': holdout_tripeptides.copy(),
            'iteration': iteration,
            'history': history,
        }, output_dir / f"model_iter_{iteration}.pt")

        # Plot confusion matrices
        plot_confusion_matrices(
            all_tri_labels, all_tri_preds, amino_acid_codes,
            f"Iteration {iteration} (+ {selected_tripeptides[-1]}) - All Tripeptides",
            output_dir, f"confusion_matrix_iter_{iteration}"
        )

        # Record results
        results.append({
            'iteration': iteration,
            'tripeptide_added': selected_tripeptides[-1],
            'train_acc': train_acc,
            'all_tripeptide_acc': all_tri_acc,
            'holdout_acc': holdout_acc,
            'num_training_tripeptides': len(selected_tripeptides),
            'num_holdout_tripeptides': len(holdout_tripeptides),
            'selected_tripeptides': ','.join(selected_tripeptides),
        })

        # Get per-tripeptide accuracy
        per_tripeptide_acc = evaluate_per_tripeptide(model, all_tripeptides_dataset, device, args.batch_size)

        print("\nPer-tripeptide accuracy:")
        for tripeptide in sorted(per_tripeptide_acc.keys()):
            acc = per_tripeptide_acc[tripeptide]
            in_training = " (training)" if tripeptide in selected_tripeptides else " (holdout)"
            print(f"  {tripeptide}: {acc:.1f}%{in_training}")

        # Find next tripeptide with lowest accuracy (not yet selected)
        if iteration < max_iterations:
            remaining = [t for t in all_tripeptides if t not in selected_tripeptides]
            if remaining:
                next_tripeptide = min(remaining, key=lambda t: per_tripeptide_acc.get(t, 0))
                selected_tripeptides.append(next_tripeptide)
                print(f"\nNext tripeptide selected: {next_tripeptide} ({per_tripeptide_acc.get(next_tripeptide, 0):.1f}%)")

    # Save results to CSV
    csv_path = output_dir / "results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['iteration', 'tripeptide_added', 'train_acc',
                                               'all_tripeptide_acc', 'holdout_acc',
                                               'num_training_tripeptides', 'num_holdout_tripeptides',
                                               'selected_tripeptides'])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {csv_path}")

    # Save tripeptide selection order
    order_path = output_dir / "tripeptide_selection_order.txt"
    with open(order_path, 'w') as f:
        f.write("Tripeptide Selection Order\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total tripeptides: {len(all_tripeptides)}\n\n")
        f.write("Selection order (worst-performing tripeptide added first):\n")
        for i, tripeptide in enumerate(selected_tripeptides, 1):
            f.write(f"  {i}. {tripeptide}\n")
    print(f"Selection order saved to {order_path}")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\nBaseline (dipeptides only, all {len(all_tripeptides)} tripeptides as holdout):")
    print(f"  Dipeptide accuracy: {results[0]['train_acc']:.2f}%")
    print(f"  Holdout (all tripeptides) accuracy: {results[0]['holdout_acc']:.2f}%")
    print(f"\nFinal model (dipeptides + all {len(all_tripeptides)} tripeptides in training):")
    print(f"  Training accuracy: {results[-1]['train_acc']:.2f}%")
    print(f"  All tripeptides accuracy: {results[-1]['all_tripeptide_acc']:.2f}%")
    print(f"\nImprovement on all tripeptides: {results[-1]['all_tripeptide_acc'] - results[0]['all_tripeptide_acc']:+.2f}%")
    print(f"\nTripeptide selection order (worst-performing first): {selected_tripeptides}")
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
