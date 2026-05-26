#!/usr/bin/env python
"""Train MultiLabelRegularizedAutoencoder with reconstruction + multi-label classification.

Phase 1 training script: Trains an autoencoder to learn spectral representations
that capture amino acid composition through combined reconstruction and
multi-label classification objectives.

Example:
    python scripts/train_autoencoder.py \
        --data-root data/processed/orpl \
        --amino-acids A D F G R S \
        --latent-dim 48 \
        --epochs 100 \
        --experiment-name exp001
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import random_split

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from primarymagic.data import SpectraDataset, MultiLabelSpectraDataset
from primarymagic.models import (
    MultiLabelAutoencoderConfig,
    MultiLabelRegularizedAutoencoder,
    TrainingConfig,
)
from primarymagic.models.trainer import MultiLabelAutoencoderTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train MultiLabelRegularizedAutoencoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Initialization
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help="Path to a previous full_model.pt checkpoint to initialize weights from",
    )

    # Data arguments
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
    parser.add_argument(
        "--amino-acids",
        type=str,
        nargs="+",
        default=["A", "D", "F", "G", "R", "S"],
        help="Target amino acid single-letter codes",
    )
    parser.add_argument(
        "--include-aminoacids",
        action="store_true",
        default=True,
        help="Include single amino acid spectra in training",
    )
    parser.add_argument(
        "--no-include-aminoacids",
        action="store_false",
        dest="include_aminoacids",
        help="Exclude single amino acid spectra from training",
    )
    parser.add_argument(
        "--include-dipeptides",
        action="store_true",
        default=True,
        help="Include dipeptide spectra in training",
    )
    parser.add_argument(
        "--no-include-dipeptides",
        action="store_false",
        dest="include_dipeptides",
        help="Exclude dipeptide spectra from training",
    )
    parser.add_argument(
        "--include-tripeptides",
        action="store_true",
        default=True,
        help="Include tripeptide spectra in training",
    )
    parser.add_argument(
        "--no-include-tripeptides",
        action="store_false",
        dest="include_tripeptides",
        help="Exclude tripeptide spectra from training",
    )
    parser.add_argument(
        "--dipeptide-codes",
        type=str,
        nargs="*",
        default=None,
        help="Specific dipeptide codes to include (e.g., AD AF). If not specified, all dipeptides are included.",
    )
    parser.add_argument(
        "--tripeptide-codes",
        type=str,
        nargs="*",
        default=None,
        help="Specific tripeptide codes to include (e.g., AFG GFS). If not specified, all tripeptides are included.",
    )
    parser.add_argument(
        "--include-tetrapeptides",
        action="store_true",
        default=False,
        help="Include tetrapeptide spectra in training",
    )
    parser.add_argument(
        "--no-include-tetrapeptides",
        action="store_false",
        dest="include_tetrapeptides",
        help="Exclude tetrapeptide spectra from training",
    )
    parser.add_argument(
        "--tetrapeptide-codes",
        type=str,
        nargs="*",
        default=None,
        help="Specific tetrapeptide codes to include (e.g., ADFG GFRS). If not specified, all tetrapeptides are included.",
    )
    parser.add_argument(
        "--include-pentapeptides",
        action="store_true",
        default=False,
        help="Include pentapeptide spectra in training",
    )
    parser.add_argument(
        "--no-include-pentapeptides",
        action="store_false",
        dest="include_pentapeptides",
        help="Exclude pentapeptide spectra from training",
    )
    parser.add_argument(
        "--pentapeptide-codes",
        type=str,
        nargs="*",
        default=None,
        help="Specific pentapeptide codes to include (e.g., ADFGR GFRSA). If not specified, all pentapeptides are included.",
    )
    parser.add_argument(
        "--samples-per-sequence",
        type=int,
        default=None,
        help="Maximum number of spectra per sequence code (for balancing). If not specified, all spectra are used.",
    )

    # Model architecture arguments
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=48,
        help="Latent space dimension",
    )
    parser.add_argument(
        "--encoder-dims",
        type=int,
        nargs="+",
        default=[512, 256],
        help="Encoder hidden layer dimensions",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "gelu", "leaky_relu", "elu", "tanh"],
        help="Activation function",
    )

    # Loss weight arguments
    parser.add_argument(
        "--recon-weight",
        type=float,
        default=1.0,
        help="Reconstruction loss weight",
    )
    parser.add_argument(
        "--class-weight",
        type=float,
        default=1.0,
        help="Classification loss weight",
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Maximum number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay (L2 regularization)",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Fraction of data for training (rest is validation)",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=20,
        help="Epochs to wait before early stopping",
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
        default="checkpoints/autoencoder",
        help="Base output directory for checkpoints",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="multilabel_autoencoder",
        help="Experiment name (creates subdirectory under output-dir)",
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 70)
    print("Phase 1: Training MultiLabelRegularizedAutoencoder")
    print("=" * 70)

    # Create output directory
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Load spectra data
    print(f"\nLoading data from: {args.data_root}")
    spectra_data = SpectraDataset(args.data_root, min_spectra=args.min_spectra)
    print(f"Loaded: {spectra_data}")

    # Filter dipeptides if specific codes are requested
    if args.dipeptide_codes is not None and args.include_dipeptides:
        original_count = len(spectra_data.dipeptides)
        filtered_dipeptides = {
            code: spectra for code, spectra in spectra_data.dipeptides.items()
            if code in args.dipeptide_codes
        }
        spectra_data.dipeptides = filtered_dipeptides
        print(f"\nFiltered dipeptides: {original_count} -> {len(filtered_dipeptides)}")
        print(f"  Included codes: {list(filtered_dipeptides.keys())}")
        missing = set(args.dipeptide_codes) - set(filtered_dipeptides.keys())
        if missing:
            print(f"  Warning: Requested codes not found: {missing}")

    # Filter tripeptides if specific codes are requested
    if args.tripeptide_codes is not None and args.include_tripeptides:
        original_count = len(spectra_data.tripeptides)
        filtered_tripeptides = {
            code: spectra for code, spectra in spectra_data.tripeptides.items()
            if code in args.tripeptide_codes
        }
        spectra_data.tripeptides = filtered_tripeptides
        print(f"\nFiltered tripeptides: {original_count} -> {len(filtered_tripeptides)}")
        print(f"  Included codes: {list(filtered_tripeptides.keys())}")
        missing = set(args.tripeptide_codes) - set(filtered_tripeptides.keys())
        if missing:
            print(f"  Warning: Requested codes not found: {missing}")

    # Filter tetrapeptides if specific codes are requested
    if args.tetrapeptide_codes is not None and args.include_tetrapeptides:
        original_count = len(spectra_data.tetrapeptides)
        filtered_tetrapeptides = {
            code: spectra for code, spectra in spectra_data.tetrapeptides.items()
            if code in args.tetrapeptide_codes
        }
        spectra_data.tetrapeptides = filtered_tetrapeptides
        print(f"\nFiltered tetrapeptides: {original_count} -> {len(filtered_tetrapeptides)}")
        print(f"  Included codes: {list(filtered_tetrapeptides.keys())}")
        missing = set(args.tetrapeptide_codes) - set(filtered_tetrapeptides.keys())
        if missing:
            print(f"  Warning: Requested codes not found: {missing}")

    # Filter pentapeptides if specific codes are requested
    if args.pentapeptide_codes is not None and args.include_pentapeptides:
        original_count = len(spectra_data.pentapeptides)
        filtered_pentapeptides = {
            code: spectra for code, spectra in spectra_data.pentapeptides.items()
            if code in args.pentapeptide_codes
        }
        spectra_data.pentapeptides = filtered_pentapeptides
        print(f"\nFiltered pentapeptides: {original_count} -> {len(filtered_pentapeptides)}")
        print(f"  Included codes: {list(filtered_pentapeptides.keys())}")
        missing = set(args.pentapeptide_codes) - set(filtered_pentapeptides.keys())
        if missing:
            print(f"  Warning: Requested codes not found: {missing}")

    # Create multi-label dataset
    amino_acid_codes = args.amino_acids
    print(f"\nTarget amino acids: {amino_acid_codes}")

    multilabel_dataset = MultiLabelSpectraDataset(
        spectra_data,
        amino_acid_codes=amino_acid_codes,
        include_aminoacids=args.include_aminoacids,
        include_dipeptides=args.include_dipeptides,
        include_tripeptides=args.include_tripeptides,
        include_tetrapeptides=args.include_tetrapeptides,
        include_pentapeptides=args.include_pentapeptides,
        samples_per_sequence=args.samples_per_sequence,
    )

    print(f"Total samples: {len(multilabel_dataset)}")
    print(f"Sequence type distribution: {multilabel_dataset.get_sequence_type_distribution()}")
    print(f"Label distribution: {multilabel_dataset.get_label_distribution()}")

    # Split into train/val
    train_size = int(args.train_split * len(multilabel_dataset))
    val_size = len(multilabel_dataset) - train_size

    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(
        multilabel_dataset, [train_size, val_size], generator=generator
    )
    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create model config
    model_config = MultiLabelAutoencoderConfig(
        encoder_dims=tuple(args.encoder_dims),
        latent_dim=args.latent_dim,
        dropout=args.dropout,
        num_classes=len(amino_acid_codes),
        activation=args.activation,
        reconstruction_weight=args.recon_weight,
        classification_weight=args.class_weight,
        amino_acid_codes=tuple(amino_acid_codes),
    )

    print(f"\nModel configuration:")
    for key, value in model_config.to_dict().items():
        print(f"  {key}: {value}")

    # Create model
    model = MultiLabelRegularizedAutoencoder(model_config)

    # Optionally load pretrained weights
    if args.init_checkpoint is not None:
        print(f"\nLoading pretrained weights from: {args.init_checkpoint}")
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu")
        pretrained_state = checkpoint["model_state_dict"]
        model_state = model.state_dict()

        loaded, skipped = [], []
        for key, param in pretrained_state.items():
            if key in model_state and param.shape == model_state[key].shape:
                model_state[key] = param
                loaded.append(key)
            else:
                skipped.append(key)

        model.load_state_dict(model_state)
        print(f"  Loaded {len(loaded)} / {len(pretrained_state)} parameter tensors")
        if skipped:
            print(f"  Skipped (shape mismatch): {skipped}")

    print(f"\nModel architecture:\n{model}")
    print(f"Total parameters: {model.count_parameters():,}")

    # Create training config
    training_config = TrainingConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stopping_patience=args.early_stopping_patience,
        checkpoint_dir=str(output_dir),
        seed=args.seed,
    )

    print(f"\nTraining configuration:")
    for key, value in training_config.to_dict().items():
        print(f"  {key}: {value}")

    # Create trainer
    trainer = MultiLabelAutoencoderTrainer(
        model=model,
        config=training_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        reconstruction_weight=args.recon_weight,
        classification_weight=args.class_weight,
    )

    # Train
    print("\n" + "-" * 70)
    print("Starting training...")
    print("-" * 70)

    history = trainer.train(verbose=True)

    # Save outputs
    print("\n" + "-" * 70)
    print("Saving model checkpoints...")
    print("-" * 70)

    # Save full model
    full_model_path = output_dir / "full_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config.to_dict(),
            "training_config": training_config.to_dict(),
            "history": history,
            "amino_acid_codes": amino_acid_codes,
        },
        full_model_path,
    )
    print(f"Saved full model: {full_model_path}")

    # Save encoder only
    encoder_path = output_dir / "encoder.pt"
    torch.save(
        {
            "encoder_state_dict": model.encoder.state_dict(),
            "input_dim": model_config.seq_length,
            "hidden_dims": model_config.encoder_dims,
            "latent_dim": model_config.latent_dim,
            "dropout": model_config.dropout,
            "activation": model_config.activation,
        },
        encoder_path,
    )
    print(f"Saved encoder: {encoder_path}")

    # Save decoder only
    decoder_path = output_dir / "decoder.pt"
    torch.save(
        {
            "decoder_state_dict": model.decoder.state_dict(),
            "latent_dim": model_config.latent_dim,
            "hidden_dims": model_config.decoder_dims,
            "output_dim": model_config.seq_length,
            "dropout": model_config.dropout,
            "activation": model_config.activation,
        },
        decoder_path,
    )
    print(f"Saved decoder: {decoder_path}")

    # Save classification head only
    head_path = output_dir / "classification_head.pt"
    torch.save(
        {
            "head_state_dict": model.classification_head.state_dict(),
            "latent_dim": model_config.latent_dim,
            "num_classes": model_config.num_classes,
            "dropout": model_config.dropout,
            "activation": model_config.activation,
        },
        head_path,
    )
    print(f"Saved classification head: {head_path}")

    # Save config as JSON
    config_path = output_dir / "config.json"
    config_dict = {
        "model_config": model_config.to_dict(),
        "training_config": training_config.to_dict(),
        "data_config": {
            "data_root": args.data_root,
            "amino_acid_codes": amino_acid_codes,
            "include_aminoacids": args.include_aminoacids,
            "include_dipeptides": args.include_dipeptides,
            "include_tripeptides": args.include_tripeptides,
            "dipeptide_codes": args.dipeptide_codes,
            "tripeptide_codes": args.tripeptide_codes,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
        },
        "final_metrics": {
            "best_val_loss": trainer.best_val_loss,
            "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
            "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
            "final_train_acc": history["train_multilabel_acc"][-1] if history["train_multilabel_acc"] else None,
            "final_val_acc": history["val_multilabel_acc"][-1] if history["val_multilabel_acc"] else None,
            "epochs_trained": len(history["train_loss"]),
        },
    }
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"Saved config: {config_path}")

    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)
    print(f"\nBest validation loss: {trainer.best_val_loss:.4f}")
    print(f"Epochs trained: {len(history['train_loss'])}")
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
