#!/usr/bin/env python
"""Train DifferentialClassifierWithPretrainedEncoder.

Phase 2 training script: Trains a differential classifier on top of a
pretrained encoder. The classifier learns to identify which amino acid was
added when comparing two spectra of different sequence lengths.

Supports two modes:
  1. Frozen encoder (default): Train classifier head only, using an encoder
     from a Phase 1 autoencoder checkpoint (--encoder-checkpoint).
  2. Fine-tuning: Load a full Phase 2 checkpoint (--model-checkpoint) and
     unfreeze the encoder (--unfreeze-encoder) for end-to-end fine-tuning
     with a separate, lower learning rate for the encoder (--encoder-lr).

Example (frozen encoder, dipeptides only):
    python scripts/train_sequencer.py \
        --encoder-checkpoint checkpoints/autoencoder/exp001/full_model.pt \
        --data-root data/processed/orpl \
        --include-dipeptide-pairs \
        --no-include-tripeptide-pairs \
        --max-samples-per-class 500 \
        --experiment-name exp001_dipep

Example (fine-tuning with tripeptide pairs):
    python scripts/train_sequencer.py \
        --model-checkpoint checkpoints/sequencer/exp04c/differential_classifier.pt \
        --data-root data/processed/orpl \
        --no-include-dipeptide-pairs \
        --include-tripeptide-pairs \
        --unfreeze-encoder \
        --encoder-lr 1e-5 \
        --experiment-name exp06a_finetune
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

from primarymagic.data import SpectraDataset, SequenceDataset
from primarymagic.models import (
    MultiLabelAutoencoderConfig,
    MultiLabelRegularizedAutoencoder,
    TrainingConfig,
)
from primarymagic.models.autoencoder import (
    DifferentialClassifierWithPretrainedEncoder,
    SpectralEncoder,
)
from primarymagic.models.trainer import PretrainedDifferentialTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train DifferentialClassifierWithPretrainedEncoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Checkpoint arguments (one of encoder-checkpoint or model-checkpoint required)
    parser.add_argument(
        "--encoder-checkpoint",
        type=str,
        default=None,
        help="Path to Phase 1 model checkpoint (full_model.pt or encoder.pt)",
    )
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default=None,
        help="Path to Phase 2 differential_classifier.pt checkpoint (for fine-tuning)",
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
        default=True,
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
        default=False,
        help="Include (XYZ, XY) -> Z tripeptide pairs",
    )
    parser.add_argument(
        "--no-include-tripeptide-pairs",
        action="store_false",
        dest="include_tripeptide_pairs",
        help="Exclude tripeptide pairs",
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
        "--max-samples-per-class",
        type=int,
        default=None,
        help="Maximum samples per target class (for balancing)",
    )
    parser.add_argument(
        "--samples-per-pair",
        type=int,
        default=None,
        help="Number of random spectra pairs per sequence pair (None = all)",
    )

    # Amino acid codes
    parser.add_argument(
        "--amino-acids",
        type=str,
        nargs="+",
        default=None,
        help="Target amino acid codes (default: from encoder checkpoint or ['A','D','F','G','R','S'])",
    )

    # Training arguments
    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        default=False,
        help="Use inverse frequency class weights",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
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

    # Model arguments (for classifier head)
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Dropout rate for classifier head (default: from encoder config)",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default=None,
        choices=["relu", "gelu", "leaky_relu", "elu", "tanh"],
        help="Activation function for classifier (default: from encoder config)",
    )

    # Fine-tuning arguments
    parser.add_argument(
        "--unfreeze-encoder",
        action="store_true",
        default=False,
        help="Unfreeze encoder for fine-tuning (requires --model-checkpoint)",
    )
    parser.add_argument(
        "--encoder-lr",
        type=float,
        default=1e-5,
        help="Learning rate for encoder when unfrozen (should be much lower than classifier LR)",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/sequencer",
        help="Base output directory for checkpoints",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="differential_classifier",
        help="Experiment name (creates subdirectory under output-dir)",
    )

    return parser.parse_args()


def load_encoder_from_checkpoint(checkpoint_path: str, device: str = "cpu"):
    """Load encoder from checkpoint file.

    Handles both full model checkpoints and encoder-only checkpoints.

    Args:
        checkpoint_path: Path to checkpoint file.
        device: Device to load model on.

    Returns:
        Tuple of (encoder, config_dict) where config_dict contains encoder parameters.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        # Full model checkpoint - create autoencoder and extract encoder
        print("Loading from full model checkpoint...")

        model_config_dict = checkpoint.get("model_config", {})
        model_config = MultiLabelAutoencoderConfig.from_dict(model_config_dict)

        autoencoder = MultiLabelRegularizedAutoencoder(model_config)
        autoencoder.load_state_dict(checkpoint["model_state_dict"])

        # Get amino acid codes from checkpoint if available
        amino_acid_codes = checkpoint.get(
            "amino_acid_codes",
            list(model_config.amino_acid_codes)
        )

        config_dict = {
            "input_dim": model_config.seq_length,
            "hidden_dims": model_config.encoder_dims,
            "latent_dim": model_config.latent_dim,
            "dropout": model_config.dropout,
            "activation": model_config.activation,
            "num_classes": model_config.num_classes,
            "amino_acid_codes": amino_acid_codes,
        }

        return autoencoder.encoder, config_dict

    elif "encoder_state_dict" in checkpoint:
        # Encoder-only checkpoint
        print("Loading from encoder-only checkpoint...")

        encoder = SpectralEncoder(
            input_dim=checkpoint.get("input_dim", 1023),
            hidden_dims=tuple(checkpoint.get("hidden_dims", (512, 256))),
            latent_dim=checkpoint.get("latent_dim", 48),
            dropout=checkpoint.get("dropout", 0.3),
            activation=checkpoint.get("activation", "relu"),
        )
        encoder.load_state_dict(checkpoint["encoder_state_dict"])

        config_dict = {
            "input_dim": checkpoint.get("input_dim", 1023),
            "hidden_dims": checkpoint.get("hidden_dims", (512, 256)),
            "latent_dim": checkpoint.get("latent_dim", 48),
            "dropout": checkpoint.get("dropout", 0.3),
            "activation": checkpoint.get("activation", "relu"),
            "num_classes": checkpoint.get("num_classes", 6),
            "amino_acid_codes": checkpoint.get("amino_acid_codes", ["A", "D", "F", "G", "R", "S"]),
        }

        return encoder, config_dict

    else:
        raise ValueError(
            f"Unrecognized checkpoint format. Expected 'model_state_dict' or "
            f"'encoder_state_dict' key. Found keys: {list(checkpoint.keys())}"
        )


def main():
    """Main training function."""
    args = parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 70)
    print("Phase 2: Training DifferentialClassifierWithPretrainedEncoder")
    print("=" * 70)

    # Create output directory
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Validate checkpoint arguments
    if args.model_checkpoint is None and args.encoder_checkpoint is None:
        print("Error: Must provide either --encoder-checkpoint or --model-checkpoint")
        sys.exit(1)
    if args.unfreeze_encoder and args.model_checkpoint is None:
        print("Error: --unfreeze-encoder requires --model-checkpoint")
        sys.exit(1)

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model or encoder
    pretrained_model = None
    if args.model_checkpoint is not None:
        # Load full differential classifier from checkpoint
        print(f"\nLoading model from: {args.model_checkpoint}")
        pretrained_model, model_checkpoint = (
            DifferentialClassifierWithPretrainedEncoder.from_checkpoint(
                args.model_checkpoint, device
            )
        )
        encoder_config = model_checkpoint["encoder_config"]
        amino_acid_codes_from_ckpt = model_checkpoint.get(
            "amino_acid_codes", ["A", "D", "F", "G", "R", "S"]
        )
        print(f"Loaded differential classifier from checkpoint")
        if args.unfreeze_encoder:
            pretrained_model.unfreeze_encoder()
            print(f"Encoder UNFROZEN for fine-tuning (encoder LR: {args.encoder_lr:.2e})")
    else:
        # Load encoder from Phase 1 checkpoint
        print(f"\nLoading encoder from: {args.encoder_checkpoint}")
        encoder, encoder_config = load_encoder_from_checkpoint(args.encoder_checkpoint, device)
        amino_acid_codes_from_ckpt = encoder_config.get(
            "amino_acid_codes", ["A", "D", "F", "G", "R", "S"]
        )

    print(f"Encoder config:")
    for key, value in encoder_config.items():
        print(f"  {key}: {value}")

    # Determine amino acid codes
    if args.amino_acids is not None:
        amino_acid_codes = args.amino_acids
    else:
        amino_acid_codes = amino_acid_codes_from_ckpt
    print(f"\nTarget amino acids: {amino_acid_codes}")

    # Load spectra data
    print(f"\nLoading data from: {args.data_root}")
    spectra_data = SpectraDataset(args.data_root, min_spectra=args.min_spectra)
    print(f"Loaded: {spectra_data}")

    # Create sequence dataset
    sequence_dataset = SequenceDataset(
        spectra_data,
        include_dipeptide_pairs=args.include_dipeptide_pairs,
        include_tripeptide_pairs=args.include_tripeptide_pairs,
        include_prefixes=args.include_prefixes,
        exclude_prefixes=args.exclude_prefixes,
        include_tripeptide_codes=args.tripeptide_codes,
        max_samples_per_class=args.max_samples_per_class,
        samples_per_pair=args.samples_per_pair,
        amino_acid_codes=amino_acid_codes,
        seed=args.seed,
    )

    print(f"\nSequence dataset:")
    print(f"  Total samples: {len(sequence_dataset)}")
    print(f"  Number of classes: {sequence_dataset.num_classes}")
    print(f"  Class distribution: {sequence_dataset.get_class_distribution()}")

    if len(sequence_dataset) == 0:
        print("\nError: No samples found. Check your data filtering options.")
        sys.exit(1)

    # Split into train/val
    train_size = int(args.train_split * len(sequence_dataset))
    val_size = len(sequence_dataset) - train_size

    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(
        sequence_dataset, [train_size, val_size], generator=generator
    )
    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create or reuse model
    dropout = args.dropout if args.dropout is not None else encoder_config.get("dropout", 0.3)
    activation = args.activation if args.activation is not None else encoder_config.get("activation", "relu")

    if pretrained_model is not None:
        model = pretrained_model
    else:
        model = DifferentialClassifierWithPretrainedEncoder(
            encoder=encoder,
            latent_dim=encoder_config["latent_dim"],
            num_classes=len(amino_acid_codes),
            dropout=dropout,
            activation=activation,
        )

    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    frozen_params = sum(p.numel() for p in model.encoder.parameters() if not p.requires_grad)
    if model.encoder_frozen:
        print(f"\nModel created with frozen encoder")
        print(f"Trainable parameters (classifier head only): {model.count_parameters():,}")
        print(f"Encoder parameters (frozen): {frozen_params:,} / {encoder_params:,}")
    else:
        print(f"\nModel loaded with unfrozen encoder for fine-tuning")
        print(f"Trainable parameters (encoder + classifier): {model.count_parameters():,}")
        print(f"Encoder parameters (trainable): {encoder_params - frozen_params:,} / {encoder_params:,}")

    # Get class weights if requested
    class_weights = None
    if args.use_class_weights:
        class_weights = sequence_dataset.get_class_weights()
        print(f"\nUsing class weights: {class_weights.tolist()}")

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
    encoder_lr = args.encoder_lr if args.unfreeze_encoder else None
    trainer = PretrainedDifferentialTrainer(
        model=model,
        config=training_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        class_weights=class_weights,
        encoder_lr=encoder_lr,
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

    # Save full classifier (encoder + head)
    classifier_path = output_dir / "differential_classifier.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "encoder_config": encoder_config,
            "classifier_config": {
                "latent_dim": encoder_config["latent_dim"],
                "num_classes": len(amino_acid_codes),
                "dropout": dropout,
                "activation": activation,
            },
            "training_config": training_config.to_dict(),
            "history": history,
            "amino_acid_codes": amino_acid_codes,
        },
        classifier_path,
    )
    print(f"Saved differential classifier: {classifier_path}")

    # Save classifier head only
    head_path = output_dir / "classifier_head.pt"
    torch.save(
        {
            "head_state_dict": model.diff_classifier.state_dict(),
            "latent_dim": encoder_config["latent_dim"],
            "num_classes": len(amino_acid_codes),
            "dropout": dropout,
            "activation": activation,
        },
        head_path,
    )
    print(f"Saved classifier head: {head_path}")

    # Save config as JSON
    config_path = output_dir / "config.json"
    config_dict = {
        "encoder_checkpoint": args.encoder_checkpoint or args.model_checkpoint,
        "model_checkpoint": args.model_checkpoint,
        "unfreeze_encoder": args.unfreeze_encoder,
        "encoder_lr": args.encoder_lr if args.unfreeze_encoder else None,
        "encoder_config": {
            k: list(v) if isinstance(v, tuple) else v
            for k, v in encoder_config.items()
        },
        "classifier_config": {
            "latent_dim": encoder_config["latent_dim"],
            "num_classes": len(amino_acid_codes),
            "dropout": dropout,
            "activation": activation,
        },
        "training_config": training_config.to_dict(),
        "data_config": {
            "data_root": args.data_root,
            "amino_acid_codes": amino_acid_codes,
            "include_dipeptide_pairs": args.include_dipeptide_pairs,
            "include_tripeptide_pairs": args.include_tripeptide_pairs,
            "include_prefixes": args.include_prefixes,
            "exclude_prefixes": args.exclude_prefixes,
            "tripeptide_codes": args.tripeptide_codes,
            "max_samples_per_class": args.max_samples_per_class,
            "samples_per_pair": args.samples_per_pair,
            "use_class_weights": args.use_class_weights,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "class_distribution": sequence_dataset.get_class_distribution(),
        },
        "final_metrics": {
            "best_val_loss": trainer.best_val_loss,
            "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
            "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
            "final_train_acc": history["train_acc"][-1] if history["train_acc"] else None,
            "final_val_acc": history["val_acc"][-1] if history["val_acc"] else None,
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
    if history["val_acc"]:
        best_val_acc = max(history["val_acc"])
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Epochs trained: {len(history['train_loss'])}")
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
