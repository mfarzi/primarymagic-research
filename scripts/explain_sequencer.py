#!/usr/bin/env python
"""Backtrack a DifferentialClassifierWithPretrainedEncoder (sequencer) head to
input wavenumbers via multi-input Integrated Gradients, plus latent-space
activation maximisation visualised through the autoencoder's decoder, plus
per-sample maps.

Usage:
    python scripts/explain_sequencer.py \\
        --sequencer-checkpoint   checkpoints/decoupled_v1/seq_step26_tp095/differential_classifier.pt \\
        --autoencoder-checkpoint checkpoints/decoupled_v1/step26_tp095/full_model.pt \\
        --data-root              data/custom/processed/primary_magic

Outputs land in ``<sequencer-parent>/explanations/`` by default.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from primarymagic.data import SpectraDataset  # noqa: E402
from primarymagic.models import (  # noqa: E402
    DifferentialClassifierWithPretrainedEncoder,
    MultiLabelAutoencoderConfig,
    MultiLabelRegularizedAutoencoder,
)
from primarymagic.interpretability.attribution import (  # noqa: E402
    compute_ig_pair,
    pair_contrastive_baseline,
    zero_baseline,
)
from primarymagic.interpretability.activation_max import synthesize_latent_prototype  # noqa: E402
from primarymagic.interpretability.pair_dataset import collect_pair_samples  # noqa: E402
from primarymagic.interpretability.plots import (  # noqa: E402
    plot_cross_class_overlay_pair,
    plot_decoded_diff_prototype,
    plot_pair_fingerprint,
    plot_pair_per_sample_grid,
)
from primarymagic.interpretability.sample_selection import (  # noqa: E402
    bottom_k_by_logit,
    top_k_by_logit,
)
from primarymagic.interpretability.summary import render_pair_summary_md  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Attribution analysis for the sequencer's differential head.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sequencer-checkpoint", type=str, required=True,
                   help="Path to differential_classifier.pt.")
    p.add_argument("--autoencoder-checkpoint", type=str, required=True,
                   help="Path to full_model.pt for the autoencoder providing the decoder.")
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--output-dir", type=str, default=None,
                   help="Default: <sequencer-parent>/explanations/")
    p.add_argument("--classes", type=str, nargs="*", default=None,
                   help="Subset of removed-AA classes to attribute. Default: all from sequencer config.")
    p.add_argument("--baselines", type=str, nargs="+",
                   default=["zero", "contrastive"], choices=["zero", "contrastive"])
    p.add_argument("--samples-per-class", type=int, default=200)
    p.add_argument("--include-dipeptide-pairs", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-tripeptide-pairs", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--include-tetrapeptide-pairs", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--include-pentapeptide-pairs", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--ig-steps", type=int, default=50)
    p.add_argument("--min-spectra", type=int, default=40)
    p.add_argument("--do-fingerprints", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--do-per-sample-maps", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--do-activation-max", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--lambda-l2", type=float, default=0.001)
    p.add_argument("--am-steps", type=int, default=500)
    p.add_argument("--am-lr", type=float, default=0.01)
    p.add_argument("--per-sample-k", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_sequencer(seq_ckpt_path: Path, device: torch.device):
    """Load DifferentialClassifierWithPretrainedEncoder + AA codes + seq_length."""
    model, ckpt = DifferentialClassifierWithPretrainedEncoder.from_checkpoint(
        str(seq_ckpt_path), device=str(device),
    )
    model = model.to(device).eval()
    # IG requires the gradient chain through the encoder. The default frozen
    # state wraps the encoder in torch.no_grad() inside forward(), which
    # severs autograd. Unfreezing only changes requires_grad + the no_grad
    # guard; we never call optimizer.step(), so encoder weights remain fixed.
    model.unfreeze_encoder()
    aa_codes = ckpt.get("amino_acid_codes")
    if aa_codes is None:
        # Fall back to classifier_config or a default 6-AA list
        aa_codes = ckpt["classifier_config"].get(
            "amino_acid_codes", ["A", "D", "F", "G", "R", "S"],
        )
    seq_length = int(ckpt["encoder_config"].get("input_dim", 1023))
    return model, list(aa_codes), seq_length


def load_decoder(
    ae_ckpt_path: Path,
    device: torch.device,
    expected_seq_length: int,
    sequencer_encoder: Optional[torch.nn.Module] = None,
):
    """Load the autoencoder's decoder, returning ``(decoder, latent_dim)``.

    Validates that the autoencoder's seq_length matches the sequencer's, and —
    if ``sequencer_encoder`` is provided — that the autoencoder's encoder
    weights are identical to the sequencer's encoder weights (the sequencer
    uses a frozen-pretrained-encoder workflow, so the two must come from the
    same training stage; mismatched checkpoints would silently produce wrong
    latent-activation-max plots).
    """
    ckpt = torch.load(ae_ckpt_path, map_location=device)
    config = MultiLabelAutoencoderConfig.from_dict(ckpt["model_config"])
    if int(config.seq_length) != expected_seq_length:
        raise ValueError(
            f"Autoencoder seq_length ({config.seq_length}) != sequencer "
            f"seq_length ({expected_seq_length}). Are these checkpoints from the same training stage?"
        )
    model = MultiLabelRegularizedAutoencoder(config)
    model.load_state_dict(ckpt["model_state_dict"])

    if sequencer_encoder is not None:
        # Compare learnable parameters only (skip BatchNorm running buffers,
        # which legitimately drift during sequencer training even with frozen
        # weights, because BN running stats update in train() mode regardless
        # of requires_grad).
        ae_params = dict(model.encoder.named_parameters())
        seq_params = dict(sequencer_encoder.named_parameters())
        if set(ae_params.keys()) != set(seq_params.keys()):
            raise ValueError(
                "Autoencoder and sequencer encoders have different parameter keys; "
                "checkpoints are not compatible."
            )
        for k, v in ae_params.items():
            if not torch.equal(v.detach().cpu(), seq_params[k].detach().cpu()):
                raise ValueError(
                    f"Autoencoder and sequencer encoders disagree on parameter {k!r}. "
                    "These checkpoints are not from the same training stage; the "
                    "autoencoder's decoder cannot meaningfully invert z values from "
                    "a different encoder. Pass matching <step>/full_model.pt and "
                    "seq_<step>/differential_classifier.pt."
                )

    return model.decoder.to(device).eval(), int(config.latent_dim)


# ---------------------------------------------------------------------------
# Logits for class on pairs (used for sample selection)
# ---------------------------------------------------------------------------
def logits_for_pair_class(model, pairs_n, pairs_n_minus_1, target_class_idx):
    model.eval()
    with torch.no_grad():
        logits = model(pairs_n, pairs_n_minus_1)
    return logits[:, target_class_idx].detach().cpu()


# ---------------------------------------------------------------------------
# Per-class IG attribution (multi-input)
# ---------------------------------------------------------------------------
def attribute_pair_class(
    model, cls, cls_idx,
    pairs_n_cls, pairs_n1_cls,
    all_pairs_n, all_pairs_n1, all_labels,
    baselines, n_steps,
):
    out: Dict[str, Dict[str, np.ndarray]] = {}
    seq_length = pairs_n_cls.shape[1]
    device = pairs_n_cls.device

    for b_name in baselines:
        if b_name == "zero":
            b_n = zero_baseline(seq_length, device=device)
            b_n1 = zero_baseline(seq_length, device=device)
        elif b_name == "contrastive":
            b_n_cpu, b_n1_cpu = pair_contrastive_baseline(
                all_pairs_n, all_pairs_n1, all_labels, target_class=cls,
            )
            b_n = b_n_cpu.to(device)
            b_n1 = b_n1_cpu.to(device)
        else:
            raise ValueError(f"Unknown baseline: {b_name!r}")

        attr_n, attr_n1 = compute_ig_pair(
            model, pairs_n_cls, pairs_n1_cls,
            baseline_n=b_n, baseline_n_minus_1=b_n1,
            target_class_idx=cls_idx, n_steps=n_steps,
        )
        out[b_name] = {
            "n": attr_n.cpu().numpy(),
            "n_minus_1": attr_n1.cpu().numpy(),
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    seq_ckpt_path = Path(args.sequencer_checkpoint)
    ae_ckpt_path = Path(args.autoencoder_checkpoint)
    out_dir = Path(args.output_dir) if args.output_dir else seq_ckpt_path.parent / "explanations"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fingerprints").mkdir(exist_ok=True)
    (out_dir / "per_sample").mkdir(exist_ok=True)
    (out_dir / "activation_max").mkdir(exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    print(f"Loading sequencer: {seq_ckpt_path}")
    model, all_codes, seq_length = load_sequencer(seq_ckpt_path, device)
    print(f"  Sequencer classes: {all_codes}  seq_length={seq_length}")

    print(f"Loading autoencoder decoder: {ae_ckpt_path}")
    decoder, latent_dim = load_decoder(
        ae_ckpt_path, device, seq_length, sequencer_encoder=model.encoder,
    )
    print(f"  Decoder latent_dim={latent_dim}")

    attribute_classes = args.classes if args.classes else all_codes
    attribute_classes = [c for c in attribute_classes if c in all_codes]

    print(f"Loading dataset: {args.data_root}")
    dataset = SpectraDataset(args.data_root, min_spectra=args.min_spectra)
    wavenumbers = dataset.wavelengths.astype(np.float32)
    if wavenumbers.shape[0] != seq_length:
        raise ValueError(
            f"Wavelength axis length {wavenumbers.shape[0]} != sequencer seq_length {seq_length}"
        )

    # Always collect a pool from all sequencer classes — needed for contrastive baselines
    pool_classes = all_codes
    pairs_n_np, pairs_n1_np, labels, pair_codes = collect_pair_samples(
        dataset, classes=pool_classes,
        include_dp=args.include_dipeptide_pairs,
        include_tp=args.include_tripeptide_pairs,
        include_4p=args.include_tetrapeptide_pairs,
        include_5p=args.include_pentapeptide_pairs,
        samples_per_class=args.samples_per_class, seed=args.seed,
    )
    pairs_n = pairs_n_np.to(device)
    pairs_n1 = pairs_n1_np.to(device)

    # Subset by attribute_classes for IG, but keep the full pool for contrastive baselines.
    per_class_pairs: Dict[str, Tuple[torch.Tensor, torch.Tensor, np.ndarray]] = {}
    for cls in attribute_classes:
        m = labels == cls
        if not m.any():
            print(f"  [WARN] no pairs for requested class {cls!r}")
            continue
        per_class_pairs[cls] = (pairs_n[m], pairs_n1[m], pair_codes[m])

    # ---- Phase A: IG attribution ---------------------------------------------
    raw_attr: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    mean_attr: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}

    for cls, (cls_n, cls_n1, _) in per_class_pairs.items():
        cls_idx = all_codes.index(cls)
        print(f"  IG  class={cls!r}  N={cls_n.shape[0]}")
        per_baseline = attribute_pair_class(
            model, cls, cls_idx, cls_n, cls_n1,
            pairs_n, pairs_n1, labels,
            args.baselines, args.ig_steps,
        )
        for b_name, sides in per_baseline.items():
            raw_attr[(cls, b_name)] = sides
            mean_attr[(cls, b_name)] = {
                "n": sides["n"].mean(axis=0),
                "n_minus_1": sides["n_minus_1"].mean(axis=0),
            }

    np.savez(
        out_dir / "attributions.npz",
        wavenumbers=wavenumbers,
        **{
            f"{cls}__{b}__attr_n": sides["n"]
            for (cls, b), sides in raw_attr.items()
        },
        **{
            f"{cls}__{b}__attr_n_minus_1": sides["n_minus_1"]
            for (cls, b), sides in raw_attr.items()
        },
    )
    print(f"  Saved raw attributions to {out_dir / 'attributions.npz'}")

    # ---- Phase B: Fingerprint plots ------------------------------------------
    if args.do_fingerprints:
        for (cls, b_name), sides in raw_attr.items():
            cls_n = per_class_pairs[cls][0].cpu().numpy()
            cls_n1 = per_class_pairs[cls][1].cpu().numpy()
            attr_n = sides["n"]
            attr_n1 = sides["n_minus_1"]
            fig = plot_pair_fingerprint(
                wavenumbers,
                cls_n.mean(axis=0), cls_n.std(axis=0),
                attr_n.mean(axis=0), attr_n.std(axis=0),
                cls_n1.mean(axis=0), cls_n1.std(axis=0),
                attr_n1.mean(axis=0), attr_n1.std(axis=0),
                class_label=cls, baseline_label=b_name, n_samples=attr_n.shape[0],
            )
            fig.savefig(out_dir / "fingerprints" / f"{cls}_{b_name}.png",
                        dpi=120, bbox_inches="tight")
            plt.close(fig)
        for b_name in args.baselines:
            data = {cls: mean_attr[(cls, b_name)]["n"] for cls in per_class_pairs}
            fig = plot_cross_class_overlay_pair(wavenumbers, data, baseline_label=b_name)
            fig.savefig(out_dir / "fingerprints" / f"all_classes_{b_name}.png",
                        dpi=120, bbox_inches="tight")
            plt.close(fig)
        print(f"  Saved fingerprint plots to {out_dir / 'fingerprints'}")

    # ---- Phase C: Per-sample maps --------------------------------------------
    if args.do_per_sample_maps:
        for cls, (cls_n, cls_n1, cls_codes) in per_class_pairs.items():
            cls_idx = all_codes.index(cls)
            logits = logits_for_pair_class(model, cls_n, cls_n1, cls_idx)
            top_idx = top_k_by_logit(logits, args.per_sample_k)
            bot_idx = bottom_k_by_logit(logits, args.per_sample_k)
            (out_dir / "per_sample" / cls).mkdir(exist_ok=True, parents=True)
            for b_name in args.baselines:
                attr_n = raw_attr[(cls, b_name)]["n"]
                attr_n1 = raw_attr[(cls, b_name)]["n_minus_1"]
                for sel_name, idx_list in [("top", top_idx), ("bottom", bot_idx)]:
                    sel_n = cls_n.cpu().numpy()[idx_list]
                    sel_n1 = cls_n1.cpu().numpy()[idx_list]
                    sel_attr_n = attr_n[idx_list]
                    sel_attr_n1 = attr_n1[idx_list]
                    sel_logits = [float(logits[i]) for i in idx_list]
                    sel_codes = [str(cls_codes[i]) for i in idx_list]
                    title = (
                        f"{cls} removed  {sel_name}-{len(idx_list)} "
                        f"({'confident' if sel_name == 'top' else 'confused'})  "
                        f"baseline={b_name}"
                    )
                    fig = plot_pair_per_sample_grid(
                        wavenumbers, sel_n, sel_attr_n, sel_n1, sel_attr_n1,
                        sel_logits, title,
                        pair_codes=sel_codes,
                    )
                    fname = (
                        f"{sel_name}{len(idx_list)}_"
                        + ("confident" if sel_name == "top" else "confused")
                        + f"_{b_name}.png"
                    )
                    fig.savefig(out_dir / "per_sample" / cls / fname,
                                dpi=120, bbox_inches="tight")
                    plt.close(fig)
        print(f"  Saved per-sample maps to {out_dir / 'per_sample'}")

    # ---- Phase D: Latent-space activation max --------------------------------
    if args.do_activation_max:
        proto_arrays: Dict[str, np.ndarray] = {}
        empirical_diff_arrays: Dict[str, np.ndarray] = {}
        decoded_empirical_arrays: Dict[str, np.ndarray] = {}

        # Encoder for computing real z values (use sequencer's encoder; same as autoencoder's)
        encoder = model.encoder

        for cls, (cls_n, cls_n1, _) in per_class_pairs.items():
            cls_idx = all_codes.index(cls)
            with torch.no_grad():
                z_n = encoder(cls_n)
                z_n1 = encoder(cls_n1)
                z_diff_real = z_n - z_n1               # (N, latent_dim)
                mean_z_diff = z_diff_real.mean(dim=0)  # (latent_dim,)
                decoded_empirical = decoder(mean_z_diff.unsqueeze(0))[0]  # (seq_length,)

            raw_diff = (cls_n - cls_n1).cpu().numpy()
            empirical_diff_arrays[cls] = raw_diff      # (N, seq_length)
            decoded_empirical_arrays[cls] = decoded_empirical.cpu().numpy()

            for init_name in ("noise", "mean_init"):
                z_diff_star, decoded_synthetic, history = synthesize_latent_prototype(
                    classifier_head=model.diff_classifier,
                    decoder=decoder,
                    target_class_idx=cls_idx,
                    latent_dim=latent_dim,
                    steps=args.am_steps, lr=args.am_lr,
                    lambda_l2=args.lambda_l2,
                    init=init_name,
                    mean_z_diff=mean_z_diff if init_name == "mean_init" else None,
                    seed=args.seed, device=device,
                )
                synthetic_np = decoded_synthetic.cpu().numpy()
                proto_arrays[f"{cls}__{init_name}"] = synthetic_np

                fig = plot_decoded_diff_prototype(
                    wavenumbers,
                    synthetic_decoded=synthetic_np,
                    decoded_empirical=decoded_empirical_arrays[cls],
                    raw_empirical_mean=raw_diff.mean(axis=0),
                    raw_empirical_std=raw_diff.std(axis=0),
                    history=history,
                    class_label=cls, init_label=init_name,
                    final_logit=history[-1], lambda_l2=args.lambda_l2,
                )
                fig.savefig(out_dir / "activation_max" / f"{cls}_{init_name}.png",
                            dpi=120, bbox_inches="tight")
                plt.close(fig)

        np.savez(
            out_dir / "activations.npz",
            wavenumbers=wavenumbers,
            **{f"synthetic__{k}": v for k, v in proto_arrays.items()},
            **{f"decoded_empirical__{k}": v for k, v in decoded_empirical_arrays.items()},
            **{f"raw_empirical_mean__{k}": v.mean(axis=0)
               for k, v in empirical_diff_arrays.items()},
        )
        print(f"  Saved activation-max prototypes to {out_dir / 'activation_max'}")

    # ---- Phase E: Summary ----------------------------------------------------
    md = render_pair_summary_md(wavenumbers, mean_attr, top_k=5)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    print(f"  Wrote summary to {out_dir / 'summary.md'}")
    print("Done.")


if __name__ == "__main__":
    main()
