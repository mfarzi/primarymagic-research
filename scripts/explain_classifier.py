#!/usr/bin/env python
"""Backtrack a MultiLabelRegularizedAutoencoder classifier head to input
wavenumbers via Integrated Gradients, plus per-sample maps and activation
maximization prototypes.

Usage:
    python scripts/explain_classifier.py \\
        --checkpoint checkpoints/decoupled_v1/step01_aa_only/full_model.pt \\
        --data-root data/custom/processed/magic_bayes_shifted

Outputs land in ``<checkpoint_parent>/explanations/`` by default.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add project root to path so 'spectra' package imports work
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from primarymagic.data import SpectraDataset  # noqa: E402
from primarymagic.models import (  # noqa: E402
    MultiLabelAutoencoderConfig,
    MultiLabelRegularizedAutoencoder,
)
from primarymagic.interpretability.attribution import (  # noqa: E402
    compute_ig,
    contrastive_baseline,
    zero_baseline,
)
from primarymagic.interpretability.activation_max import synthesize_prototype  # noqa: E402
from primarymagic.interpretability.plots import (  # noqa: E402
    plot_activation_max,
    plot_cross_class_overlay,
    plot_fingerprint,
    plot_per_sample_grid,
)
from primarymagic.interpretability.sample_selection import (  # noqa: E402
    bottom_k_by_logit,
    logits_for_class,
    top_k_by_logit,
)
from primarymagic.interpretability.summary import render_summary_md  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Attribution analysis for the multi-label classifier head.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to full_model.pt for a MultiLabelRegularizedAutoencoder.")
    p.add_argument("--data-root", type=str, required=True,
                   help="SpectraDataset root (e.g. data/custom/processed/magic_bayes_shifted).")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output directory. Default: <checkpoint_parent>/explanations/")
    p.add_argument("--classes", type=str, nargs="*", default=None,
                   help="Subset of classes to attribute. Default: all from checkpoint config.")
    p.add_argument("--baselines", type=str, nargs="+",
                   default=["zero", "contrastive"],
                   choices=["zero", "contrastive"])
    p.add_argument("--samples-per-class", type=int, default=200)
    p.add_argument("--ig-steps", type=int, default=50)
    p.add_argument("--min-spectra", type=int, default=40)
    p.add_argument("--do-fingerprints", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--do-per-sample-maps", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--do-activation-max", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--lambda-smooth", type=float, default=1.0)
    p.add_argument("--lambda-mag", type=float, default=0.001)
    p.add_argument("--am-steps", type=int, default=500)
    p.add_argument("--am-lr", type=float, default=0.01)
    p.add_argument("--per-sample-k", type=int, default=10,
                   help="Number of top/bottom samples to plot per class.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None,
                   help="Device. Default: cuda if available else cpu.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(checkpoint_path: Path, device: torch.device):
    """Load MultiLabelRegularizedAutoencoder from checkpoint and return
    (model, amino_acid_codes, seq_length)."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = MultiLabelAutoencoderConfig.from_dict(ckpt["model_config"])
    model = MultiLabelRegularizedAutoencoder(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    return model, list(config.amino_acid_codes), int(config.seq_length)


# ---------------------------------------------------------------------------
# Spectra collection (single AA spectra labeled by class)
# ---------------------------------------------------------------------------
def collect_class_spectra(
    dataset: SpectraDataset,
    classes: List[str],
    samples_per_class: int,
    seed: int,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """For each class in ``classes``, draw up to ``samples_per_class`` spectra.

    Returns:
        per_class: dict label -> array of shape (n, seq_length).
        all_spectra: concatenation of all collected spectra (N, seq_length).
        all_labels: array of length N with class labels (strings).
    """
    rng = np.random.default_rng(seed)
    per_class: Dict[str, np.ndarray] = {}
    chunks: List[np.ndarray] = []
    label_chunks: List[np.ndarray] = []

    for cls in classes:
        if cls not in dataset.aminoacids:
            print(f"  [SKIP] class {cls!r}: not present in dataset.aminoacids")
            continue
        spectra = dataset.aminoacids[cls]
        n = min(samples_per_class, len(spectra))
        idx = rng.choice(len(spectra), size=n, replace=False)
        subset = spectra[idx].astype(np.float32)
        per_class[cls] = subset
        chunks.append(subset)
        label_chunks.append(np.full(n, cls, dtype=object))

    if not chunks:
        raise RuntimeError("No spectra collected for any requested class.")

    all_spectra = np.concatenate(chunks, axis=0)
    all_labels = np.concatenate(label_chunks, axis=0)
    return per_class, all_spectra, all_labels


# ---------------------------------------------------------------------------
# Per-class attribution loop
# ---------------------------------------------------------------------------
def attribute_class(
    model,
    cls: str,
    cls_idx: int,
    cls_spectra: torch.Tensor,
    all_spectra: torch.Tensor,
    all_labels: np.ndarray,
    baselines: List[str],
    n_steps: int,
) -> Dict[str, np.ndarray]:
    """Compute IG attributions for class ``cls`` under each requested baseline.

    Returns:
        Dict baseline_name -> attribution array (n_cls_samples, seq_length).
    """
    out: Dict[str, np.ndarray] = {}
    seq_length = cls_spectra.shape[1]
    device = cls_spectra.device

    for b_name in baselines:
        if b_name == "zero":
            baseline = zero_baseline(seq_length, device=device)
        elif b_name == "contrastive":
            baseline = contrastive_baseline(all_spectra, all_labels, target_class=cls).to(device)
        else:
            raise ValueError(f"Unknown baseline: {b_name!r}")

        attr = compute_ig(
            model, cls_spectra, baseline=baseline,
            target_class_idx=cls_idx, n_steps=n_steps,
        )
        out[b_name] = attr.cpu().numpy()
    return out


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    ckpt_path = Path(args.checkpoint)
    if args.output_dir is None:
        out_dir = ckpt_path.parent / "explanations"
    else:
        out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fingerprints").mkdir(exist_ok=True)
    (out_dir / "per_sample").mkdir(exist_ok=True)
    (out_dir / "activation_max").mkdir(exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    print(f"Loading checkpoint: {ckpt_path}")
    model, all_codes, seq_length = load_model(ckpt_path, device)
    attribute_classes = args.classes if args.classes else all_codes
    attribute_classes = [c for c in attribute_classes if c in all_codes]
    print(f"Classes to attribute: {attribute_classes}")

    print(f"Loading dataset: {args.data_root}")
    dataset = SpectraDataset(args.data_root, min_spectra=args.min_spectra)
    wavenumbers = dataset.wavelengths.astype(np.float32)
    if wavenumbers.shape[0] != seq_length:
        raise ValueError(
            f"Wavelength axis length {wavenumbers.shape[0]} != model seq_length {seq_length}"
        )

    # Always load the contrastive pool from ALL classes the model knows about,
    # so the contrastive baseline for class c = mean of all 25 other AAs (not
    # just other classes the user requested via --classes).
    pool_classes = all_codes
    pool_per_class, all_spectra_np, all_labels = collect_class_spectra(
        dataset, pool_classes, args.samples_per_class, args.seed,
    )
    all_spectra = torch.tensor(all_spectra_np, device=device)
    per_class = {c: pool_per_class[c] for c in attribute_classes if c in pool_per_class}
    missing = [c for c in attribute_classes if c not in pool_per_class]
    if missing:
        print(f"  [WARN] requested classes not present in dataset, skipped: {missing}")

    # ---- Phase A: IG attributions ------------------------------------------------
    # raw_attr[(cls, baseline)] = (n_samples, seq_length) numpy array
    raw_attr: Dict[Tuple[str, str], np.ndarray] = {}
    mean_attr: Dict[Tuple[str, str], np.ndarray] = {}

    for cls in per_class:
        cls_idx = all_codes.index(cls)
        cls_spectra = torch.tensor(per_class[cls], device=device)
        print(f"  IG  class={cls!r}  N={cls_spectra.shape[0]}")
        per_baseline = attribute_class(
            model, cls, cls_idx, cls_spectra, all_spectra, all_labels,
            args.baselines, args.ig_steps,
        )
        for b_name, attr in per_baseline.items():
            raw_attr[(cls, b_name)] = attr
            mean_attr[(cls, b_name)] = attr.mean(axis=0)

    # Save raw arrays for reuse
    np.savez(
        out_dir / "attributions.npz",
        wavenumbers=wavenumbers,
        **{f"{cls}__{b}": arr for (cls, b), arr in raw_attr.items()},
    )
    print(f"  Saved raw attributions to {out_dir / 'attributions.npz'}")

    # ---- Phase B: Fingerprint plots ---------------------------------------------
    if args.do_fingerprints:
        for (cls, b_name), attr in raw_attr.items():
            mean_a = attr.mean(axis=0)
            std_a = attr.std(axis=0)
            mean_s = per_class[cls].mean(axis=0)
            std_s = per_class[cls].std(axis=0)
            fig = plot_fingerprint(
                wavenumbers, mean_s, std_s, mean_a, std_a,
                class_label=cls, baseline_label=b_name, n_samples=attr.shape[0],
            )
            out_path = out_dir / "fingerprints" / f"{cls}_{b_name}.png"
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
        # Cross-class overlays
        for b_name in args.baselines:
            data = {cls: mean_attr[(cls, b_name)] for cls in per_class}
            fig = plot_cross_class_overlay(wavenumbers, data, baseline_label=b_name)
            out_path = out_dir / "fingerprints" / f"all_classes_{b_name}.png"
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
        print(f"  Saved fingerprint plots to {out_dir / 'fingerprints'}")

    # ---- Phase C: Per-sample maps -----------------------------------------------
    if args.do_per_sample_maps:
        for cls in per_class:
            cls_idx = all_codes.index(cls)
            cls_spectra = torch.tensor(per_class[cls], device=device)
            logits = logits_for_class(model, cls_spectra, cls_idx)
            top_idx = top_k_by_logit(logits, args.per_sample_k)
            bot_idx = bottom_k_by_logit(logits, args.per_sample_k)
            (out_dir / "per_sample" / cls).mkdir(exist_ok=True, parents=True)
            for b_name in args.baselines:
                attr = raw_attr[(cls, b_name)]
                for selection_name, idx_list in [("top", top_idx), ("bottom", bot_idx)]:
                    sel_spectra = per_class[cls][idx_list]
                    sel_attr = attr[idx_list]
                    sel_logits = [float(logits[i]) for i in idx_list]
                    title = (
                        f"{cls}  {selection_name}-{len(idx_list)}  "
                        f"({'confident' if selection_name == 'top' else 'confused'})  "
                        f"baseline={b_name}"
                    )
                    fig = plot_per_sample_grid(
                        wavenumbers, sel_spectra, sel_attr, sel_logits, title,
                    )
                    fname = f"{selection_name}{len(idx_list)}_" + (
                        "confident" if selection_name == "top" else "confused"
                    ) + f"_{b_name}.png"
                    fig.savefig(out_dir / "per_sample" / cls / fname, dpi=120, bbox_inches="tight")
                    plt.close(fig)
        print(f"  Saved per-sample maps to {out_dir / 'per_sample'}")

    # ---- Phase D: Activation maximization ---------------------------------------
    if args.do_activation_max:
        proto_arrays: Dict[str, np.ndarray] = {}
        mean_train = all_spectra.mean(dim=0).cpu()
        for cls in per_class:
            cls_idx = all_codes.index(cls)
            mean_s = per_class[cls].mean(axis=0)
            std_s = per_class[cls].std(axis=0)
            for init_name in ("noise", "mean_init"):
                proto, history = synthesize_prototype(
                    model, target_class_idx=cls_idx, seq_length=seq_length,
                    steps=args.am_steps, lr=args.am_lr,
                    lambda_smooth=args.lambda_smooth, lambda_mag=args.lambda_mag,
                    init=init_name,
                    mean_spectrum=mean_train if init_name == "mean_init" else None,
                    seed=args.seed, device=device,
                )
                proto_np = proto.cpu().numpy()
                proto_arrays[f"{cls}__{init_name}"] = proto_np
                fig = plot_activation_max(
                    wavenumbers, proto_np, mean_s, std_s,
                    class_label=cls, init_label=init_name,
                    final_logit=history[-1], lambda_smooth=args.lambda_smooth,
                )
                fig.savefig(out_dir / "activation_max" / f"{cls}_{init_name}.png", dpi=120, bbox_inches="tight")
                plt.close(fig)
        np.savez(out_dir / "activations.npz", wavenumbers=wavenumbers, **proto_arrays)
        print(f"  Saved activation-max prototypes to {out_dir / 'activation_max'}")

    # ---- Phase E: Summary --------------------------------------------------------
    md = render_summary_md(wavenumbers, mean_attr, top_k=5)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    print(f"  Wrote summary to {out_dir / 'summary.md'}")
    print("Done.")


if __name__ == "__main__":
    main()
