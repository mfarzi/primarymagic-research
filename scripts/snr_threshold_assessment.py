"""Visual assessment of SNR threshold for FARG/rep1.

Generates two figures:
  1. SNR histogram with candidate thresholds and retention table.
  2. Grid of example stage3 spectra at SNR bands {~10, ~20, ~30, ~50, ~80},
     so the eye can judge what each threshold actually preserves/rejects.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

BASE = Path(r'C:\Users\mfarzi\mycodes\visiogen\spectra\data\custom\processed\magic_new\4-letter\FARG\rep1')
OUT = Path(r'C:\Users\mfarzi\mycodes\visiogen\spectra')

mask = np.load(BASE / 'mask.npz')
snr = mask['snr']
passed = mask['passed']
clean_idx = mask['clean_index']  # raw row index for each spectrum entry
stored_thr = float(mask['snr_threshold'])

s3 = np.load(BASE / 'stage3_baseline_removed.npz')
wn = s3['wavelengths']
ints = s3['intensities']  # (N, W) — order matches mask rows since stage3 is pre-filter

# -------------------------------------------------------------- Figure 1: histogram
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.hist(snr, bins=60, color='steelblue', alpha=0.75, edgecolor='black')
colors = {10: 'tab:gray', 20: 'tab:green', 30: 'tab:orange', 50: 'tab:red', 75: 'tab:purple'}
for t, c in colors.items():
    n_keep = int((snr > t).sum())
    ax.axvline(t, color=c, linestyle='--', linewidth=1.4,
               label=f'SNR>{t}: keep {n_keep}/{len(snr)} ({100*n_keep/len(snr):.1f}%)')
ax.set_xlabel('SNR  (peak / MAD-σ of stage1−stage2 residual)')
ax.set_ylabel('count')
ax.set_title(f'FARG rep1 — per-spectrum SNR distribution (N={len(snr)})')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / 'farg_rep1_snr_histogram.png', dpi=130)
plt.close(fig)
print('Wrote', OUT / 'farg_rep1_snr_histogram.png')

# -------------------------------------------------------------- Figure 2: example spectra by SNR band
bands = [
    ('~5  (very low)',    5),
    ('~10 (LOQ-ish)',    10),
    ('~20',              20),
    ('~30',              30),
    ('~50',              50),
    ('~80 (top)',        80),
]
rng = np.random.default_rng(0)
n_examples = 4

fig, axes = plt.subplots(len(bands), n_examples, figsize=(13, 2.0 * len(bands)),
                          sharex=True)
for r, (lbl, target) in enumerate(bands):
    # find spectra with SNR closest to target (within ±15% of target, else nearest)
    tol = max(2.0, 0.15 * target)
    pool = np.where(np.abs(snr - target) <= tol)[0]
    if len(pool) < n_examples:
        pool = np.argsort(np.abs(snr - target))[:n_examples * 4]
    pick = rng.choice(pool, size=min(n_examples, len(pool)), replace=False)

    for c, idx in enumerate(pick):
        ax = axes[r, c]
        ax.plot(wn, ints[idx], linewidth=0.7, color='black')
        ax.set_title(f'SNR={snr[idx]:.1f}', fontsize=9)
        ax.tick_params(labelsize=7)
        if c == 0:
            ax.set_ylabel(lbl, fontsize=9)
        ax.grid(True, alpha=0.25)
    for c in range(len(pick), n_examples):
        axes[r, c].axis('off')

for ax in axes[-1, :]:
    ax.set_xlabel('wavenumber (cm$^{-1}$)', fontsize=8)

fig.suptitle('FARG rep1 — stage3 (baseline-removed) spectra at different SNR levels',
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT / 'farg_rep1_snr_examples.png', dpi=130)
plt.close(fig)
print('Wrote', OUT / 'farg_rep1_snr_examples.png')
