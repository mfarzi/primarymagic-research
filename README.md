# primarymagic-research

Research notebooks and analysis scripts using the [primarymagic](https://github.com/mfarzi/primarymagic) Raman-spectroscopy library. Extracted from the [spectra](https://github.com/mfarzi/spectra) monorepo in 2026-05.

## Layout

```
primarymagic-research/
├── notebooks/                 # Jupyter notebooks (demos, exploration, analysis)
├── scripts/                   # Analysis, diagnostic, training, evaluation scripts
├── config/
│   └── paths.example.toml     # Template for external data paths (committed)
├── pyproject.toml             # Pins primarymagic + orplib + jupyter
├── LICENSE
└── README.md
```

## Install

Requires Python 3.10+ and SSH access to `mfarzi/primarymagic` on GitHub.

```bash
git clone git@github.com:mfarzi/primarymagic-research.git
cd primarymagic-research

python -m venv .venv
source .venv/Scripts/activate    # Windows Git Bash; on Unix use bin/activate
pip install --upgrade pip
pip install -e .[ml]             # core + ML extras (torch, scikit-learn, pywt, captum)
```

`primarymagic` itself is pinned to `v0.1.1` in `pyproject.toml`. To run scripts against a different primarymagic version, edit the pin and `pip install -e .` again.

## External data paths

Heavy artefacts (`data/`, `checkpoints/`, `results/`, `analysis_outputs/`) are **not** in this repo. They live on your local filesystem and are referenced via env-var-driven config.

Copy the template:

```bash
cp config/paths.example.toml config/paths.toml    # paths.toml is gitignored
```

Then edit `config/paths.toml` to point at your local data:

```toml
data_root        = "C:/Users/mfarzi/data/spectra"
checkpoints_root = "C:/Users/mfarzi/data/checkpoints"
results_root     = "C:/Users/mfarzi/data/results"
```

Or set env vars (these override the TOML file):

- `PRIMARYMAGIC_DATA_ROOT`
- `PRIMARYMAGIC_CHECKPOINTS_ROOT`
- `PRIMARYMAGIC_RESULTS_ROOT`

## Running a notebook

```bash
source .venv/Scripts/activate
jupyter notebook notebooks/
```

## Running a script

```bash
source .venv/Scripts/activate
python scripts/<name>.py [args...]
```

## Version pinning

The repo pins a **single** version of `primarymagic` (currently `v0.1.1`). All scripts and notebooks run against that version. If a future script needs a newer primarymagic:

1. Bump the pin in `pyproject.toml`.
2. `pip install -e .` to upgrade.
3. Older scripts MAY break. If they do, you can re-run them against the historical tag via a sibling worktree:
   ```bash
   git worktree add ../primarymagic-research-v0.1.0 v0.1.0
   cd ../primarymagic-research-v0.1.0 && python -m venv .venv && source .venv/Scripts/activate && pip install -e .
   ```

## What's NOT in this repo

- **The library itself.** It's at [`mfarzi/primarymagic`](https://github.com/mfarzi/primarymagic).
- **The Django + React app.** It's at [`mfarzi/primarymagic-app`](https://github.com/mfarzi/primarymagic-app).
- **Library tests.** Live in `mfarzi/primarymagic` (under `tests/`).
- **Publications (manuscripts, presentations, powerpoint slides).** These are managed separately (e.g. in Google Drive).
- **Data, checkpoints, results.** Configured externally via the path mechanism above.

## License

MIT — see [LICENSE](LICENSE).
