# primarymagic-research

Research notebooks and analysis scripts using the [primarymagic](https://github.com/mfarzi/primarymagic) Raman-spectroscopy library. Extracted from the [spectra](https://github.com/mfarzi/spectra) monorepo in 2026-05.

This is a **workspace**, not an installable Python package. Environment is managed by [`uv`](https://docs.astral.sh/uv/) from `pyproject.toml`; the repo itself is never built or installed (`[tool.uv] package = false`).

## Layout

```
primarymagic-research/
├── notebooks/                 # Jupyter notebooks (demos, exploration, analysis)
├── scripts/                   # Analysis, diagnostic, training, evaluation scripts
├── config/
│   └── paths.example.toml     # Template for external data paths (committed)
├── pyproject.toml             # uv project — default env spec, not installable
├── uv.lock                    # Pinned env — committed for reproducibility
├── LICENSE
└── README.md
```

## Prerequisites

Install [`uv`](https://docs.astral.sh/uv/) — fast Python package manager from Astral:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

You also need SSH access to `mfarzi/primarymagic` on GitHub (it's private). Verify with `ssh -T git@github.com` — you should see "Hi mfarzi!".

## Setup (once per clone)

```bash
git clone git@github.com:mfarzi/primarymagic-research.git
cd primarymagic-research
uv sync               # creates .venv from pyproject.toml + uv.lock
```

That's it. No `pip install`, no manual `python -m venv`.

## Running a script

```bash
uv run scripts/preprocess_magic.py [args]
```

`uv run` ensures the env is up to date (cheap if nothing changed) and runs the script inside it. Equivalent to `.venv/bin/python scripts/preprocess_magic.py`, but you never have to activate the venv.

## Running a notebook

```bash
uv run jupyter notebook notebooks/
```

Or open a specific notebook:

```bash
uv run jupyter notebook notebooks/demoio.ipynb
```

## Bumping the `primarymagic` version (or any other dep)

Single source of truth: `pyproject.toml`. Edit the pin, then resync:

```bash
# Edit pyproject.toml: change @v0.1.1 → @v0.2.0 (for example)
uv sync                    # re-resolves and updates .venv + uv.lock
git add pyproject.toml uv.lock
git commit -m "chore(deps): bump primarymagic to v0.2.0"
```

Commit the new `uv.lock` so collaborators get the same pin.

## Per-script version overrides (rare, but supported)

If a specific script needs a *different* primarymagic version from what `pyproject.toml` declares, add a [PEP 723](https://peps.python.org/pep-0723/) inline metadata header at the top of that script. `uv run` will use the inline spec for that script, ignoring the project's env:

```python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "primarymagic[ml] @ git+ssh://git@github.com/mfarzi/primarymagic@v0.0.9",
#     "orplib>=1.0.10",
#     "pandas>=2.0",
# ]
# ///

import primarymagic
print(primarymagic.__version__)   # 0.0.9, not the project default
```

This is the right tool for **historical compatibility** ("this old script needs v0.0.9 because primarymagic's API changed in v0.1.0") — not for everyday scripts that should just use the project default.

## External data paths

Heavy artefacts (`data/`, `checkpoints/`, `results/`, `analysis_outputs/`) are **not** in this repo. They live on your local filesystem and are referenced via env-var-driven config.

```bash
cp config/paths.example.toml config/paths.toml    # paths.toml is gitignored
```

Edit `config/paths.toml`:

```toml
data_root        = "C:/Users/mfarzi/data/spectra"
checkpoints_root = "C:/Users/mfarzi/data/checkpoints"
results_root     = "C:/Users/mfarzi/data/results"
```

Or set env vars (these override the TOML file):

- `PRIMARYMAGIC_DATA_ROOT`
- `PRIMARYMAGIC_CHECKPOINTS_ROOT`
- `PRIMARYMAGIC_RESULTS_ROOT`

## What's NOT in this repo

- **The library itself.** It's at [`mfarzi/primarymagic`](https://github.com/mfarzi/primarymagic).
- **The Django + React app.** It's at [`mfarzi/primarymagic-app`](https://github.com/mfarzi/primarymagic-app).
- **Library tests.** They live in `mfarzi/primarymagic` (under `tests/`).
- **Publications (manuscripts, presentations, powerpoint slides).** Managed separately (e.g. in Google Drive).
- **Data, checkpoints, results.** Configured externally via the path mechanism above.

## License

MIT — see [LICENSE](LICENSE).
