# Repository Guidelines

## Project Structure & Module Organization
- Source code lives in `src/social_xlstm/` (models, dataset, training, metrics, utils). Deprecated modules are under `src/social_xlstm/deprecated/`.
- Configurations are in `cfgs/` with dataset/model profiles under `cfgs/profiles/`.
- Training and utilities are in `scripts/` (e.g., `scripts/train/with_social_pooling/train_multi_vd.py`).
- Data artifacts and results go under `blob/` (e.g., `blob/dataset/processed/*.h5`, `blob/experiments/...`).
- Docs live in `docs/` (see references for configuration and testing guides).

## Build, Test, and Development Commands
- Create environment: `conda env create -f environment.yaml && conda activate social_xlstm`.
- Editable install (src layout): `pip install -e .`.
- Train (PEMS-BAY dev): `python scripts/train/with_social_pooling/train_multi_vd.py --config cfgs/profiles/pems_bay_dev.yaml`.
- Quick test run: use `cfgs/profiles/pems_bay_10vd_fast.yaml` (≈20s on CPU/GPU) and generate plots with `scripts/utils/generate_metrics_plots.py`.
- Tests (if present): `pytest -m "not slow"` or `pytest --cov=src` (see `docs/reference/testing-guide.md`).

## Agent Runtime
- Always run commands inside the conda env: `conda activate social_xlstm`.
- Use the env’s Python and tooling (do not use system Python).
- If dependencies change, run `pip install -e .` again after activation.

## Coding Style & Naming Conventions
- Python 3.11+, PEP 8, 4-space indentation, type hints encouraged. Prefer docstrings for public APIs.
- Naming: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`.
- Config files: YAML in `cfgs/`, lowercase with underscores (e.g., `pems_bay_fast_test.yaml`).
- Keep patches minimal and focused; avoid renames unless necessary (src layout depends on package paths).

## Testing Guidelines
- Framework: `pytest` with markers: `unit`, `integration`, `functional`, `slow` (see `pytest.ini`).
- Structure: `tests/` with files `test_*.py`, classes `Test*`, functions `test_*`.
- Aim for coverage on changed code; add fixtures under `tests/fixtures/` when needed.
- Prefer fast, deterministic tests; gate GPU-heavy tests with `@pytest.mark.gpu`.

## Commit & Pull Request Guidelines
- Commit style: imperative mood with type prefix (e.g., `Add: ...`, `Fix: ...`, `Refactor: ...`, `docs:` observed in history). Group related changes.
- PRs must include: purpose/summary, key commands to reproduce (configs, seeds, data paths), linked issues, and screenshots/plots for training metrics when applicable.
- Update docs/configs when changing training interfaces or profiles; avoid committing large data—use `blob/` paths and `.gitignore`.

## Security & Configuration Tips
- Do not commit credentials or private datasets. Large artifacts belong in `blob/` and are ignored.
- Prefer profile-based configs (`cfgs/profiles/*.yaml`) and document new options in `docs/reference/configuration-reference.md`.
