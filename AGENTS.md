# Repository Guidelines

## Project Structure & Module Organization
- Source code lives in `src/social_xlstm/` (models, dataset, training, metrics, utils). Deprecated modules are under `src/social_xlstm/deprecated/`.
- Configurations are in `cfgs/` with dataset/model profiles under `cfgs/profiles/`.
- Training and utilities are in `scripts/` (e.g., `scripts/train/with_social_pooling/train_multi_vd.py`).
- Data artifacts and results go under `blob/` (e.g., `blob/dataset/processed/*.h5`, `blob/experiments/...`).
- Docs live in `docs/` (see references for configuration and testing guides).

## Build, Test, and Development Commands (UV-only)
- Execution: always run via `uv run` (no conda activation, no system Python).
- Train (PEMS-BAY dev): `uv run python scripts/train/with_social_pooling/train_multi_vd.py --config cfgs/profiles/pems_bay_dev.yaml`.
- Quick test run: use `cfgs/profiles/pems_bay_10vd_fast.yaml` and generate plots with `uv run python scripts/utils/generate_metrics_plots.py`.
- Tests (if present): `uv run pytest -m "not slow"` or `uv run pytest --cov=src` (see `docs/reference/testing-guide.md`).
- With approval only: editable install or deps changes (e.g., `uv run pip install -e .`).

## Agent Runtime (UV-only)
- Use `uv run` for all commands; do not use conda or system Python.
- No package changes by default. Only `uv run` is allowed without consent.
- If a package change is needed, request explicit approval first, then use `uv run pip ...` or `uv add ...` as approved.

## Environment Management (UV-only)
- Tooling: `uv` is the sole environment tool in this repo.
- Allowed by default: `uv run` only.
- Requires explicit user consent before execution: `uv add ...`, `uv run pip ...` (or any command that changes packages).
- The agent must not modify environments or install/upgrade/uninstall packages without prior approval.

## Package Change Policy
- No implicit changes. The agent must not install, uninstall, or upgrade packages without explicit user consent.
- If a dependency is missing, report the missing module and propose an exact `uv run pip ...` or `uv add ...` command for approval.
- After approval, use the approved command and record it in the conversation for traceability.

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
