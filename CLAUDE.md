# CLAUDE.md

Guidance for working in this repository.

## What this is

`cvx-linalg` — NaN-aware linear algebra utilities for portfolio optimization,
part of the [jebel-quant](https://github.com/jebel-quant) ecosystem. Pure-Python
on top of `numpy`; the `ewm_covariance` helper additionally needs the optional
`polars` extra. The public API is re-exported flat from `cvx.linalg` (see
`README.md`); internally the code is split into subpackages under
`src/cvx/linalg/`: `core`, `covariance`, `decomposition`, `kkt`, `norm`,
`operators`, `solve`.

## Ownership: locally owned vs Rhiza-managed

This repo syncs its dev infrastructure from the
[`jebel-quant/rhiza`](https://github.com/jebel-quant/rhiza) template. The pinned
version lives in `.rhiza/template.yml` (`ref:`), and `make sync` re-applies the
template. **The authoritative, machine-generated list of synced files is the
`files:` block of `.rhiza/template.lock`** — when in doubt, consult it. The split
below summarizes it.

### Locally owned — edit these freely

- `src/` — the library source (all of `cvx/linalg/**`)
- `tests/` — the test suite for the library
- `pyproject.toml` — project metadata, dependencies, dependency-groups, tool config
- `README.md` — the public-facing docs (kept in sync with the Makefile help by a hook)
- `.rhiza/template.yml` — the template pin + `profiles:`/`templates:` selection (owned here, though it lives under `.rhiza/`)
- `mkdocs.yml`, `CHANGELOG.md`, `CLAUDE.md`, and any project-specific docs
- `book/` marimo notebooks and other project content

### Rhiza-managed — do NOT edit in place; fix upstream

These are overwritten on the next `make sync`. To change them, open a PR against
`jebel-quant/rhiza` (or adjust `.rhiza/template.yml`), then re-sync:

- `.github/workflows/rhiza_*.yml` — all CI/CD workflows
- `.github/` scaffolding — issue/PR/discussion templates, `dependabot.yml`,
  `release.yml`, rulesets, `secret_scanning.yml`, `CONFIG.md`
- `Makefile` and `.rhiza/make.d/*.mk`, `.rhiza/rhiza.mk` — the make targets
- `.pre-commit-config.yaml`, `ruff.toml`, `pytest.ini`, `.bandit`,
  `.editorconfig`, `.python-version`, `cliff.toml`, `.rhiza/semgrep.yml` — tooling config
- `.claude/commands/rhiza_*.md` — synced Claude commands
- `.rhiza/**` (except `template.yml`) — `.rhiza-version`, completions, tests,
  `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, assets
- `LICENSE`, `SECURITY.md`, `docs/index.md`, `docs/development/*.md`,
  `docs/mkdocs-base.yml`, `docs/assets/*`

Two independent version fields under `.rhiza/`: `template.yml`'s `ref` tracks
`jebel-quant/rhiza` *template content* releases; `.rhiza/.rhiza-version` tracks
the *rhiza CLI tool* release. They are decoupled — do not derive one from the other.

## Quality gates

All gates run via `make <target>` (never call `.venv/bin/...` directly):

- `make fmt` — pre-commit hooks: ruff format/check, markdownlint, bandit, actionlint
- `make typecheck` — `ty` + `mypy --strict` over `src/`
- `make docs-coverage` — interrogate docstring coverage (100% required)
- `make deptry` — unused/missing dependency analysis
- `make security` — pip-audit + bandit
- `make validate` — check the repo against the Rhiza template
- `make test` — full pytest suite with the coverage gate (100% line+branch)

## Conventions

- Keep the public API flat: new user-facing symbols get re-exported from
  `src/cvx/linalg/__init__.py`. `ewm_covariance` is the deliberate exception (it
  requires the optional `polars` extra) and stays importable only from
  `cvx.linalg.covariance.ewm_cov`.
- Every public symbol needs a docstring — docstring coverage must stay at 100%.
- Maintain 100% test + branch coverage; add tests alongside any source change.
- Functions are NaN-aware by design; preserve that when extending them.
