## Makefile (repo-owned)
# Keep this file small. It can be edited without breaking template sync.

DEFAULT_AI_MODEL=claude-sonnet-4.6
LOGO_FILE=.rhiza/assets/rhiza-logo.svg
GH_AW_ENGINE ?= copilot  # Default AI engine for gh-aw workflows (copilot, claude, or codex)

# Override template default: fix quoting bug and typo (mkdocstring -> mkdocstrings)
MKDOCS_EXTRA_PACKAGES = --with-editable . --with 'mkdocstrings[python]'

# Always include the Rhiza API (template-managed)
include .rhiza/rhiza.mk

.PHONY: test-linux
test-linux: ## run tests in a Linux/OpenBLAS container (mirrors CI numerics; Accelerate on macOS suppresses some overflow warnings)
	docker run --rm \
		-v "$(CURDIR)":/repo -w /repo \
		-e UV_PROJECT_ENVIRONMENT=/tmp/.venv -e VIRTUAL_ENV=/tmp/.venv -e UV_CACHE_DIR=/tmp/uv-cache \
		ghcr.io/astral-sh/uv:python3.13-bookworm-slim \
		sh -c "uv sync --all-extras --all-groups && uv pip install -r .rhiza/requirements/tests.txt && uv run pytest -n auto --ignore=tests/benchmarks --ignore=tests/stress"

# Optional: developer-local extensions (not committed)
-include local.mk
