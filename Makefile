## Makefile (repo-owned)
# Keep this file small. It can be edited without breaking template sync.

LOGO_FILE=.rhiza/assets/rhiza-logo.svg

# Override template default: include mkdocstrings plugin for API docs
MKDOCS_EXTRA_PACKAGES = --with 'mkdocstrings[python]'

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
