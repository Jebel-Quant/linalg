"""Tests for package-level metadata."""

from __future__ import annotations

import re
from pathlib import Path

import cvx.linalg

README = Path(__file__).resolve().parents[2] / "README.md"


def test_version_is_exposed() -> None:
    """Test that the package exposes a non-empty __version__ string."""
    assert isinstance(cvx.linalg.__version__, str)
    assert cvx.linalg.__version__


def test_readme_documents_every_export() -> None:
    """Test that every name in ``cvx.linalg.__all__`` appears in README.md as a backticked identifier."""
    readme = README.read_text(encoding="utf-8")
    missing = [name for name in cvx.linalg.__all__ if not re.search(rf"`{re.escape(name)}\b", readme)]
    assert not missing, f"exported but not documented in README.md: {missing}"
