"""Tests for package-level metadata."""

from __future__ import annotations

import cvx.linalg


def test_version_is_exposed() -> None:
    """Test that the package exposes a non-empty __version__ string."""
    assert isinstance(cvx.linalg.__version__, str)
    assert cvx.linalg.__version__
