"""Tests for type aliases defined in types.py."""

from __future__ import annotations

import numpy as np

from cvx.linalg import Matrix


def test_matrix_is_ndarray() -> None:
    """Test that Matrix can be used as a float64 ndarray annotation."""
    a: Matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert isinstance(a, np.ndarray)
    assert a.dtype == np.float64
