"""Tests for type aliases defined in types.py."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from cvx.linalg import Matrix, Vector


def test_matrix_is_ndarray() -> None:
    """Test that Matrix can be used as a float64 ndarray annotation."""
    a: Matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert isinstance(a, np.ndarray)
    assert a.dtype == np.float64


def test_vector_is_ndarray() -> None:
    """Test that Vector can be used as a float64 ndarray annotation."""
    v: Vector = np.array([1.0, 2.0, 3.0])
    assert isinstance(v, np.ndarray)
    assert v.dtype == np.float64


def test_aliases_are_float64_ndarray_types() -> None:
    """Matrix and Vector are precisely NDArray[float64] at runtime."""
    assert Matrix == npt.NDArray[np.float64]
    assert Vector == npt.NDArray[np.float64]
