"""Tests for type aliases defined in types.py."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from cvx.linalg import DenseOperator, GramOperator, Matrix, SupportsMatvec, Vector


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


def test_supports_matvec_matches_operators() -> None:
    """Operators exposing n and matvec conform to the SupportsMatvec protocol."""
    assert isinstance(DenseOperator(np.eye(3)), SupportsMatvec)
    assert isinstance(GramOperator(np.ones((4, 3))), SupportsMatvec)


def test_supports_matvec_rejects_arrays_and_callables() -> None:
    """Plain arrays and bare callables deliberately do not conform to SupportsMatvec."""
    assert not isinstance(np.eye(3), SupportsMatvec)
    assert not isinstance(lambda v: v, SupportsMatvec)
