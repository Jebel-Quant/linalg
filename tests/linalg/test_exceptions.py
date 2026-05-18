"""Tests for the exception types and warning classes exported from cvx.linalg."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cvx.linalg import (
    DimensionMismatchError,
    IllConditionedMatrixWarning,
    NonSquareMatrixError,
    SingularMatrixError,
    cond,
)


def test_exception_hierarchy() -> None:
    """Test that linalg exceptions are ValueError subclasses."""
    assert issubclass(NonSquareMatrixError, ValueError)
    assert issubclass(DimensionMismatchError, ValueError)
    assert issubclass(SingularMatrixError, ValueError)


def test_non_square_matrix_error_attributes() -> None:
    """Test that NonSquareMatrixError stores the offending dimensions."""
    exc = NonSquareMatrixError(3, 2)
    assert exc.rows == 3
    assert exc.cols == 2
    assert "3" in str(exc)
    assert "2" in str(exc)


def test_dimension_mismatch_error_attributes() -> None:
    """Test that DimensionMismatchError stores the offending sizes."""
    exc = DimensionMismatchError(5, 3)
    assert exc.vector_size == 5
    assert exc.matrix_size == 3
    assert "5" in str(exc)
    assert "3" in str(exc)


def test_ill_conditioned_warning_is_user_warning_subclass() -> None:
    """Test that IllConditionedMatrixWarning inherits from UserWarning."""
    assert issubclass(IllConditionedMatrixWarning, UserWarning)


# ---------------------------------------------------------------------------
# cond()
# ---------------------------------------------------------------------------


def test_cond_identity() -> None:
    """Condition number of the identity matrix is 1."""
    assert cond(np.eye(3)) == pytest.approx(1.0)


def test_cond_diagonal() -> None:
    """Condition number of a diagonal matrix equals max/min eigenvalue ratio."""
    m = np.diag([1.0, 100.0])
    assert cond(m) == pytest.approx(100.0)


def test_cond_p_norm() -> None:
    """cond() respects the optional p parameter."""
    m = np.diag([1.0, 1e6])
    result = cond(m, p=1)
    assert result == pytest.approx(np.linalg.cond(m, p=1))


def test_cond_nan_matrix_returns_nan() -> None:
    """cond() returns nan when the matrix contains NaN entries."""
    m = np.array([[float("nan"), 1.0], [1.0, 2.0]])
    assert math.isnan(cond(m))


def test_cond_inf_matrix_returns_nan() -> None:
    """cond() returns nan when the matrix contains infinite entries."""
    m = np.array([[float("inf"), 0.0], [0.0, 1.0]])
    assert math.isnan(cond(m))
