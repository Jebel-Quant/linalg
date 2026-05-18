"""Tests for the exception types and warning classes exported from cvx.linalg."""

from __future__ import annotations

from cvx.linalg import (
    DimensionMismatchError,
    IllConditionedMatrixWarning,
    NonSquareMatrixError,
    SingularMatrixError,
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
