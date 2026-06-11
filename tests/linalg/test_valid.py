"""Tests for the matrix validation utility."""

from __future__ import annotations

import re

import numpy as np
import pytest

from cvx.linalg import NonSquareMatrixError, valid


def test_valid() -> None:
    """Test that the valid function correctly identifies valid rows/columns in a matrix.

    This test verifies that:
    1. The valid function can process a matrix with NaN values
    2. The function returns the correct boolean vector indicating valid rows/columns
    3. The function returns the correct submatrix with invalid rows/columns removed
    """
    a = np.array([[np.nan, np.nan], [np.nan, 4]])
    v, mat = valid(a)

    assert np.allclose(mat, np.array([[4]]))
    assert np.allclose(v, np.array([False, True]))


def test_invalid() -> None:
    """Test that the valid function raises NonSquareMatrixError for non-square matrices."""
    a = np.zeros((3, 2))
    with pytest.raises(NonSquareMatrixError):
        valid(a)


def test_valid_eye() -> None:
    """Test that valid returns an all-True mask and the full matrix for an identity matrix."""
    a = np.eye(2)
    val, submatrix = valid(a)

    np.testing.assert_array_equal(val, np.array([True, True]))
    np.testing.assert_array_equal(submatrix, np.eye(2))


def test_valid_non_square_raises_exact_message() -> None:
    """valid() reports the actual (rows, cols) of a non-square input."""
    with pytest.raises(NonSquareMatrixError, match=re.escape("Matrix must be square, got shape (2, 3).")):
        valid(np.ones((2, 3)))
