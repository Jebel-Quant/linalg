"""Tests for the eigvals() utility."""

from __future__ import annotations

import re

import numpy as np
import pytest

from cvx.linalg import NonSquareMatrixError, NotAMatrixError, eigvals


def test_eigvals_general_matrix_can_be_complex() -> None:
    """Test eigvals on a non-symmetric matrix with complex eigenvalues."""
    matrix = np.array([[0.0, -1.0], [1.0, 0.0]])
    values = eigvals(matrix)

    assert np.allclose(np.sort_complex(values), np.sort_complex(np.array([1j, -1j])))


def test_eigvals_matches_numpy_for_general_matrix() -> None:
    """Test eigvals matches NumPy for a general square matrix."""
    matrix = np.array([[1.0, 2.0, 3.0], [0.0, -4.0, 5.0], [2.0, 1.0, 0.0]])

    assert np.allclose(np.sort_complex(eigvals(matrix)), np.sort_complex(np.linalg.eigvals(matrix)))


def test_eigvals_requires_square_matrix() -> None:
    """Test eigvals raises NonSquareMatrixError for non-square input."""
    matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    with pytest.raises(NonSquareMatrixError):
        eigvals(matrix)


def test_eigvals_requires_2d_input() -> None:
    """Test eigvals raises NotAMatrixError for non-2-D input."""
    matrix = np.array([1.0, 2.0, 3.0])

    with pytest.raises(NotAMatrixError):
        eigvals(matrix)


def test_eigvals_one_dimensional_raises_exact_message() -> None:
    """eigvals() names itself in the dimensionality error."""
    with pytest.raises(NotAMatrixError, match=re.escape("eigvals() expected a 2-D matrix, got 1-D input.")):
        eigvals(np.ones(3))


def test_eigvals_non_square_raises_exact_message() -> None:
    """eigvals() reports the actual (rows, cols) of a non-square input."""
    with pytest.raises(NonSquareMatrixError, match=re.escape("Matrix must be square, got shape (2, 3).")):
        eigvals(np.ones((2, 3)))
