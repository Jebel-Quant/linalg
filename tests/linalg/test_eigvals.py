"""Tests for the eigvals() utility."""

from __future__ import annotations

import numpy as np
import pytest

from cvx.linalg import NonSquareMatrixError, eigvals


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
    """Test eigvals raises IndexError for non-2-D input."""
    matrix = np.array([1.0, 2.0, 3.0])

    with pytest.raises(IndexError):
        eigvals(matrix)
