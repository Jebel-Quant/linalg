"""Tests for symmetric/Hermitian eigendecomposition utilities."""

from __future__ import annotations

import numpy as np
import pytest

from cvx.linalg import NonSquareMatrixError, eigh, eigvalsh


def test_eigh_matches_numpy_for_symmetric_matrix() -> None:
    """Test eigh matches NumPy for a real symmetric matrix."""
    matrix = np.array([[4.0, 2.0], [2.0, 3.0]])

    expected_values, expected_vectors = np.linalg.eigh(matrix)
    values, vectors = eigh(matrix)

    np.testing.assert_allclose(values, expected_values, atol=1e-12)
    np.testing.assert_allclose(np.abs(vectors), np.abs(expected_vectors), atol=1e-12)


def test_eigh_supports_hermitian_matrix() -> None:
    """Test eigh supports complex Hermitian matrices."""
    matrix = np.array([[2.0 + 0.0j, 1.0 - 2.0j], [1.0 + 2.0j, 5.0 + 0.0j]])

    values, vectors = eigh(matrix)

    np.testing.assert_allclose(matrix @ vectors, vectors @ np.diag(values), atol=1e-12)
    assert np.all(np.diff(values) >= 0.0)


def test_eigh_uses_valid_submatrix() -> None:
    """Test eigh removes rows/columns with non-finite diagonal entries."""
    matrix = np.array([[4.0, 1.0, 0.0], [1.0, np.nan, 2.0], [0.0, 2.0, 3.0]])

    values, vectors = eigh(matrix)
    expected_values, expected_vectors = np.linalg.eigh(np.array([[4.0, 0.0], [0.0, 3.0]]))

    np.testing.assert_allclose(values, expected_values, atol=1e-12)
    np.testing.assert_allclose(np.abs(vectors), np.abs(expected_vectors), atol=1e-12)


def test_eigvalsh_returns_only_eigenvalues() -> None:
    """Test eigvalsh returns the same values as eigh."""
    matrix = np.array([[3.0, -1.0], [-1.0, 3.0]])

    values = eigvalsh(matrix)
    expected_values, _ = np.linalg.eigh(matrix)

    np.testing.assert_allclose(values, expected_values, atol=1e-12)


def test_eigh_all_invalid_returns_empty() -> None:
    """Test eigh returns empty outputs when all rows/columns are invalid."""
    matrix = np.array([[np.nan, 0.0], [0.0, np.nan]])
    values, vectors = eigh(matrix)

    assert values.shape == (0,)
    assert vectors.shape == (0, 0)


def test_eigh_requires_square_matrix() -> None:
    """Test eigh raises NonSquareMatrixError for non-square matrices."""
    matrix = np.zeros((2, 3))

    with pytest.raises(NonSquareMatrixError):
        eigh(matrix)
