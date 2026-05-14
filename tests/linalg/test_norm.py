"""Tests for the norm utilities."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cvx.linalg import a_norm, inv_a_norm


def test_a_norm_without_matrix_ignores_non_finite_entries() -> None:
    """Test that the Euclidean norm uses only finite vector entries."""
    vector = np.array([3.0, np.nan, 4.0])

    assert a_norm(vector) == pytest.approx(5.0)


def test_a_norm_uses_valid_submatrix() -> None:
    """Test that invalid matrix rows and columns are filtered before evaluation."""
    vector = np.array([10.0, 1.0, 2.0])
    matrix = np.array([[np.nan, 0.0, 0.0], [0.0, 4.0, 1.0], [0.0, 1.0, 9.0]])

    assert a_norm(vector, matrix) == pytest.approx(math.sqrt(44.0))


def test_inv_a_norm_uses_valid_submatrix() -> None:
    """Test that the inverse norm is computed on the finite submatrix only."""
    vector = np.array([10.0, 1.0, 2.0])
    matrix = np.array([[np.nan, 0.0, 0.0], [0.0, 4.0, 1.0], [0.0, 1.0, 9.0]])

    assert inv_a_norm(vector, matrix) == pytest.approx(math.sqrt(3.0 / 5.0))


def test_norms_require_matching_dimensions() -> None:
    """Test that the norm utilities validate matrix compatibility."""
    vector = np.array([1.0, 2.0])
    matrix = np.eye(3)

    with pytest.raises(AssertionError):
        a_norm(vector, matrix)

    with pytest.raises(AssertionError):
        inv_a_norm(vector, matrix)
