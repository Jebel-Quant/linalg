"""Tests for the linear solve utility."""

from __future__ import annotations

import numpy as np
import pytest

from cvx.linalg import solve


def test_solve_uses_valid_submatrix() -> None:
    """Test that solve returns NaN for invalid rows and solves the valid block."""
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, np.nan, 0.0], [0.0, 0.0, 4.0]])
    rhs = np.array([2.0, 3.0, 8.0])

    result = solve(matrix, rhs)

    assert np.isnan(result[1])
    assert np.allclose(result[[0, 2]], np.array([2.0, 2.0]))


def test_solve_requires_matching_dimensions() -> None:
    """Test that solve validates matrix compatibility."""
    matrix = np.eye(2)
    rhs = np.array([1.0, 2.0, 3.0])

    with pytest.raises(AssertionError):
        solve(matrix, rhs)
