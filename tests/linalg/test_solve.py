"""Tests for the linear solve utility."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from cvx.linalg import (
    DimensionMismatchError,
    IllConditionedMatrixWarning,
    NonSquareMatrixError,
    SingularMatrixError,
    solve,
)


def test_solve_uses_valid_submatrix() -> None:
    """Test that solve returns NaN for invalid rows and solves the valid block."""
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, np.nan, 0.0], [0.0, 0.0, 4.0]])
    rhs = np.array([2.0, 3.0, 8.0])

    result = solve(matrix, rhs)

    assert np.isnan(result[1])
    assert np.allclose(result[[0, 2]], np.array([2.0, 2.0]))


def test_solve_requires_matching_dimensions() -> None:
    """Test that solve raises DimensionMismatchError when rhs size does not match."""
    matrix = np.eye(2)
    rhs = np.array([1.0, 2.0, 3.0])

    with pytest.raises(DimensionMismatchError):
        solve(matrix, rhs)


def test_solve_requires_square_matrix() -> None:
    """Test that solve raises NonSquareMatrixError for non-square matrices."""
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    rhs = np.array([1.0, 2.0])

    with pytest.raises(NonSquareMatrixError):
        solve(matrix, rhs)


def test_solve_singular_matrix_raises() -> None:
    """Test that solve raises SingularMatrixError for a singular system."""
    matrix = np.array([[1.0, 1.0], [1.0, 1.0]])
    rhs = np.array([1.0, 2.0])

    with pytest.raises(SingularMatrixError):
        solve(matrix, rhs)


def test_solve_ill_conditioned_warns() -> None:
    """Test that solve emits IllConditionedMatrixWarning when condition exceeds threshold."""
    rng = np.random.default_rng(0)
    base = rng.standard_normal((4, 4))
    matrix = base @ base.T + 1e-14 * np.eye(4)
    rhs = rng.standard_normal(4)

    with pytest.warns(IllConditionedMatrixWarning):
        solve(matrix, rhs, cond_threshold=1.0)


def test_solve_custom_threshold_suppresses_warning() -> None:
    """Test that a high cond_threshold suppresses IllConditionedMatrixWarning."""
    matrix = np.eye(2)
    rhs = np.array([1.0, 2.0])

    with warnings.catch_warnings():
        warnings.simplefilter("error", IllConditionedMatrixWarning)
        result = solve(matrix, rhs, cond_threshold=1e20)

    assert np.allclose(result, rhs)
