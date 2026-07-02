"""Tests for the linear solve utility."""

from __future__ import annotations

import re
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
    """Test that solve warns and raises SingularMatrixError for a singular system."""
    matrix = np.array([[1.0, 1.0], [1.0, 1.0]])
    rhs = np.array([1.0, 2.0])

    with pytest.warns(IllConditionedMatrixWarning), pytest.raises(SingularMatrixError):
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


def test_solve_scaled_identity() -> None:
    """Test solve with a scaled identity matrix."""
    rhs = np.array([3.0, 4.0])
    matrix = 0.5 * np.eye(2)
    x = solve(matrix=matrix, rhs=rhs)
    np.testing.assert_allclose(matrix @ x, rhs, atol=1e-12)


def test_solve_pd_matrix_correct_result() -> None:
    """Test solve returns correct result for a positive-definite matrix."""
    matrix = np.array([[4.0, 2.0], [2.0, 3.0]])
    rhs = np.array([1.0, 2.0])
    x = solve(matrix=matrix, rhs=rhs)
    np.testing.assert_allclose(matrix @ x, rhs, atol=1e-12)


def test_solve_low_threshold_triggers_warning() -> None:
    """Test that IllConditionedMatrixWarning is triggered when threshold is set below cond(I)."""
    matrix = np.eye(2)
    rhs = np.array([1.0, 2.0])

    with pytest.warns(IllConditionedMatrixWarning):
        solve(matrix=matrix, rhs=rhs, cond_threshold=0.5)


def test_solve_matrix_rhs() -> None:
    """Test that solve supports matrix right-hand sides with NaN masking."""
    matrix = np.array([[4.0, 0.0], [0.0, np.nan]])
    rhs = np.array([[8.0, 4.0], [1.0, 2.0]])

    result = solve(matrix, rhs)

    assert result.shape == (2, 2)
    np.testing.assert_allclose(result[0], np.array([2.0, 1.0]))
    assert np.all(np.isnan(result[1]))


def test_solve_non_positive_definite_fallback() -> None:
    """Test that solve falls back to LU and succeeds for an indefinite matrix."""
    # [[1, 2], [2, 1]] has eigenvalues 3 and -1, so it is indefinite.
    matrix = np.array([[1.0, 2.0], [2.0, 1.0]])
    rhs = np.array([1.0, 0.0])
    x = solve(matrix=matrix, rhs=rhs)
    np.testing.assert_allclose(matrix @ x, rhs, atol=1e-12)


def test_solve_all_invalid_matrix_returns_nan() -> None:
    """When no row/column is valid, solve returns an all-NaN solution."""
    matrix = np.full((2, 2), np.nan)
    rhs = np.array([1.0, 2.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = solve(matrix, rhs)
    assert np.all(np.isnan(result))


def test_solve_non_square_raises_exact_message() -> None:
    """solve() reports the actual (rows, cols) of a non-square input."""
    with pytest.raises(NonSquareMatrixError, match=re.escape("Matrix must be square, got shape (2, 3).")):
        solve(np.ones((2, 3)), np.ones(2))
