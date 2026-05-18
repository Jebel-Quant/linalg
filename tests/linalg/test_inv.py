"""Tests for the guarded matrix inversion utility."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from cvx.linalg import (
    IllConditionedMatrixWarning,
    NonSquareMatrixError,
    SingularMatrixError,
    inv,
)


def test_inv_identity() -> None:
    """Test that inv of an identity matrix returns the identity matrix."""
    result = inv(np.eye(3))
    np.testing.assert_allclose(result, np.eye(3), atol=1e-12)


def test_inv_scaled_identity() -> None:
    """Test inv with a scaled identity matrix."""
    matrix = 4.0 * np.eye(2)
    result = inv(matrix)
    np.testing.assert_allclose(result, 0.25 * np.eye(2), atol=1e-12)


def test_inv_pd_matrix() -> None:
    """Test that inv returns the correct inverse for a positive-definite matrix."""
    matrix = np.array([[4.0, 2.0], [2.0, 3.0]])
    result = inv(matrix)
    np.testing.assert_allclose(matrix @ result, np.eye(2), atol=1e-12)


def test_inv_uses_valid_submatrix() -> None:
    """Test that inv returns NaN rows/columns for invalid entries."""
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, np.nan, 0.0], [0.0, 0.0, 4.0]])
    result = inv(matrix)

    # Row/column 1 should be NaN throughout
    assert np.all(np.isnan(result[1, :]))
    assert np.all(np.isnan(result[:, 1]))

    # Valid block should be correct
    assert np.isclose(result[0, 0], 1.0)
    assert np.isclose(result[2, 2], 0.25)


def test_inv_nan_diagonal_fully_invalid() -> None:
    """Test inv on a fully invalid matrix returns all-NaN."""
    matrix = np.array([[np.nan, 0.0], [0.0, np.nan]])
    result = inv(matrix)
    assert np.all(np.isnan(result))


def test_inv_requires_square_matrix() -> None:
    """Test that inv raises NonSquareMatrixError for non-square matrices."""
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    with pytest.raises(NonSquareMatrixError):
        inv(matrix)


def test_inv_singular_matrix_raises() -> None:
    """Test that inv raises SingularMatrixError for a singular matrix."""
    matrix = np.array([[1.0, 1.0], [1.0, 1.0]])
    with pytest.raises(SingularMatrixError):
        inv(matrix)


def test_inv_ill_conditioned_warns() -> None:
    """Test that inv emits IllConditionedMatrixWarning when condition exceeds threshold."""
    with pytest.warns(IllConditionedMatrixWarning):
        inv(np.eye(2), cond_threshold=0.5)


def test_inv_custom_threshold_suppresses_warning() -> None:
    """Test that a high cond_threshold suppresses IllConditionedMatrixWarning."""
    matrix = np.eye(2)

    with warnings.catch_warnings():
        warnings.simplefilter("error", IllConditionedMatrixWarning)
        result = inv(matrix, cond_threshold=1e20)

    np.testing.assert_allclose(result, np.eye(2), atol=1e-12)


def test_inv_result_shape() -> None:
    """Test that inv returns a matrix with the same shape as the input."""
    matrix = np.eye(4)
    result = inv(matrix)
    assert result.shape == matrix.shape


def test_inv_nan_partial_block_correct() -> None:
    """Test inv with a partial NaN block inverts the valid submatrix correctly."""
    matrix = np.array([[2.0, 0.0, 0.0], [0.0, np.nan, 0.0], [0.0, 0.0, 5.0]])
    result = inv(matrix)

    # Check valid entries
    np.testing.assert_allclose(result[0, 0], 0.5, atol=1e-12)
    np.testing.assert_allclose(result[2, 2], 0.2, atol=1e-12)
    np.testing.assert_allclose(result[0, 2], 0.0, atol=1e-12)
    np.testing.assert_allclose(result[2, 0], 0.0, atol=1e-12)

    # Invalid row/column
    assert np.all(np.isnan(result[1, :]))
    assert np.all(np.isnan(result[:, 1]))
