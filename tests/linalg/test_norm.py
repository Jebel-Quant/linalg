"""Tests for the norm utilities."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from cvx.linalg import (
    DimensionMismatchError,
    IllConditionedMatrixWarning,
    NonSquareMatrixError,
    SingularMatrixError,
    a_norm,
    inv_a_norm,
    norm,
)


def test_norm_vector_l2() -> None:
    """Test that norm() computes the 2-norm of a vector, ignoring NaN."""
    assert norm(np.array([3.0, np.nan, 4.0])) == pytest.approx(5.0)


def test_norm_vector_l1() -> None:
    """Test that norm() computes the 1-norm, treating NaN as zero."""
    assert norm(np.array([1.0, np.nan, 2.0]), ord=1) == pytest.approx(3.0)


def test_norm_vector_inf() -> None:
    """Test that norm() computes the inf-norm, treating NaN as zero."""
    assert norm(np.array([3.0, np.nan, 4.0]), ord=np.inf) == pytest.approx(4.0)


def test_norm_matrix_frobenius() -> None:
    """Test Frobenius norm of a matrix with NaN entries treated as zero."""
    m = np.array([[1.0, np.nan], [np.nan, 1.0]])
    assert norm(m, ord="fro") == pytest.approx(math.sqrt(2.0))


def test_norm_matrix_nuclear() -> None:
    """Test nuclear norm of an identity matrix (sum of singular values = n)."""
    assert norm(np.eye(3), ord="nuc") == pytest.approx(3.0)


def test_norm_all_nan_returns_zero() -> None:
    """Test that a fully NaN array returns 0 (all entries treated as zero)."""
    assert norm(np.array([np.nan, np.nan])) == pytest.approx(0.0)


def test_norm_finite_vector_matches_numpy() -> None:
    """Test that norm() matches np.linalg.norm for fully finite inputs."""
    rng = np.random.default_rng(42)
    v = rng.standard_normal(10)
    assert norm(v) == pytest.approx(float(np.linalg.norm(v)))


def test_norm_finite_matrix_matches_numpy() -> None:
    """Test that norm() matches np.linalg.norm for fully finite matrix."""
    rng = np.random.default_rng(42)
    m = rng.standard_normal((4, 4))
    assert norm(m) == pytest.approx(float(np.linalg.norm(m)))


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
    """Test that the norm utilities raise DimensionMismatchError for incompatible shapes."""
    vector = np.array([1.0, 2.0])
    matrix = np.eye(3)

    with pytest.raises(DimensionMismatchError):
        a_norm(vector, matrix)

    with pytest.raises(DimensionMismatchError):
        inv_a_norm(vector, matrix)


def test_norms_require_square_matrix() -> None:
    """Test that the norm utilities raise NonSquareMatrixError for non-square matrices."""
    vector = np.array([1.0, 2.0])
    matrix = np.array([[2.0, 0.0, 1.0], [0.0, 3.0, 4.0]])

    with pytest.raises(NonSquareMatrixError):
        a_norm(vector, matrix)

    with pytest.raises(NonSquareMatrixError):
        inv_a_norm(vector, matrix)


def test_inv_a_norm_without_matrix_ignores_non_finite_entries() -> None:
    """Test that inv_a_norm without a matrix falls back to the Euclidean norm."""
    vector = np.array([3.0, np.nan, 4.0])

    assert inv_a_norm(vector) == pytest.approx(5.0)


def test_a_norm_all_invalid_matrix_returns_nan() -> None:
    """Test that a_norm returns NaN when no matrix diagonal entry is finite."""
    vector = np.array([1.0, 2.0])
    matrix = np.array([[np.nan, 0.0], [0.0, np.nan]])

    assert math.isnan(a_norm(vector, matrix))


def test_inv_a_norm_all_invalid_matrix_returns_nan() -> None:
    """Test that inv_a_norm returns NaN when no matrix diagonal entry is finite."""
    vector = np.array([1.0, 2.0])
    matrix = np.array([[np.nan, 0.0], [0.0, np.nan]])

    assert math.isnan(inv_a_norm(vector, matrix))


def test_inv_a_norm_singular_matrix_raises() -> None:
    """Test that inv_a_norm raises SingularMatrixError for a singular system."""
    vector = np.array([1.0, 2.0])
    matrix = np.array([[1.0, 1.0], [1.0, 1.0]])

    with pytest.raises(SingularMatrixError):
        inv_a_norm(vector, matrix)


def test_inv_a_norm_ill_conditioned_warns() -> None:
    """Test that inv_a_norm emits IllConditionedMatrixWarning when condition exceeds threshold."""
    rng = np.random.default_rng(0)
    base = rng.standard_normal((4, 4))
    matrix = base @ base.T + 1e-14 * np.eye(4)
    vector = rng.standard_normal(4)

    with pytest.warns(IllConditionedMatrixWarning):
        inv_a_norm(vector, matrix, cond_threshold=1.0)


def test_inv_a_norm_custom_threshold_suppresses_warning() -> None:
    """Test that a high cond_threshold suppresses IllConditionedMatrixWarning."""
    matrix = np.eye(2)
    vector = np.array([1.0, 2.0])

    with warnings.catch_warnings():
        warnings.simplefilter("error", IllConditionedMatrixWarning)
        result = inv_a_norm(vector, matrix, cond_threshold=1e20)

    assert result == pytest.approx(math.sqrt(5.0))


def test_inv_a_norm_with_scaled_identity() -> None:
    """Test inv_a_norm with a scaled identity matrix."""
    v = np.array([3.0, 4.0])
    a = 0.5 * np.eye(2)
    assert inv_a_norm(vector=v, matrix=a) == pytest.approx(np.sqrt(2) * 5.0)


def test_inv_a_norm_without_matrix_returns_euclidean_norm() -> None:
    """Test that inv_a_norm without a matrix returns the Euclidean norm."""
    v = np.array([3.0, 4.0])
    assert inv_a_norm(vector=v) == pytest.approx(5.0)


def test_inv_a_norm_pd_matrix_correct_result() -> None:
    """Test inv_a_norm gives correct result via Cholesky for a positive-definite matrix."""
    matrix = np.array([[4.0, 2.0], [2.0, 3.0]])
    v = np.array([1.0, 0.0])
    expected = float(np.sqrt(np.dot(v, np.linalg.solve(matrix, v))))
    assert inv_a_norm(vector=v, matrix=matrix) == pytest.approx(expected)
