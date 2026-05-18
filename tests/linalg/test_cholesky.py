"""Tests for the Cholesky decomposition utility."""

from __future__ import annotations

import numpy as np
import pytest

from cvx.linalg import cholesky, cholesky_solve, is_positive_definite, rand_cov


def test_cholesky() -> None:
    """Test that the cholesky function correctly decomposes a covariance matrix.

    This test verifies that:
    1. A random covariance matrix can be generated
    2. The cholesky function returns an upper triangular matrix
    3. The product of the transpose of the Cholesky factor and the Cholesky factor
       equals the original covariance matrix
    """
    a = rand_cov(10)
    u = cholesky(a)
    # test numpy arrays are equivalent
    assert np.allclose(a, np.transpose(u) @ u)


def test_is_positive_definite_true() -> None:
    """Test that is_positive_definite returns True for a positive-definite matrix."""
    assert is_positive_definite(np.eye(3))
    assert is_positive_definite(np.array([[1.0, 0.5], [0.5, 1.0]]))


def test_is_positive_definite_false() -> None:
    """Test that is_positive_definite returns False for non-positive-definite matrices."""
    assert not is_positive_definite(np.array([[1.0, 2.0], [2.0, 1.0]]))
    assert not is_positive_definite(np.zeros((2, 2)))


def test_cholesky_solve_identity() -> None:
    """Test that cholesky_solve correctly solves an identity system."""
    rhs = np.array([1.0, 2.0])
    result = cholesky_solve(np.eye(2), rhs)
    assert np.allclose(result, rhs)


def test_cholesky_solve_pd_matrix() -> None:
    """Test that cholesky_solve gives the correct solution for a positive-definite matrix."""
    matrix = np.array([[4.0, 0.0], [0.0, 9.0]])
    rhs = np.array([8.0, 27.0])
    result = cholesky_solve(matrix, rhs)
    assert np.allclose(result, np.array([2.0, 3.0]))


def test_cholesky_solve_non_pd_falls_back_to_lu() -> None:
    """Test that cholesky_solve falls back to LU for a non-positive-definite matrix."""
    matrix = np.array([[1.0, 0.0], [0.0, -1.0]])
    rhs = np.array([2.0, -3.0])
    result = cholesky_solve(matrix, rhs)
    assert np.allclose(result, np.array([2.0, 3.0]))


def test_cholesky_solve_random() -> None:
    """Test that cholesky_solve satisfies matrix @ x = rhs for a random PD matrix."""
    a = rand_cov(5)
    rhs = np.random.default_rng(0).standard_normal(5)
    x = cholesky_solve(a, rhs)
    assert np.allclose(a @ x, rhs)


def test_cholesky_solve_singular_raises() -> None:
    """Test that cholesky_solve raises LinAlgError for a singular matrix."""
    matrix = np.array([[1.0, 1.0], [1.0, 1.0]])
    rhs = np.array([1.0, 2.0])
    with pytest.raises(np.linalg.LinAlgError):
        cholesky_solve(matrix, rhs)


def test_is_positive_definite_correlation_like() -> None:
    """Test that a shrunk correlation matrix is positive-definite."""
    matrix = np.array([[1.0, 0.3], [0.3, 1.0]])
    assert is_positive_definite(matrix) is True


def test_is_positive_definite_negative_definite() -> None:
    """Test that a negative-definite matrix returns False."""
    assert is_positive_definite(-1.0 * np.eye(2)) is False


def test_is_positive_definite_singular() -> None:
    """Test that a rank-deficient matrix returns False."""
    assert is_positive_definite(np.array([[1.0, 1.0], [1.0, 1.0]])) is False


def test_is_positive_definite_scalar_positive() -> None:
    """Test that a 1x1 positive matrix is positive-definite."""
    assert is_positive_definite(np.array([[5.0]])) is True


def test_is_positive_definite_scalar_zero() -> None:
    """Test that a 1x1 zero matrix is not positive-definite."""
    assert is_positive_definite(np.array([[0.0]])) is False
