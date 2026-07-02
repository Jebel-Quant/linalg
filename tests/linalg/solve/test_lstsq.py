"""Tests for the least-squares solver utility."""

from __future__ import annotations

import re
import warnings

import numpy as np
import pytest

from cvx.linalg import DimensionMismatchError, IllConditionedMatrixWarning, lstsq


def test_lstsq_overdetermined_basic() -> None:
    """Test lstsq returns correct solution for a simple overdetermined system."""
    mat = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    b = np.array([6.0, 5.0, 7.0])
    x, _residuals, rank, sv = lstsq(mat, b)
    assert rank == 2
    assert sv.shape == (2,)
    # Verify residuals match numpy directly.
    x_np, _res_np, rank_np, _sv_np = np.linalg.lstsq(mat, b, rcond=None)
    np.testing.assert_allclose(x, x_np, atol=1e-12)
    assert rank == rank_np


def test_lstsq_exact_solution() -> None:
    """Test lstsq recovers the exact solution when system is consistent."""
    mat = np.eye(3)
    b = np.array([1.0, 2.0, 3.0])
    x, _, rank, _ = lstsq(mat, b)
    np.testing.assert_allclose(x, b, atol=1e-12)
    assert rank == 3


def test_lstsq_nan_rows_filtered() -> None:
    """Test that NaN rows in the matrix are excluded before solving."""
    mat = np.array([[1.0, 0.0], [np.nan, 1.0], [0.0, 1.0]])
    b = np.array([2.0, 99.0, 3.0])
    x, _, rank, _ = lstsq(mat, b)
    # Should be solved on rows 0 and 2 only (identity system).
    np.testing.assert_allclose(x, np.array([2.0, 3.0]), atol=1e-12)
    assert rank == 2


def test_lstsq_nan_rhs_filtered() -> None:
    """Test that rows where rhs is NaN are excluded before solving."""
    mat = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    b = np.array([2.0, np.nan, 3.0])
    x, _, rank, _ = lstsq(mat, b)
    # Row 1 dropped → least-squares on rows 0 and 2.
    assert np.isfinite(x).all()
    assert rank > 0


def test_lstsq_all_nan_rows_returns_nan_solution() -> None:
    """Test that all-NaN input returns NaN solution and empty residuals/sv."""
    mat = np.full((3, 2), np.nan)
    b = np.array([1.0, 2.0, 3.0])
    x, residuals, rank, sv = lstsq(mat, b)
    assert np.all(np.isnan(x))
    assert x.shape == (2,)
    assert rank == 0
    assert residuals.size == 0
    assert sv.size == 0


def test_lstsq_dimension_mismatch_raises() -> None:
    """Test that DimensionMismatchError is raised when rhs length != matrix rows."""
    mat = np.eye(3)
    b = np.array([1.0, 2.0])
    with pytest.raises(DimensionMismatchError):
        lstsq(mat, b)


def test_lstsq_ill_conditioned_warns() -> None:
    """Test that IllConditionedMatrixWarning is emitted when condition exceeds threshold."""
    rng = np.random.default_rng(42)
    mat = rng.standard_normal((10, 4))
    # Make mat nearly rank-deficient by repeating a column.
    mat[:, 3] = mat[:, 2] + 1e-15 * rng.standard_normal(10)
    b = rng.standard_normal(10)
    with pytest.warns(IllConditionedMatrixWarning):
        lstsq(mat, b, cond_threshold=1.0)


def test_lstsq_high_threshold_suppresses_warning() -> None:
    """Test that a very high cond_threshold suppresses IllConditionedMatrixWarning."""
    mat = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    b = np.array([1.0, 2.0, 3.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error", IllConditionedMatrixWarning)
        lstsq(mat, b, cond_threshold=1e20)


def test_lstsq_return_tuple_structure() -> None:
    """Test that lstsq always returns a 4-tuple (x, residuals, rank, sv)."""
    mat = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    b = np.array([1.0, 2.0, 3.0])
    result = lstsq(mat, b)
    assert len(result) == 4
    x, _residuals, rank, sv = result
    assert isinstance(x, np.ndarray)
    assert isinstance(rank, (int, np.integer))
    assert isinstance(sv, np.ndarray)


def test_lstsq_underdetermined() -> None:
    """Test lstsq handles underdetermined systems (more columns than rows)."""
    mat = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    b = np.array([1.0, 2.0])
    x, _, rank, _ = lstsq(mat, b)
    # Verify the solution satisfies mat @ x ≈ b (minimum-norm solution).
    np.testing.assert_allclose(mat @ x, b, atol=1e-12)
    assert rank == 2


def test_lstsq_rank_deficient_warns_with_inf_cond() -> None:
    """Test that a rank-deficient matrix (sv[-1] == 0) triggers inf condition warning."""
    # Second column is all zeros → sv[-1] == 0.0 exactly, cond treated as inf.
    mat = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    b = np.array([1.0, 2.0, 3.0])
    with pytest.warns(IllConditionedMatrixWarning):
        lstsq(mat, b, cond_threshold=1.0)


def test_lstsq_zero_columns_returns_empty_solution() -> None:
    """Test lstsq with a zero-column matrix returns empty x and sv without warning."""
    mat = np.zeros((3, 0))
    b = np.array([1.0, 2.0, 3.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error", IllConditionedMatrixWarning)
        x, _, _rank, sv = lstsq(mat, b)
    assert x.shape == (0,)
    assert sv.size == 0


def test_lstsq_dimension_mismatch_reports_row_count() -> None:
    """lstsq() reports the matrix ROW count in the mismatch error, not columns."""
    with pytest.raises(
        DimensionMismatchError,
        match=re.escape("Vector length 3 does not match matrix dimension 4."),
    ):
        lstsq(np.ones((4, 2)), np.ones(3))


def test_lstsq_exactly_singular_single_column_warns() -> None:
    """A zero single-column system has an infinite condition number and warns."""
    with pytest.warns(IllConditionedMatrixWarning):
        lstsq(np.zeros((2, 1)), np.array([1.0, 2.0]))


def test_lstsq_zero_columns_has_unit_condition_number() -> None:
    """With no columns there are no singular values; cond defaults to exactly 1."""
    x, _residuals, rank, sv = lstsq(np.empty((2, 0)), np.array([1.0, 2.0]), cond_threshold=1.5)
    assert x.size == 0
    assert rank == 0
    assert sv.size == 0
