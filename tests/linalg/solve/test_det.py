"""Tests for the det() matrix determinant utility."""

from __future__ import annotations

import re
import warnings

import numpy as np
import pytest

from cvx.linalg import (
    IllConditionedMatrixWarning,
    NonSquareMatrixError,
    det,
)


def test_det_identity() -> None:
    """Test that the determinant of an identity matrix is 1."""
    assert det(np.eye(3)) == pytest.approx(1.0)


def test_det_known_value() -> None:
    """Test det against a matrix with a known determinant."""
    # [[2, 0], [0, 3]] -> det = 6
    matrix = np.diag([2.0, 3.0])
    assert det(matrix) == pytest.approx(6.0)


def test_det_requires_square_matrix() -> None:
    """Test that det raises NonSquareMatrixError for non-square input."""
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    with pytest.raises(NonSquareMatrixError):
        det(matrix)


def test_det_nan_entry_excluded() -> None:
    """Test that rows/columns with NaN diagonal are excluded before computing det."""
    # Valid sub-block is [[2.0]], det = 2.0
    matrix = np.array([[2.0, 0.0], [0.0, np.nan]])
    assert det(matrix) == pytest.approx(2.0)


def test_det_all_nan_returns_nan() -> None:
    """Test that det returns NaN when every diagonal entry is NaN."""
    matrix = np.array([[np.nan, 0.0], [0.0, np.nan]])
    assert np.isnan(det(matrix))


def test_det_ill_conditioned_warns() -> None:
    """Test that det emits IllConditionedMatrixWarning for a near-singular matrix."""
    matrix = np.eye(2)
    with pytest.warns(IllConditionedMatrixWarning):
        det(matrix, cond_threshold=0.5)


def test_det_high_threshold_suppresses_warning() -> None:
    """Test that a high cond_threshold suppresses IllConditionedMatrixWarning."""
    matrix = np.eye(2)
    with warnings.catch_warnings():
        warnings.simplefilter("error", IllConditionedMatrixWarning)
        result = det(matrix, cond_threshold=1e20)
    assert result == pytest.approx(1.0)


def test_det_singular_matrix() -> None:
    """Test that det returns 0 (and warns) for a singular matrix."""
    matrix = np.array([[1.0, 1.0], [1.0, 1.0]])
    with pytest.warns(IllConditionedMatrixWarning):
        result = det(matrix)
    assert result == pytest.approx(0.0, abs=1e-12)


def test_det_non_square_raises_exact_message() -> None:
    """det() reports the actual (rows, cols) of a non-square input."""
    with pytest.raises(NonSquareMatrixError, match=re.escape("Matrix must be square, got shape (2, 3).")):
        det(np.ones((2, 3)))
