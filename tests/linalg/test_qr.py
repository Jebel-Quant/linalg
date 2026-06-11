"""Tests for the QR decomposition utility."""

from __future__ import annotations

import numpy as np
import pytest

from cvx.linalg import NotAMatrixError, qr


@pytest.mark.parametrize(
    "matrix",
    [
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    ],
)
def test_qr_matches_numpy_reduced(matrix: np.ndarray) -> None:
    """Test that qr returns reduced QR matching numpy."""
    q, r = qr(matrix)
    expected_q, expected_r = np.linalg.qr(matrix, mode="reduced")

    np.testing.assert_allclose(q, expected_q)
    np.testing.assert_allclose(r, expected_r)


def test_qr_requires_two_dimensional_matrix() -> None:
    """Test that qr raises NotAMatrixError for non-2D inputs."""
    with pytest.raises(NotAMatrixError, match="qr"):
        qr(np.array([1.0, 2.0, 3.0]))
