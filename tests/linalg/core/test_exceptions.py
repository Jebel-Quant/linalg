"""Tests for the exception types and warning classes exported from cvx.linalg."""

from __future__ import annotations

import math
import re
import warnings

import numpy as np
import pytest

from cvx.linalg import (
    DimensionMismatchError,
    IllConditionedMatrixWarning,
    InvalidComponentsError,
    NegativeWarmupError,
    NonIntegerWarmupError,
    NonSquareMatrixError,
    NotAMatrixError,
    SingularMatrixError,
    cond,
)
from cvx.linalg.core.exceptions import DEFAULT_COND_THRESHOLD, warn_ill_conditioned


def test_exception_hierarchy() -> None:
    """Test that linalg exceptions are ValueError subclasses."""
    assert issubclass(NonSquareMatrixError, ValueError)
    assert issubclass(DimensionMismatchError, ValueError)
    assert issubclass(SingularMatrixError, ValueError)
    assert issubclass(NegativeWarmupError, ValueError)
    assert issubclass(InvalidComponentsError, ValueError)
    assert issubclass(NonIntegerWarmupError, TypeError)
    assert issubclass(NotAMatrixError, TypeError)


def test_not_a_matrix_error_function_name() -> None:
    """Test that NotAMatrixError includes the rejecting function's name."""
    assert "qr()" in str(NotAMatrixError(3, func="qr"))
    assert "eigvals()" in str(NotAMatrixError(3))


def test_warmup_errors_have_messages() -> None:
    """Test that warmup errors carry informative messages."""
    assert "-3" in str(NegativeWarmupError(-3))
    assert "bool" in str(NonIntegerWarmupError(True))


def test_invalid_components_error_attributes() -> None:
    """Test that InvalidComponentsError stores the offending values."""
    exc = InvalidComponentsError(10, 5)
    assert exc.n_components == 10
    assert exc.max_components == 5
    assert "10" in str(exc)
    assert "5" in str(exc)


def test_non_square_matrix_error_attributes() -> None:
    """Test that NonSquareMatrixError stores the offending dimensions."""
    exc = NonSquareMatrixError(3, 2)
    assert exc.rows == 3
    assert exc.cols == 2
    assert "3" in str(exc)
    assert "2" in str(exc)


def test_dimension_mismatch_error_attributes() -> None:
    """Test that DimensionMismatchError stores the offending sizes."""
    exc = DimensionMismatchError(5, 3)
    assert exc.vector_size == 5
    assert exc.matrix_size == 3
    assert "5" in str(exc)
    assert "3" in str(exc)


def test_ill_conditioned_warning_is_user_warning_subclass() -> None:
    """Test that IllConditionedMatrixWarning inherits from UserWarning."""
    assert issubclass(IllConditionedMatrixWarning, UserWarning)


# ---------------------------------------------------------------------------
# cond()
# ---------------------------------------------------------------------------


def test_cond_identity() -> None:
    """Condition number of the identity matrix is 1."""
    assert cond(np.eye(3)) == pytest.approx(1.0)


def test_cond_diagonal() -> None:
    """Condition number of a diagonal matrix equals max/min eigenvalue ratio."""
    m = np.diag([1.0, 100.0])
    assert cond(m) == pytest.approx(100.0)


def test_cond_p_norm() -> None:
    """cond() respects the optional p parameter."""
    m = np.diag([1.0, 1e6])
    result = cond(m, p=1)
    assert result == pytest.approx(np.linalg.cond(m, p=1))


def test_cond_nan_matrix_returns_nan() -> None:
    """cond() returns nan when the matrix contains NaN entries."""
    m = np.array([[float("nan"), 1.0], [1.0, 2.0]])
    assert math.isnan(cond(m))


def test_cond_inf_matrix_returns_nan() -> None:
    """cond() returns nan when the matrix contains infinite entries."""
    m = np.array([[float("inf"), 0.0], [0.0, 1.0]])
    assert math.isnan(cond(m))


# ---------------------------------------------------------------------------
# default messages
# ---------------------------------------------------------------------------


def test_singular_matrix_error_without_detail() -> None:
    """SingularMatrixError without detail uses the bare default message."""
    err = SingularMatrixError()
    assert str(err) == "Matrix is singular and cannot be solved."


def test_negative_warmup_error_without_value() -> None:
    """NegativeWarmupError without a value uses the generic message."""
    err = NegativeWarmupError()
    assert str(err) == "warmup must be non-negative."
    assert err.warmup is None


# ---------------------------------------------------------------------------
# exact messages and attributes (mutation-hardening)
# ---------------------------------------------------------------------------


def test_default_cond_threshold_value() -> None:
    """The default condition-number threshold is pinned to 1e12."""
    assert DEFAULT_COND_THRESHOLD == 1e12


def test_not_a_matrix_error_attributes_and_message() -> None:
    """NotAMatrixError carries the exact message and its attributes."""
    err = NotAMatrixError(3, func="qr")
    assert str(err) == "qr() expected a 2-D matrix, got 3-D input."
    assert err.ndim == 3
    assert err.func == "qr"


def test_non_square_matrix_error_message() -> None:
    """NonSquareMatrixError reports rows and columns in order."""
    assert str(NonSquareMatrixError(2, 3)) == "Matrix must be square, got shape (2, 3)."


def test_dimension_mismatch_error_message() -> None:
    """DimensionMismatchError reports vector length then matrix dimension."""
    assert str(DimensionMismatchError(3, 4)) == "Vector length 3 does not match matrix dimension 4."


def test_singular_matrix_error_with_detail() -> None:
    """SingularMatrixError appends the detail after the default message."""
    assert str(SingularMatrixError("boom")) == "Matrix is singular and cannot be solved. boom"


def test_negative_warmup_error_with_value() -> None:
    """NegativeWarmupError reports the offending value and keeps it as attribute."""
    err = NegativeWarmupError(-3)
    assert str(err) == "warmup must be non-negative, got -3."
    assert err.warmup == -3


def test_non_integer_warmup_error_attributes() -> None:
    """NonIntegerWarmupError reports the offending type and keeps the value."""
    err = NonIntegerWarmupError(1.5)
    assert str(err) == "warmup must be an integer, got float."
    assert err.value == 1.5


def test_invalid_components_error_message() -> None:
    """InvalidComponentsError reports the allowed range and the request."""
    assert str(InvalidComponentsError(5, 3)) == "n_components must be between 1 and 3, got 5."


def test_warn_ill_conditioned_exact_message() -> None:
    """warn_ill_conditioned emits the exact documented warning text."""
    msg = "Matrix condition number 2.000e+12 exceeds threshold 1.000e+12; results may be numerically unreliable."
    with pytest.warns(IllConditionedMatrixWarning, match=rf"^{re.escape(msg)}$"):
        warn_ill_conditioned(2e12, 1e12)


def test_warn_ill_conditioned_not_at_exact_threshold() -> None:
    """A condition number exactly at the threshold does not warn."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_ill_conditioned(1e12, 1e12)
