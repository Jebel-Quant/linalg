"""Tests for the exception types and warning classes exported from cvx.linalg."""

from __future__ import annotations

import importlib
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
    check_and_warn_condition,
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
# exact messages and attributes
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


def test_check_and_warn_condition_warns_when_cond_exceeds_threshold() -> None:
    """check_and_warn_condition computes the matrix condition number, then warns above threshold."""
    # cond(I) == 1, so a threshold below 1 must trip the warning.
    with pytest.warns(IllConditionedMatrixWarning):
        check_and_warn_condition(np.eye(2), 0.5)


def test_check_and_warn_condition_silent_when_well_conditioned() -> None:
    """check_and_warn_condition does not warn when the condition number is within threshold."""
    # cond(I) == 1 <= 2.0, so no warning is emitted.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_and_warn_condition(np.eye(2), 2.0)


def test_warn_ill_conditioned_none_threshold_never_warns() -> None:
    """A ``None`` threshold disables the warning whatever the condition number."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", IllConditionedMatrixWarning)
        warn_ill_conditioned(math.inf, None)


def test_check_and_warn_condition_none_threshold_skips_cond(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``None`` threshold returns before computing the condition number."""

    def fail(*_args: object) -> float:
        """Stand in for ``cond``; any call fails the test."""
        raise AssertionError("cond() called despite threshold=None")  # noqa: TRY003

    monkeypatch.setattr("cvx.linalg.core.exceptions.cond", fail)
    with warnings.catch_warnings():
        warnings.simplefilter("error", IllConditionedMatrixWarning)
        check_and_warn_condition(np.diag([1.0, 1e-14]), None)


def _spd(n: int, seed: int = 0) -> np.ndarray:
    """Return a well-conditioned random symmetric positive-definite matrix."""
    a = np.random.default_rng(seed).standard_normal((n, n))
    return a @ a.T + n * np.eye(n)


def test_check_and_warn_condition_screen_skips_svd(monkeypatch: pytest.MonkeyPatch) -> None:
    """A well-conditioned SPD matrix is cleared by the LAPACK estimate without calling ``cond``."""

    def fail(*_args: object) -> float:
        """Stand in for ``cond``; any call fails the test."""
        raise AssertionError("cond() called for a well-conditioned SPD matrix")  # noqa: TRY003

    monkeypatch.setattr("cvx.linalg.core.exceptions.cond", fail)
    with warnings.catch_warnings():
        warnings.simplefilter("error", IllConditionedMatrixWarning)
        check_and_warn_condition(_spd(20), DEFAULT_COND_THRESHOLD)


@pytest.mark.parametrize(
    "matrix",
    [
        pytest.param(np.array([[1.0, 2.0], [0.0, 1.0]]), id="non-symmetric"),
        pytest.param(np.array([[1.0, 0.0], [0.0, -1.0]]), id="indefinite"),
        pytest.param(np.array([[1.0, np.inf], [np.inf, 1.0]]), id="non-finite"),
        pytest.param(np.diag([1.0, 1e-8]), id="estimate-too-close"),
    ],
)
def test_certainly_within_not_proven(matrix: np.ndarray) -> None:
    """The screen declines to vouch for matrices it cannot bound, leaving them to the exact check."""
    exceptions = importlib.import_module("cvx.linalg.core.exceptions")
    assert not exceptions._certainly_within(matrix, 1e8)


def test_certainly_within_without_scipy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without SciPy the screen never vouches, so the exact check always runs."""
    exceptions = importlib.import_module("cvx.linalg.core.exceptions")
    monkeypatch.setattr(exceptions, "_HAVE_SCIPY", False)
    assert not exceptions._certainly_within(np.eye(3), DEFAULT_COND_THRESHOLD)


@pytest.mark.parametrize("have_scipy", [True, False])
def test_check_and_warn_condition_same_verdict_both_paths(monkeypatch: pytest.MonkeyPatch, have_scipy: bool) -> None:
    """With or without the screen, the warning fires exactly when the 2-norm condition number exceeds threshold."""
    exceptions = importlib.import_module("cvx.linalg.core.exceptions")
    monkeypatch.setattr(exceptions, "_HAVE_SCIPY", have_scipy)
    rng = np.random.default_rng(7)
    for _ in range(50):
        n = int(rng.integers(2, 30))
        q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        matrix = (q * np.logspace(0, rng.uniform(0, 12), n)) @ q.T
        matrix = (matrix + matrix.T) / 2
        threshold = 10.0 ** rng.uniform(0, 12)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            check_and_warn_condition(matrix, threshold)
        assert bool(caught) == (np.linalg.cond(matrix) > threshold)
