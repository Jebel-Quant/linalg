"""Tests for the condition-number helpers in cvx.linalg.core.condition."""

from __future__ import annotations

import importlib
import math
import re
import warnings

import numpy as np
import pytest

from cvx.linalg import IllConditionedMatrixWarning, check_and_warn_condition, cond
from cvx.linalg.core.condition import DEFAULT_COND_THRESHOLD, warn_ill_conditioned

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


def test_default_cond_threshold_value() -> None:
    """The default condition-number threshold is pinned to 1e12."""
    assert DEFAULT_COND_THRESHOLD == 1e12


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

    monkeypatch.setattr("cvx.linalg.core.condition.cond", fail)
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

    monkeypatch.setattr("cvx.linalg.core.condition.cond", fail)
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
    condition = importlib.import_module("cvx.linalg.core.condition")
    assert not condition._certainly_within(matrix, 1e8)


def test_certainly_within_without_scipy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without SciPy the screen never vouches, so the exact check always runs."""
    condition = importlib.import_module("cvx.linalg.core.condition")
    monkeypatch.setattr(condition, "_HAVE_SCIPY", False)
    assert not condition._certainly_within(np.eye(3), DEFAULT_COND_THRESHOLD)


@pytest.mark.parametrize("have_scipy", [True, False])
def test_check_and_warn_condition_same_verdict_both_paths(monkeypatch: pytest.MonkeyPatch, have_scipy: bool) -> None:
    """With or without the screen, the warning fires exactly when the 2-norm condition number exceeds threshold."""
    condition = importlib.import_module("cvx.linalg.core.condition")
    monkeypatch.setattr(condition, "_HAVE_SCIPY", have_scipy)
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
