"""Fuzz the cvx.linalg numeric routines against arbitrary square matrices.

The linalg utilities ingest covariance/correlation-shaped float matrices that,
in production, may contain non-finite values (NaN/inf) or be singular,
ill-conditioned or degenerate. Each routine is contracted to either return a
result or raise one of its *documented* domain errors (or numpy's
``LinAlgError``) — never to crash with an unexpected exception. This harness
exercises that contract with coverage-guided input.

Run locally:
    pip install atheris numpy
    python tests/fuzz/fuzz_linalg.py -atheris_runs=20000

Run in ClusterFuzzLite: this file is built by .clusterfuzzlite/build.sh.
"""

from __future__ import annotations

import contextlib
import sys

import atheris
import numpy as np

with atheris.instrument_imports():
    from cvx.linalg import (
        cholesky,
        cond,
        cov_to_corr,
        det,
        eigh,
        eigvals,
        eigvalsh,
        inv,
        norm,
        solve,
        valid,
    )
    from cvx.linalg.exceptions import (
        DimensionMismatchError,
        InvalidComponentsError,
        NonSquareMatrixError,
        NotAMatrixError,
        SingularMatrixError,
    )

# Domain errors the routines are *allowed* to raise on adversarial input.
# Anything outside this set propagates and is recorded by Atheris as a crash.
_ALLOWED = (
    DimensionMismatchError,
    InvalidComponentsError,
    NonSquareMatrixError,
    NotAMatrixError,
    SingularMatrixError,
    np.linalg.LinAlgError,
)


def _matrix(fdp: atheris.FuzzedDataProvider) -> np.ndarray:
    """Build an ``n x n`` float64 matrix from fuzzed bytes (n in [0, 8])."""
    n = fdp.ConsumeIntInRange(0, 8)
    # ConsumeFloat may yield NaN/inf, which is exactly the non-finite input the
    # validation/cleaning routines are designed to tolerate.
    floats = [fdp.ConsumeFloat() for _ in range(n * n)]
    return np.array(floats, dtype=np.float64).reshape(n, n)


def test_one_input(data: bytes) -> None:
    """Run a fuzzed square matrix through the public linalg routines."""
    fdp = atheris.FuzzedDataProvider(data)
    matrix = _matrix(fdp)
    # A conforming right-hand side (column vector of matching length) so solve()
    # exercises its numeric path rather than only the shape guard.
    rhs = matrix[:, 0].copy() if matrix.shape[0] else matrix.reshape(0)

    for call in (
        lambda: valid(matrix),
        lambda: cond(matrix),
        lambda: det(matrix),
        lambda: norm(matrix),
        lambda: inv(matrix),
        lambda: solve(matrix, rhs),
        lambda: cholesky(matrix),
        lambda: eigh(matrix),
        lambda: eigvals(matrix),
        lambda: eigvalsh(matrix),
        lambda: cov_to_corr(matrix),
    ):
        with contextlib.suppress(_ALLOWED):
            call()


def main() -> None:
    """Run the Atheris fuzz loop."""
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
