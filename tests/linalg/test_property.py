"""Property-based tests for cvx.linalg using Hypothesis.

Properties under test
---------------------
valid
  - The returned mask always has the same length as the matrix dimension.
  - A matrix whose diagonal is entirely finite yields an all-True mask.
  - The returned sub-matrix has side length equal to the number of True entries.

inv_a_norm (matrix=None)
  - Result equals ``np.linalg.norm(v[np.isfinite(v)], 2)`` for every input.
  - Result is always non-negative.

inv_a_norm (matrix=identity)
  - With an identity matrix the A-norm collapses to the Euclidean norm.

solve (matrix=identity)
  - The solution of I·x = b equals b.

solve (diagonal matrix)
  - The solution satisfies matrix @ x ≈ rhs at valid (non-NaN) positions.

Every other public function has at least one algebraic property test below:
cholesky/cholesky_solve/is_positive_definite, cov_to_corr, det, eigh/eigvalsh/
eigvals, inv, lstsq, qr, rand_cov, svd, pca, a_norm, cond, and norm
(ewm_covariance is property-tested in test_ewm_cov.py, which owns the polars
dependency).
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as np_st

from cvx.linalg import (
    a_norm,
    cholesky,
    cholesky_solve,
    cond,
    cov_to_corr,
    det,
    eigh,
    eigvals,
    eigvalsh,
    inv,
    inv_a_norm,
    is_positive_definite,
    lstsq,
    norm,
    pca,
    qr,
    rand_cov,
    solve,
    svd,
    valid,
)

# ─── Shared strategies ────────────────────────────────────────────────────────

_SIZES = st.integers(min_value=1, max_value=8)
_FINITE_FLOAT = st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False)
_MAYBE_NAN_FLOAT = st.floats(allow_nan=True, allow_infinity=False)
# Magnitudes are capped so that squaring (norm computes sqrt(v·v)) stays below
# the float64 max; unbounded values overflow in `dot` and OpenBLAS surfaces
# that as a RuntimeWarning, which strict filterwarnings promotes to an error.
_BOUNDED_OR_NAN_FLOAT = st.one_of(
    st.just(float("nan")),
    st.floats(min_value=-1e150, max_value=1e150, allow_nan=False, allow_infinity=False),
)


def _square_matrix(n: int) -> st.SearchStrategy[np.ndarray]:
    """Strategy for an (n, n) float64 matrix with possibly-NaN entries."""
    return np_st.arrays(dtype=np.float64, shape=(n, n), elements=_MAYBE_NAN_FLOAT)


def _finite_square_matrix(n: int) -> st.SearchStrategy[np.ndarray]:
    """Strategy for an (n, n) finite float64 matrix."""
    return np_st.arrays(dtype=np.float64, shape=(n, n), elements=_FINITE_FLOAT)


def _finite_vector(n: int) -> st.SearchStrategy[np.ndarray]:
    """Strategy for a length-n finite float64 vector."""
    return np_st.arrays(dtype=np.float64, shape=(n,), elements=_FINITE_FLOAT)


# ─── valid() ─────────────────────────────────────────────────────────────────


@given(n=_SIZES, data=st.data())
@settings(max_examples=200)
def test_valid_mask_length_equals_matrix_size(n: int, data: st.DataObject) -> None:
    """valid() must return a mask whose length equals the matrix dimension."""
    matrix = data.draw(_square_matrix(n))
    mask, _ = valid(matrix)
    assert len(mask) == n


@given(n=_SIZES, data=st.data())
@settings(max_examples=200)
def test_valid_finite_diagonal_gives_all_true_mask(n: int, data: st.DataObject) -> None:
    """A matrix whose diagonal is entirely finite must yield an all-True mask."""
    matrix = data.draw(_finite_square_matrix(n))
    mask, submatrix = valid(matrix)
    assert mask.all()
    assert submatrix.shape == (n, n)


@given(n=_SIZES, data=st.data())
@settings(max_examples=200)
def test_valid_submatrix_shape_matches_true_count(n: int, data: st.DataObject) -> None:
    """The sub-matrix side length must equal the number of True entries in the mask."""
    matrix = data.draw(_square_matrix(n))
    mask, submatrix = valid(matrix)
    k = int(mask.sum())
    assert submatrix.shape == (k, k)


# ─── inv_a_norm (no matrix) ───────────────────────────────────────────────────


@given(
    vector=np_st.arrays(
        dtype=np.float64,
        shape=st.integers(min_value=1, max_value=20),
        elements=_BOUNDED_OR_NAN_FLOAT,
    )
)
@settings(max_examples=300)
def test_inv_a_norm_no_matrix_equals_euclidean_norm(vector: np.ndarray) -> None:
    """inv_a_norm(v) must equal the Euclidean norm of the finite entries in v."""
    result = inv_a_norm(vector)
    expected = float(np.linalg.norm(vector[np.isfinite(vector)], 2))
    assert result == pytest.approx(expected, abs=1e-10)


@given(
    vector=np_st.arrays(
        dtype=np.float64,
        shape=st.integers(min_value=1, max_value=20),
        elements=_FINITE_FLOAT,
    )
)
@settings(max_examples=300)
def test_inv_a_norm_no_matrix_is_non_negative(vector: np.ndarray) -> None:
    """The Euclidean norm of any finite vector is always >= 0."""
    result = inv_a_norm(vector)
    assert result >= 0.0


# ─── inv_a_norm (identity matrix) ────────────────────────────────────────────


@given(n=_SIZES, data=st.data())
@settings(max_examples=200)
def test_inv_a_norm_with_identity_equals_euclidean_norm(n: int, data: st.DataObject) -> None:
    """With an identity matrix, the inverse A-norm collapses to the Euclidean norm."""
    vector = data.draw(_finite_vector(n))
    result = inv_a_norm(vector=vector, matrix=np.eye(n))
    expected = float(np.linalg.norm(vector, 2))
    assert result == pytest.approx(expected, rel=1e-9, abs=1e-12)


# ─── solve (identity) ────────────────────────────────────────────────────────


@given(n=_SIZES, data=st.data())
@settings(max_examples=200)
def test_solve_identity_returns_rhs(n: int, data: st.DataObject) -> None:
    """Solving I·x = b must return b exactly."""
    rhs = data.draw(_finite_vector(n))
    result = solve(matrix=np.eye(n), rhs=rhs)
    np.testing.assert_allclose(result, rhs, rtol=1e-10, atol=1e-12)


# ─── solve (positive diagonal matrix) ────────────────────────────────────────


@given(n=_SIZES, data=st.data())
@settings(max_examples=200)
def test_solve_diagonal_satisfies_linear_system(n: int, data: st.DataObject) -> None:
    """For a positive-definite diagonal matrix, matrix @ solve(matrix, rhs) == rhs."""
    diag_vals = data.draw(
        np_st.arrays(
            dtype=np.float64,
            shape=(n,),
            elements=st.floats(min_value=0.1, max_value=1e3, allow_nan=False, allow_infinity=False),
        )
    )
    matrix = np.diag(diag_vals)
    rhs = data.draw(_finite_vector(n))

    x = solve(matrix=matrix, rhs=rhs)

    assert np.all(np.isfinite(x))
    np.testing.assert_allclose(matrix @ x, rhs, rtol=1e-9, atol=1e-9)


# ─── Shared bounded strategies for numerical stability ───────────────────────
# Entries are kept small so products like A @ A.T stay far from the float64
# max on every BLAS backend (OpenBLAS promotes overflow to RuntimeWarning).

_SMALL_FLOAT = st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)


def _bounded_matrix(rows: int, cols: int) -> st.SearchStrategy[np.ndarray]:
    """Strategy for a (rows, cols) float64 matrix with entries in [-10, 10]."""
    return np_st.arrays(dtype=np.float64, shape=(rows, cols), elements=_SMALL_FLOAT)


def _pd_matrix(n: int) -> st.SearchStrategy[np.ndarray]:
    """Strategy for a well-conditioned (n, n) positive-definite matrix."""
    return _bounded_matrix(n, n).map(lambda a: a @ a.T + np.eye(n))


# ─── cholesky / cholesky_solve / is_positive_definite ────────────────────────


@given(n=_SIZES, data=st.data())
@settings(max_examples=100)
def test_cholesky_factor_reconstructs_matrix(n: int, data: st.DataObject) -> None:
    """cholesky() returns an upper triangular R with R.T @ R == cov."""
    cov = data.draw(_pd_matrix(n))
    r = cholesky(cov)
    assert np.allclose(r, np.triu(r))
    np.testing.assert_allclose(r.T @ r, cov, rtol=1e-8, atol=1e-8)


@given(n=_SIZES, data=st.data())
@settings(max_examples=100)
def test_cholesky_solve_satisfies_system(n: int, data: st.DataObject) -> None:
    """cholesky_solve() returns x with cov @ x == rhs for PD cov."""
    cov = data.draw(_pd_matrix(n))
    rhs = data.draw(_finite_vector(n))
    x = cholesky_solve(cov, rhs)
    np.testing.assert_allclose(cov @ x, rhs, rtol=1e-6, atol=1e-6)


@given(n=_SIZES, data=st.data())
@settings(max_examples=100)
def test_is_positive_definite_sign(n: int, data: st.DataObject) -> None:
    """A @ A.T + I is positive-definite and its negation is not."""
    cov = data.draw(_pd_matrix(n))
    assert is_positive_definite(cov)
    assert not is_positive_definite(-cov)


# ─── cov_to_corr ──────────────────────────────────────────────────────────────


@given(n=_SIZES, data=st.data())
@settings(max_examples=100)
def test_cov_to_corr_is_correlation_matrix(n: int, data: st.DataObject) -> None:
    """cov_to_corr() yields a symmetric matrix with unit diagonal, entries in [-1, 1]."""
    cov = data.draw(_pd_matrix(n))
    corr = cov_to_corr(cov)
    np.testing.assert_allclose(np.diag(corr), np.ones(n), rtol=1e-10)
    np.testing.assert_allclose(corr, corr.T, rtol=1e-10, atol=1e-12)
    assert np.all(np.abs(corr) <= 1.0 + 1e-10)


# ─── det ─────────────────────────────────────────────────────────────────────


@given(n=_SIZES, data=st.data())
@settings(max_examples=100)
def test_det_diagonal_equals_product(n: int, data: st.DataObject) -> None:
    """The determinant of a diagonal matrix equals the product of its diagonal."""
    diag_vals = data.draw(
        np_st.arrays(
            dtype=np.float64,
            shape=(n,),
            elements=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
        )
    )
    result = det(np.diag(diag_vals))
    assert result == pytest.approx(float(np.prod(diag_vals)), rel=1e-9)


# ─── eigh / eigvalsh / eigvals ───────────────────────────────────────────────


@given(n=_SIZES, data=st.data())
@settings(max_examples=100)
def test_eigh_ascending_and_reconstructs(n: int, data: st.DataObject) -> None:
    """eigh() returns ascending eigenvalues w and eigenvectors V with S @ V == V @ diag(w)."""
    a = data.draw(_bounded_matrix(n, n))
    sym = (a + a.T) / 2.0
    w, v = eigh(sym)
    assert np.all(np.diff(w) >= -1e-10)
    np.testing.assert_allclose(sym @ v, v @ np.diag(w), rtol=1e-7, atol=1e-7)


@given(n=_SIZES, data=st.data())
@settings(max_examples=100)
def test_eigvalsh_matches_eigh(n: int, data: st.DataObject) -> None:
    """eigvalsh() returns exactly the eigenvalues computed by eigh()."""
    a = data.draw(_bounded_matrix(n, n))
    sym = (a + a.T) / 2.0
    np.testing.assert_array_equal(eigvalsh(sym), eigh(sym)[0])


@given(n=_SIZES, data=st.data())
@settings(max_examples=100)
def test_eigvals_sum_equals_trace(n: int, data: st.DataObject) -> None:
    """The eigenvalues of any square matrix sum to its trace."""
    a = data.draw(_bounded_matrix(n, n))
    w = eigvals(a)
    total = complex(np.sum(w))
    assert total.real == pytest.approx(float(np.trace(a)), abs=1e-6)
    assert abs(total.imag) < 1e-6


# ─── inv ─────────────────────────────────────────────────────────────────────


@given(n=_SIZES, data=st.data())
@settings(max_examples=100)
def test_inv_times_matrix_is_identity(n: int, data: st.DataObject) -> None:
    """inv() of a well-conditioned PD matrix is its true inverse."""
    cov = data.draw(_pd_matrix(n))
    np.testing.assert_allclose(cov @ inv(cov), np.eye(n), rtol=1e-6, atol=1e-6)


# ─── lstsq ───────────────────────────────────────────────────────────────────


@given(n=_SIZES, extra=st.integers(min_value=0, max_value=4), data=st.data())
@settings(max_examples=100)
def test_lstsq_residual_is_orthogonal_to_columns(n: int, extra: int, data: st.DataObject) -> None:
    """The least-squares residual is orthogonal to the column space of the matrix."""
    m = n + extra
    a = data.draw(_bounded_matrix(m, n))
    b = data.draw(_finite_vector(m))
    # Near-singular systems warn by design (IllConditionedMatrixWarning, an
    # error under strict filterwarnings); restrict to well-conditioned draws.
    singular_values = np.linalg.svd(a, compute_uv=False)
    assume(float(singular_values.min()) > 1e-3)
    x, _, _, _ = lstsq(a, b)
    np.testing.assert_allclose(a.T @ (a @ x - b), np.zeros(n), atol=1e-5)


# ─── qr ──────────────────────────────────────────────────────────────────────


@given(n=_SIZES, extra=st.integers(min_value=0, max_value=4), data=st.data())
@settings(max_examples=100)
def test_qr_reconstructs_and_q_is_orthonormal(n: int, extra: int, data: st.DataObject) -> None:
    """qr() returns orthonormal Q and upper triangular R with Q @ R == A."""
    a = data.draw(_bounded_matrix(n + extra, n))
    q, r = qr(a)
    np.testing.assert_allclose(q @ r, a, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(q.T @ q, np.eye(n), rtol=1e-8, atol=1e-8)
    assert np.allclose(r, np.triu(r))


# ─── rand_cov ────────────────────────────────────────────────────────────────


@given(n=_SIZES, seed=st.integers(min_value=0, max_value=2**16))
@settings(max_examples=100)
def test_rand_cov_is_symmetric_pd_and_reproducible(n: int, seed: int) -> None:
    """rand_cov() returns a symmetric PD matrix, reproducible for a fixed seed."""
    cov = rand_cov(n, seed=seed)
    assert cov.shape == (n, n)
    np.testing.assert_allclose(cov, cov.T, rtol=1e-12, atol=1e-12)
    assert is_positive_definite(cov)
    np.testing.assert_array_equal(cov, rand_cov(n, seed=seed))


# ─── svd ─────────────────────────────────────────────────────────────────────


@given(n=_SIZES, extra=st.integers(min_value=0, max_value=4), data=st.data())
@settings(max_examples=100)
def test_svd_reconstructs_with_sorted_singular_values(n: int, extra: int, data: st.DataObject) -> None:
    """svd() reconstructs the input; singular values are non-negative descending."""
    a = data.draw(_bounded_matrix(n + extra, n))
    u, s, vt = svd(a)
    np.testing.assert_allclose((u * s) @ vt, a, rtol=1e-8, atol=1e-8)
    assert np.all(s >= 0.0)
    assert np.all(np.diff(s) <= 1e-12)


# ─── pca ─────────────────────────────────────────────────────────────────────


@given(
    n_assets=st.integers(min_value=2, max_value=5),
    n_components=st.integers(min_value=1, max_value=2),
    data=st.data(),
)
@settings(max_examples=50)
def test_pca_decomposes_returns(n_assets: int, n_components: int, data: st.DataObject) -> None:
    """pca() splits returns into systematic + idiosyncratic with valid explained variance."""
    returns = data.draw(_bounded_matrix(20, n_assets))
    assume(float(np.linalg.norm(returns - returns.mean(axis=0))) > 1e-6)
    result = pca(returns, n_components=n_components)
    np.testing.assert_allclose(result.systematic + result.idiosyncratic, returns, atol=1e-8)
    assert np.all(result.explained_variance >= 0.0)
    assert float(result.explained_variance.sum()) <= 1.0 + 1e-10


# ─── a_norm / cond / norm ────────────────────────────────────────────────────


@given(n=_SIZES, data=st.data())
@settings(max_examples=100)
def test_a_norm_identity_equals_euclidean(n: int, data: st.DataObject) -> None:
    """With the identity matrix, a_norm collapses to the Euclidean norm."""
    vector = data.draw(_finite_vector(n))
    result = a_norm(vector, matrix=np.eye(n))
    assert result == pytest.approx(float(np.linalg.norm(vector, 2)), rel=1e-9, abs=1e-12)


@given(
    n=_SIZES,
    scale=st.floats(min_value=0.5, max_value=10.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_cond_scaled_identity_is_one(n: int, scale: float) -> None:
    """Any non-zero multiple of the identity has condition number 1."""
    assert cond(scale * np.eye(n)) == pytest.approx(1.0, rel=1e-12)


@given(n=_SIZES, k=st.integers(min_value=1, max_value=5), data=st.data())
@settings(max_examples=100)
def test_norm_ignores_nan_entries(n: int, k: int, data: st.DataObject) -> None:
    """Appending NaN entries to a vector leaves its norm unchanged."""
    vector = data.draw(_finite_vector(n))
    padded = np.concatenate([vector, np.full(k, np.nan)])
    assert norm(padded) == pytest.approx(norm(vector), rel=1e-12, abs=1e-12)
