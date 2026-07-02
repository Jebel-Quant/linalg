---
icon: material/decimal
---

# Numerical behavior

This page collects the numerical conventions that apply across the package, so
they live in one place rather than being repeated in every docstring.

## NaN-awareness

Most functions are *NaN-aware*: before computing, they reduce the input to its
**valid submatrix** via [`valid`][cvx.linalg.core.valid.valid]. A row/column is
valid when its **diagonal entry is finite**; rows and columns with NaN or
infinity on the diagonal are dropped, the computation runs on the remaining
square submatrix, and (where applicable) results are re-expanded with NaN at
the invalid positions.

| Function | NaN handling |
|---|---|
| `solve`, `inv` | Compute on the valid submatrix; invalid positions of the result are NaN. |
| `det`, `cond` | Computed on the valid submatrix; return `nan` when nothing valid remains (or when entries are non-finite, for `cond`). |
| `eigh`, `eigvalsh` | Eigen-decomposition of the valid symmetric submatrix. |
| `lstsq` | Rows with non-finite right-hand side or matrix entries are filtered before solving. |
| `norm`, `a_norm`, `inv_a_norm` | Non-finite entries are treated as zero (they contribute nothing). |
| `pca` | **Not NaN-aware.** Inputs must be entirely finite; clean or impute first. |
| `cholesky`, `cholesky_solve`, `qr`, `svd`, `eigvals` | Standard NumPy semantics; non-finite inputs are not filtered. |

## Condition-number convention

Functions that solve or invert (`solve`, `inv`, `det`, `lstsq`) accept a
`cond_threshold` parameter, defaulting to `DEFAULT_COND_THRESHOLD = 1e12`.
When the condition number of the (valid) matrix exceeds the threshold, an
[`IllConditionedMatrixWarning`][cvx.linalg.core.exceptions.IllConditionedMatrixWarning]
is emitted and the computation proceeds — the warning signals that results may
be numerically unreliable, it does not abort.

The test suite runs with `filterwarnings = error`, so any unexpected warning
is a test failure; expected warnings are asserted explicitly with
`pytest.warns`.

## dtype contract

The public API is annotated with the [`Matrix`][cvx.linalg.core.types] and
[`Vector`][cvx.linalg.core.types] aliases — both `numpy.typing.NDArray[np.float64]`
(2-D and 1-D respectively). `eigvals` may return complex eigenvalues for
non-symmetric input. The package ships a `py.typed` marker, so these
annotations are checked in downstream code.

## BLAS backends

Numerical warnings can differ between BLAS backends: OpenBLAS (Linux/CI)
surfaces overflow in matrix products as a `RuntimeWarning`, while Accelerate
(macOS) often does not. Because the test suite promotes warnings to errors, a
suite that is green on macOS is not automatically green on CI. Run

```bash
make test-linux
```

to execute the test suite in a Linux/OpenBLAS container that mirrors CI
numerics. Property-based tests bound input magnitudes so that intermediate
products stay below the float64 maximum on every backend.
