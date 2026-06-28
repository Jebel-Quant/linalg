---
icon: material/code-tags
---

# API Reference

NaN-aware linear algebra utilities for risk models. All public symbols are
importable directly from the top-level package:

```python
from cvx.linalg import cholesky, cholesky_solve, eigh, eigvalsh, eigvals, qr, svd, pca
from cvx.linalg import solve, lstsq, inv
from cvx.linalg import norm, a_norm, inv_a_norm, cond, det
from cvx.linalg import rand_cov, valid, is_positive_definite
from cvx.linalg.ewm_cov import ewm_covariance  # requires the 'ewm' extra (polars)
```

---

## Decompositions

::: cvx.linalg.cholesky.cholesky

::: cvx.linalg.cholesky.cholesky_solve

::: cvx.linalg.cholesky.is_positive_definite

::: cvx.linalg.eigh.eigh

::: cvx.linalg.eigh.eigvalsh

::: cvx.linalg.eigvals.eigvals

::: cvx.linalg.qr.qr

::: cvx.linalg.svd.svd

::: cvx.linalg.svd.svd_k

::: cvx.linalg.pca.pca

::: cvx.linalg.power_iteration.power_iteration

---

## Solvers

::: cvx.linalg.solve.solve

::: cvx.linalg.lstsq.lstsq

::: cvx.linalg.inv.inv

---

## Norms & Metrics

::: cvx.linalg.norm.norm

::: cvx.linalg.norm.a_norm

::: cvx.linalg.norm.inv_a_norm

::: cvx.linalg.exceptions.cond

::: cvx.linalg.det.det

---

## Covariance

::: cvx.linalg.ewm_cov.ewm_covariance

::: cvx.linalg.rand_cov.rand_cov

::: cvx.linalg.cov_to_corr.cov_to_corr

---

## Validation

::: cvx.linalg.valid.valid

---

## Exceptions & Warnings

::: cvx.linalg.exceptions.SingularMatrixError

::: cvx.linalg.exceptions.IllConditionedMatrixWarning

::: cvx.linalg.exceptions.DimensionMismatchError

::: cvx.linalg.exceptions.NonSquareMatrixError

::: cvx.linalg.exceptions.NotAMatrixError

::: cvx.linalg.exceptions.NegativeWarmupError

::: cvx.linalg.exceptions.NonIntegerWarmupError

::: cvx.linalg.exceptions.InvalidComponentsError

::: cvx.linalg.exceptions.check_and_warn_condition

::: cvx.linalg.exceptions.warn_ill_conditioned

::: cvx.linalg.exceptions.DEFAULT_COND_THRESHOLD

---

## Types

::: cvx.linalg.types.Matrix

::: cvx.linalg.types.Vector
