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
from cvx.linalg.covariance.ewm_cov import ewm_covariance  # requires the 'ewm' extra (polars)
```

---

## Decompositions

::: cvx.linalg.decomposition.cholesky.cholesky

::: cvx.linalg.decomposition.cholesky.cholesky_solve

::: cvx.linalg.decomposition.cholesky.is_positive_definite

::: cvx.linalg.decomposition.eigh.eigh

::: cvx.linalg.decomposition.eigh.eigvalsh

::: cvx.linalg.decomposition.eigvals.eigvals

::: cvx.linalg.decomposition.qr.qr

::: cvx.linalg.decomposition.svd.svd

::: cvx.linalg.decomposition.svd.svd_k

::: cvx.linalg.covariance.pca.pca

::: cvx.linalg.decomposition.power_iteration.power_iteration

---

## Solvers

::: cvx.linalg.solve.solve.solve

::: cvx.linalg.solve.lstsq.lstsq

::: cvx.linalg.solve.inv.inv

---

## Norms & Metrics

::: cvx.linalg.norm.norm.norm

::: cvx.linalg.norm.norm.a_norm

::: cvx.linalg.norm.norm.inv_a_norm

::: cvx.linalg.core.exceptions.cond

::: cvx.linalg.solve.det.det

---

## Covariance

::: cvx.linalg.covariance.ewm_cov.ewm_covariance

::: cvx.linalg.covariance.rand_cov.rand_cov

::: cvx.linalg.covariance.cov_to_corr.cov_to_corr

---

## Validation

::: cvx.linalg.core.valid.valid

---

## Exceptions & Warnings

::: cvx.linalg.core.exceptions.SingularMatrixError

::: cvx.linalg.core.exceptions.IllConditionedMatrixWarning

::: cvx.linalg.core.exceptions.DimensionMismatchError

::: cvx.linalg.core.exceptions.NonSquareMatrixError

::: cvx.linalg.core.exceptions.NotAMatrixError

::: cvx.linalg.core.exceptions.NegativeWarmupError

::: cvx.linalg.core.exceptions.NonIntegerWarmupError

::: cvx.linalg.core.exceptions.InvalidComponentsError

::: cvx.linalg.core.exceptions.check_and_warn_condition

::: cvx.linalg.core.exceptions.warn_ill_conditioned

::: cvx.linalg.core.exceptions.DEFAULT_COND_THRESHOLD

---

## Types

::: cvx.linalg.core.types.Matrix

::: cvx.linalg.core.types.Vector
