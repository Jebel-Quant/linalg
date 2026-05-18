---
icon: material/code-tags
---

# API Reference

NaN-aware linear algebra utilities for risk models. All public symbols are
importable directly from the top-level package:

```python
from cvx.linalg import cholesky, eigh, eigvalsh, eigvals, qr, svd, pca
from cvx.linalg import solve, lstsq, inv
from cvx.linalg import norm, a_norm, inv_a_norm, cond, det
from cvx.linalg import ewm_covariance, rand_cov
from cvx.linalg import valid, is_positive_definite
```

---

## Decompositions

::: cvx.linalg.cholesky.cholesky

::: cvx.linalg.cholesky.is_positive_definite

::: cvx.linalg.eigh.eigh

::: cvx.linalg.eigh.eigvalsh

::: cvx.linalg.eigvals.eigvals

::: cvx.linalg.qr.qr

::: cvx.linalg.svd.svd

::: cvx.linalg.pca.pca

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

::: cvx.linalg.ewm_cov.NegativeWarmupError
