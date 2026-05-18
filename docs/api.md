---
icon: material/code-tags
---

# API Reference

NaN-aware linear algebra utilities for risk models. All public symbols are
importable directly from the top-level package:

```python
from cvx.linalg import (
    # decompositions
    cholesky, eigh, eigvalsh, eigvals, qr, svd, pca,
    # solvers
    solve, lstsq, inv,
    # norms & metrics
    norm, a_norm, inv_a_norm, cond, det,
    # covariance
    ewm_covariance, rand_cov,
    # validation
    valid, is_positive_definite,
    # exceptions
    SingularMatrixError, IllConditionedMatrixWarning,
    DimensionMismatchError, NonSquareMatrixError,
    NotAMatrixError, NegativeWarmupError,
)
```

---

::: cvx.linalg
