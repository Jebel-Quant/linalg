# Mutation testing

Run mutation testing with:

```bash
make mutation
```

This uses [mutmut](https://mutmut.readthedocs.io) to mutate `src/` and re-run
the test suite against each mutant. A mutant *survives* when no test fails.

## Baseline (2026-06-11)

262 mutants: 236 killed, 26 survived. Every survivor was reviewed and falls
into one of these categories of **equivalent or unobservable mutants** — they
do not indicate missing test coverage:

| Category | Mutants | Why they cannot be killed |
|---|---|---|
| `typing.cast("…")` type strings | cholesky 84/88/89, eigvals 227 | The string has no runtime effect; `cast` returns its argument unchanged. |
| `stacklevel=` arguments | cholesky 86, exceptions 201/205 | Only affects which source line a warning is attributed to. |
| polars-missing import message | ewm_cov 102–104 | Observable only in an environment without polars; the test env installs it. |
| `warmup` 0↔1 equivalence | ewm_cov 107/115 | `min_samples = 1 if warmup == 0 else warmup` yields 1 for both warmup=0 and warmup=1. |
| Internal column alias | ewm_cov 120 | `f"{a}_{b}"` names a transient polars column; never user-visible. |
| Redundant square-guards | det 3, inv 16, norm 32/45, solve 206 | If the guard is disabled, `valid()` raises the identical `NonSquareMatrixError` with the identical message. Defense in depth. |
| Post-guard index swaps | inv 21, norm 38/39/51/52, solve 213/215 | After the square check, `shape[0] == shape[1]`, so swapping the indices is a no-op. |
| `tril_indices(k=-1)` → `k=+1` | cov_to_corr 74 | The widened index set adds diagonal/mirrored pairs whose average equals the original value; the symmetrization result is unchanged. |

When new survivors appear outside this list, treat them as missing-test
signals: strengthen assertions (exact error messages, attribute checks,
boundary values) rather than adding redundant tests.
