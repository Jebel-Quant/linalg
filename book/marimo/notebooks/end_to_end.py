# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo==0.23.2",
#     "cvx-linalg",
#     "numpy>=2.0.0",
# ]
# [tool.uv.sources]
# cvx-linalg = { path = "../../..", editable = true }
# ///

import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    from cvx.linalg import cholesky, inv, rand_cov, solve

    return cholesky, inv, mo, np, rand_cov, solve


@app.cell
def _(mo):
    mo.md(
        r"""
        # cvx-linalg — Marimo End-to-End Example

        This notebook demonstrates a small linear-system workflow with `cvx-linalg`.
        """
    )
    return


@app.cell
def _(cholesky, inv, np, rand_cov, solve):
    cov = rand_cov(4, seed=7)
    rhs = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    chol = cholesky(cov)
    inv_cov = inv(cov)
    solution = solve(cov, rhs)
    return chol, cov, inv_cov, rhs, solution


@app.cell
def _(chol, cov, inv_cov, mo, rhs, solution):
    mo.vstack(
        [
            mo.md("## Inputs and outputs"),
            mo.md(f"**cov**\n\n```python\n{cov}\n```"),
            mo.md(f"**cholesky(cov)**\n\n```python\n{chol}\n```"),
            mo.md(f"**inv(cov)**\n\n```python\n{inv_cov}\n```"),
            mo.md(f"**rhs**\n\n```python\n{rhs}\n```"),
            mo.md(f"**solve(cov, rhs)**\n\n```python\n{solution}\n```"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
