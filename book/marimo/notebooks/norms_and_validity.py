import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    from cvx.linalg import a_norm, inv_a_norm, valid

    return a_norm, inv_a_norm, mo, np, valid


@app.cell
def _(mo):
    mo.md(
        r"""
        # cvx-linalg — Norms and validity

        This notebook shows how invalid (non-finite diagonal) rows/columns are
        removed by `valid`, then reused in `a_norm` and `inv_a_norm`.
        """
    )
    return


@app.cell
def _(a_norm, inv_a_norm, np, valid):
    matrix = np.array(
        [
            [2.0, 0.3, 0.1],
            [0.3, np.nan, 0.2],
            [0.1, 0.2, 1.5],
        ]
    )
    vector = np.array([1.0, 2.0, -1.0], dtype=float)
    mask, reduced = valid(matrix)
    norm_value = a_norm(vector, matrix)
    inv_norm_value = inv_a_norm(vector, matrix)
    return inv_norm_value, mask, matrix, norm_value, reduced, vector


@app.cell
def _(inv_norm_value, mask, matrix, mo, norm_value, reduced, vector):
    mo.vstack(
        [
            mo.md("## Result summary"),
            mo.md(f"**matrix**\n\n```python\n{matrix}\n```"),
            mo.md(f"**valid mask**\n\n```python\n{mask}\n```"),
            mo.md(f"**reduced submatrix**\n\n```python\n{reduced}\n```"),
            mo.md(f"**vector**\n\n```python\n{vector}\n```"),
            mo.md(f"**a_norm(vector, matrix)**: `{norm_value}`"),
            mo.md(f"**inv_a_norm(vector, matrix)**: `{inv_norm_value}`"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
