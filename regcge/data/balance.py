"""SAM balancing: RAS and cross-entropy.

* :func:`ras` -- classic bi-proportional scaling of a non-negative matrix to
  given row and column totals.
* :func:`balance_sam_ras` -- RAS for a SAM whose account totals are unknown:
  iterates towards totals equal to the average of row and column sums.
* :func:`balance_sam_cross_entropy` -- minimum cross-entropy (generalised
  Kullback-Leibler) adjustment of a prior SAM subject to row = column totals,
  optional fixed cells and optional known account totals (Golan, Judge &
  Robinson 1994; Robinson, Cattaneo & El-Said 2001).  Negative prior cells
  (e.g. negative savings) keep their sign.  Solved with IPOPT via CasADi.

Structural zeros of the prior are always preserved.
"""

from __future__ import annotations

import casadi as ca
import numpy as np
import pandas as pd

from .sam import canonical


def ras(prior: np.ndarray, row_targets: np.ndarray, col_targets: np.ndarray,
        tol: float = 1e-10, max_iter: int = 10_000) -> np.ndarray:
    A = np.asarray(prior, float).copy()
    if (A < 0).any():
        raise ValueError("RAS requires a non-negative matrix")
    r_t = np.asarray(row_targets, float)
    c_t = np.asarray(col_targets, float)
    if not np.isclose(r_t.sum(), c_t.sum(), rtol=1e-9):
        raise ValueError("Row and column targets must have the same grand total")
    for _ in range(max_iter):
        rs = A.sum(axis=1)
        A *= np.where(rs > 0, r_t / np.where(rs > 0, rs, 1), 1.0)[:, None]
        cs = A.sum(axis=0)
        A *= np.where(cs > 0, c_t / np.where(cs > 0, cs, 1), 1.0)[None, :]
        if np.max(np.abs(A.sum(axis=1) - r_t)) < tol * max(1.0, r_t.max()):
            return A
    raise RuntimeError("RAS did not converge")


def balance_sam_ras(sam: pd.DataFrame, tol: float = 1e-10, max_iter: int = 100_000) -> pd.DataFrame:
    """Balance a non-negative SAM by alternately scaling rows and columns
    towards the average of current row and column totals."""
    S = canonical(sam).to_numpy().copy()
    if (S < 0).any():
        raise ValueError("balance_sam_ras requires non-negative cells; use cross-entropy instead")
    for _ in range(max_iter):
        t = 0.5 * (S.sum(axis=1) + S.sum(axis=0))
        rs = S.sum(axis=1)
        S *= np.where(rs > 0, t / np.where(rs > 0, rs, 1), 1.0)[:, None]
        t = 0.5 * (S.sum(axis=1) + S.sum(axis=0))
        cs = S.sum(axis=0)
        S *= np.where(cs > 0, t / np.where(cs > 0, cs, 1), 1.0)[None, :]
        if np.abs(S.sum(axis=1) - S.sum(axis=0)).max() < tol * max(1.0, S.sum(axis=1).max()):
            return pd.DataFrame(S, index=sam.index, columns=sam.index)
    raise RuntimeError("SAM RAS balancing did not converge")


def balance_sam_cross_entropy(sam: pd.DataFrame, fixed: pd.DataFrame | np.ndarray | None = None,
                              totals: dict[str, float] | None = None,
                              verbose: bool = False) -> pd.DataFrame:
    """Minimum cross-entropy balancing of a SAM.

    Parameters
    ----------
    sam : prior SAM (need not balance).
    fixed : boolean mask (same shape) of cells to keep at their prior value.
    totals : optional ``{account: total}`` targets for row (= column) totals.
    """
    S = canonical(sam)
    lab = list(S.index)
    X0 = S.to_numpy()
    n = len(lab)
    nz = np.argwhere(np.abs(X0) > 0)
    fixed_mask = np.zeros_like(X0, dtype=bool) if fixed is None else np.asarray(fixed, bool)
    free = [(i, j) for i, j in nz if not fixed_mask[i, j]]
    m = len(free)
    y = ca.SX.sym("y", m)  # absolute value of free cells, relative to prior
    a0 = np.array([abs(X0[i, j]) for i, j in free])
    sgn = np.array([np.sign(X0[i, j]) for i, j in free])
    # cells = sign * a0 * y ; objective: sum a0 * (y log y - y + 1)
    obj = ca.sum1(ca.DM(a0) * (y * ca.log(y) - y + 1))
    row = [0] * n
    col = [0] * n
    for k, (i, j) in enumerate(free):
        v = sgn[k] * a0[k] * y[k]
        row[i] = row[i] + v
        col[j] = col[j] + v
    fixed_row = np.where(fixed_mask, X0, 0).sum(axis=1)
    fixed_col = np.where(fixed_mask, X0, 0).sum(axis=0)
    g, lbg, ubg = [], [], []
    scale = max(1.0, np.abs(X0).max())
    for i in range(n):
        g.append((row[i] + fixed_row[i] - col[i] - fixed_col[i]) / scale)
        lbg.append(0.0)
        ubg.append(0.0)
    if totals:
        for acc, tot in totals.items():
            i = lab.index(acc)
            g.append((row[i] + fixed_row[i] - tot) / scale)
            lbg.append(0.0)
            ubg.append(0.0)
    g = ca.vertcat(*[ca.SX(gi) for gi in g])
    nlp = {"x": y, "f": obj / scale, "g": g}
    opts = {"ipopt.print_level": 5 if verbose else 0, "print_time": False, "ipopt.sb": "yes",
            "ipopt.tol": 1e-12, "ipopt.constr_viol_tol": 1e-12, "ipopt.max_iter": 3000}
    solver = ca.nlpsol("ce", "ipopt", nlp, opts)
    sol = solver(x0=np.ones(m), lbx=1e-8, ubx=np.inf, lbg=lbg, ubg=ubg)
    st = solver.stats()
    if not st["success"]:
        raise RuntimeError(f"Cross-entropy balancing failed: {st['return_status']}")
    yv = np.array(sol["x"]).ravel()
    X = np.where(fixed_mask, X0, 0.0)
    for k, (i, j) in enumerate(free):
        X[i, j] = sgn[k] * a0[k] * yv[k]
    return pd.DataFrame(X, index=lab, columns=lab)
