"""Functional forms in calibrated share form (Rutherford 1998).

All aggregators are written relative to their benchmark point:

* CES unit-cost index:   c = [ sum_i theta_i (p_i/p0_i)^(1-sigma) ]^(1/(1-sigma))
  with Leontief (sigma = 0) and Cobb-Douglas (sigma = 1) limits;
* CET unit-revenue index: r = [ sum_i theta_i (p_i/p0_i)^(1+eta) ]^(1/(1+eta)).

Compensated demands follow from Shephard's lemma:
``x_i = x0_i (Q/Q0) (c / (p_i/p0_i))^sigma`` and supplies from Hotelling's
lemma: ``y_i = y0_i (Q/Q0) ((p_i/p0_i) / r)^eta``.  Elements with zero
benchmark share drop out automatically.
"""

from __future__ import annotations

import casadi as ca
import numpy as np


def ces_index(thetas, rels, sigma: float):
    terms = [(float(t), r) for t, r in zip(thetas, rels) if float(t) > 1e-14]
    if not terms:
        return ca.SX(1.0)
    if abs(sigma - 1.0) < 1e-8:
        out = 1
        for t, r in terms:
            out = out * r ** t
        return out
    if sigma < 1e-10:
        return sum(t * r for t, r in terms)
    s = 0
    for t, r in terms:
        s = s + t * r ** (1.0 - sigma)
    return s ** (1.0 / (1.0 - sigma))


def cet_index(thetas, rels, eta: float):
    terms = [(float(t), r) for t, r in zip(thetas, rels) if float(t) > 1e-14]
    if not terms:
        return ca.SX(1.0)
    s = 0
    for t, r in terms:
        s = s + t * r ** (1.0 + eta)
    return s ** (1.0 / (1.0 + eta))


def vcat(items):
    return ca.vertcat(*[i if isinstance(i, ca.SX) else ca.SX(i) for i in items])


def rep_col(v, n: int):
    """Column vector v (m x 1) -> m x n matrix with v in every column."""
    return ca.repmat(v, 1, n)


def rep_row(v, m: int):
    """Column vector v (n x 1) -> m x n matrix with v' in every row."""
    return ca.repmat(v.T, m, 1)


def colsum(M):
    """Sum over rows -> column vector of column totals (n x 1)."""
    return ca.sum1(M).T


def rowsum(M):
    return ca.sum2(M)


def DM(x):
    return ca.DM(np.asarray(x, dtype=float))
