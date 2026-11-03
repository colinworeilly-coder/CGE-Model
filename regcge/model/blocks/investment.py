"""Investment by sector of destination (STAMP eqs. 18-20, Tobin's-q form).

    IND_a = INVSCALE * (g + delta_a) * K_a * [(RK_a/PK_a) / (RK0_a/PK0_a)]^eta_a

On the baseline the relative return equals one and investment exactly
replaces depreciation and finances growth, so K grows at g.  A policy that
raises (lowers) the return to capital in a sector raises (lowers) its
investment until the capital stock has adjusted.  Investment goods are bought
through a capital-coefficient matrix (eq. 20).  INVSCALE is fixed under the
open-capital closure and adjusts to available saving under the
savings-driven closure.
"""

from __future__ import annotations

import casadi as ca
import numpy as np

from ..functional import DM


def declare(M):
    m, c, S = M.m, M.c, M.S
    A = S["A"]
    m.add_var("IND", (A,), level=c.bench["IND"], lb=0.0, grow="q", desc="investment by sector of destination (real)")
    m.add_var("INVSCALE", (), level=1.0, fixed=True, desc="investment scale (savings-driven closure)")


def equations(M):
    m, c, S, x = M.m, M.c, M.S, M.x
    A = S["A"]
    P = c.par
    g = P["g"]
    rk0 = c.bench["RK"]
    q = (m.V("RK") / x["PK"]) / DM(rk0 / P["PK0"])
    rhs = m.V("INVSCALE") * DM(g + P["delta"]) * m.V("K") * q ** DM(P["eta_inv"])
    m.add_eq("EQ_IND", m.V("IND"), rhs, (A,), desc="investment by destination (eq. 19)")
