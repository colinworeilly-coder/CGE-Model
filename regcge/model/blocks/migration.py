"""Population and migration (STAMP eqs. 7-8, Harris-Todaro spirit).

    N_h = NBAR_h * NLR_h^(1 - lambda_h) * (X_h / XBAR_h)^(lambda_h * eta_h)

X is real disposable income per household (default) or utility per household;
bars denote values on the baseline path, NLR the previous period's ratio of
actual to baseline households (partial adjustment; lambda = 1 reproduces
STAMP's instantaneous migration).

Closure logic: on the baseline, N is fixed at NBAR and XBAR is free (it simply
records X); in a scenario XBAR is fixed at its baseline value and N is free.
Households with zero elasticity or speed never migrate and XBAR = X for them.
"""

from __future__ import annotations

import casadi as ca
import numpy as np

from ..functional import DM


def declare(M):
    m, c, S = M.m, M.c, M.S
    H = S["H"]
    b = c.bench
    m.add_var("N", (H,), level=b["N"], lb=0.0, grow="n", fixed=True, desc="number of households")
    xbar = b["XBAR"] if M.migration_driver == "real_income" else b["U"] / b["N"]
    m.add_var("XBAR", (H,), level=xbar, lb=0.0, grow="c", desc="baseline real income (or utility) per household")
    m.add_param("NBAR", (H,), b["N"], "baseline number of households")
    m.add_param("NLR", (H,), 1.0, "previous-period ratio of actual to baseline households")


def equations(M):
    m, c, S = M.m, M.c, M.S
    H = S["H"]
    P = c.par
    N, XBAR = m.V("N"), m.V("XBAR")
    if M.migration_driver == "real_income":
        X = m.V("YD") / (N * m.V("CPI"))
    else:
        X = m.V("U") / N
    lam, eta = P["mig_speed"], P["mig_eta"]
    mig = (lam * eta) != 0
    M.migrating = mig
    if mig.any():
        rhs = m.P("NBAR") * m.P("NLR") ** DM(1 - lam) * (X / XBAR) ** DM(lam * eta)
        m.add_eq("EQ_MIG", N, rhs, (H,), where=mig, desc="households: natural growth + net migration (eq. 7)")
    if (~mig).any():
        m.add_eq("EQ_XBAR", XBAR, X, (H,), where=~mig, desc="record real income for non-migrating households")
