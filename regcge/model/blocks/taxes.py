"""Tax revenues by instrument (STAMP eqs. 21-22) and household-paid amounts.

Every instrument has a scale variable TSCALE (normally fixed at one).  Freeing
it in a closure turns the instrument into the endogenous, revenue-balancing
tax of an equal-yield experiment.
"""

from __future__ import annotations

import casadi as ca
import numpy as np

from ..functional import DM, colsum, rep_col, rep_row, vcat


def declare(M):
    m, c, S = M.m, M.c, M.S
    K = S["K"]
    m.add_var("TAXR", (K,), level=c.bench["TAXR"], grow="q", desc="tax revenue by instrument")
    m.add_var("TSCALE", (K,), level=1.0, fixed=True, desc="scale factor on each instrument's rates")


def equations(M):
    m, c, S, x = M.m, M.c, M.S, M.x
    A, H, G, F, FL, K = S["A"], S["H"], S["G"], S["F"], S["FL"], S["K"]
    nA, nH, nG = len(A), len(H), len(G)
    P = c.par
    ts = x["ts"]
    PQ, PX, QX, QH, QINTA, IND, QGL, K_, RK, QF, W = (
        m.V(n) for n in ("PQ", "PX", "QX", "QH", "QINTA", "IND", "QGL", "K", "RK", "QF", "W"))
    x["QINT"] = DM(P["icoef"]) * rep_row(QINTA, nA)
    x["QINV"] = ca.mtimes(DM(P["B"]), IND)
    x["QG"] = DM(P["qgc"]) * rep_row(QGL, nA)
    PQh = rep_col(PQ, nH)
    PQa = rep_col(PQ, nA)
    PQg = rep_col(PQ, nG)
    fac = M.ds.accounts["factors"]
    rev = []
    paid = {}
    TAXH = m.V("TAXH") if len(S["KI"]) else None
    for k in K:
        base = c.taxes[k]["base"]
        if base == "sales":
            hh = m.P(f"tsh_{k}") * PQh * QH
            paid[k] = ts[k] * colsum(hh)
            r = (ca.sum1(ca.sum2(hh))
                 + ca.sum1(ca.sum2(m.P(f"tsi_{k}") * PQa * x["QINT"]))
                 + ca.dot(m.P(f"tsv_{k}") * PQ, x["QINV"])
                 + ca.sum1(ca.sum2(m.P(f"tsg_{k}") * PQg * x["QG"])))
            rev.append(ts[k] * r)
        elif base == "output":
            rev.append(ts[k] * ca.dot(m.P(f"tout_{k}") * PX, QX))
        elif base == "factor":
            f = c.par["tax_factor"][k]
            if fac[f] == "labor":
                pay = W[FL.index(f)] * QF[F.index(f), :].T
                gpay = W[FL.index(f)] * x["QFG"][FL.index(f), :].T
                r = ca.dot(m.P(f"tf_{k}"), pay) + ca.dot(m.P(f"tfg_{k}"), gpay)
            elif fac[f] == "capital":
                r = ca.dot(m.P(f"tf_{k}"), RK * K_)
            else:
                ix_ = S["FX"].index(f)
                r = ca.dot(m.P(f"tf_{k}"), m.V("RFX")[ix_, :].T * QF[F.index(f), :].T)
            rev.append(ts[k] * r)
        elif base == "property":
            rev.append(ts[k] * ca.dot(m.P(f"tprop_{k}") * x["PK"], K_))
        elif base == "income":
            i = S["KI"].index(k)
            paid[k] = TAXH[i, :].T
            rev.append(ca.sum1(paid[k]))
    x["paid"] = paid
    x["rev_list"] = rev
    m.add_eq("EQ_TAXR", m.V("TAXR"), vcat(rev), (K,), desc="revenue by tax instrument (eq. 22)")
