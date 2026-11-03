"""Market clearing, the rest-of-world account and saving = investment
(STAMP eqs. 31-32 and the model closure).

The savings-investment identity carries a WALRAS slack: since every other
account's budget and every market are explicit, it is implied by Walras' law
and the slack must be zero at a solution (checked by the tests).
"""

from __future__ import annotations

import casadi as ca
import numpy as np

from ..functional import DM, colsum


def declare(M):
    m, c, S = M.m, M.c, M.S
    m.add_var("SROW", (), level=c.bench["SROW"], grow="q", desc="rest-of-world saving (net capital inflow)")
    m.add_var("WALRAS", (), level=0.0, desc="Walras' law check (must be zero)")


def equations(M):
    m, c, S, x = M.m, M.c, M.S, M.x
    A, H, G, FL, FX = S["A"], S["H"], S["G"], S["FL"], S["FX"]
    P = c.par
    GI, ER = m.P("GI"), m.V("ER")
    PQ, QQ, QH, QM, QE, PE, W = (m.V(n) for n in ("PQ", "QQ", "QH", "QM", "QE", "PE", "W"))
    # commodity markets
    demand = ca.sum2(x["QINT"]) + ca.sum2(QH) + ca.sum2(x["QG"]) + x["QINV"]
    m.add_eq("EQ_MKT", QQ, demand, (A,), desc="commodity market clearing (eqs. 31-32)")
    # rest of world
    th = lambda key, d: float(d.get(key, 0.0))
    pay = (ca.dot(x["PM"], QM) + x["PROW"] * ca.sum1(m.V("QROWC"))
           + ca.dot(W * DM(P["nrsh"]), x["LD"])
           + th(c.row, P["thK"]) * m.V("YFK")
           + float(P["divROW"]) * m.V("YENT")
           + float(P["GROW0"].sum()) * GI * ER)
    for ix_, xf in enumerate(FX):
        pay = pay + th(c.row, P["thX"][xf]) * m.V("YFX")[ix_]
    rec = (ca.dot(PE, QE)
           + float(P["YLROW0"].sum()) * GI * ER
           + float(P["YKROW0"]) * GI * ER
           + sum(float(v) for v in P["YXROW0"].values()) * GI * ER
           + float(P["TRROW0"].sum()) * GI * ER
           + float(P["ROWG0"].sum()) * GI * ER
           + float(P["ENTROW0"]) * GI * ER)
    m.add_eq("EQ_SROW", m.V("SROW"), pay - rec, desc="rest-of-world balance: saving = current-account deficit")
    # saving = investment
    inv_value = ca.dot(x["PK"], m.V("IND"))
    saving = ca.sum1(m.V("SAVH")) + m.V("SENT") + ca.sum1(m.V("SG")) + m.V("SROW")
    m.add_eq("EQ_SI", inv_value, saving + m.V("WALRAS"), desc="saving = investment (+ Walras slack)")
