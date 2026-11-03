"""Production: nested CES technology with zero profits (STAMP eqs. 9-12).

Top nest: CES (default Leontief) between value added and an intermediate
bundle; the bundle is Leontief across commodities.  Value added is a CES over
factors with sector-specific elasticities.  Factor demands use user costs that
include factor taxes and the property tax on the value of capital.
"""

from __future__ import annotations

import casadi as ca
import numpy as np

from ..functional import ces_index, vcat


def declare(M):
    m, c, S = M.m, M.c, M.S
    A, F = S["A"], S["F"]
    b = c.bench
    m.add_var("QX", (A,), level=b["QX"], lb=0.0, grow="q", desc="gross output")
    m.add_var("QVA", (A,), level=b["QVA"], lb=0.0, grow="q", desc="value added (at user cost)")
    m.add_var("QINTA", (A,), level=b["QINTA"], lb=0.0, grow="q", desc="intermediate input bundle")
    m.add_var("QF", (F, A), level=b["QF"], lb=0.0, grow="q", desc="factor demand (capital: stock units)")
    m.add_var("K", (A,), level=b["K"], lb=0.0, grow="q", fixed=True, desc="capital stock by sector")


def equations(M):
    m, c, S, x = M.m, M.c, M.S, M.x
    A, F = S["A"], S["F"]
    nA, nF = len(A), len(F)
    P = c.par
    PX, PVA, QX, QVA, QINTA, QF = (m.V(n) for n in ("PX", "PVA", "QX", "QVA", "QINTA", "QF"))
    TFP = m.P("TFP")
    UC, UC0 = x["UC"], P["UC0"]
    QF0, QVA0, QINTA0, QX0 = c.bench["QF"], c.bench["QVA"], c.bench["QINTA"], c.bench["QX"]
    pva, ctop, qf_rhs, qva_rhs, qint_rhs = [], [], [], [], []
    for j in range(nA):
        sva = P["sigma_va"][j]
        rel = [UC[i, j] / UC0[i, j] for i in range(nF)]
        idx = ces_index(P["thva"][:, j], rel, sva)
        pva.append(idx / TFP[j])
        st = P["sigma_top"][j]
        tv = P["thv_top"][j]
        ct = P["c0"][j] * ces_index([tv, 1 - tv], [PVA[j], x["PINTA"][j]], st)
        ctop.append(ct)
        for i in range(nF):
            qf_rhs.append(QF0[i, j] * (QVA[j] / QVA0[j]) / TFP[j]
                          * (PVA[j] * TFP[j] / (UC[i, j] / UC0[i, j])) ** sva)
        rc = ct / P["c0"][j]
        qva_rhs.append(QX[j] * (QVA0[j] / QX0[j]) * (rc / PVA[j]) ** st)
        qint_rhs.append(QX[j] * (QINTA0[j] / QX0[j]) * (rc / x["PINTA"][j]) ** st)
    x["CTOP"] = vcat(ctop)
    m.add_eq("EQ_PVA", PVA, vcat(pva), (A,), desc="price of value added = CES unit cost of factors (eq. 11)")
    # factor demands, element order (f, a) row-major
    QFmat = QF
    qf_r = ca.reshape(vcat(qf_rhs), nF, nA)  # built as (a outer, f inner) -> column-major nF x nA
    m.add_eq("EQ_QF", QFmat, qf_r, (F, A), desc="factor demands (Shephard's lemma, eq. 12)")
    m.add_eq("EQ_ZP", PX * (1 - x["tout"]), x["CTOP"], (A,), desc="zero profit net of output taxes")
    m.add_eq("EQ_QVA", QVA, vcat(qva_rhs), (A,), desc="value-added demand (top nest)")
    m.add_eq("EQ_QINTA", QINTA, vcat(qint_rhs), (A,), desc="intermediate bundle demand (top nest, eq. 9)")
