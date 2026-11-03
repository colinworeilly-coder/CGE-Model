"""Factor markets and factor incomes (STAMP eqs. 13, 29, 30, 39).

* Labor (one or more types) is measured in efficiency units with an
  economy-wide wage per type (fixed inter-sectoral differentials, eq. 39);
  a fixed share of jobs is filled by non-residents (commuters).
* Capital is sector-specific within a period (rental rate clears each
  sector's capital market, eq. 30) or, in the long-run static mode, mobile
  across sectors with fixed relative returns.
* Fixed factors (land, resources) are sector-specific and grow with the economy.
"""

from __future__ import annotations

import casadi as ca
import numpy as np

from ..functional import DM, colsum, rep_row, vcat


def declare(M):
    m, c, S = M.m, M.c, M.S
    FX = S["FX"]
    m.add_var("YFK", (), level=sum_capital_income(c), grow="q", desc="capital income (net of business taxes)")
    if len(FX):
        lev = [c.bench["QF_X"][i].sum() + c.par["YXROW0"][x] for i, x in enumerate(FX)]
        m.add_var("YFX", (FX,), level=lev, grow="q", desc="income of fixed factors")
    if M.capital_mobile:
        rk0, k0 = c.bench["RK"], c.bench["K"]
        m.add_var("RKAVG", (), level=(rk0 * k0).sum() / k0.sum(), lb=0.0,
                  desc="economy-wide average capital rental rate")
        m.add_param("KTOT", (), c.bench["K"].sum(), "total capital stock (mobile-capital mode)")


def sum_capital_income(c):
    return c.bench["KINC"].sum() + c.par["YKG0"].sum() + c.par["YKROW0"]


def equations(M):
    m, c, S, x = M.m, M.c, M.S, M.x
    A, F, FL, FX, G = S["A"], S["F"], S["FL"], S["FX"], S["G"]
    nA, nG = len(A), len(G)
    P = c.par
    QF, QGL, LS = m.V("QF"), m.V("QGL"), m.V("LS")
    GI, ER = m.P("GI"), m.V("ER")
    # government employment by labor type (FL x G)
    x["QFG"] = DM(P["qfg"]) * rep_row(QGL, len(FL))
    # labor markets
    ld, ls = [], []
    for i, l in enumerate(FL):
        demand = ca.sum2(QF[F.index(l), :]) + ca.sum2(x["QFG"][i, :])
        ld.append(demand)
        ls.append(ca.dot(DM(P["lshare"][i]), LS))
    x["LD"] = vcat(ld)
    m.add_eq("EQ_LAB", (1 - DM(P["nrsh"])) * x["LD"], vcat(ls), (FL,),
             desc="labor market: resident supply = demand net of commuters (eq. 29)")
    # capital
    iK = F.index(c.cap)
    K, RK = m.V("K"), m.V("RK")
    hasK = P["hasK"]
    m.add_eq("EQ_KAP", QF[iK, :].T, K, (A,), where=hasK, desc="capital market by sector (eq. 30)")
    if M.capital_mobile:
        rk0 = c.bench["RK"]
        avg0 = (rk0 * c.bench["K"]).sum() / c.bench["K"].sum()
        m.add_eq("EQ_RKMOB", RK, m.V("RKAVG") * DM(rk0 / avg0), (A,), where=hasK,
                 desc="mobile capital: fixed relative rental rates")
        m.add_eq("EQ_KTOT", ca.sum1(K), m.P("KTOT"), desc="mobile capital: total stock")
    # fixed factors
    for ix_, xf in enumerate(FX):
        q0 = c.bench["QF_X"][ix_]
        m.add_eq(f"EQ_FIX_{xf}", QF[F.index(xf), :].T, DM(q0) * GI, (A,), where=q0 > 0,
                 desc=f"supply of fixed factor {xf}")
    # factor incomes
    yfk = ca.dot(RK, K) + float(P["YKG0"].sum()) * GI * ER + float(P["YKROW0"]) * GI * ER
    m.add_eq("EQ_YFK", m.V("YFK"), yfk, desc="capital income (eq. 13)")
    if len(FX):
        RFX = m.V("RFX")
        rows = []
        for ix_, xf in enumerate(FX):
            rows.append(ca.dot(RFX[ix_, :].T, QF[F.index(xf), :].T) + float(P["YXROW0"][xf]) * GI * ER)
        m.add_eq("EQ_YFX", m.V("YFX"), vcat(rows), (FX,), desc="fixed-factor income")
