"""Governments and the enterprise sector (STAMP eqs. 22-27).

Each government (e.g. FED, STATE, LOCAL) receives shares of tax revenues,
intergovernmental transfers, capital income and transfers from the rest of the
world.  It buys a fixed-proportion bundle of goods and labor (real level QGL),
pays indexed transfers to households and transfers to other governments, and
saves.  Exogenous nominal flows (fixed grants, government capital rentals,
transfers to enterprises) are indexed to growth and to the numeraire (ER), so
the model is homogeneous of degree zero in prices.  Closures (see
closures.py) choose which of QGL, SG, TRSCALE or a tax instrument's TSCALE
adjusts to satisfy the budget.
"""

from __future__ import annotations

import casadi as ca
import numpy as np

from ..functional import DM, colsum, rep_row, vcat


def declare(M):
    m, c, S = M.m, M.c, M.S
    G = S["G"]
    b = c.bench
    m.add_var("YG", (G,), level=b["YG"], grow="q", desc="government receipts")
    m.add_var("QGL", (G,), level=b["QGL"], lb=0.0, grow="q", desc="real government spending on goods and labor")
    m.add_var("SG", (G,), level=b["SG"], grow="q", desc="government saving")
    m.add_var("YENT", (), level=b["YENT"], grow="q", desc="enterprise income")
    m.add_var("SENT", (), level=b["SENT"], grow="q", desc="enterprise saving (retained earnings)")


def equations(M):
    m, c, S, x = M.m, M.c, M.S, M.x
    A, G, H, FL, FX, K = S["A"], S["G"], S["H"], S["FL"], S["FX"], S["K"]
    nG = len(G)
    P = c.par
    GI, ER = m.P("GI"), m.V("ER")
    YG, QGL, SG, W = m.V("YG"), m.V("QGL"), m.V("SG"), m.V("W")
    YFK, YENT = m.V("YFK"), m.V("YENT")
    # intergovernmental transfers TRGG[to, from]
    share_mode = P["intergov_share"]
    cols = []
    for j in range(nG):
        if share_mode[j]:
            cols.append(DM(P["mu"][:, j]) * YG[j])
        else:
            cols.append(DM(P["TRGG0"][:, j]) * GI * ER)
    TRGG = ca.horzcat(*cols)
    x["TRGG"] = TRGG
    # receipts
    yg = ca.mtimes(DM(P["shTG"]).T, m.V("TAXR")) + ca.sum2(TRGG)
    yg = yg + DM([P["thK"].get(g, 0.0) for g in G]) * YFK
    for ix_, xf in enumerate(FX):
        yg = yg + DM([P["thX"][xf].get(g, 0.0) for g in G]) * m.V("YFX")[ix_]
    yg = yg + DM(P["divG"]) * YENT + DM(P["ROWG0"]) * GI * ER
    m.add_eq("EQ_YG", YG, yg, (G,), desc="government receipts (eq. 22)")
    # spending
    goods = colsum(x["PG"] * x["QG"])
    labor = 0
    for i, l in enumerate(FL):
        labor = labor + W[i] * (1 + x["tfg"][l]) * x["QFG"][i, :].T
    transfers_hh = colsum(x["TRG"])
    spend = (goods + labor + DM(P["YKG0"]) * GI * ER + transfers_hh + colsum(TRGG)
             + DM(P["GENT0"]) * GI * ER + DM(P["GROW0"]) * GI * ER)
    x["GSPEND"] = spend
    m.add_eq("EQ_SG", SG, YG - spend, (G,), desc="government saving = receipts - spending (eq. 25)")
    # enterprise
    yent = DM([P["thK"].get(c.ent, 0.0)]) * YFK if c.ent else ca.SX(0)
    if c.ent:
        for ix_, xf in enumerate(FX):
            yent = yent + P["thX"][xf].get(c.ent, 0.0) * m.V("YFX")[ix_]
        yent = yent + float(P["GENT0"].sum()) * GI * ER + float(P["ENTROW0"]) * GI * ER
    m.add_eq("EQ_YENT", YENT, yent, desc="enterprise income")
    retain = 1 - P["div"].sum() - P["divROW"] - P["divG"].sum()
    m.add_eq("EQ_SENT", m.V("SENT"), retain * YENT, desc="enterprise saving")
