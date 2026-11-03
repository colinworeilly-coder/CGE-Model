"""Prices, exogenous parameters and purchaser-price wedges.

Declares commodity and factor prices, the exchange rate (numeraire by
default) and all time-varying parameters, and builds the tax-inclusive
purchaser prices used by the other blocks (STAMP eqs. 5, 11, 17, 20).
"""

from __future__ import annotations

import casadi as ca
import numpy as np

from ..functional import DM, rep_col, colsum


def declare(M):
    m, c, S = M.m, M.c, M.S
    A, FL, FX, H, G = S["A"], S["FL"], S["FX"], S["H"], S["G"]
    b = c.bench
    for nm in ("PX", "PD", "PE", "PQ", "PVA"):
        m.add_var(nm, (A,), level=1.0, lb=0.0, desc=f"price {nm}")
    m.add_var("W", (FL,), level=1.0, lb=0.0, desc="wage per efficiency unit, by labor type")
    m.add_var("RK", (A,), level=b["RK"], lb=0.0, desc="rental rate of capital by sector (net of business taxes)")
    if len(FX):
        m.add_var("RFX", (FX, A), level=1.0, lb=0.0, desc="rent of sector-specific fixed factors")
    m.add_var("ER", (), level=1.0, lb=0.0, fixed=True, desc="exchange rate (price of rest-of-world goods)")
    m.add_var("CPI", (H,), level=1.0, lb=0.0, desc="consumer price index by household")
    m.add_var("CPIAVG", (), level=1.0, lb=0.0, desc="aggregate CPI")
    # --- time-varying parameters ---------------------------------------------
    m.add_param("GI", (), 1.0, "aggregate growth index (1+g)^t")
    m.add_param("PCI", (), 1.0, "per-capita/productivity index (1+a)^t")
    m.add_param("PWM", (A,), 1.0, "rest-of-world import price")
    m.add_param("PWE", (A,), 1.0, "rest-of-world competing export price")
    m.add_param("PROW", (), 1.0, "price of direct out-of-state household purchases")
    m.add_param("TFP", (A,), 1.0, "total factor productivity (value added)")
    m.add_param("EXS", (A,), 1.0, "export demand shifter")
    for k in S["KS"]:
        m.add_param(f"tsh_{k}", (A, H), c.par["tsh"][k], f"{k} rate on household purchases")
        m.add_param(f"tsi_{k}", (A, A), c.par["tsi"][k], f"{k} rate on intermediate purchases (c x buyer)")
        m.add_param(f"tsv_{k}", (A,), c.par["tsv"][k], f"{k} rate on investment purchases")
        m.add_param(f"tsg_{k}", (A, G), c.par["tsg"][k], f"{k} rate on government purchases")
    for k in S["KO"]:
        m.add_param(f"tout_{k}", (A,), c.par["tout"][k], f"{k} rate on gross output")
    for k in S["KF"]:
        m.add_param(f"tf_{k}", (A,), c.par["tf"][k], f"{k} rate on factor payments by sector")
        m.add_param(f"tfg_{k}", (G,), c.par["tfg"][k], f"{k} rate on government factor payments")
    for k in S["KP"]:
        m.add_param(f"tprop_{k}", (A,), c.par["tprop"][k], f"{k} rate on value of capital stock")
    for k in S["KI"]:
        m.add_param(f"mtr_{k}", (H,), c.par["mtr0"][k], f"{k} marginal rate")
        m.add_param(f"apar_{k}", (H,), c.par["A0"][k], f"{k} linearisation intercept (base-year $)")


def equations(M):
    m, c, S, x = M.m, M.c, M.S, M.x
    A, H, G, FL = S["A"], S["H"], S["G"], S["FL"]
    nA, nH, nG = len(A), len(H), len(G)
    K = S["K"]
    TS = m.V("TSCALE")
    ts = {k: TS[K.index(k)] for k in K}
    x["ts"] = ts
    zA = ca.SX.zeros(nA, 1)

    def tot(prefix, ks, shape):
        out = ca.SX.zeros(*shape)
        for k in ks:
            out = out + ts[k] * m.P(f"{prefix}_{k}")
        return out

    x["tsh"] = tot("tsh", S["KS"], (nA, nH))
    x["tsi"] = tot("tsi", S["KS"], (nA, nA))
    x["tsv"] = tot("tsv", S["KS"], (nA, 1))
    x["tsg"] = tot("tsg", S["KS"], (nA, nG))
    x["tout"] = tot("tout", S["KO"], (nA, 1))
    x["tprop"] = tot("tprop", S["KP"], (nA, 1))
    # factor taxes aggregated by factor
    tf, tfg = {}, {}
    for f in S["F"]:
        ks = [k for k in S["KF"] if c.par["tax_factor"][k] == f]
        tf[f] = tot("tf", ks, (nA, 1))
        tfg[f] = tot("tfg", ks, (nG, 1))
    x["tf"], x["tfg"] = tf, tfg

    PQ, ER = m.V("PQ"), m.V("ER")
    x["PM"] = m.P("PWM") * ER
    x["PROW"] = m.P("PROW") * ER
    x["PH"] = rep_col(PQ, nH) * (1 + x["tsh"])                       # household purchaser prices
    x["PINTA"] = colsum(DM(c.par["icoef"]) * rep_col(PQ, nA) * (1 + x["tsi"]))
    x["PK"] = colsum(DM(c.par["B"]) * rep_col(PQ * (1 + x["tsv"]), nA))  # price of capital goods
    x["PSAV"] = ca.dot(DM(c.par["wsav"] / c.par["PK0"]), x["PK"])      # price index of saving
    x["PG"] = rep_col(PQ, nG) * (1 + x["tsg"])
    # factor user costs (F x A)
    rows = []
    fac = M.ds.accounts["factors"]
    W = m.V("W")
    for f in S["F"]:
        if fac[f] == "labor":
            rows.append((W[FL.index(f)] * (1 + tf[f])).T)
        elif fac[f] == "capital":
            rows.append((m.V("RK") * (1 + tf[f]) + x["tprop"] * x["PK"]).T)
        else:
            rows.append((m.V("RFX")[S["FX"].index(f), :].T * (1 + tf[f])).T)
    x["UC"] = ca.vertcat(*rows)
    # consumer prices
    CPI = m.V("CPI")
    cpi = colsum(DM(c.par["wc0"] / c.par["PH0"]) * x["PH"]) + DM(c.par["wr0"]) * x["PROW"]
    m.add_eq("EQ_CPI", CPI, cpi, (H,), desc="Laspeyres CPI by household (STAMP eq. 5)")
    m.add_eq("EQ_CPIAVG", m.V("CPIAVG"), ca.dot(DM(c.par["wcpi"]), CPI), desc="aggregate CPI")
