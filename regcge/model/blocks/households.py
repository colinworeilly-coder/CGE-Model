"""Households: income, direct taxes, and utility-consistent demand and labor supply.

Replaces STAMP eqs. 1-6 with a utility-consistent system:

* Income = labor income (wage x own labor supply, plus exogenous out-of-state
  earnings) + capital income shares + dividends + transfers (eq. 1).
* Direct taxes per income-tax instrument are linear in the tax base with a
  marginal rate and an intercept (virtual-income linearisation).  The base
  subtracts deductible taxes (SALT), so a new state tax lowers the federal
  liability of itemizers (eq. 21).
* Preferences: CES between leisure and an ELES bundle of commodities, direct
  out-of-state purchases and saving (with subsistence quantities).  Labor
  supply = time endowment - leisure.  Because demand comes from an explicit
  utility function, welfare is measured by exact equivalent variation.
"""

from __future__ import annotations

import casadi as ca
import numpy as np

from ..calibrate import effective_labor_mtr
from ..functional import DM, ces_index, colsum, rep_col, vcat


def declare(M):
    m, c, S = M.m, M.c, M.S
    A, H, KI = S["A"], S["H"], S["KI"]
    b = c.bench
    m.add_var("YH", (H,), level=b["YH"], grow="q", desc="household gross income")
    m.add_var("YD", (H,), level=b["YD"], lb=0.0, grow="q", desc="disposable income")
    m.add_var("U", (H,), level=b["U"], lb=0.0, grow="q", desc="utility (real full supernumerary income)")
    m.add_var("LEIS", (H,), level=b["LEIS"], lb=0.0, grow="q", desc="leisure (efficiency units)")
    m.add_var("LS", (H,), level=b["LS"], lb=0.0, grow="q", desc="resident labor supply (efficiency units)")
    m.add_var("ZB", (H,), level=b["Z"], lb=0.0, grow="q", desc="supernumerary consumption-saving bundle")
    m.add_var("QH", (A, H), level=b["QH"], lb=0.0, grow="q", desc="household consumption by commodity")
    m.add_var("SAVH", (H,), level=b["SAVH"], grow="q", desc="household saving")
    m.add_var("QROWC", (H,), level=b["QROWC"], lb=0.0, grow="q", desc="direct out-of-state purchases")
    if len(KI):
        m.add_var("TAXH", (KI, H), level=b["TAXH"], grow="q", desc="direct taxes paid by households")
    m.add_var("TRSCALE", (S["G"],), level=1.0, fixed=True, desc="scale of government transfers to households")


def equations(M):
    m, c, S, x = M.m, M.c, M.S, M.x
    A, H, G, FL, FX, KI = S["A"], S["H"], S["G"], S["FL"], S["FX"], S["KI"]
    nA, nH, nG = len(A), len(H), len(G)
    P, b = c.par, c.bench
    GI, PCI, ER = m.P("GI"), m.P("PCI"), m.V("ER")
    W, LS, N, CPI = m.V("W"), m.V("LS"), m.V("N"), m.V("CPI")
    YH, YD, U, LEIS, ZB, QH, SAVH, QROWC = (m.V(n) for n in ("YH", "YD", "U", "LEIS", "ZB", "QH", "SAVH", "QROWC"))
    N0 = b["N"]
    # ---- income components
    lab = 0
    for i, l in enumerate(FL):
        lab = lab + W[i] * DM(P["lshare"][i]) * LS + DM(P["rowlab"][i] * P["YLROW0"][i]) * GI * ER
    x["YL"] = lab
    thK = P["thK"]
    capinc = DM([thK.get(h, 0.0) for h in H]) * m.V("YFK")
    for ix_, xf in enumerate(FX):
        capinc = capinc + DM([P["thX"][xf].get(h, 0.0) for h in H]) * m.V("YFX")[ix_]
    capinc = capinc + DM(P["div"]) * m.V("YENT")
    x["YK"] = capinc
    TRS = m.V("TRSCALE")
    trg = DM(P["trpc"]) * ca.repmat(TRS.T, nH, 1) * rep_col(N * PCI * CPI, nG)    # (H x G)
    x["TRG"] = trg
    x["TR"] = ca.sum2(trg) + DM(P["TRROW0"]) * GI * ER
    m.add_eq("EQ_YH", YH, x["YL"] + x["YK"] + x["TR"], (H,), desc="household gross income (eq. 1)")
    # ---- direct taxes
    comp = {"labor": x["YL"], "capital": x["YK"], "transfers": x["TR"]}
    ded = DM(P["ded"])
    if len(KI):
        TAXH = m.V("TAXH")
        rhs = []
        base_expr = {}
        for i, k in enumerate(KI):
            base = 0
            for cname, w in P["wcomp"][k].items():
                if w:
                    base = base + w * comp[cname]
            for d in P["dlist"][k]:
                base = base - ded * x["paid"][d]
            base_expr[k] = base
            liab = x["ts"][k] * (m.P(f"mtr_{k}") * base - m.P(f"apar_{k}") * PCI * (N / DM(N0)) * CPI)
            rhs.append(liab.T)
        x["BASE"] = base_expr
        x["TAXH_RHS"] = {k: rhs[i].T for i, k in enumerate(KI)}
        m.add_eq("EQ_TAXH", TAXH, ca.vertcat(*rhs), (KI, H), desc="direct taxes by instrument (eq. 21)")
        m.add_eq("EQ_YD", YD, YH - ca.sum1(TAXH).T, (H,), desc="disposable income (eq. 2)")
        mtr = {k: x["ts"][k] * m.P(f"mtr_{k}") for k in KI}
        x["MTRL"] = effective_labor_mtr(KI.labels, P["wcomp"], P["dlist"], mtr, P["ded"], nH, xp=ca)
    else:
        m.add_eq("EQ_YD", YD, YH, (H,), desc="disposable income (eq. 2)")
        x["MTRL"] = ca.SX.zeros(nH, 1)
    # ---- prices faced by the household
    wbar = 0
    for i, l in enumerate(FL):
        wbar = wbar + DM(P["lshare"][i]) * W[i]
    x["PL"] = wbar * (1 - x["MTRL"])
    items_P = ca.vertcat(x["PH"], ca.repmat(x["PROW"], 1, nH), ca.repmat(x["PSAV"], 1, nH))  # (nA+2) x H
    P0 = P["items_P0"]
    beta = P["beta"]
    gam = P["gamma_pc"]
    pz, phi, committed = [], [], []
    for j in range(nH):
        lp = 0
        for i in range(nA + 2):
            if beta[i, j] > 0:
                lp = lp + beta[i, j] * ca.log(items_P[i, j] / P0[i, j])
        pz.append(ca.exp(lp))
    PZ = vcat(pz)
    x["PZ"] = PZ
    committed = colsum(items_P * DM(gam)) * N * PCI
    SUP = YD - committed
    x["SUP"] = SUP
    FS = SUP + x["PL"] * LEIS
    PLrel = x["PL"] / DM(P["PL0"])
    for j in range(nH):
        phi.append(ces_index([P["thZ"][j], P["thL"][j]], [PZ[j], PLrel[j]], P["sigma_ls"][j]))
    PHI = vcat(phi)
    x["PHI"] = PHI
    sig = DM(P["sigma_ls"])
    U0, Z0, LEIS0 = b["U"], b["Z"], b["LEIS"]
    m.add_eq("EQ_U", U * PHI, FS, (H,), desc="utility = full supernumerary income / CES price index")
    m.add_eq("EQ_LEIS", LEIS, U * DM(LEIS0 / U0) * (PHI / PLrel) ** sig, (H,), desc="leisure demand")
    m.add_eq("EQ_ZB", ZB, U * DM(Z0 / U0) * (PHI / PZ) ** sig, (H,), desc="consumption-saving bundle demand")
    m.add_eq("EQ_LS", LS, N * PCI * DM(P["epc"]) - LEIS, (H,), desc="labor supply = endowment - leisure (replaces eq. 6)")
    pzz = PZ * ZB
    qh_rhs = DM(gam[:nA]) * rep_col(N * PCI, nA).T + DM(beta[:nA]) * ca.repmat(pzz.T, nA, 1) / x["PH"]
    m.add_eq("EQ_QH", QH, qh_rhs, (A, H), desc="ELES consumption demand (replaces eq. 3)")
    m.add_eq("EQ_SAVH", SAVH, x["PSAV"] * DM(gam[nA + 1]) * N * PCI + DM(beta[nA + 1]) * pzz, (H,),
             desc="household saving (ELES; replaces eq. 4)")
    m.add_eq("EQ_QROWC", QROWC, DM(gam[nA]) * N * PCI + DM(beta[nA]) * pzz / x["PROW"], (H,),
             desc="direct out-of-state purchases")
