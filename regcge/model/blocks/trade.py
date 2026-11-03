"""Trade with the rest of the United States and the world (STAMP eqs. 14-17).

* Armington CES composite of in-state and imported goods.
* Output is either a single good sold at home and abroad (perfect
  transformation, as in STAMP) or split by a CET function.
* Export demand has a finite price elasticity (regional price setting) or the
  state is a price taker (infinite elasticity).

Zero benchmark flows get replacement equations so the system stays
non-singular (e.g. a sector without exports pins its export price to PX).
"""

from __future__ import annotations

import numpy as np

from ..functional import ces_index, cet_index, vcat


def declare(M):
    m, c, S = M.m, M.c, M.S
    A = S["A"]
    b = c.bench
    m.add_var("QD", (A,), level=b["QD"], lb=0.0, grow="q", desc="in-state sales of in-state output")
    m.add_var("QE", (A,), level=b["QE"], lb=0.0, grow="q", desc="exports (out-of-state sales)")
    m.add_var("QM", (A,), level=b["QM"], lb=0.0, grow="q", desc="imports")
    m.add_var("QQ", (A,), level=b["QQ"], lb=0.0, grow="q", desc="Armington composite supply")


def equations(M):
    m, c, S, x = M.m, M.c, M.S, M.x
    A = S["A"]
    nA = len(A)
    P, b = c.par, c.bench
    PX, PD, PE, PQ = (m.V(n) for n in ("PX", "PD", "PE", "PQ"))
    QX, QD, QE, QM, QQ = (m.V(n) for n in ("QX", "QD", "QE", "QM", "QQ"))
    PM, ER = x["PM"], m.V("ER")
    GI, EXS, PWE = m.P("GI"), m.P("EXS"), m.P("PWE")
    QD0, QE0, QM0, QQ0, QX0 = b["QD"], b["QE"], b["QM"], b["QQ"], b["QX"]
    arm_p, arm_d, arm_m, exp_l, exp_r, t1l, t1r, t2l, t2r, t3l, t3r = ([] for _ in range(11))
    for j in range(nA):
        sa = P["sigma_arm"][j]
        # ---- Armington
        if QQ0[j] > 0:
            td, tm = QD0[j] / QQ0[j], QM0[j] / QQ0[j]
            pq = ces_index([td, tm], [PD[j], PM[j]], sa)
            arm_p.append((PQ[j], pq))
            arm_d.append((QD[j], QD0[j] * (QQ[j] / QQ0[j]) * (PQ[j] / PD[j]) ** sa))
            arm_m.append((QM[j], QM0[j] * (QQ[j] / QQ0[j]) * (PQ[j] / PM[j]) ** sa))
        else:  # no in-state absorption
            arm_p.append((PQ[j], PD[j]))
            arm_d.append((QD[j], 0.0))
            arm_m.append((QM[j], 0.0))
        # ---- transformation and export demand
        cet = P["cet"][j]
        eta_e = P["eta_exp"][j]
        if not np.isfinite(cet):
            t1l.append(PD[j]); t1r.append(PX[j])
            t2l.append(PE[j]); t2r.append(PX[j])
            t3l.append(QX[j]); t3r.append(QD[j] + QE[j])
            if QE0[j] <= 0:
                exp_l.append(QE[j]); exp_r.append(0.0)
            elif np.isfinite(eta_e):
                exp_l.append(QE[j]); exp_r.append(QE0[j] * GI * EXS[j] * (PE[j] / (PWE[j] * ER)) ** (-eta_e))
            else:
                exp_l.append(PE[j]); exp_r.append(PWE[j] * ER)
        else:
            th_d, th_e = QD0[j] / QX0[j], QE0[j] / QX0[j]
            t1l.append(PX[j]); t1r.append(cet_index([th_d, th_e], [PD[j], PE[j]], cet))
            if QD0[j] > 0:
                t2l.append(QD[j]); t2r.append(QD0[j] * (QX[j] / QX0[j]) * (PD[j] / PX[j]) ** cet)
            else:
                t2l.append(PD[j]); t2r.append(PX[j])
            t3l.append(QE[j]); t3r.append(QE0[j] * (QX[j] / QX0[j]) * (PE[j] / PX[j]) ** cet)
            if QE0[j] <= 0:
                exp_l.append(PE[j]); exp_r.append(PX[j])
            elif np.isfinite(eta_e):
                exp_l.append(QE[j]); exp_r.append(QE0[j] * GI * EXS[j] * (PE[j] / (PWE[j] * ER)) ** (-eta_e))
            else:
                exp_l.append(PE[j]); exp_r.append(PWE[j] * ER)

    def add(name, pairs, desc):
        m.add_eq(name, vcat([p[0] for p in pairs]), vcat([p[1] for p in pairs]), (A,), desc=desc)

    add("EQ_ARM_P", arm_p, "Armington composite price (eq. 17)")
    add("EQ_ARM_D", arm_d, "demand for in-state goods (eq. 15)")
    add("EQ_ARM_M", arm_m, "import demand (eq. 16)")
    m.add_eq("EQ_EXP", vcat(exp_l), vcat(exp_r), (A,), desc="export demand (eq. 14)")
    m.add_eq("EQ_TR1", vcat(t1l), vcat(t1r), (A,), desc="transformation: price")
    m.add_eq("EQ_TR2", vcat(t2l), vcat(t2r), (A,), desc="transformation: in-state supply")
    m.add_eq("EQ_TR3", vcat(t3l), vcat(t3r), (A,), desc="transformation: export supply / output balance")
