"""RegionalCGE: assembles the equation blocks into a square system.

Typical use::

    from regcge import Dataset, RegionalCGE
    ds = Dataset.load("data/illustrative_state")
    model = RegionalCGE(ds)
    z, p = model.benchmark()
    res = model.solve(z, p)          # one period

Multi-period baseline and scenario paths are run by :mod:`regcge.model.dynamics`.
"""

from __future__ import annotations

import casadi as ca
import numpy as np

from ..core.model_base import CoreModel
from ..core.solver import Solver
from ..core.system import check_square
from ..data.dataset import Dataset
from .blocks import (factors, government, households, investment, markets, migration, prices,
                     production, taxes, trade)
from .calibrate import Calibration
from .closures import Closure, apply_closure
from .functional import DM, colsum

BLOCKS = [prices, production, trade, factors, taxes, households, migration, government, investment, markets]


class RegionalCGE:
    """A recursive-dynamic regional tax CGE model built from a Dataset."""

    def __init__(self, data: Dataset | Calibration, spec: dict | None = None, blocks=None):
        cal = data if isinstance(data, Calibration) else Calibration.from_dataset(data, spec)
        self.c = self.cal = cal
        self.ds = cal.ds
        self.spec = cal.spec
        cl = self.spec.get("closures", {})
        self.capital_mobile = cl.get("capital", "sector_specific") == "mobile"
        self.migration_driver = cl.get("migration_driver", "real_income")
        self.S = cal.sets
        self.m = CoreModel("regcge")
        self.x: dict = {}
        self.blocks = list(blocks or BLOCKS)
        for B in self.blocks:
            B.declare(self)
        for B in self.blocks:
            B.equations(self)
        self._report_expressions()
        self.m.compile()
        self.closure = Closure.from_spec(self.spec, self.S["G"].labels)
        self._solvers: dict = {}
        sol = self.spec.get("solver", {})
        self.solver_method = sol.get("method", "auto")
        self.solver_tol = float(sol.get("tol", 1e-8))

    # ------------------------------------------------------------------ setup
    def benchmark(self):
        """Scaled benchmark solution vector and base parameter vector."""
        return self.m.z_benchmark(), self.m.p_benchmark()

    def fixed_mask(self, closure: Closure | None = None, migration_active: bool = False) -> np.ndarray:
        return apply_closure(self, closure or self.closure, migration_active)

    def check_closure(self, closure: Closure | None = None, migration_active: bool = False) -> str:
        return check_square(self.m, self.fixed_mask(closure, migration_active), raise_error=False)

    def solver(self, fixed: np.ndarray) -> Solver:
        key = fixed.tobytes()
        if key not in self._solvers:
            self._solvers[key] = Solver(self.m, fixed, method=self.solver_method, tol=self.solver_tol)
        return self._solvers[key]

    def solve(self, z_guess: np.ndarray, p: np.ndarray, closure: Closure | None = None,
              migration_active: bool = False, z_start=None, p_start=None):
        fixed = self.fixed_mask(closure, migration_active)
        return self.solver(fixed).solve(z_guess, p, z_start, p_start)

    # --------------------------------------------------------------- reports
    def _report_expressions(self):
        m, c, S, x = self.m, self.c, self.S, self.x
        A, H, G, FL, K = S["A"], S["H"], S["G"], S["FL"], S["K"]
        P, b = c.par, c.bench
        nA = len(A)
        GI = m.P("GI")
        PCI = m.P("PCI")
        QH, IND, QE, QM, PE = m.V("QH"), m.V("IND"), m.V("QE"), m.V("QM"), m.V("PE")
        gov_labor = 0
        gov_labor0 = 0
        for i, l in enumerate(FL):
            gov_labor = gov_labor + m.V("W")[i] * (1 + x["tfg"][l]) * x["QFG"][i, :].T
            tfg0 = sum(P["tfg"][k] for k in S["KF"] if P["tax_factor"][k] == l) if len(S["KF"]) else 0
            gov_labor0 = gov_labor0 + DM(1 + tfg0) * x["QFG"][i, :].T
        gcons = colsum(x["PG"] * x["QG"]) + gov_labor + DM(P["YKG0"]) * GI * m.V("ER")
        hcons = ca.sum1(ca.vec(x["PH"] * QH))
        gsp = hcons + ca.dot(x["PK"], IND) + ca.sum1(gcons) + ca.dot(PE, QE) - ca.dot(x["PM"], QM)
        PG0 = 1 + sum(P["tsg"].values()) if P["tsg"] else np.ones((nA, len(G)))
        rgsp = (ca.sum1(ca.vec(DM(P["PH0"]) * QH)) + ca.dot(DM(P["PK0"]), IND)
                + ca.sum1(colsum(DM(PG0) * x["QG"]) + gov_labor0 + DM(P["YKG0"]) * GI)
                + ca.sum1(QE) - ca.sum1(QM))
        m.add_expr("GSP", gsp)
        m.add_expr("RGSP", rgsp)
        m.add_expr("SPI", ca.sum1(m.V("YH")))
        m.add_expr("RYD", m.V("YD") / m.V("CPI"), (H,))
        # jobs (thousands): efficiency units / productivity index x base jobs per unit
        sec_jobs = self.ds.sectors["jobs"].to_numpy(float)
        F = S["F"]
        lab_rows = [F.index(l) for l in FL]
        QFL0 = b["QF"][lab_rows].sum(axis=0)
        jpu = np.where(np.isnan(sec_jobs), 1.0, sec_jobs) / np.where(QFL0 > 0, QFL0, 1)
        qfl = 0
        for r in lab_rows:
            qfl = qfl + m.V("QF")[r, :].T
        m.add_expr("JOBS", qfl * DM(jpu) / PCI, (A,))
        gjobs = c.info["gov_jobs"]
        QFG0 = b["QFG"].sum(axis=0)
        gjpu = np.where(np.isnan(gjobs), 1.0, gjobs) / np.where(QFG0 > 0, QFG0, 1)
        m.add_expr("GJOBS", ca.sum1(x["QFG"]).T * DM(gjpu) / PCI, (G,))
        m.add_expr("PK", x["PK"], (A,))
        m.add_expr("PM", x["PM"], (A,))
        m.add_expr("PHI", x["PHI"], (H,))
        m.add_expr("PZ", x["PZ"], (H,))
        m.add_expr("PL", x["PL"], (H,))
        m.add_expr("MTRL", x["MTRL"], (H,))
        m.add_expr("YL", x["YL"], (H,))
        m.add_expr("YK", x["YK"], (H,))
        m.add_expr("TR", x["TR"], (H,))
        m.add_expr("SUP", x["SUP"], (H,))
        m.add_expr("GSPEND", x["GSPEND"], (G,))
        m.add_expr("INVEST", ca.dot(x["PK"], IND))
        m.add_expr("LD", x["LD"], (FL,))
        m.add_expr("PSAV", x["PSAV"])
        m.add_expr("COMMITTED", m.V("YD") - x["SUP"], (H,))
        for k in S["KI"]:
            m.add_expr(f"BASE_{k}", x["BASE"][k], (H,))
        # revenue at current rates holding the tax base (used for static estimates)
        rs = []
        for i, k in enumerate(K):
            rs.append(ca.sum1(x["TAXH_RHS"][k]) if k in S["KI"] else x["rev_list"][i])
        m.add_expr("REVSTATIC", ca.vertcat(*rs), (K,))
        for k, v in x["paid"].items():
            m.add_expr(f"PAID_{k}", v, (H,))
