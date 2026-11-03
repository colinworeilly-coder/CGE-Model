"""Recursive dynamics: baseline and scenario paths.

Each year is a static equilibrium.  Between years:

* capital accumulates, ``K_{t+1} = (1 - delta) K_t + IND_t`` (STAMP eq. 18);
* growth indices advance: GI = (1+g)^t for aggregates, PCI = (1+a)^t for
  per-capita quantities, NBAR = N0 (1+n)^t for households;
* the migration lag NLR = N_{t-1}/NBAR_{t-1} is updated.

Because base-year investment equals (g + delta) K0 by construction, the
baseline with no shocks is an exact balanced-growth path: real quantities grow
at g and prices are constant.  A scenario path re-solves every year with the
policy applied; variables that are exogenous under the scenario closure take
their baseline values unless the scenario shocks them, so a scenario without
shocks reproduces the baseline exactly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..core.solver import SolverError
from .closures import Closure


@dataclass
class PathResult:
    model: object
    years: list
    z: list
    p: list
    stats: list
    closure: Closure
    label: str = "baseline"
    _levels: list = field(default=None, repr=False)
    _exprs: list = field(default=None, repr=False)

    @property
    def levels(self) -> list[dict]:
        if self._levels is None:
            self._levels = [self.model.m.levels(z) for z in self.z]
        return self._levels

    @property
    def exprs(self) -> list[dict]:
        if self._exprs is None:
            self._exprs = [self.model.m.evaluate_exprs(z, p) for z, p in zip(self.z, self.p)]
        return self._exprs

    def get(self, name: str):
        """Array of a variable or reporting expression over time (time first)."""
        if name in self.model.m.vars:
            return np.array([lv[name] for lv in self.levels])
        if name in self.model.m.exprs:
            return np.array([ex[name] for ex in self.exprs])
        raise KeyError(f"{name} is neither a variable nor a reporting expression")

    def param(self, name: str):
        return np.array([self.model.m.get_param(p, name) for p in self.p])

    @property
    def seconds(self) -> float:
        return float(sum(s.seconds for s in self.stats))


def _growth_factors(M, t: int) -> dict:
    P = M.c.par
    return {"p": 1.0, "q": (1 + P["g"]) ** t, "n": (1 + P["n"]) ** t, "c": (1 + P["a"]) ** t}


def _grown_vector(M, z_ref: np.ndarray, t_from: int, t_to: int) -> np.ndarray:
    """Scale a solution vector from period t_from to t_to by growth tags."""
    f0, f1 = _growth_factors(M, t_from), _growth_factors(M, t_to)
    z = z_ref.copy()
    for v in M.m.vars.values():
        r = f1[v.grow] / f0[v.grow]
        if r != 1.0:
            z[v.offset:v.offset + v.n] *= r
    return z


class Simulator:
    """Runs baseline and scenario paths for a RegionalCGE model."""

    def __init__(self, model, periods: int | None = None, verbose: bool = False):
        self.M = model
        self.periods = int(periods or model.spec.get("periods", 11))
        self.base_year = int(model.spec.get("base_year", model.ds.accounts.get("base_year", 2024)))
        self.verbose = verbose

    @property
    def years(self) -> list[int]:
        return [self.base_year + t for t in range(self.periods)]

    # --------------------------------------------------------------- helpers
    def _params(self, t: int, nlr: np.ndarray | None = None) -> np.ndarray:
        M = self.M
        p = M.m.p_benchmark().copy()
        f = _growth_factors(M, t)
        M.m.set_param(p, "GI", f["q"])
        M.m.set_param(p, "PCI", f["c"])
        M.m.set_param(p, "NBAR", M.c.bench["N"] * f["n"])
        if nlr is not None:
            M.m.set_param(p, "NLR", nlr)
        if M.capital_mobile:
            M.m.set_param(p, "KTOT", M.c.bench["K"].sum() * f["q"])
        return p

    def _accumulate(self, K: np.ndarray, IND: np.ndarray) -> np.ndarray:
        return (1 - self.M.c.par["delta"]) * K + IND

    def _solve(self, z, p, closure, migration_active, label, year, z_start=None, p_start=None):
        res = self.M.solve(z, p, closure, migration_active, z_start, p_start)
        if not res.success:
            worst = self.M.m.worst_residuals(res.z, p, 5)
            raise SolverError(f"{label}: no solution in {year} ({res.status}). Largest residuals: {worst}")
        if self.verbose:
            print(f"  {label} {year}: {res.method} {res.iterations} it, {res.seconds:.2f}s, max res {res.max_residual:.1e}")
        return res

    def _anchor(self, base, t, zs):
        """A point that solves the scenario closure at baseline parameters: the
        baseline solution of period t (valid while capital equals the baseline's;
        otherwise continuation still starts close to the solution)."""
        return base.z[t]

    # -------------------------------------------------------------- baseline
    def baseline(self, scenario=None, closure: Closure | None = None) -> PathResult:
        """Balanced-growth baseline (optionally with exogenous baseline shocks)."""
        M = self.M
        closure = closure or M.closure
        z0 = M.m.z_benchmark()
        zs, ps, stats = [], [], []
        K = M.c.bench["K"].copy()
        for t, year in enumerate(self.years):
            p = self._params(t)
            z = z0.copy() if t == 0 else _grown_vector(M, zs[-1], t - 1, t)
            M.m.set_level(z, "K", K)
            if scenario is not None:
                scenario.apply(M, t, year, p, z, M.m.levels(z), {})
            res = self._solve(z, p, closure, False, "baseline", year)
            zs.append(res.z)
            ps.append(p)
            stats.append(res)
            K = self._accumulate(M.m.get_level(res.z, "K"), M.m.get_level(res.z, "IND"))
        return PathResult(M, self.years, zs, ps, stats, closure, "baseline")

    # -------------------------------------------------------------- scenario
    def scenario(self, base: PathResult, scenario, label: str | None = None) -> PathResult:
        M = self.M
        closure = M.closure.updated(scenario.closure, M.S["G"].labels)
        if scenario.migration is not None:
            closure.migration = bool(scenario.migration)
        zs, ps, stats = [], [], []
        K = M.c.bench["K"].copy()
        nlr = np.ones(len(M.S["H"]))
        for t, year in enumerate(self.years):
            p = self._params(t, nlr)
            # exogenous values from the baseline; free values start from last solution
            z = base.z[t].copy()
            if t > 0:
                guess = _grown_vector(M, zs[-1], t - 1, t)
                fixed = M.fixed_mask(closure, migration_active=True)
                z[~fixed] = guess[~fixed]
            M.m.set_level(z, "K", K)
            scenario.apply(M, t, year, p, z, base.levels[t], base.exprs[t])
            # anchor for continuation: the baseline solution of this year (with scenario K)
            res = self._solve(z, p, closure, True, label or scenario.name, year,
                              z_start=self._anchor(base, t, zs), p_start=base.p[t])
            zs.append(res.z)
            ps.append(p)
            stats.append(res)
            K = self._accumulate(M.m.get_level(res.z, "K"), M.m.get_level(res.z, "IND"))
            nlr = M.m.get_level(res.z, "N") / base.levels[t]["N"]
        return PathResult(M, self.years, zs, ps, stats, closure, label or scenario.name)
