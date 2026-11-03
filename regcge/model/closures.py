"""Closures: which variables are exogenous.

Every closure option is implemented purely by fixing or freeing variables,
so closures can be changed between runs without rebuilding the model; the
squareness/structural check in :mod:`regcge.core.system` guards against
inconsistent choices.

Government budget rules (per government)
    deficit_adjusts    real spending QGL fixed; saving SG adjusts
    spending_adjusts   saving SG fixed; real spending QGL adjusts (STAMP default
                       for state and local governments: spending follows revenue)
    tax_adjusts        QGL and SG fixed; the scale of a designated tax
                       instrument adjusts (equal-yield experiments)
    transfers_adjust   QGL and SG fixed; the scale of transfers to households adjusts

Savings-investment
    open_capital       investment follows returns; rest-of-world saving adjusts (STAMP)
    savings_driven     rest-of-world saving fixed; investment scales to saving

Numeraire
    row_price          the exchange rate (rest-of-world price level) is fixed
    cpi                the aggregate CPI is fixed; the exchange rate adjusts
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

GOV_RULES = ("deficit_adjusts", "spending_adjusts", "tax_adjusts", "transfers_adjust")


@dataclass
class Closure:
    governments: dict = field(default_factory=dict)
    tax_adjusts: dict = field(default_factory=dict)
    savings_investment: str = "open_capital"
    numeraire: str = "row_price"
    migration: bool = True

    @classmethod
    def from_spec(cls, spec: dict, govs: list[str]) -> "Closure":
        cl = spec.get("closures", {})
        g = {}
        for gv in govs:
            g[gv] = cl.get("governments", {}).get(gv, "deficit_adjusts" if gv.upper().startswith("FED") else "spending_adjusts")
        out = cls(governments=g, tax_adjusts=dict(cl.get("tax_adjusts", {}) or {}),
                  savings_investment=cl.get("savings_investment", "open_capital"),
                  numeraire=spec.get("numeraire", "row_price"),
                  migration=bool(cl.get("migration", True)))
        out.validate(govs)
        return out

    def updated(self, overrides: dict | None, govs: list[str]) -> "Closure":
        if not overrides:
            return copy.deepcopy(self)
        out = copy.deepcopy(self)
        for k, v in overrides.items():
            if k == "governments":
                out.governments.update(v)
            elif k == "tax_adjusts":
                out.tax_adjusts.update(v)
            elif hasattr(out, k):
                setattr(out, k, v)
            else:
                raise KeyError(f"Unknown closure option {k!r}")
        out.validate(govs)
        return out

    def validate(self, govs: list[str]) -> None:
        for g, r in self.governments.items():
            if g not in govs:
                raise KeyError(f"Closure refers to unknown government {g!r}")
            if r not in GOV_RULES:
                raise ValueError(f"Government {g}: rule must be one of {GOV_RULES}, got {r!r}")
            if r == "tax_adjusts" and g not in self.tax_adjusts:
                raise ValueError(f"Government {g} uses tax_adjusts but no instrument is given in tax_adjusts")
        if self.savings_investment not in ("open_capital", "savings_driven"):
            raise ValueError("savings_investment must be open_capital or savings_driven")
        if self.numeraire not in ("row_price", "cpi"):
            raise ValueError("numeraire must be row_price or cpi")


def apply_closure(M, closure: Closure, migration_active: bool) -> np.ndarray:
    """Set fixed/free flags on the model's variables; return the fixed mask."""
    m, c, S = M.m, M.c, M.S
    for v in m.vars.values():
        v.fixed[:] = False
    hasK = c.par["hasK"]
    # structural (permanent) fixes
    if M.capital_mobile:
        m.fix("K", ~hasK)
    else:
        m.fix("K")
    m.fix("RK", ~hasK)
    if len(S["FX"]):
        m.fix("RFX", c.bench["QF_X"] <= 0)
    for nm in ("TSCALE", "TRSCALE", "INVSCALE"):
        m.fix(nm)
    # numeraire
    m.fix("ER" if closure.numeraire == "row_price" else "CPIAVG")
    # savings-investment
    if closure.savings_investment == "savings_driven":
        m.free("INVSCALE")
        m.fix("SROW")
    # governments
    for g, rule in closure.governments.items():
        if rule == "deficit_adjusts":
            m.fix("QGL", g)
        elif rule == "spending_adjusts":
            m.fix("SG", g)
        elif rule == "tax_adjusts":
            m.fix("QGL", g)
            m.fix("SG", g)
            m.free("TSCALE", closure.tax_adjusts[g])
        elif rule == "transfers_adjust":
            m.fix("QGL", g)
            m.fix("SG", g)
            m.free("TRSCALE", g)
    # migration
    mig = M.migrating
    H = S["H"].labels
    for j, h in enumerate(H):
        if not mig[j]:
            m.fix("N", h)
        elif migration_active and closure.migration:
            m.fix("XBAR", h)
        else:
            m.fix("N", h)
    return m.fixed_mask()
