"""Policy scenarios defined in YAML (or dicts).

Example::

    name: sales_tax_services
    description: Extend the state sales tax to services
    start: 2025                     # first year the policy applies (default base_year + 1)
    closure:                        # optional closure overrides
      governments: {STATE: spending_adjusts}
    parameters:                     # optional overrides of parameter tables (re-calibrates)
      households: {migration_elasticity: {"*": 0.5}}
    shocks:
      - tax: ST_SALES
        purchaser: [household, intermediate]
        commodities: [FIN, BUS, PSV]
        op: set                     # set | add | scale
        value: 0.0625
        phase_in: {2025: 0.5}       # optional fraction of the change by year (default 1)
      - tax: ST_PIT
        schedule: {brackets: [[0, 0.02], [10000, 0.03], [75000, 0.04]],
                   standard_deduction: {single: 4750, head_of_household: 7000, married: 9500}}
      - param: TFP                  # TFP | PWM | PWE | EXS | PROW
        labels: [MFG]
        op: scale
        value: 1.02
      - variable: QGL               # shock to an exogenous (fixed) variable
        labels: [STATE]
        op: scale
        value: 1.05

Selections: ``commodities`` (sales taxes), ``buyers`` (buying sectors for
intermediate purchases, households for household purchases, governments for
government purchases), ``sectors`` (output, factor and property taxes),
``employers`` (governments paying factor taxes), ``households`` (income taxes).
Omitted selections mean "all".
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from ..tools.tax_schedule import effective_rates

PARAM_SHOCKS = ("TFP", "PWM", "PWE", "EXS", "PROW")


@dataclass
class Scenario:
    name: str = "scenario"
    description: str = ""
    start: int | None = None
    end: int | None = None
    closure: dict = field(default_factory=dict)
    parameters: dict = field(default_factory=dict)
    shocks: list = field(default_factory=list)
    migration: bool | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        d = copy.deepcopy(d or {})
        return cls(name=d.get("name", "scenario"), description=d.get("description", ""),
                   start=d.get("start"), end=d.get("end"), closure=d.get("closure", {}) or {},
                   parameters=d.get("parameters", {}) or {}, shocks=d.get("shocks", []) or [],
                   migration=d.get("migration"))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Scenario":
        with open(path) as fh:
            return cls.from_dict(yaml.safe_load(fh))

    # ------------------------------------------------------------ application
    def _fraction(self, shock: dict, year: int, start: int) -> float:
        s = int(shock.get("start", start))
        e = shock.get("end", self.end)
        if year < s or (e is not None and year > int(e)):
            return 0.0
        ph = shock.get("phase_in") or {}
        if ph:
            ph = {int(k): float(v) for k, v in ph.items()}
            past = [y for y in ph if y <= year]
            return ph[max(past)] if past else 0.0
        return 1.0

    def apply(self, M, t: int, year: int, p: np.ndarray, z: np.ndarray, base_levels: dict,
              base_exprs: dict) -> None:
        """Modify the parameter vector ``p`` and fixed values in ``z`` for period t."""
        start = self.start if self.start is not None else int(M.spec.get("base_year", 2024)) + 1
        for sh in self.shocks:
            frac = self._fraction(sh, year, start)
            if frac == 0.0:
                continue
            if "tax" in sh:
                if "schedule" in sh or "avg" in sh or "mtr" in sh:
                    _income_tax(M, sh, frac, t, year, p, base_levels, base_exprs)
                else:
                    _tax_rates(M, sh, frac, p)
            elif "param" in sh:
                name = sh["param"]
                if name not in PARAM_SHOCKS:
                    raise ValueError(f"param shocks must be one of {PARAM_SHOCKS}")
                sel = sh.get("labels")
                _op_param(M, p, name, sel, sh.get("op", "scale"), float(sh["value"]), frac)
            elif "variable" in sh:
                name = sh["variable"]
                pos = M.m.select(name, sh.get("labels"))
                v = M.m.vars[name]
                cur = z[v.offset + pos] * v.scale[pos]
                new = _op(cur, sh.get("op", "scale"), float(sh["value"]))
                z[v.offset + pos] = (cur + frac * (new - cur)) / v.scale[pos]
            else:
                raise ValueError(f"Unrecognised shock {sh}")


def _op(cur, op, value):
    if op == "set":
        return np.full_like(cur, value)
    if op == "add":
        return cur + value
    if op == "scale":
        return cur * value
    raise ValueError(f"op must be set, add or scale; got {op!r}")


def _op_param(M, p, name, sel, op, value, frac):
    info = M.m.params[name]
    pos = M.m.select(name, sel, kind="param")
    cur = p[info.offset + pos]
    new = _op(cur, op, value)
    p[info.offset + pos] = cur + frac * (new - cur)


def _listify(x):
    if x is None:
        return None
    return [x] if isinstance(x, str) else list(x)


def _tax_rates(M, sh, frac, p):
    k = sh["tax"]
    if k not in M.c.taxes:
        raise KeyError(f"Unknown tax {k}")
    base = M.c.taxes[k]["base"]
    op, value = sh.get("op", "scale"), float(sh["value"])
    if base == "sales":
        purch = _listify(sh.get("purchaser")) or ["household", "intermediate", "investment", "government"]
        com = _listify(sh.get("commodities"))
        buyers = _listify(sh.get("buyers"))
        for pu in purch:
            if pu == "household":
                _op_param(M, p, f"tsh_{k}", (com, buyers), op, value, frac)
            elif pu == "intermediate":
                _op_param(M, p, f"tsi_{k}", (com, buyers), op, value, frac)
            elif pu == "investment":
                _op_param(M, p, f"tsv_{k}", com, op, value, frac)
            elif pu == "government":
                _op_param(M, p, f"tsg_{k}", (com, buyers), op, value, frac)
            else:
                raise ValueError(f"Unknown purchaser {pu}")
    elif base == "output":
        _op_param(M, p, f"tout_{k}", _listify(sh.get("sectors")), op, value, frac)
    elif base == "factor":
        _op_param(M, p, f"tf_{k}", _listify(sh.get("sectors")), op, value, frac)
        if sh.get("employers", "all") is not None:
            emp = sh.get("employers", "all")
            _op_param(M, p, f"tfg_{k}", None if emp == "all" else _listify(emp), op, value, frac)
    elif base == "property":
        _op_param(M, p, f"tprop_{k}", _listify(sh.get("sectors")), op, value, frac)
    elif base == "income":
        hh = _listify(sh.get("households"))
        if op == "scale":
            _op_param(M, p, f"mtr_{k}", hh, op, value, frac)
            _op_param(M, p, f"apar_{k}", hh, op, value, frac)
        else:
            raise ValueError("Use 'schedule', 'avg'/'mtr', or op: scale for income taxes")


def _income_tax(M, sh, frac, t, year, p, base_levels, base_exprs):
    """Set marginal rates and intercepts so that, at baseline incomes of period t,
    the average rate equals the target and the marginal rate is as specified."""
    k = sh["tax"]
    if M.c.taxes[k]["base"] != "income":
        raise ValueError(f"{k} is not an income tax")
    H = M.S["H"].labels
    hh = M.ds.households
    base = np.asarray(base_exprs[f"BASE_{k}"], float)            # $ scale of the SAM
    N = np.asarray(base_levels["N"], float)
    CPI = np.asarray(base_levels["CPI"], float)
    PCI = (1 + M.c.par["a"]) ** t
    N0 = M.c.bench["N"]
    if "schedule" in sh:
        dollars = float(M.ds.accounts.get("dollar_scale", 1e9))
        infl = (1 + float(M.spec.get("inflation", 0.0))) ** t
        per_hh = base * dollars / N * infl
        avg, mtr = effective_rates(sh["schedule"], per_hh,
                                   hh.get("agi_ratio", np.ones(len(H))).to_numpy(float) if "agi_ratio" in hh else np.ones(len(H)),
                                   hh["share_married"].to_numpy(float) if "share_married" in hh else np.zeros(len(H)),
                                   hh["share_hoh"].to_numpy(float) if "share_hoh" in hh else np.zeros(len(H)))
    else:
        def vec(v):
            if isinstance(v, dict):
                return np.array([float(v.get(h, 0.0)) for h in H])
            return np.full(len(H), float(v))
        avg = vec(sh.get("avg", sh.get("mtr")))
        mtr = vec(sh.get("mtr", sh.get("avg")))
    info_m = M.m.params[f"mtr_{k}"]
    info_a = M.m.params[f"apar_{k}"]
    cur_m = p[info_m.offset:info_m.offset + info_m.n].copy()
    cur_a = p[info_a.offset:info_a.offset + info_a.n].copy()
    new_a = (mtr - avg) * base / (PCI * (N / N0) * CPI)
    p[info_m.offset:info_m.offset + info_m.n] = cur_m + frac * (mtr - cur_m)
    p[info_a.offset:info_a.offset + info_a.n] = cur_a + frac * (new_a - cur_a)
