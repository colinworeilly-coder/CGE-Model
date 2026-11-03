"""Systematic sensitivity analysis over parameters (elasticities etc.).

Results of CGE tax models hinge on elasticities that are imprecisely known
(STAMP itself stressed this).  :func:`monte_carlo` draws parameter values,
re-calibrates, re-runs baseline and scenario for each draw, and collects
headline indicators; :func:`grid` evaluates an explicit list of overrides.

Distributions are specified per ``table.column``, applied multiplicatively to
every row by default (so relative differences across sectors or households
are preserved)::

    dists = {
        "sectors.sigma_armington": {"dist": "uniform", "low": 0.5, "high": 1.5},
        "sectors.export_elasticity": {"dist": "triangular", "low": 0.5, "mode": 1, "high": 2},
        "households.labor_supply_elasticity": {"dist": "lognormal", "sigma": 0.3},
        "households.migration_elasticity": {"dist": "uniform", "low": 0.0, "high": 1.0},
    }
    df = monte_carlo("data/illustrative_state", "scenarios/sales_tax_services.yaml", dists, n=50)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..runner import Runner, _as_scenario

DEFAULT_INDICATORS = (
    "Total employment, % change",
    "Real disposable income, % change",
    "Real GSP, % change",
    "Equivalent variation, % of disposable income",
)


def _draw(spec: dict, rng) -> float:
    d = spec.get("dist", "uniform")
    if d == "uniform":
        return rng.uniform(spec["low"], spec["high"])
    if d == "triangular":
        return rng.triangular(spec["low"], spec.get("mode", 1.0), spec["high"])
    if d == "lognormal":
        return float(np.exp(rng.normal(spec.get("mu", 0.0), spec.get("sigma", 0.2))))
    if d == "normal":
        return rng.normal(spec.get("mean", 1.0), spec.get("sd", 0.1))
    raise ValueError(f"Unknown distribution {d}")


def _overrides(values: dict, dists: dict) -> dict:
    out: dict = {}
    for key, val in values.items():
        table, col = key.split(".", 1)
        mode = dists.get(key, {}).get("mode_apply", "scale") if dists else "scale"
        out.setdefault(table, {})[col] = {"*": val} if mode == "scale" else val
    return out


def grid(data, scenario, overrides: list[dict], indicators=DEFAULT_INDICATORS, year: int | None = None,
         runner: Runner | None = None) -> pd.DataFrame:
    """Run the scenario once per parameter override dict."""
    runner = runner or Runner(data)
    sc = _as_scenario(scenario)
    rows = []
    for i, ov in enumerate(overrides):
        sc_i = _as_scenario({**sc.__dict__, "parameters": ov})
        comp = runner.run(sc_i)
        sm = comp.summary()
        y = year or comp.years[-1]
        rows.append({"run": i, **{k: sm.loc[k, y] for k in indicators}, "overrides": repr(ov)})
    return pd.DataFrame(rows).set_index("run")


def monte_carlo(data, scenario, dists: dict, n: int = 50, seed: int = 0, indicators=DEFAULT_INDICATORS,
                year: int | None = None) -> pd.DataFrame:
    """Monte Carlo sensitivity analysis; returns one row per draw (failed draws
    are recorded with NaN indicators)."""
    rng = np.random.default_rng(seed)
    runner = Runner(data)
    sc = _as_scenario(scenario)
    rows = []
    for i in range(n):
        vals = {k: _draw(spec, rng) for k, spec in dists.items()}
        ov = _overrides(vals, dists)
        try:
            comp = Runner(runner.ds).run(_as_scenario({**sc.__dict__, "parameters": ov}))
            sm = comp.summary()
            y = year or comp.years[-1]
            res = {k: sm.loc[k, y] for k in indicators}
        except Exception as exc:  # keep going; report the failure
            res = {k: np.nan for k in indicators}
            res["error"] = str(exc)[:200]
        rows.append({"draw": i, **vals, **res})
    return pd.DataFrame(rows).set_index("draw")


def summarize(df: pd.DataFrame, indicators=DEFAULT_INDICATORS) -> pd.DataFrame:
    cols = [c for c in indicators if c in df.columns]
    return df[cols].describe(percentiles=[0.05, 0.5, 0.95]).T
