"""Welfare analysis: marginal cost of public funds by tax instrument and a
small Monte Carlo sensitivity analysis over key elasticities.

    python examples/welfare_and_mcf.py
"""

from pathlib import Path

import pandas as pd

from regcge import Dataset, RegionalCGE
from regcge.results.welfare import marginal_cost_of_funds
from regcge.tools.sensitivity import monte_carlo, summarize

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "illustrative_state"
pd.set_option("display.width", 200)

ds = Dataset.load(DATA)

# 1. MCF with the default (regional, finite export demand) and near price-taking trade
for label, overrides in [("default trade elasticities", None),
                         ("near price-taker", {"sectors": {"export_elasticity": 30.0, "sigma_armington": 6.0}})]:
    M = RegionalCGE(ds.with_parameters(overrides))
    rows = [marginal_cost_of_funds(M, k) for k in ("FED_PIT", "ST_SALES", "ST_FRAN", "LO_PROPB", "LO_PROPR")]
    print(f"\nMarginal cost of public funds ({label}; revenue returned lump-sum, population fixed)")
    print(pd.DataFrame(rows).set_index("tax")[["government", "MEB", "MCF"]].round(3).to_string())

# 2. Monte Carlo sensitivity of the services sales-tax experiment
dists = {
    "sectors.sigma_armington": {"dist": "uniform", "low": 0.5, "high": 2.0},
    "sectors.export_elasticity": {"dist": "uniform", "low": 0.5, "high": 2.0},
    "households.labor_supply_elasticity": {"dist": "uniform", "low": 0.25, "high": 1.5},
    "households.migration_elasticity": {"dist": "uniform", "low": 0.0, "high": 1.0},
}
df = monte_carlo(DATA, ROOT / "scenarios" / "sales_tax_services.yaml", dists, n=20, seed=1)
print("\nMonte Carlo (20 draws; multiplicative factors on the calibrated values), 2034:")
print(summarize(df).round(3).to_string())
