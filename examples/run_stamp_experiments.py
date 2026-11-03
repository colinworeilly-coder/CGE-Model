"""Run the two Texas-STAMP experiments (and an equal-yield swap) on the
illustrative state data and write Excel/Markdown reports to results/.

    python examples/run_stamp_experiments.py
"""

from pathlib import Path

import pandas as pd

from regcge.runner import Runner

ROOT = Path(__file__).resolve().parents[1]
pd.set_option("display.width", 200)

runner = Runner(ROOT / "data" / "illustrative_state")
rows = [
    "Private employment, change (jobs)",
    "Government employment, change (jobs)",
    "Gross wage rate, % change",
    "Capital stock, % change",
    "Real disposable income per capita, % change",
    "Equivalent variation, % of disposable income",
]
for name in ("state_income_tax", "sales_tax_services", "property_to_sales_swap"):
    comp = runner.run(ROOT / "scenarios" / f"{name}.yaml")
    out = ROOT / "results"
    comp.to_excel(out / f"{name}.xlsx")
    (out / f"{name}.md").write_text(comp.to_markdown())
    print(f"\n=== {name}: {comp.scenario.description}")
    print(comp.summary().loc[rows, [2025, 2029, 2034]].round(2).to_string())
    print("\nDistribution, 2029 (EV as % of disposable income):")
    print(comp.distribution(2029)[["description", "EV, % of disposable income"]].round(2).to_string())
