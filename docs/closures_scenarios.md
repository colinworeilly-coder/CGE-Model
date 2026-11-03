# Closures, scenarios and outputs

## Closures

A closure states which variables are exogenous. In `regcge` every closure
option fixes or frees variables, so a closure can change between runs without
rebuilding the model. Before solving, the system is checked:
* it must be square (free variables = equations);
* it must be structurally non-singular, checked with a maximum matching on the
  Jacobian sparsity pattern.

Errors name the variable and equation blocks involved.

Set defaults in `model.yaml` under `closures:`. Override them per scenario
under `closure:`.

| option | values | meaning |
|---|---|---|
| `governments.<G>` | `deficit_adjusts` | real spending fixed; saving adjusts (default for FED) |
| | `spending_adjusts` | saving fixed; real spending (goods and jobs) adjusts (STAMP; default for others) |
| | `tax_adjusts` | spending and saving fixed; the instrument named in `tax_adjusts.<G>` adjusts (equal yield) |
| | `transfers_adjust` | spending and saving fixed; transfers to households adjust (lump-sum recycling) |
| `savings_investment` | `open_capital` | investment follows returns; rest-of-world saving adjusts (STAMP) |
| | `savings_driven` | rest-of-world saving fixed; investment scales to available saving |
| `numeraire` (top level of model.yaml) | `row_price` / `cpi` | fix the rest-of-world price level, or the state CPI with the exchange rate free |
| `migration` | true / false | population responds to relative real income |

These are build-time options in `model.yaml` (they change equations, so a new
model is built):

| option | values |
|---|---|
| `capital` | `sector_specific` (recursive dynamics) or `mobile` (long-run static comparisons) |
| `labor_supply` | `endogenous` or `exogenous` |
| `migration_driver` | `real_income` or `utility` |

Other alternative closures can be set directly on the fixed mask. For example,
using the wage as the numeraire instead of the rest-of-world price level:

```python
fixed = model.fixed_mask()
fixed[model.m.vars["W"].offset] = True     # fix the wage ...
fixed[model.m.vars["ER"].offset] = False    # ... and free the exchange rate to keep the system square
model.solver(fixed).solve(z, p)            # check_square() runs automatically
```

A closure with involuntary unemployment (fixed real wage or a wage curve)
needs one extra variable and equation: an unemployment rate in the labor
market and a wage rule. Add both in `blocks/factors.py`, then fix or free
the wage accordingly.

## Scenarios

```yaml
name: my_reform
description: What the reform does
start: 2025                # first policy year (default: base year + 1)
end: null                  # last policy year (default: horizon)
migration: true            # optional override
closure:                   # optional closure overrides
  governments: {STATE: tax_adjusts}
  tax_adjusts: {STATE: ST_SALES}
parameters:                # optional overrides of parameter tables (triggers recalibration)
  sectors: {sigma_armington: {"*": 1.2}}        # multiply all rows
  households: {migration_elasticity: 0.5}       # set all rows
  model: {growth: {population: 0.01}}
shocks:
  - tax: ST_SALES          # sales tax: selections by purchaser, commodity, buyer
    purchaser: [household, intermediate]
    commodities: [BUS, PSV]
    buyers: null           # sectors (intermediate), households, or governments
    op: set                # set | add | scale
    value: 0.0625
    phase_in: {2025: 0.5, 2026: 1.0}
  - tax: ST_FRAN           # output, factor or property taxes: select sectors
    sectors: [MIN]
    op: add
    value: 0.01
  - tax: ST_PIT            # income tax from a statutory schedule
    schedule:
      brackets: [[0, 0.02], [10000, 0.03], [75000, 0.04]]
      standard_deduction: {single: 4750, head_of_household: 7000, married: 9500}
      threshold_scale: 1.0
  - tax: FED_PIT           # or directly with average and marginal rates by class
    avg: {H1: 0.01, H7: 0.20}
    mtr: {H1: 0.12, H7: 0.37}
  - param: TFP             # TFP | PWM | PWE | EXS | PROW
    labels: [MFG]
    op: scale
    value: 1.02
  - variable: QGL          # shock an exogenous variable (it must be fixed by the closure)
    labels: [STATE]
    op: scale
    value: 1.05
```

How shocks are applied:
* Rates are modified relative to their baseline values in each year.
* An income-tax schedule is converted each year into average and marginal
  rates for each class, at baseline incomes (`regcge.tools.tax_schedule`).
* The intercept of the linear tax function is then chosen so the average
  rate matches at baseline income, while the marginal rate governs behavior.

## Running

```bash
python -m regcge run scenarios/state_income_tax.yaml --out results/
python -m regcge check                 # calibration, closures, baseline
python -m regcge mcf --taxes ST_SALES LO_PROPB
```

```python
from regcge.runner import Runner
r = Runner("data/illustrative_state")
comp = r.run("scenarios/sales_tax_services.yaml")
comp.summary(); comp.revenue(2025); comp.distribution(2030); comp.sectors()
comp.to_excel("results/services.xlsx")
```

## Outputs (`regcge.results.report.Comparison`)

* **`summary()`**: STAMP-style indicators by year, as changes from baseline:
  * private and government jobs;
  * gross wage (nominal \$ per worker and %);
  * investment and capital stock;
  * state personal income;
  * real disposable income (total and per capita);
  * population, real GSP and CPI;
  * tax revenue by government;
  * aggregate equivalent variation.
* **`revenue(year)`**: by instrument, the baseline, scenario and dynamic change,
  together with the static change (new rates on baseline bases) and the
  dynamic feedback in %.
* **`governments(year)`**: receipts, saving and real spending by government. The federal row is the federal offset.
* **`distribution(year)`**: by income class:
  * EV per household and as % of disposable income;
  * real disposable income per household;
  * CPI and labor supply;
  * migration;
  * direct and sales taxes paid.
* **`welfare()`**: EV per household by year, plus present values.
* **`sectors(year)`**: output, prices, jobs, exports, imports, capital and investment.
* **`paths()`**: long-format baseline and scenario paths of key variables.
* **`to_excel(path)`** and **`to_markdown()`**: reports.

Inflation (`model.yaml: inflation`) is used only to express money results in
nominal dollars. The model itself is solved in real terms.

## Welfare and sensitivity tools

* `regcge.results.welfare.marginal_cost_of_funds(model, tax, recycle="transfers"|"spending", periods=2|31)` gives the short-run or long-run marginal excess burden and MCF.
* `regcge.tools.sensitivity.monte_carlo(data, scenario, dists, n)` and `grid(...)` give systematic sensitivity analysis over elasticities. Each draw is recalibrated with its own baseline.
