# regcge: a flexible regional tax CGE model

`regcge` is a recursive-dynamic computable general equilibrium (CGE) model of
a state economy, built for tax analysis. Its starting point is
**Texas-STAMP** (State Tax Analysis Modeling Program; Tuerck, Haughton,
Barrett & Codreanu, Beacon Hill Institute, 2004), which builds on the
California DRAM model (Berck et al. 1996). From there it moves toward current
practice in applied general equilibrium modeling:

* **Households are utility-consistent:**
  * an extended linear expenditure system over goods and saving, nested in a CES with leisure;
  * labor supply calibrated to target elasticities;
  * exact **equivalent variation by income class**.
* **A generic tax layer.** Any number of instruments are declared in data, with five base types: sales (by commodity and purchaser), output, factor, property and income.
* **Income taxes** have marginal and average rates, state and local taxes are deductible from the federal tax (**federal offset**), and statutory schedules are converted to effective rates.
* **Closures are switches.** Government budget rules (spending, deficit, equal-yield tax or transfers adjust), the saving–investment closure and the numeraire are set in configuration. Closures are checked for squareness and structural singularity, with named diagnostics.
* **The baseline is an exact balanced-growth path.** Tobin's-q investment, capital accumulation, and partial-adjustment migration.
* **Reports:**
  * STAMP-style summary tables;
  * dynamic vs. static revenue;
  * distribution tables;
  * marginal excess burden and marginal cost of public funds;
  * Monte Carlo sensitivity analysis.
* **Open source, written in Python:**
  * equations are written in vectorised, GAMS-like form;
  * CasADi provides exact sparse Jacobians;
  * damped Newton is the default solver, with continuation and IPOPT as fallbacks;
  * a 10-year scenario of the 11-sector model solves in well under a second.

The full specification, including a STAMP equation-by-equation crosswalk, is
in [`docs/model.md`](docs/model.md).

## Quick start

```bash
pip install -e ".[dev]"             # numpy, pandas, scipy, casadi (bundles IPOPT), pyyaml, openpyxl
python -m regcge check              # calibrate; check benchmark replication, closures, baseline
python -m regcge run scenarios/state_income_tax.yaml scenarios/sales_tax_services.yaml
python -m regcge template --out my_state   # blank, annotated SAM workbook (starter accounts; see docs/data.md)
python -m regcge validate --data my_state  # check a filled dataset: lists every problem at once
pytest                              # 48 tests (about 20 s)
```

Results are written to `results/` as an Excel workbook, a Markdown report and
a CSV summary. In Python:

```python
from regcge.runner import Runner

r = Runner("data/illustrative_state")
comp = r.run("scenarios/sales_tax_services.yaml")
comp.summary()             # headline indicators by year (cf. STAMP Tables 4-5)
comp.revenue(2025)         # by instrument: dynamic vs static change
comp.distribution(2030)    # EV and real income by income class
comp.to_excel("results/services.xlsx")
```

## What is included

| Path | Contents |
|---|---|
| `regcge/core/` | thin modeling layer on CasADi: variables, parameters, equations, closure diagnostics, solvers |
| `regcge/data/` | SAM I/O, RAS and cross-entropy balancing, dataset loader with defaults, SAM disaggregation, SAM rulebook, template generator, validator |
| `regcge/model/calibrate.py` | benchmark extraction and calibration from the SAM |
| `regcge/model/blocks/` | equation blocks: prices, production, trade, factors, taxes, households, migration, government, investment, markets |
| `regcge/model/closures.py`, `dynamics.py` | closures; recursive baseline and scenario paths |
| `regcge/scenarios/` | YAML scenario definitions (tax rates, schedules, parameter and closure overrides, phase-ins) |
| `regcge/results/` | welfare (EV, MCF) and report tables |
| `regcge/tools/` | statutory tax schedules → effective rates; sensitivity analysis |
| `data/illustrative_state/` | **stylized** balanced SAM (11 sectors, 7 income classes, FED/STATE/LOCAL, 14 taxes) and parameter files |
| `scenarios/` | STAMP experiment 1 (state income tax), experiment 2 (sales tax on services), a property-to-sales-tax equal-yield swap |
| `examples/` | scripts reproducing the experiments, MCF tables, Monte Carlo sensitivity |
| `docs/` | model specification, data guide, closures and scenarios |

## Illustrative results

These come from the stylized data and illustrate the model's mechanics. They
are not estimates for any real state.

| 2029, change vs. baseline | State income tax (2/3/4%) | Sales tax on services | Property → sales swap |
|---|---:|---:|---:|
| Private employment (jobs) | −440,000 | −688,000 | −90,000 |
| Government employment (jobs) | +134,000 | +351,000 | −1,900 |
| Gross wage rate | +4.5% | +4.8% | +0.15% |
| Capital stock | −0.4% | −0.9% | +0.1% |
| Real disposable income per capita | −1.4% | −1.7% | −0.1% |
| Equivalent variation (% of disposable income) | −0.63% | −1.03% | −0.03% |

How these scenarios are set up:
* Under STAMP's closure, new state revenue is spent on public goods and jobs, which is why government employment rises.
* The equivalent variation does not value public goods.
* The income tax is progressive in its welfare effects: the lowest classes gain from higher wages, and the top class loses 1.3% of income.
* The services tax costs between 0.3% (lowest class) and 1.4% (highest class) of disposable income. High-income budgets are more service-intensive in the data, and the revenue funds public jobs that raise wages.

Run `python examples/run_stamp_experiments.py` for full tables, and
`python examples/welfare_and_mcf.py` for MCFs and sensitivity ranges.

## Adapting the model

| Task | How |
|---|---|
| Use your own state, sectors or households | Declare accounts in `accounts.yaml`, then `python -m regcge template --accounts accounts.yaml --out my_state` writes an annotated Excel template (allowed cells marked, live balance checks, cell guide, parameter sheets). Fill it in, then run `python -m regcge validate --data my_state`, which lists every problem at once. See [`docs/data.md`](docs/data.md) |
| Add a tax instrument | Add an account to the SAM and declare it in `accounts.yaml` with its base type; for sales taxes, add a structure file |
| Change elasticities | Edit `sectors.csv` / `households.csv`, or override per scenario with `parameters:` |
| Change closures | `model.yaml` defaults, or per scenario with `closure:`; see [`docs/closures_scenarios.md`](docs/closures_scenarios.md) |
| Long-run comparative statics | `closures: {capital: mobile}` with `periods: 1`, or run 30+ periods |
| Turn off labor-supply or migration responses | `labor_supply: exogenous`, `migration: false` |
| New behavior (unemployment, public capital, energy nests, ...) | Edit or add a block in `regcge/model/blocks/` (each declares its variables and equations); `RegionalCGE(ds, blocks=[...])` accepts a custom block list |
| New reported indicators | Add `m.add_expr(...)` in `RegionalCGE._report_expressions` |
| Scale up | Tested with a 55-sector, 14-class synthetic SAM (1,950 variables): build plus an 11-year scenario takes a few seconds |

## Caveats

* The illustrative data are stylized. Elasticities follow STAMP where an analogue exists.
* STAMP's migration elasticities (1.3 to 2.3) are high relative to recent evidence and drive much of the employment response; explore alternatives with the sensitivity tools.
* Low export-demand elasticities give the state market power. Combined with out-of-state ownership of capital, this allows part of the tax burden to be exported, and short-run MCFs can fall below one. Long-run MCFs after capital adjusts exceed one; see `docs/model.md` §6.
* Households are myopic, as in STAMP. Within-year welfare includes saving and leisure, but there is no intertemporal optimisation.
* Each income class is represented by its average household for statutory schedules, as in STAMP.

## License

MIT.
