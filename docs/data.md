# Data: SAM conventions and parameter files

A dataset is a directory. Only the SAM and `accounts.yaml` are required; every
other file has documented defaults.

```
my_state/
  accounts.yaml               account roles and tax instruments          (required)
  sam.csv | sam.xlsx | template.xlsx
                              Social Accounting Matrix                   (required)
  model.yaml                  growth, closures, horizon, solver
  sectors.csv                 per-sector elasticities, depreciation, jobs
  households.csv              per-class counts and behavioral parameters
  governments.csv             government jobs, intergovernmental transfer rules
  tax_structure/<TAX>.csv     relative rate structure of each sales-type tax
  income_elasticities.csv     LES income elasticities (optional)
  capital_coefficients.csv    investment-goods composition by destination (optional)
```

If several SAM files are present, `sam.csv` takes precedence over `sam.xlsx`,
which takes precedence over `template.xlsx`. A parameter table is read from
its CSV file if it exists, otherwise from the sheet of the same name in the
workbook.

## 0. Starting from the template

The quickest route to a usable dataset:

1. **Declare your accounts.** Copy `data/illustrative_state/accounts.yaml` and edit the lists of sectors, factors, household classes, governments and tax instruments (section 2 below).
2. **Generate the template:**

   ```bash
   python -m regcge template --accounts my_state/accounts.yaml --out my_state
   ```

   This writes `my_state/template.xlsx` and a default `model.yaml` (and copies `accounts.yaml` if it lives elsewhere). Use `--format csv` for CSV files, or `--format both`, which puts the CSVs in `my_state/csv/`. An existing workbook is never overwritten unless you add `--force`, so after changing `accounts.yaml`, generate into a new directory and copy your numbers across.
3. **Fill in the workbook:**

   | Sheet | What to do |
   |---|---|
   | **SAM** | Enter flows in the pale-yellow cells only. Grey cells are payments the model does not interpret. Header colours show account types; hover over a header for its role. The *Imbalance* column turns red until an account's row and column totals match. |
   | **Cell guide** | Explains every block: what goes there, the expected sign, typical data sources. |
   | **Accounts** | Lists every account's role, tax base and receiving government. |
   | **sectors**, **households**, **governments**, **tax_structure_&lt;TAX&gt;** | Pre-filled with defaults. The **number of households per class is required**; jobs and marginal income-tax rates (`mtr_<TAX>`) are recommended. **Parameters guide** describes every column. |
4. **Validate:**

   ```bash
   python -m regcge validate --data my_state --out my_state/validation.md
   ```

   The validator lists *every* problem at once, which calibration does not:
   * structure and labels, and account balance (with the largest gaps);
   * cells the model cannot interpret, and taxes paid by the wrong kind of account;
   * sectors with no output or with exports above output;
   * missing investment, non-positive disposable income, zero-revenue taxes without a receiving government;
   * missing household counts and other parameter problems.

   It adds plausibility tables: a macro SAM; GSP and its components; taxes as % of GSP; household tax and saving rates; sector trade and labor shares; government budgets; and the out-of-state share of capital income. If there are no errors, it also calibrates the model and checks benchmark replication and closures. The command exits with code 1 when errors are found.
5. **Fix small imbalances** with `python -m regcge balance` (cross-entropy). Then run `python -m regcge check --data my_state` and your scenarios with `--data my_state`.

To annotate an existing dataset, use `python -m regcge template --from-data data/illustrative_state --out annotated`. The SAM, parameter tables and tax structures are pre-filled, which makes a worked example of a complete template. `--from-sam file.csv` pre-fills only the SAM.

The allowed cells come from one rulebook (`regcge/data/rules.py`), shared by
the template, the validator and the calibration, so they cannot disagree.

Common validation messages:

| Message | Usual cause and fix |
|---|---|
| `X <- H1: households pay governments only through tax accounts` | A fee or tax was booked directly to a government. Route it through a tax account (add one to `accounts.yaml` if needed). |
| `FED_PIT is a 'income' tax, which can only be paid by: household` | An income-tax payment sits in a sector column. Move it to the household columns, or redeclare the tax with the right base. |
| `accounts do not balance` | Row ≠ column totals. Fix data errors, then remove residual gaps by cross-entropy. |
| `has no revenue in the SAM: declare its 'government'` | A tax with zero base-year revenue needs `government:` in `accounts.yaml`. |
| `exports exceed its output` | Re-exports or a valuation mismatch. Exports must come from in-state production. |
| `earns capital income but has no investment share` | Leave `sectors.inv_share` blank (proportional to capital income) or give the sector a positive share. |

## 1. The SAM

The SAM is square, with identical row and column labels. Cell (r, c) is a
payment received by row account r from column account c. Every account's row
total must equal its column total. Values are in one unit throughout (for
example \$ billions); declare that unit's size in dollars as `dollar_scale`.

Account types and the cells the model interprets:

| Column (payer) → | pays these rows |
|---|---|
| sector *a* | intermediate inputs (sectors), factors, taxes (sales on its intermediate purchases, output, factor, property), imports of commodity *a* (ROW) |
| factor *f* | households, enterprise, governments, ROW (non-residents). Optionally factor or income taxes (re-attributed, see below) |
| household *h* | consumption (sectors), taxes (sales on its purchases, income), saving (INV), direct out-of-state purchases (ROW) |
| enterprise | dividends to households, governments and ROW; retained earnings (INV) |
| government *g* | purchases (sectors), employment (labor), capital rentals, sales and payroll taxes, transfers (households, governments, enterprise, ROW), saving (INV) |
| tax account *k* | governments that receive the revenue |
| INV | investment demand (sectors), sales taxes on investment goods |
| ROW | exports (sectors), residents' out-of-state factor income, transfers, ROW saving (INV) |

Sectors are combined activity–commodity accounts. Flows are valued at basic
(producer) prices, and sales taxes are separate payments by the purchaser to a
tax account. Cells without an interpretation (for example a household paying a
government directly) are reported as errors with their names.

**Re-attribution.** Some SAMs book factor taxes or income taxes in other
columns. The calibration re-attributes them while keeping the SAM balanced:
* factor taxes paid from a factor column or from the enterprise column (e.g. `CAP → FED_CIT`) are moved to the employing sectors and governments, in proportion to their factor payments;
* income taxes paid from a factor column (e.g. `LAB → FED_SSC`) are moved to households, in proportion to their factor income.

**Balancing.** `regcge.data.balance` provides two methods: RAS for
non-negative SAMs, and minimum cross-entropy, which keeps signs and zeros and
allows fixed cells and known totals:

```python
from regcge.data.sam import read_sam, write_sam
from regcge.data.balance import balance_sam_cross_entropy
sam = read_sam("raw_sam.csv")
write_sam(balance_sam_cross_entropy(sam), "sam.csv")
```

The same is available from the command line: `python -m regcge balance raw.csv sam.csv`.
`regcge.data.sam.aggregate(sam, mapping)` aggregates accounts, and
`regcge.data.synthetic.split_accounts` disaggregates them.

## 2. `accounts.yaml`

```yaml
units: billion USD
dollar_scale: 1.0e9
base_year: 2024
sectors: [AGR, MIN, ...]
sector_names: {AGR: Agriculture, ...}     # optional, used in reports
factors: {LAB: labor, CAP: capital}       # types: labor (one or more), capital (exactly one), fixed
households: [H1, ..., H7]
enterprises: [ENT]                        # optional, at most one
governments: [FED, STATE, LOCAL]
investment: INV
row: ROW
taxes:
  ST_SALES: {base: sales, government: STATE}
  ST_FRAN:  {base: output, government: STATE}
  FED_PAYROLL: {base: factor, factor: LAB, government: FED}
  LO_PROPB: {base: property, government: LOCAL}
  FED_PIT:  {base: income, government: FED, components: [labor, capital, transfers],
             deductible: [ST_PIT, LO_PROPR]}
  ST_PIT:   {base: income, government: STATE, components: [labor, capital]}
```

* **Revenue shares** across governments come from the tax account's SAM column. `government` is only needed for instruments with zero base-year revenue, such as a state PIT that a scenario will introduce; it can be a dict of shares.
* **Income taxes:** `components` lists which income enters the base. A dict of weights is also allowed, for example to tax half of transfers. `deductible` lists income or sales taxes that are subtracted from the base by itemizers, at the household's `salt_deduction_share`.

## 3. Parameter files

### `sectors.csv` (index: sector)

| column | default | meaning |
|---|---|---|
| `sigma_va` | 0.9 | CES between factors (STAMP SIGMA) |
| `sigma_top` | 0 | value added vs intermediates (0 = Leontief) |
| `sigma_armington` | 1.5 | domestic vs imported (STAMP ETAM) |
| `cet_elasticity` | inf | domestic vs export transformation; `inf` = same good (STAMP) |
| `export_elasticity` | 1.65 | absolute price elasticity of export demand (STAMP ETAE); `inf` = price taker |
| `depreciation` | 0.06 | annual rate |
| `inv_elasticity` | 0.6 | investment response to relative return |
| `inv_share` | (blank) | share of investment by destination; blank = proportional to capital income |
| `jobs` | (blank) | base-year jobs in thousands, used to report employment |

### `households.csv` (index: household)

| column | default | meaning |
|---|---|---|
| `households` | required | number of households |
| `persons_per_household` | 2.5 | used for per-capita results |
| `labor_supply_elasticity` | 0.2 | target uncompensated elasticity w.r.t. the marginal net wage |
| `time_endowment_ratio` | 1.75 | (labor + leisure) / labor |
| `frisch` | -1 | LES Frisch parameter (-1 with unit income elasticities = Cobb-Douglas) |
| `migration_elasticity` | 0 | elasticity of households to relative real income |
| `migration_speed` | 1 | partial-adjustment speed (1 = instantaneous) |
| `salt_deduction_share` | 0 | share of deductible taxes deducted at the margin |
| `mtr_<TAX>` | avg | marginal rate of an income tax; must exceed the average rate for progressivity |
| `agi_ratio` | 1 | AGI / model income, used to apply statutory schedules |
| `share_married`, `share_hoh` | 0 | filer mix, used for standard deductions in schedules |

A note on the illustrative migration elasticities: they follow STAMP's ETAYD
(1.3 to 2.3) and are large relative to recent evidence on tax-induced
migration, for example Moretti & Wilson (2017) for star scientists, and Young
& Varner (2011) and Young et al. (2016) for millionaires. They strongly affect
employment and welfare results; use the `parameters:` key of a scenario, or
the sensitivity tools, to explore alternatives.

### `governments.csv` (index: government)

| column | default | meaning |
|---|---|---|
| `jobs` | (blank) | base-year employment in thousands |
| `intergov_mode` | share | transfers this government pays to others: `share` of its revenue (STAMP) or `fixed` real amounts |

### `tax_structure/<TAX>.csv` (index: commodity)

Columns are `household`, `intermediate`, `investment` and `government`. The
entries are relative rates (taxable shares) of a sales-type tax by commodity
and purchaser. The calibration scales them for every paying account so that
the SAM revenue is reproduced. With the default structure file, the scale is
approximately statutory rate × compliance. If a payer pays the tax but has no
taxable purchases under the structure, a uniform rate is used and a warning is
issued.

### `income_elasticities.csv` (optional)

Rows are the commodities, plus `ROW` (direct out-of-state purchases) and
`SAV`. Columns are household labels, or a single `all` column. Together with
the Frisch parameters, these determine the LES subsistence quantities.

### `capital_coefficients.csv` (optional)

Rows are source commodities and columns are destination sectors, giving the
composition of investment goods (STAMP used the BEA capital-flow table). The
table is RAS-adjusted to the SAM's investment column.

## 4. Building a dataset for a real state

The model only needs a balanced SAM with the account structure above. Two
routes are common.

**IMPLAN.** Export the state SAM, aggregate industries to the sector list, and
map the institutions:
* households by income class;
* federal, state and local government accounts;
* IMPLAN's tax accounts to tax instruments (indirect business taxes become `output` or `sales` instruments; employee and employer social insurance become `income` and `factor` instruments; corporate profits tax becomes `factor` on CAP).

Then re-balance by cross-entropy.

**Public data.**

| Source | Provides |
|---|---|
| BEA state GDP by industry and state personal income | value added, compensation, transfers |
| BEA national input-output tables, regionalised (e.g. location quotients) | intermediate flows |
| Census Annual Survey of State and Local Government Finances | revenue by tax, spending by function |
| IRS SOI state data | income and federal income tax by AGI class |
| BLS Consumer Expenditure Survey | budget shares by income class |
| BEA capital-flow and fixed-asset tables | investment composition and depreciation |

STAMP was built this way. Assemble an unbalanced SAM and balance it with
cross-entropy, fixing the cells you trust most (e.g. tax revenues).

The illustrative SAM in `data/illustrative_state/` is stylized. Its
`build_sam.py` shows the steps on assumed shares.
