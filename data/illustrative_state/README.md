# Illustrative state dataset

**Stylized data, not official statistics.** This dataset describes a large U.S.
state with no personal income tax: a sales tax, a franchise (margins) tax, a
severance tax, motor-fuel and other excises, and local property taxes. It is
loosely Texas-like in structure so that the Texas-STAMP experiments make
sense. Every number is an assumption, documented in `build_sam.py`.

| file | contents |
|---|---|
| `sam.csv` | balanced SAM, \$ billions, base year 2024: 11 sectors, LAB and CAP, 7 income classes, an enterprise account, FED/STATE/LOCAL, 14 tax accounts, INV, ROW |
| `accounts.yaml` | account roles and tax-instrument definitions |
| `sectors.csv` | elasticities (STAMP Table 2 analogues), depreciation, jobs |
| `households.csv` | counts, labor-supply and migration elasticities (STAMP Table 3 analogues), SALT deduction shares, marginal federal rates, AGI ratios, filer mix |
| `governments.csv` | government jobs; intergovernmental transfer rules |
| `tax_structure/*.csv` | taxable shares of the sales-type taxes by commodity and purchaser |
| `model.yaml` | growth (population 1.5%, productivity 1%), horizon 2024–2034, closures, solver |
| `build_sam.py` | regenerates `sam.csv` and `tax_structure/` from the stated assumptions |

Approximate magnitudes:

| Item | Value |
|---|---|
| GSP | \$1.95 trillion |
| Gross output | \$3.4 trillion |
| Household income | \$1.64 trillion |
| Investment | \$350 billion |
| Households | 11 million |
| Jobs | 14 million |
| State sales tax | \$44 billion |
| Local property taxes | \$73 billion |
| Federal personal income tax | \$190 billion |

To regenerate the SAM: `python data/illustrative_state/build_sam.py`.
