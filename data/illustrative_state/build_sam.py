"""Build the illustrative state SAM (billions of dollars, base year 2024).

THIS IS NOT OFFICIAL DATA.  The SAM describes a stylized large U.S. state with
no personal income tax: a sales tax, a franchise (margins) tax, a severance
tax, motor-fuel and other excises, and a local property tax.  It is loosely
Texas-like in scale and structure so that the two STAMP experiments make
sense, but every number below is an assumption chosen for plausibility.  Use
it to learn and test the model; replace it with a SAM built from IMPLAN or
BEA/Census/IRS data for real analysis (see docs/data.md).

Construction
------------
1. Stylized shares (value-added ratios, input mixes, export and import shares,
   household budget shares, tax rates ...) are set below.
2. Gross output is solved from final demand with an input-output (Leontief)
   model; household income, taxes and consumption are iterated to a fixed
   point; government, enterprise and rest-of-world savings close the accounts.
3. The resulting SAM is rounded to two decimals (which unbalances it slightly)
   and re-balanced by minimum cross-entropy, exercising regcge.data.balance.

Run:  python data/illustrative_state/build_sam.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from regcge.data.balance import balance_sam_cross_entropy  # noqa: E402
from regcge.data.sam import check_balance, write_sam  # noqa: E402

S = ["AGR", "MIN", "UTL", "CON", "MFG", "TRD", "TRN", "FIN", "REA", "BUS", "PSV"]
H = ["H1", "H2", "H3", "H4", "H5", "H6", "H7"]
G = ["FED", "STATE", "LOCAL"]
TAXES = ["FED_PIT", "FED_SSC", "FED_PAYROLL", "FED_CIT", "FED_EXC",
         "ST_SALES", "ST_MFUEL", "ST_EXC", "ST_FRAN", "ST_SEV", "ST_PIT",
         "LO_PROPR", "LO_PROPB", "LO_SALES"]
ACCOUNTS = S + ["LAB", "CAP"] + H + ["ENT"] + G + TAXES + ["INV", "ROW"]
n = len(S)
ix = {s: i for i, s in enumerate(S)}

# --------------------------------------------------------------------------
# 1. Production structure
# --------------------------------------------------------------------------
# value added (market prices: factor payments + production taxes) / output
va_ratio = dict(AGR=0.40, MIN=0.55, UTL=0.55, CON=0.45, MFG=0.30, TRD=0.62,
                TRN=0.45, FIN=0.50, REA=0.70, BUS=0.58, PSV=0.60)
# labor share of factor cost (labor incl. employer payroll tax)
lab_share = dict(AGR=0.30, MIN=0.25, UTL=0.30, CON=0.70, MFG=0.50, TRD=0.60,
                 TRN=0.62, FIN=0.50, REA=0.12, BUS=0.68, PSV=0.75)
# composition of intermediate inputs (columns = buying sector, sum to 1)
mix_cols = {
    "AGR": dict(AGR=.25, MIN=.02, UTL=.04, CON=.02, MFG=.30, TRD=.10, TRN=.07, FIN=.06, REA=.06, BUS=.05, PSV=.03),
    "MIN": dict(MIN=.25, UTL=.04, CON=.06, MFG=.20, TRD=.05, TRN=.08, FIN=.06, REA=.06, BUS=.17, PSV=.03),
    "UTL": dict(MIN=.35, UTL=.10, CON=.05, MFG=.10, TRD=.03, TRN=.10, FIN=.05, REA=.04, BUS=.15, PSV=.03),
    "CON": dict(MIN=.03, UTL=.01, CON=.02, MFG=.45, TRD=.15, TRN=.05, FIN=.04, REA=.05, BUS=.17, PSV=.03),
    "MFG": dict(AGR=.06, MIN=.15, UTL=.03, CON=.01, MFG=.45, TRD=.08, TRN=.06, FIN=.03, REA=.02, BUS=.09, PSV=.02),
    "TRD": dict(UTL=.04, CON=.02, MFG=.10, TRD=.08, TRN=.10, FIN=.10, REA=.16, BUS=.30, PSV=.10),
    "TRN": dict(MIN=.02, UTL=.02, CON=.02, MFG=.25, TRD=.05, TRN=.25, FIN=.08, REA=.08, BUS=.18, PSV=.05),
    "FIN": dict(UTL=.01, CON=.01, MFG=.02, TRD=.02, TRN=.02, FIN=.40, REA=.12, BUS=.35, PSV=.05),
    "REA": dict(UTL=.05, CON=.20, MFG=.03, TRD=.03, TRN=.01, FIN=.25, REA=.15, BUS=.20, PSV=.08),
    "BUS": dict(UTL=.02, CON=.01, MFG=.05, TRD=.03, TRN=.04, FIN=.08, REA=.12, BUS=.55, PSV=.10),
    "PSV": dict(AGR=.01, UTL=.04, CON=.02, MFG=.15, TRD=.08, TRN=.03, FIN=.07, REA=.14, BUS=.30, PSV=.16),
}
mix = np.zeros((n, n))
for a, col in mix_cols.items():
    for c, v in col.items():
        mix[ix[c], ix[a]] = v
assert np.allclose(mix.sum(axis=0), 1.0)

export_share = dict(AGR=.50, MIN=.58, UTL=.08, CON=.02, MFG=.58, TRD=.12, TRN=.30, FIN=.20, REA=.03, BUS=.25, PSV=.06)
import_share = dict(AGR=.40, MIN=.20, UTL=.08, CON=.02, MFG=.58, TRD=.10, TRN=.22, FIN=.22, REA=.03, BUS=.24, PSV=.07)

# --------------------------------------------------------------------------
# 2. Taxes (statutory-like rates x taxable-share weights)
# --------------------------------------------------------------------------
PURCH = ["household", "intermediate", "investment", "government"]
sales_weights = {  # commodity -> weights by purchaser (shared by ST_SALES and LO_SALES)
    "AGR": [0.00, 0.05, 0.00, 0], "MIN": [0.00, 0.10, 0.00, 0], "UTL": [0.15, 0.30, 0.00, 0],
    "CON": [0.00, 0.05, 0.10, 0], "MFG": [0.75, 0.25, 0.60, 0], "TRD": [0.70, 0.20, 0.60, 0],
    "TRN": [0.05, 0.05, 0.10, 0], "FIN": [0.00, 0.00, 0.00, 0], "REA": [0.00, 0.00, 0.00, 0],
    "BUS": [0.25, 0.25, 0.30, 0], "PSV": [0.25, 0.05, 0.00, 0],
}
sales_like = {
    # tax: (rate, {commodity: weights by purchaser})
    "ST_SALES": (0.0625, sales_weights),
    "LO_SALES": (0.0175, sales_weights),
    "ST_MFUEL": (0.006, {"MFG": [1, 1, 0, 0]}),
    "ST_EXC": (0.02, {"MFG": [1, 0, 0, 0], "TRD": [0.5, 0, 0, 0], "PSV": [0.1, 0, 0, 0]}),
    "FED_EXC": (0.025, {"MFG": [1, 0.3, 0, 0], "TRN": [0, 0.3, 0, 0]}),
    "LO_PROPR": (0.10, {"REA": [1, 0, 0, 0]}),  # residential property tax on housing services
}


def weights(tax: str, purchaser: str) -> np.ndarray:
    _, w = sales_like[tax]
    j = PURCH.index(purchaser)
    return np.array([w.get(c, [0, 0, 0, 0])[j] for c in S], float)


def rate_vec(purchaser: str) -> dict[str, np.ndarray]:
    return {t: sales_like[t][0] * weights(t, purchaser) for t in sales_like}


R_HH = rate_vec("household")
R_INT = rate_vec("intermediate")
R_INV = rate_vec("investment")
tot_hh = sum(R_HH.values())
tot_int = sum(R_INT.values())
tot_inv = sum(R_INV.values())

out_tax = {"ST_FRAN": np.full(n, 0.0025), "ST_SEV": np.where(np.array(S) == "MIN", 0.035, 0.0)}
payroll_rate = 0.0765
cit_rate = 0.05       # FED_CIT as a share of net capital income
propb_rate = np.where(np.array(S) == "REA", 0.10, 0.045)  # business property tax / net capital income

# --------------------------------------------------------------------------
# 3. Households, governments, rest of world
# --------------------------------------------------------------------------
lab_dist = np.array([.02, .07, .11, .12, .21, .15, .32])
cap_dist = np.array([.01, .03, .05, .06, .12, .12, .61])
div_dist = np.array([.005, .02, .04, .06, .12, .13, .625])
tr_dist = {"FED": np.array([.20, .22, .17, .12, .13, .07, .09]),
           "STATE": np.array([.45, .25, .12, .07, .06, .03, .02]),
           "LOCAL": np.array([.45, .25, .12, .07, .06, .03, .02])}
fed_pit_avg = np.array([0.005, 0.025, 0.05, 0.07, 0.09, 0.12, 0.19])   # share of gross income
ssc_rate = np.array([.0765] * 6 + [0.05])
save_rate = np.array([-0.05, 0.0, 0.02, 0.04, 0.06, 0.09, 0.15])     # share of disposable income
direct_row_share = 0.025                                             # out-of-state purchases / YD
budget = {  # purchaser-value consumption shares by commodity (columns H1..H7)
    "AGR": [.02, .015, .012, .01, .008, .007, .005], "MIN": [0] * 7, "CON": [0] * 7,
    "UTL": [.05, .045, .04, .035, .03, .025, .02], "MFG": [.16, .16, .155, .15, .145, .14, .13],
    "TRD": [.17, .17, .165, .16, .155, .15, .14], "TRN": [.04] * 7,
    "FIN": [.05, .055, .06, .07, .08, .09, .11], "REA": [.26, .25, .245, .24, .235, .23, .22],
    "BUS": [.06, .065, .07, .075, .08, .085, .09], "PSV": [.19, .20, .213, .22, .227, .233, .245],
}
B = np.array([budget[c] for c in S])
assert np.allclose(B.sum(axis=0), 1.0)

gov_purch = {"FED": 70.0, "STATE": 38.0, "LOCAL": 32.0}
gov_comp = {"FED": 40.0, "STATE": 38.0, "LOCAL": 90.0}
gov_mix = {
    "FED": dict(MFG=.35, BUS=.30, CON=.10, PSV=.10, TRN=.05, UTL=.03, REA=.02, FIN=.02, TRD=.03),
    "STATE": dict(PSV=.55, CON=.15, BUS=.15, MFG=.06, UTL=.03, TRN=.03, REA=.03),
    "LOCAL": dict(CON=.25, BUS=.20, PSV=.20, MFG=.12, UTL=.10, TRN=.05, REA=.05, TRD=.03),
}
gov_transfers_hh = {"FED": 190.0, "STATE": 12.0, "LOCAL": 4.0}
grants = {("STATE", "FED"): 55.0, ("LOCAL", "FED"): 8.0, ("LOCAL", "STATE"): 35.0}  # (to, from)
cap_owner = {"H": 0.33, "ENT": 0.52, "ROW": 0.12, "STATE": 0.01, "LOCAL": 0.02}
ent_out = {"H": 0.40, "ROW": 0.25}  # rest retained (enterprise saving)
inv_mix = dict(CON=.45, MFG=.30, BUS=.15, TRD=.07, TRN=.03)
inv_gdp = 0.18
nonres_labor = 0.01      # share of in-state wage bill earned by non-residents
res_out_labor = 0.008    # residents' out-of-state earnings / in-state wage bill
cap_inflow = 0.03        # residents' out-of-state capital income / in-state capital income
row_transfers_hh = 8.0

# --------------------------------------------------------------------------
# 4. Fixed point: consumption -> output -> income -> consumption
# --------------------------------------------------------------------------
va = np.array([va_ratio[s] for s in S])
es = np.array([export_share[s] for s in S])
ms = np.array([import_share[s] for s in S])
lsh = np.array([lab_share[s] for s in S])
A = mix * (1 - va)[None, :] / (1 + tot_int)[:, None]     # basic-value input coefficients
gvec = {g: np.array([gov_mix[g].get(s, 0.0) for s in S]) * gov_purch[g] for g in G}
G_fd = sum(gvec.values())
inv_comp = np.array([inv_mix.get(s, 0.0) for s in S])

C_basic = np.full(n, 100.0)
I_purch = 400.0
for it in range(200):
    I_basic = inv_comp * I_purch / (1 + tot_inv)
    FD = C_basic + G_fd + I_basic
    X = np.linalg.solve(np.diag(1 - es) - np.diag(1 - ms) @ A, (1 - ms) * FD)
    Z = A * X[None, :]
    VA = va * X
    otax = {k: v * X for k, v in out_tax.items()}
    FC = VA - sum(otax.values())
    LABC = lsh * FC                      # labor cost incl. payroll tax
    LAB = LABC / (1 + payroll_rate)
    CAPC = FC - LABC
    CAPI = CAPC / (1 + cit_rate + propb_rate)  # net capital income
    wage_bill = LAB.sum() + sum(gov_comp.values())
    cap_total = CAPI.sum() * (1 + cap_inflow)
    ent_inc = cap_owner["ENT"] * cap_total
    YL = lab_dist * wage_bill * (1 - nonres_labor + res_out_labor)
    YK = cap_dist * cap_owner["H"] * cap_total + div_dist * ent_out["H"] * ent_inc
    TR = sum(tr_dist[g] * gov_transfers_hh[g] for g in G) + row_transfers_hh * lab_dist
    YH = YL + YK + TR
    PIT = fed_pit_avg * YH
    SSC = ssc_rate * YL
    YD = YH - PIT - SSC
    SAV = save_rate * YD
    DROW = direct_row_share * YD
    CONS = YD - SAV - DROW                       # purchaser value
    Cp = B * CONS[None, :]                        # purchaser value by commodity
    Cb = Cp / (1 + tot_hh)[:, None]               # basic value
    C_new = Cb.sum(axis=1)
    gsp = VA.sum() + sum(gov_comp.values()) * (1 + payroll_rate)
    I_new = inv_gdp * gsp
    if np.max(np.abs(C_new - C_basic)) < 1e-10 and abs(I_new - I_purch) < 1e-10:
        break
    C_basic, I_purch = C_new, I_new
else:
    raise RuntimeError("Fixed point did not converge")

E = es * X
Q = Z.sum(axis=1) + FD
M = ms * Q
assert np.allclose(X, Q - M + E)

# --------------------------------------------------------------------------
# 5. Fill the SAM
# --------------------------------------------------------------------------
sam = pd.DataFrame(0.0, index=ACCOUNTS, columns=ACCOUNTS)
sam.loc[S, S] = Z
for j, a in enumerate(S):
    for t in sales_like:
        sam.loc[t, a] += (R_INT[t] * Z[:, j]).sum()
    for t in out_tax:
        sam.loc[t, a] += otax[t][j]
    sam.loc["LAB", a] = LAB[j]
    sam.loc["FED_PAYROLL", a] = LAB[j] * payroll_rate
    sam.loc["CAP", a] = CAPI[j]
    sam.loc["FED_CIT", a] = CAPI[j] * cit_rate
    sam.loc["LO_PROPB", a] = CAPI[j] * propb_rate[j]
    sam.loc["ROW", a] = M[j]
# households
for k, h in enumerate(H):
    sam.loc[S, h] = Cb[:, k]
    for t in sales_like:
        sam.loc[t, h] = (R_HH[t] * Cb[:, k]).sum()
    sam.loc["FED_PIT", h] = PIT[k]
    sam.loc["FED_SSC", h] = SSC[k]
    sam.loc["INV", h] = SAV[k]
    sam.loc["ROW", h] = DROW[k]
    sam.loc[h, "LAB"] = lab_dist[k] * wage_bill * (1 - nonres_labor + res_out_labor)
    sam.loc[h, "CAP"] = cap_dist[k] * cap_owner["H"] * cap_total
    sam.loc[h, "ENT"] = div_dist[k] * ent_out["H"] * ent_inc
    for g in G:
        sam.loc[h, g] = tr_dist[g][k] * gov_transfers_hh[g]
    sam.loc[h, "ROW"] = row_transfers_hh * lab_dist[k]
# factors
for g in G:
    sam.loc["LAB", g] = gov_comp[g]
    sam.loc["FED_PAYROLL", g] = gov_comp[g] * payroll_rate
sam.loc["ROW", "LAB"] = nonres_labor * wage_bill
sam.loc["LAB", "ROW"] = res_out_labor * wage_bill
sam.loc["CAP", "ROW"] = CAPI.sum() * cap_inflow
sam.loc["ENT", "CAP"] = cap_owner["ENT"] * cap_total
sam.loc["ROW", "CAP"] = cap_owner["ROW"] * cap_total
sam.loc["STATE", "CAP"] = cap_owner["STATE"] * cap_total
sam.loc["LOCAL", "CAP"] = cap_owner["LOCAL"] * cap_total
# enterprise
sam.loc["ROW", "ENT"] = ent_out["ROW"] * ent_inc
sam.loc["INV", "ENT"] = (1 - ent_out["H"] - ent_out["ROW"]) * ent_inc
# government spending
for g in G:
    sam.loc[S, g] = gvec[g]
for (to, frm), v in grants.items():
    sam.loc[to, frm] = v
# investment column
I_b = inv_comp * I_purch / (1 + tot_inv)
sam.loc[S, "INV"] = I_b
for t in sales_like:
    sam.loc[t, "INV"] = (R_INV[t] * I_b).sum()
# exports
sam.loc[S, "ROW"] = E
# tax accounts -> governments
gov_of = {t: ("FED" if t.startswith("FED") else "STATE" if t.startswith("ST") else "LOCAL") for t in TAXES}
for t in TAXES:
    sam.loc[gov_of[t], t] = sam.loc[t].sum()
# government saving = receipts - payments (excluding the INV cell)
for g in G:
    sam.loc["INV", g] = sam.loc[g].sum() - sam[g].sum()
# rest-of-world saving closes the investment account
sam.loc["INV", "ROW"] = sam[["INV"]].sum().iloc[0] - sam.loc["INV"].sum()

gap = (sam.sum(axis=1) - sam.sum(axis=0)).abs().max()
print(f"Constructed SAM: max imbalance before rounding {gap:.2e}")

# --------------------------------------------------------------------------
# 6. Round and re-balance with cross-entropy
# --------------------------------------------------------------------------
prior = sam.round(2)
print(f"After rounding: max imbalance {(prior.sum(axis=1) - prior.sum(axis=0)).abs().max():.3f}")
balanced = balance_sam_cross_entropy(prior)
check_balance(balanced, rel_tol=1e-9)
write_sam(balanced, HERE / "sam.csv")

# tax-structure files
(HERE / "tax_structure").mkdir(exist_ok=True)
for t, (rate, w) in sales_like.items():
    df = pd.DataFrame([w.get(c, [0, 0, 0, 0]) for c in S], index=S, columns=PURCH)
    df.index.name = "commodity"
    with open(HERE / "tax_structure" / f"{t}.csv", "w") as fh:
        fh.write(f"# Relative rate structure (taxable shares) of {t}; illustrative statutory rate {rate}\n")
        df.to_csv(fh)

print(f"Gross output {X.sum():,.0f}  |  GSP (approx.) {gsp:,.0f}  |  "
      f"household income {balanced.loc[H].to_numpy().sum():,.0f}  |  investment {I_purch:,.0f}")
print("Wrote", HERE / "sam.csv")
