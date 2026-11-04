"""The SAM rulebook: which cells the model interprets, and what goes in them.

One set of rules is shared by the calibration (which raises on violations),
the template generator (which marks input cells) and the validator (which
reports every violation at once), so the three can never disagree.

Two modes:

* ``raw=True`` (template, validation): also allows the alternative bookings
  that the calibration re-attributes -- factor taxes paid from a factor or
  enterprise column, income taxes paid from a factor column.
* ``raw=False`` (calibration, after :func:`canonicalize_sam`): strict rules.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TAX_BASES = ("sales", "output", "factor", "property", "income")

# Which payer columns may pay each tax base (canonical form)
TAX_PAYERS = {
    "sales": ("sector", "household", "government", "investment"),
    "output": ("sector",),
    "factor": ("sector", "government"),
    "property": ("sector",),
    "income": ("household",),
}


def account_roles(acc: dict) -> dict[str, str]:
    """Label -> role.  Factors get their type (labor/capital/fixed); taxes 'tax'."""
    roles = {}
    for a in acc.get("sectors", []):
        roles[a] = "sector"
    for f, t in (acc.get("factors") or {}).items():
        roles[f] = t
    for h in acc.get("households", []):
        roles[h] = "household"
    for e in acc.get("enterprises") or []:
        roles[e] = "enterprise"
    for g in acc.get("governments", []):
        roles[g] = "government"
    for k in (acc.get("taxes") or {}):
        roles[k] = "tax"
    if acc.get("investment"):
        roles[acc["investment"]] = "investment"
    if acc.get("row"):
        roles[acc["row"]] = "row"
    return roles


def role_detail(acc: dict, label: str) -> str:
    """Role with the tax base for tax accounts, e.g. 'tax:sales'."""
    r = account_roles(acc)[label]
    if r == "tax":
        return f"tax:{acc['taxes'][label].get('base')}"
    return r


def ordered_accounts(acc: dict) -> list[str]:
    return (list(acc.get("sectors", [])) + list((acc.get("factors") or {}).keys())
            + list(acc.get("households", [])) + list(acc.get("enterprises") or [])
            + list(acc.get("governments", [])) + list((acc.get("taxes") or {}).keys())
            + [acc["investment"], acc["row"]])


def allowed_cells(acc: dict, raw: bool = True) -> dict[str, set[str]]:
    """Column (payer) -> set of row accounts (receivers) the model interprets."""
    A = list(acc.get("sectors", []))
    fac = acc.get("factors") or {}
    F = list(fac.keys())
    H = list(acc.get("households", []))
    E = list(acc.get("enterprises") or [])
    G = list(acc.get("governments", []))
    taxes = acc.get("taxes") or {}
    inv, row = acc["investment"], acc["row"]

    def taxes_paid_by(role: str) -> set[str]:
        return {k for k, s in taxes.items() if role in TAX_PAYERS.get(s.get("base"), ())}

    out: dict[str, set[str]] = {}
    for a in A:
        out[a] = set(A) | set(F) | taxes_paid_by("sector") | {row}
    for h in H:
        out[h] = set(A) | taxes_paid_by("household") | {inv, row}
    for e in E:
        out[e] = set(H) | set(G) | {inv, row}
        if raw:  # corporate taxes booked in the enterprise column
            out[e] |= {k for k, s in taxes.items() if s.get("base") == "factor"}
    labor = [f for f, t in fac.items() if t == "labor"]
    capital = [f for f, t in fac.items() if t == "capital"]
    for g in G:  # governments hire labor and may rent capital services
        out[g] = set(A) | set(labor) | set(capital) | taxes_paid_by("government") | set(H) | set(G) | set(E) | {inv, row}
    for f in F:
        # labor income goes to resident households or non-residents; capital and
        # fixed-factor income may also go to enterprises and governments
        out[f] = (set(H) | {row}) if fac[f] == "labor" else (set(H) | set(E) | set(G) | {row})
        if raw:  # factor taxes on f and income taxes booked in the factor column
            out[f] |= {k for k, s in taxes.items()
                       if (s.get("base") == "factor" and s.get("factor") == f) or s.get("base") == "income"}
    for k in taxes:
        out[k] = set(G)
    out[inv] = set(A) | taxes_paid_by("investment")
    out[row] = set(A) | set(F) | set(H) | set(E) | set(G) | {inv}
    return out


def allowed_mask(acc: dict, raw: bool = True) -> pd.DataFrame:
    labels = ordered_accounts(acc)
    al = allowed_cells(acc, raw)
    M = pd.DataFrame(False, index=labels, columns=labels)
    for col, rows in al.items():
        M.loc[list(rows), col] = True
    return M


@dataclass
class CellIssue:
    row: str
    col: str
    value: float
    message: str


def disallowed_nonzero(sam: pd.DataFrame, acc: dict, raw: bool = True, tol: float = 1e-12) -> list[CellIssue]:
    al = allowed_cells(acc, raw)
    roles = account_roles(acc)
    issues = []
    for col in sam.columns:
        if col not in al:
            continue
        nz = sam.index[sam[col].abs() > tol]
        for r in nz:
            if r not in al[col]:
                what = _explain(acc, roles, r, col)
                issues.append(CellIssue(r, col, float(sam.loc[r, col]), what))
    return issues


def _explain(acc, roles, r, col) -> str:
    rr, cr = roles.get(r, "?"), roles.get(col, "?")
    if rr == "tax":
        base = acc["taxes"][r].get("base")
        allowed = ", ".join(TAX_PAYERS.get(base, ()))
        return f"{r} is a '{base}' tax, which can only be paid by: {allowed}; {col} is a {cr}"
    if cr == "household" and rr == "government":
        return "households pay governments only through tax accounts (declare the payment as a tax)"
    if cr == "sector" and rr in ("household", "government", "enterprise", "investment"):
        return f"sectors pay {rr}s only through factor or tax accounts"
    return f"the model has no interpretation for a payment from a {cr} to a {rr}"


def tax_payer_issues(sam: pd.DataFrame, acc: dict, tol: float = 1e-12) -> list[CellIssue]:
    """Data-dependent payer checks: a factor tax must be paid by an account that
    uses that factor."""
    issues = []
    for k, spec in (acc.get("taxes") or {}).items():
        if spec.get("base") != "factor" or k not in sam.index:
            continue
        f = spec.get("factor")
        if f not in sam.index:
            continue
        roles = account_roles(acc)
        for p in sam.columns[sam.loc[k].abs() > tol]:
            if roles.get(p) in ("sector", "government") and sam.loc[f, p] <= 0:
                issues.append(CellIssue(k, p, float(sam.loc[k, p]),
                                        f"factor tax {k} is paid by {p}, which pays no {f}"))
    return issues


# ---------------------------------------------------------------------------
# Cell guide: what goes in each block of the SAM
# ---------------------------------------------------------------------------
# (row role, column role) -> (meaning, sign, typical sources)
GUIDE = {
    ("sector", "sector"): ("Intermediate input: commodity (row) bought by sector (column), basic prices", ">= 0",
                           "IMPLAN use table; BEA input-output table regionalized (location quotients, RPCs)"),
    ("labor", "sector"): ("Wages and salaries paid by the sector, excluding employer payroll taxes", ">= 0",
                          "BEA SAGDP/SAINC compensation by industry; IMPLAN employee compensation"),
    ("capital", "sector"): ("Gross operating surplus: capital income incl. depreciation, net of the business "
                            "taxes recorded in tax rows", ">= 0", "BEA SAGDP gross operating surplus; IMPLAN"),
    ("fixed", "sector"): ("Rent to a sector-specific fixed factor (land, natural resources)", ">= 0", "USDA, BEA"),
    ("tax:sales", "sector"): ("Sales and excise taxes paid on the sector's intermediate purchases", ">= 0",
                              "State revenue department collections by industry; IMPLAN"),
    ("tax:output", "sector"): ("Gross receipts, margins or severance tax paid by the sector", ">= 0",
                               "State revenue department; Census of Governments"),
    ("tax:factor", "sector"): ("Employer payroll tax (labor) or corporate/franchise tax (capital) paid by the sector",
                               ">= 0", "IRS SOI; SSA; BEA contributions for government social insurance; IMPLAN"),
    ("tax:property", "sector"): ("Business property tax", ">= 0",
                                 "Census of Governments property tax, split residential/business with assessor data"),
    ("row", "sector"): ("Imports of the commodity from other states and abroad", ">= 0",
                        "IMPLAN regional purchase coefficients; Freight Analysis Framework; Census foreign trade"),
    ("sector", "household"): ("Household consumption of the commodity at basic prices (excluding sales taxes)", ">= 0",
                              "BLS Consumer Expenditure Survey by income class; BEA PCE by state"),
    ("tax:sales", "household"): ("Sales and excise taxes paid on the household's purchases (and residential "
                                 "property tax if it is modeled as a tax on housing services)", ">= 0",
                                 "CE Survey x statutory rates; Census of Governments; ITEP incidence tables"),
    ("tax:income", "household"): ("Income taxes and employee social insurance contributions paid by the class", ">= 0",
                                  "IRS SOI state data by AGI class; NBER TAXSIM"),
    ("investment", "household"): ("Household saving (residual; may be negative)", "any", "Residual of the household account"),
    ("row", "household"): ("Direct out-of-state purchases (travel, online, remittances)", ">= 0", "CE Survey; tourism data"),
    ("household", "labor"): ("Wages received by the class, including residents' out-of-state earnings", ">= 0",
                             "IRS SOI wages by AGI; ACS"),
    ("household", "capital"): ("Capital income received directly (proprietors' income, rent, interest)", ">= 0",
                               "IRS SOI by AGI class"),
    ("household", "fixed"): ("Fixed-factor income received by the class", ">= 0", ""),
    ("enterprise", "capital"): ("Capital income accruing to corporations", ">= 0", "BEA corporate profits; IMPLAN"),
    ("government", "capital"): ("Capital income of governments (state enterprises, land trusts)", ">= 0",
                                "Census of Governments"),
    ("row", "labor"): ("Wages earned in the state by non-residents (commuters)", ">= 0", "BEA residence adjustment"),
    ("row", "capital"): ("Capital income paid to out-of-state owners (key for tax exporting)", ">= 0",
                         "Assumption from ownership data; IMPLAN"),
    ("household", "enterprise"): ("Dividends to the class", ">= 0", "IRS SOI dividends by AGI"),
    ("government", "enterprise"): ("Non-tax payments by enterprises to governments", ">= 0", ""),
    ("investment", "enterprise"): ("Retained earnings (enterprise saving)", "any", "Residual"),
    ("row", "enterprise"): ("Dividends to out-of-state shareholders", ">= 0", "Assumption from ownership data"),
    ("sector", "government"): ("Government purchases of goods and services", ">= 0",
                               "Census of Governments current operations by function; BEA"),
    ("labor", "government"): ("Government employee compensation", ">= 0", "Census of Governments payroll; BEA"),
    ("capital", "government"): ("Government rental of capital services (often zero)", ">= 0", ""),
    ("tax:sales", "government"): ("Sales taxes paid by government on its purchases (often zero)", ">= 0", ""),
    ("tax:factor", "government"): ("Employer payroll taxes paid by government as an employer", ">= 0", "BEA"),
    ("household", "government"): ("Transfers to households (Social Security, Medicaid, assistance ...)", ">= 0",
                                  "BEA SAINC35 transfer receipts by program, allocated by class with IRS/CPS"),
    ("government", "government"): ("Intergovernmental transfers (row receives from column)", ">= 0",
                                   "Census of Governments intergovernmental revenue"),
    ("enterprise", "government"): ("Transfers and subsidies to enterprises", ">= 0", ""),
    ("investment", "government"): ("Government saving (surplus; negative = deficit)", "any", "Residual"),
    ("row", "government"): ("Government transfers to the rest of the world", ">= 0", ""),
    ("government", "tax"): ("Revenue of the tax instrument received by the government", ">= 0",
                            "Census of Governments tax revenue by type"),
    ("sector", "investment"): ("Investment demand by commodity (gross fixed capital formation)", ">= 0",
                               "BEA; IMPLAN capital formation"),
    ("tax:sales", "investment"): ("Sales taxes on investment goods", ">= 0", ""),
    ("sector", "row"): ("Exports to other states and abroad", ">= 0", "IMPLAN; FAF; Census foreign trade"),
    ("labor", "row"): ("Residents' wages earned out of state", ">= 0", "BEA residence adjustment"),
    ("capital", "row"): ("Residents' capital income from out of state", ">= 0", "Assumption; IRS SOI"),
    ("fixed", "row"): ("Fixed-factor income from out of state", ">= 0", ""),
    ("household", "row"): ("Transfers to households from outside the state", ">= 0", ""),
    ("enterprise", "row"): ("Enterprise income from outside the state", ">= 0", ""),
    ("government", "row"): ("Transfers to governments from outside the state", ">= 0", ""),
    ("investment", "row"): ("Net capital inflow (rest-of-world saving = current-account deficit)", "any", "Residual"),
    # alternative bookings re-attributed by the calibration
    ("tax:factor", "labor"): ("Payroll tax booked in the labor column (re-attributed to employers)", ">= 0", ""),
    ("tax:factor", "capital"): ("Corporate tax booked in the capital column (re-attributed to sectors)", ">= 0", ""),
    ("tax:income", "labor"): ("Social insurance booked in the labor column (re-attributed to households)", ">= 0", ""),
    ("tax:income", "capital"): ("Income tax booked in the capital column (re-attributed to households)", ">= 0", ""),
    ("tax:income", "fixed"): ("Income tax booked in a fixed-factor column (re-attributed to households)", ">= 0", ""),
    ("tax:factor", "fixed"): ("Factor tax booked in a fixed-factor column (re-attributed to sectors)", ">= 0", ""),
    ("enterprise", "fixed"): ("Fixed-factor income accruing to enterprises", ">= 0", ""),
    ("government", "fixed"): ("Fixed-factor income of governments (e.g. state lands)", ">= 0", ""),
    ("row", "fixed"): ("Fixed-factor income paid to out-of-state owners", ">= 0", ""),
    ("tax:factor", "enterprise"): ("Corporate tax booked in the enterprise column (re-attributed to sectors)", ">= 0", ""),
}


def cell_guide(acc: dict) -> pd.DataFrame:
    """One row per (row type, column type) block that the model interprets."""
    al = allowed_cells(acc, raw=True)
    blocks: dict[tuple[str, str], tuple[list, list]] = {}
    for col, rows in al.items():
        cr = role_detail(acc, col)
        cr_key = "tax" if cr.startswith("tax:") else cr
        for r in rows:
            rr = role_detail(acc, r)
            key = (rr, cr_key)
            blocks.setdefault(key, ([], []))
            if r not in blocks[key][0]:
                blocks[key][0].append(r)
            if col not in blocks[key][1]:
                blocks[key][1].append(col)
    recs = []
    order = ordered_accounts(acc)
    for (rr, cr), (rows, cols) in sorted(blocks.items(), key=lambda kv: (order.index(kv[1][1][0]), order.index(kv[1][0][0]))):
        meaning, sign, src = GUIDE.get((rr, cr), ("", ">= 0", ""))
        recs.append({"row type": rr, "column type": cr,
                     "rows (receivers)": ", ".join(sorted(rows, key=order.index)),
                     "columns (payers)": ", ".join(sorted(cols, key=order.index)),
                     "what goes here": meaning, "sign": sign, "typical sources": src})
    return pd.DataFrame(recs)
