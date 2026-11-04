"""Validate a dataset (SAM + accounts + parameter files) before calibration.

Unlike calibration, which stops at the first problem, the validator collects
*every* issue (errors, warnings, notes) and adds plausibility tables so a
SAM can be debugged in one pass::

    from regcge.data.validate import validate_dataset
    rep = validate_dataset("my_state")
    print(rep.to_markdown())

or ``python -m regcge validate --data my_state``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .rules import GUIDE, account_roles, disallowed_nonzero, ordered_accounts, role_detail, tax_payer_issues

LEVELS = ("error", "warning", "info")


@dataclass
class Issue:
    level: str
    category: str
    message: str


@dataclass
class ValidationReport:
    path: Path
    issues: list = field(default_factory=list)
    tables: dict = field(default_factory=dict)
    model_check: list = field(default_factory=list)

    def add(self, level: str, category: str, message: str) -> None:
        self.issues.append(Issue(level, category, message))

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def issues_frame(self) -> pd.DataFrame:
        return pd.DataFrame([(i.level, i.category, i.message) for i in self.issues],
                            columns=["level", "category", "message"])

    def to_markdown(self) -> str:
        from ..results.report import to_md
        n = {lv: sum(i.level == lv for i in self.issues) for lv in LEVELS}
        status = "PASSED" if self.ok else "FAILED"
        out = [f"# Validation report: {self.path}", "",
               f"**{status}**: {n['error']} errors, {n['warning']} warnings, {n['info']} notes.", ""]
        for lv, title in (("error", "Errors"), ("warning", "Warnings"), ("info", "Notes")):
            items = [i for i in self.issues if i.level == lv]
            if items:
                out += [f"## {title}", ""]
                out += [f"- **{i.category}**: {i.message}" for i in items]
                out.append("")
        if self.model_check:
            out += ["## Model check", ""] + [f"- {m}" for m in self.model_check] + [""]
        for name, df in self.tables.items():
            out += [f"## {name}", "", to_md(df), ""]
        return "\n".join(out)

    def to_excel(self, path: str | Path) -> Path:
        path = Path(path)
        with pd.ExcelWriter(path, engine="openpyxl") as xl:
            self.issues_frame().to_excel(xl, sheet_name="Issues", index=False)
            if self.model_check:
                pd.DataFrame({"model check": self.model_check}).to_excel(xl, sheet_name="Model check", index=False)
            for name, df in self.tables.items():
                df.to_excel(xl, sheet_name=name[:31])
        return path


# ---------------------------------------------------------------------------
def _accounts_issues(rep: ValidationReport, acc: dict) -> bool:
    """Check accounts.yaml; return False if the structure is unusable."""
    ok = True
    for key in ("sectors", "factors", "households", "governments", "investment", "row"):
        if not acc.get(key):
            rep.add("error", "accounts.yaml", f"missing required key '{key}'")
            ok = False
    if not ok:
        return False
    labels = ordered_accounts(acc)
    dup = sorted({a for a in labels if labels.count(a) > 1})
    if dup:
        rep.add("error", "accounts.yaml", f"accounts declared more than once: {dup}")
        ok = False
    ftypes = list(acc["factors"].values())
    bad = sorted(set(ftypes) - {"labor", "capital", "fixed"})
    if bad:
        rep.add("error", "accounts.yaml", f"factor types must be labor, capital or fixed (got {bad})")
        ok = False
    if ftypes.count("capital") != 1:
        rep.add("error", "accounts.yaml", f"exactly one factor of type 'capital' is required (found {ftypes.count('capital')})")
        ok = False
    if "labor" not in ftypes:
        rep.add("error", "accounts.yaml", "at least one factor of type 'labor' is required")
        ok = False
    if len(acc.get("enterprises") or []) > 1:
        rep.add("error", "accounts.yaml", "at most one enterprise account is supported (aggregate them)")
    taxes = acc.get("taxes") or {}
    for k, s in taxes.items():
        b = (s or {}).get("base")
        if b not in ("sales", "output", "factor", "property", "income"):
            rep.add("error", "accounts.yaml", f"tax {k}: base must be sales, output, factor, property or income (got {b!r})")
            ok = False
            continue
        if b == "factor" and s.get("factor") not in acc["factors"]:
            rep.add("error", "accounts.yaml", f"factor tax {k} needs 'factor:' naming one of {list(acc['factors'])}")
            ok = False
        if b == "income":
            comp = s.get("components", ["labor", "capital", "transfers"])
            keys = comp.keys() if isinstance(comp, dict) else comp
            badc = [c for c in keys if c not in ("labor", "capital", "transfers")]
            if badc:
                rep.add("error", "accounts.yaml", f"income tax {k}: unknown components {badc}")
            for d in s.get("deductible", []) or []:
                if d not in taxes or taxes[d].get("base") not in ("income", "sales"):
                    rep.add("error", "accounts.yaml", f"income tax {k}: deductible tax {d} must be a declared income or sales tax")
        g = s.get("government")
        if g is not None:
            gs = [g] if isinstance(g, str) else list(g)
            for gg in gs:
                if gg not in acc["governments"]:
                    rep.add("error", "accounts.yaml", f"tax {k}: government {gg!r} is not declared")
    return ok


def validate_dataset(directory: str | Path, tol: float = 1e-6, model_check: bool = True) -> ValidationReport:
    from .dataset import Dataset, find_sam
    from .sam import read_sam

    d = Path(directory)
    rep = ValidationReport(d)
    # ---------------------------------------------------------------- accounts
    try:
        with open(d / "accounts.yaml") as fh:
            acc = yaml.safe_load(fh)
    except Exception as exc:
        rep.add("error", "accounts.yaml", f"cannot read: {exc}")
        return rep
    if not _accounts_issues(rep, acc):
        return rep
    # ---------------------------------------------------------------- SAM file
    try:
        sam_path = find_sam(d)
        S = read_sam(sam_path)
    except Exception as exc:
        rep.add("error", "SAM", f"cannot read the SAM: {exc}")
        return rep
    rep.add("info", "SAM", f"read {sam_path.name}: {len(S)} accounts, {int((S.abs() > 0).to_numpy().sum())} non-zero cells")
    if not (S.abs() > 0).to_numpy().any():
        rep.add("error", "SAM", "the SAM is empty: fill in the pale-yellow input cells of the template")
        return rep
    labels = ordered_accounts(acc)
    missing = [a for a in labels if a not in S.index]
    extra = [a for a in S.index if a not in labels]
    if missing:
        rep.add("error", "SAM", f"accounts declared in accounts.yaml but missing from the SAM: {missing}")
    if extra:
        rep.add("error", "SAM", f"SAM accounts not declared in accounts.yaml: {extra}")
    # ---------------------------------------------------------------- balance
    rows, cols = S.sum(axis=1), S.sum(axis=0)
    tot = np.maximum(rows.abs(), cols.abs())
    gap = rows - cols
    rel = gap.abs() / tot.where(tot > 0, 1.0)
    bad = rel[rel > tol].sort_values(ascending=False)
    if len(bad):
        lst = ", ".join(f"{a} ({gap[a]:+,.4g})" for a in bad.index[:12])
        rep.add("error", "balance", f"{len(bad)} accounts do not balance (row total - column total): {lst}. "
                                    f"Fix the data or run `python -m regcge balance`.")
        rep.tables["Imbalances"] = pd.DataFrame({"row total": rows, "column total": cols, "imbalance": gap,
                                                 "relative": rel}).loc[bad.index]
    if missing or extra:
        return rep
    S = S.loc[labels, labels]
    roles = account_roles(acc)
    A = list(acc["sectors"])
    H = list(acc["households"])
    G = list(acc["governments"])
    E = list(acc.get("enterprises") or [])
    taxes = acc.get("taxes") or {}
    fac = acc["factors"]
    cap = [f for f, t in fac.items() if t == "capital"][0]
    labor = [f for f, t in fac.items() if t == "labor"]
    inv, row = acc["investment"], acc["row"]
    # ---------------------------------------------------------------- cells
    cells = disallowed_nonzero(S, acc, raw=True)
    for c in cells[:50]:
        rep.add("error", "cells", f"{c.row} <- {c.col} = {c.value:,.4g}: {c.message}")
    if len(cells) > 50:
        rep.add("error", "cells", f"... and {len(cells) - 50} more cells without a model interpretation")
    for c in tax_payer_issues(S, acc):
        rep.add("error", "taxes", f"{c.row} <- {c.col} = {c.value:,.4g}: {c.message}")
    neg = []
    for col in S.columns:
        cr = role_detail(acc, col)
        cr = "tax" if cr.startswith("tax:") else cr
        for r in S.index[S[col] < 0]:
            meaning, sign, _ = GUIDE.get((role_detail(acc, r), cr), ("", ">= 0", ""))
            if sign == ">= 0":
                neg.append(f"{r} <- {col} = {S.loc[r, col]:,.4g}")
    if neg:
        rep.add("warning", "signs", f"{len(neg)} negative cells where a non-negative flow is expected: "
                                    + "; ".join(neg[:15]) + (" ..." if len(neg) > 15 else ""))
    # ---------------------------------------------------------------- parameters
    try:
        ds = Dataset.load(d, check=False)
    except Exception as exc:
        rep.add("error", "parameters", f"cannot read parameter tables: {exc}")
        ds = None
    # ---------------------------------------------------------------- economics
    QM = S.loc[row, A]
    QE = S.loc[A, row]
    QX = S[A].sum(axis=0) - QM
    for a in A:
        if QX[a] <= 0:
            rep.add("error", "sectors", f"sector {a} has zero or negative output ({QX[a]:,.4g}); remove or aggregate it")
        elif QE[a] > QX[a] * (1 + 1e-9):
            rep.add("error", "sectors", f"sector {a} exports ({QE[a]:,.4g}) exceed its output ({QX[a]:,.4g})")
    kinc = S.loc[cap, A]
    if kinc.sum() <= 0:
        rep.add("error", "capital", f"no capital income ({cap} row is empty for sectors)")
    if S[inv].sum() <= 0:
        rep.add("error", "investment", f"the {inv} column (investment demand) is empty")
    if ds is not None and not ds.sectors["inv_share"].isna().all():
        sh = ds.sectors["inv_share"].fillna(0.0)
        for a in A:
            if kinc[a] > 0 and sh[a] <= 0:
                rep.add("error", "capital", f"sector {a} earns capital income but has no investment share (sectors.inv_share)")
    for k, s in taxes.items():
        if abs(S.loc[k].sum()) < 1e-12 and not s.get("government"):
            rep.add("error", "taxes", f"tax {k} has no revenue in the SAM: declare its 'government' in accounts.yaml")
        elif abs(S.loc[k].sum()) < 1e-12:
            rep.add("info", "taxes", f"tax {k} has zero base-year revenue (it can be introduced by a scenario)")
    inc_taxes = [k for k, s in taxes.items() if s.get("base") == "income"]
    YH = S.loc[H].sum(axis=1)
    YD = YH - (S.loc[inc_taxes, H].sum(axis=0) if inc_taxes else 0.0)
    for h in H:
        if YD[h] <= 0:
            rep.add("error", "households", f"household {h} has non-positive disposable income ({YD[h]:,.4g})")
    lab_hh = S.loc[H, labor].sum().sum()
    lab_in = S.loc[labor, row].sum()
    if lab_in > lab_hh:
        rep.add("error", "labor", "residents' out-of-state earnings (ROW -> labor) exceed households' labor income")
    for g in G:
        if S.loc[A, g].sum() + S.loc[labor, g].sum().sum() <= 0:
            rep.add("warning", "governments", f"government {g} buys no goods or labor; spending_adjusts would be degenerate "
                                              f"for it (use deficit_adjusts)")
    if ds is not None:
        _structure_checks(rep, S, acc, ds)
        hh = ds.households
        if hh["households"].isna().any():
            rep.add("error", "parameters", "households.households (number of households) is missing for "
                                           f"{list(hh.index[hh['households'].isna()])}")
        if ds.sectors["jobs"].isna().any():
            rep.add("warning", "parameters", "sectors.jobs is missing for "
                                             f"{list(ds.sectors.index[ds.sectors['jobs'].isna()])}: employment will be "
                                             "reported as if $1 billion of wages = 1,000 jobs")
        if ds.governments["jobs"].isna().any():
            rep.add("warning", "parameters", "governments.jobs is missing for "
                                             f"{list(ds.governments.index[ds.governments['jobs'].isna()])}")
        if (hh["time_endowment_ratio"] < 1).any():
            rep.add("error", "parameters", "households.time_endowment_ratio must be >= 1")
        if (hh["frisch"] >= 0).any():
            rep.add("error", "parameters", "households.frisch must be negative")
        for k in inc_taxes:
            if f"mtr_{k}" not in hh.columns or hh[f"mtr_{k}"].isna().all():
                if abs(S.loc[k].sum()) > 0:
                    rep.add("info", "parameters", f"no marginal rates for {k} (households.mtr_{k}): the tax is "
                                                  "treated as proportional (marginal = average rate)")
    # ---------------------------------------------------------------- tables
    rep.tables.update(_summary_tables(S, acc, ds))
    # ---------------------------------------------------------------- model
    if model_check and rep.ok:
        _model_check(rep, d)
    return rep


def _structure_checks(rep, S, acc, ds):
    A = list(acc["sectors"])
    roles = account_roles(acc)
    for k, s in (acc.get("taxes") or {}).items():
        if s.get("base") != "sales":
            continue
        st = ds.tax_structure.get(k)
        if st is None:
            if abs(S.loc[k].sum()) > 0:
                rep.add("info", "taxes", f"no tax_structure for sales tax {k}: uniform rates across commodities")
            continue
        st = st.reindex(index=A).fillna(0.0)
        for p in S.columns[S.loc[k].abs() > 1e-12]:
            r = roles[p]
            col = {"sector": "intermediate", "household": "household", "government": "government",
                   "investment": "investment"}.get(r)
            if col is None or col not in st.columns:
                continue
            base = (st[col].to_numpy() * S.loc[A, p].to_numpy()).sum()
            if base <= 0:
                rep.add("warning", "taxes", f"{p} pays {k} but buys nothing taxable under tax_structure ({col} column); "
                                            "a uniform rate will be used for it")


def _summary_tables(S: pd.DataFrame, acc: dict, ds) -> dict:
    roles = account_roles(acc)
    A, H, G = list(acc["sectors"]), list(acc["households"]), list(acc["governments"])
    E = list(acc.get("enterprises") or [])
    fac = acc["factors"]
    taxes = acc.get("taxes") or {}
    cap = [f for f, t in fac.items() if t == "capital"][0]
    labor = [f for f, t in fac.items() if t == "labor"]
    inv, row = acc["investment"], acc["row"]
    out = {}
    # macro SAM
    def grp(a):
        r = roles[a]
        return {"sector": "Sectors", "household": "Households", "tax": "Taxes"}.get(r, a)
    order = ["Sectors"] + list(fac) + ["Households"] + E + G + ["Taxes", inv, row]
    m = S.groupby(S.index.map(grp)).sum().T.groupby(S.columns.map(grp)).sum().T
    out["Macro SAM"] = m.loc[order, order]
    # key figures
    K = list(taxes)
    prod_taxes = [k for k in K if taxes[k]["base"] in ("output", "factor", "property")]
    gsp = (S.loc[list(fac), A].to_numpy().sum() + S.loc[prod_taxes, A].to_numpy().sum()
           + S.loc[labor, G].to_numpy().sum() + S.loc[[k for k in K if taxes[k]["base"] == "factor"], G].to_numpy().sum())
    capinc = S[cap].sum()
    ow_direct = S.loc[row, cap] / capinc if capinc else 0.0
    ow_ent = 0.0
    if E:
        e = E[0]
        ow_ent = (S.loc[e, cap] / capinc) * (S.loc[row, e] / S[e].sum()) if capinc and S[e].sum() else 0.0
    key = {
        "Gross state product (value added + production taxes)": gsp,
        "Gross output": (S[A].sum(axis=0) - S.loc[row, A]).sum(),
        "Household income": S.loc[H].sum(axis=1).sum(),
        "Total tax revenue": S.loc[K].sum(axis=1).sum() if K else 0.0,
        "Investment": S[inv].sum(),
        "Rest-of-world saving (net capital inflow)": S.loc[inv, row],
        "Exports": S.loc[A, row].sum(),
        "Imports": S.loc[row, A].sum(),
    }
    kf = pd.DataFrame({"value": key})
    kf["% of GSP"] = kf["value"] / gsp * 100 if gsp else np.nan
    kf.loc["Out-of-state share of capital income (%)", "value"] = (ow_direct + ow_ent) * 100
    out["Key figures"] = kf
    # taxes
    if K:
        rev = S.loc[K].sum(axis=1)
        tg = []
        for k in K:
            col = S.loc[G, k]
            tg.append(", ".join(g for g in G if col[g] > 0) or str(taxes[k].get("government", "")))
        out["Taxes"] = pd.DataFrame({"base": [taxes[k]["base"] for k in K], "government": tg,
                                     "revenue": rev.to_numpy(), "% of GSP": (rev / gsp * 100).to_numpy() if gsp else np.nan},
                                    index=pd.Index(K, name="tax"))
    # households
    inc_t = [k for k in K if taxes[k]["base"] == "income"]
    sales_t = [k for k in K if taxes[k]["base"] == "sales"]
    YH = S.loc[H].sum(axis=1)
    dtax = S.loc[inc_t, H].sum(axis=0) if inc_t else pd.Series(0.0, index=H)
    stax = S.loc[sales_t, H].sum(axis=0) if sales_t else pd.Series(0.0, index=H)
    YD = YH - dtax
    hh = pd.DataFrame({
        "income": YH, "transfers, % of income": S.loc[H, G + [row]].sum(axis=1) / YH * 100,
        "direct taxes, % of income": dtax / YH * 100, "sales taxes, % of income": stax / YH * 100,
        "saving, % of disposable income": S.loc[inv, H] / YD * 100,
    })
    if ds is not None and not ds.households["households"].isna().all():
        scale = float(acc.get("dollar_scale", 1.0))
        hh.insert(1, "income per household ($)", YH * scale / ds.households["households"])
    hh.index.name = "household"
    out["Households"] = hh
    # sectors
    QM, QE = S.loc[row, A], S.loc[A, row]
    QX = S[A].sum(axis=0) - QM
    va = S.loc[list(fac), A].sum(axis=0)
    labinc = S.loc[labor, A].sum(axis=0)
    out["Sectors"] = pd.DataFrame({
        "output": QX, "value added, % of output": va / QX * 100,
        "labor share of factor income, %": labinc / va.where(va > 0, np.nan) * 100,
        "export share of output, %": QE / QX * 100,
        "import share of absorption, %": QM / (QX - QE + QM).where((QX - QE + QM) > 0, np.nan) * 100,
    }).rename_axis("sector")
    # governments
    out["Governments"] = pd.DataFrame({
        "receipts": S.loc[G].sum(axis=1), "tax revenue": S.loc[G, K].sum(axis=1) if K else 0.0,
        "purchases": S.loc[A, G].sum(axis=0), "employee compensation": S.loc[labor, G].sum(axis=0),
        "transfers to households": S.loc[H, G].sum(axis=0), "saving": S.loc[inv, G],
    }).rename_axis("government")
    return out


def _model_check(rep: ValidationReport, d: Path) -> None:
    import warnings as _w

    from ..model.model import RegionalCGE
    from .dataset import Dataset
    try:
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            ds = Dataset.load(d)
            M = RegionalCGE(ds)
        z, p = M.benchmark()
        res = float(np.abs(M.m.residuals(z, p)).max())
        rep.model_check.append(f"calibration succeeded: {M.m.nz} variables, {M.m.neq} equations")
        rep.model_check.append(f"benchmark replication: max scaled residual {res:.2e}"
                               + (" (OK)" if res < 1e-8 else " (TOO LARGE)"))
        if res >= 1e-8:
            rep.add("error", "model", f"benchmark not replicated (max residual {res:.2e})")
        for lab, active in (("baseline", False), ("scenario", True)):
            r = M.check_closure(migration_active=active)
            rep.model_check.append(f"closure ({lab}): {'OK' if not r else r}")
            if r:
                rep.add("error", "model", f"closure ({lab}) is not valid: {r}")
        for w in M.c.warnings:
            rep.add("warning", "calibration", w)
        P = M.c.par
        for k in M.S["KI"]:
            lowered = [h for j, h in enumerate(M.S["H"])
                       if P["mtr0"][k][j] < P["avg0"][k][j] - max(0.002, 0.02 * P["avg0"][k][j])]
            if lowered:
                rep.add("info", "parameters", f"{k}: marginal rate below the average rate for {lowered} "
                                              "(a regressive schedule, as with capped payroll taxes; check it is intended)")
    except Exception as exc:
        rep.add("error", "model", f"calibration failed: {exc}")
