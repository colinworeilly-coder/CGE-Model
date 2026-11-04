"""Blank, annotated SAM templates generated from ``accounts.yaml``.

``write_template`` produces a dataset directory you fill in:

* ``template.xlsx`` (default) with sheets
    - Instructions   workflow, colour legend, units
    - SAM            square matrix; input cells pale yellow, cells the model does
                     not interpret grey; live row/column totals and imbalances
    - Cell guide     what belongs in each block, sign, typical data sources
    - Accounts       every account with its role (and tax base / government)
    - Parameters guide, sectors, households, governments
                     parameter tables pre-filled with the model defaults
    - tax_structure_<TAX>  taxable shares of each sales-type tax
* or the equivalent CSV files (``fmt='csv'``), the format of the shipped datasets;
* ``accounts.yaml`` (copied) and a default ``model.yaml``.

``Dataset.load`` reads a filled ``template.xlsx`` directly (SAM and parameter
sheets), so a workbook plus ``accounts.yaml`` is a complete dataset.  Run
``python -m regcge validate --data DIR`` to check it.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .dataset import GOVERNMENT_DEFAULTS, HOUSEHOLD_DEFAULTS, SECTOR_DEFAULTS, structure_sheet_name
from .rules import account_roles, allowed_mask, cell_guide, ordered_accounts, role_detail

PURCHASERS = ("household", "intermediate", "investment", "government")
ILLUSTRATIVE = Path(__file__).resolve().parents[2] / "data" / "illustrative_state" / "accounts.yaml"

PARAM_GUIDE = [
    ("sectors", "sigma_va", "CES elasticity between factors (STAMP SIGMA)"),
    ("sectors", "sigma_top", "Elasticity between value added and intermediates (0 = Leontief, as STAMP)"),
    ("sectors", "sigma_armington", "Elasticity between in-state and imported goods (STAMP ETAM)"),
    ("sectors", "cet_elasticity", "Transformation between in-state sales and exports; inf = same good (STAMP)"),
    ("sectors", "export_elasticity", "Absolute price elasticity of export demand (STAMP ETAE); inf = price taker"),
    ("sectors", "depreciation", "Annual depreciation rate of capital"),
    ("sectors", "inv_elasticity", "Investment response to the relative return on capital (STAMP 0.6)"),
    ("sectors", "inv_share", "Share of investment by sector of destination; blank = proportional to capital income"),
    ("sectors", "jobs", "Base-year jobs in thousands (used to report employment); recommended"),
    ("households", "description", "Label for reports, e.g. 'Under $25k'"),
    ("households", "households", "Number of households in the class (REQUIRED)"),
    ("households", "persons_per_household", "Average household size (for per-capita results)"),
    ("households", "labor_supply_elasticity", "Target uncompensated labor-supply elasticity w.r.t. the marginal net wage"),
    ("households", "time_endowment_ratio", "(labor + leisure) / labor at the benchmark (Ballard 2000)"),
    ("households", "frisch", "LES Frisch parameter (-1 with unit income elasticities = Cobb-Douglas)"),
    ("households", "migration_elasticity", "Elasticity of the number of households to relative real income (0 = none)"),
    ("households", "migration_speed", "Partial adjustment of migration per year (1 = instantaneous)"),
    ("households", "salt_deduction_share", "Share of deductible state/local taxes deducted at the margin (itemizers)"),
    ("households", "mtr_<TAX>", "Marginal rate of each income tax; blank = average rate (no progressivity)"),
    ("households", "agi_ratio", "AGI / model household income (to apply statutory schedules)"),
    ("households", "share_married", "Share of married-joint filers (standard deductions in schedules)"),
    ("households", "share_hoh", "Share of head-of-household filers"),
    ("governments", "jobs", "Base-year government employment in thousands"),
    ("governments", "intergov_mode", "Transfers this government pays to others: 'share' of its revenue or 'fixed' real"),
    ("tax_structure_<TAX>", "household ... government",
     "Relative rate (taxable share) of a sales-type tax by commodity and purchaser; scaled to SAM revenue"),
]

ROLE_COLORS = {
    "sector": "DDEBF7", "labor": "E2EFDA", "capital": "E2EFDA", "fixed": "E2EFDA",
    "household": "FCE4D6", "enterprise": "EDE7F6", "government": "FFF2CC",
    "tax": "F8CBAD", "investment": "D9E1F2", "row": "E7E6E6",
}
INPUT_FILL = "FFFBE6"
GREY_FILL = "D0D0D0"


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def load_accounts(accounts) -> tuple[dict, Path | None]:
    if accounts is None:
        accounts = ILLUSTRATIVE
    if isinstance(accounts, dict):
        return copy.deepcopy(accounts), None
    p = Path(accounts)
    with open(p) as fh:
        return yaml.safe_load(fh), p


def parameter_tables(acc: dict) -> dict[str, pd.DataFrame]:
    sec = pd.DataFrame({c: [v] * len(acc["sectors"]) for c, v in SECTOR_DEFAULTS.items()},
                       index=pd.Index(acc["sectors"], name="sector"))
    hdef = dict(HOUSEHOLD_DEFAULTS)
    hh = pd.DataFrame({"description": [""] * len(acc["households"])},
                      index=pd.Index(acc["households"], name="household"))
    for c, v in hdef.items():
        hh[c] = v
    for k, s in (acc.get("taxes") or {}).items():
        if s.get("base") == "income":
            hh[f"mtr_{k}"] = np.nan
    hh["agi_ratio"] = 1.0
    hh["share_married"] = 0.0
    hh["share_hoh"] = 0.0
    gov = pd.DataFrame({c: [v] * len(acc["governments"]) for c, v in GOVERNMENT_DEFAULTS.items()},
                       index=pd.Index(acc["governments"], name="government"))
    return {"sectors": sec, "households": hh, "governments": gov}


def tax_structure_tables(acc: dict) -> dict[str, pd.DataFrame]:
    out = {}
    for k, s in (acc.get("taxes") or {}).items():
        if s.get("base") == "sales":
            out[k] = pd.DataFrame(1.0, index=pd.Index(acc["sectors"], name="commodity"), columns=list(PURCHASERS))
    return out


def accounts_table(acc: dict) -> pd.DataFrame:
    roles = account_roles(acc)
    names = acc.get("sector_names", {}) or {}
    recs = []
    for a in ordered_accounts(acc):
        r = roles[a]
        rec = {"account": a, "role": r, "base / type": "", "government": "", "factor": "",
               "components": "", "deductible": "", "description": names.get(a, "")}
        if r in ("labor", "capital", "fixed"):
            rec["role"] = "factor"
            rec["base / type"] = r
        if r == "tax":
            s = acc["taxes"][a]
            rec["base / type"] = s.get("base", "")
            g = s.get("government", "")
            rec["government"] = g if isinstance(g, str) else ", ".join(f"{k}:{v}" for k, v in (g or {}).items())
            rec["factor"] = s.get("factor", "")
            comp = s.get("components", "")
            rec["components"] = ", ".join(comp) if isinstance(comp, list) else str(comp or "")
            rec["deductible"] = ", ".join(s.get("deductible", []) or [])
            rec["description"] = s.get("description", "")
        recs.append(rec)
    return pd.DataFrame(recs)


def default_model_yaml(acc: dict) -> str:
    govs = acc.get("governments", [])
    rules = {g: ("deficit_adjusts" if g.upper().startswith("FED") else "spending_adjusts") for g in govs}
    gov_s = ", ".join(f"{g}: {r}" for g, r in rules.items())
    return f"""# Model specification (see docs/closures_scenarios.md)
base_year: {acc.get('base_year', 2024)}
periods: 11                 # t = 0 (base year) ... 10
growth:
  population: 0.015         # households and labor endowment
  productivity: 0.010       # labor-augmenting
inflation: 0.02             # only used to report nominal dollars
discount_rate: 0.04         # present values of welfare changes
numeraire: row_price        # row_price | cpi
closures:
  governments: {{{gov_s}}}
  tax_adjusts: {{}}
  savings_investment: open_capital   # open_capital | savings_driven
  capital: sector_specific  # sector_specific | mobile
  labor_supply: endogenous  # endogenous | exogenous
  migration: true
  migration_driver: real_income      # real_income | utility
solver:
  method: auto
  tol: 1.0e-8
"""


# ---------------------------------------------------------------------------
# writers
# ---------------------------------------------------------------------------
def write_template(accounts=None, out_dir: str | Path = "my_state", fmt: str = "excel",
                   sam: pd.DataFrame | None = None, params: dict | None = None,
                   structures: dict | None = None, overwrite: bool = False) -> dict[str, Path]:
    """Write a fill-in dataset for the accounts in ``accounts`` (path or dict;
    default: the illustrative state's accounts as a starter).

    ``fmt``: 'excel' (template.xlsx), 'csv' (sam.csv + parameter CSVs) or
    'both' (Excel in out_dir, CSVs in out_dir/csv).  ``sam`` pre-fills SAM values
    (e.g. to annotate an existing SAM); ``params`` pre-fills parameter tables
    ({'sectors': df, ...}).
    """
    acc, src = load_accounts(accounts)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    labels = ordered_accounts(acc)
    if sam is not None:
        missing = [a for a in labels if a not in sam.index]
        if missing:
            raise ValueError(f"The SAM lacks accounts declared in accounts.yaml: {missing}")
        sam = sam.loc[labels, labels]
    tables = parameter_tables(acc)
    for k, v in (params or {}).items():
        if k in tables:
            t = tables[k]
            v = v.reindex(t.index)
            for c in v.columns:
                t[c] = v[c]
    st_tables = tax_structure_tables(acc)
    for k, v in (structures or {}).items():
        if k in st_tables:
            st_tables[k] = v.reindex(index=st_tables[k].index, columns=st_tables[k].columns).fillna(0.0)
    structures = st_tables

    src_text = Path(src).read_text() if src is not None else None   # read before any write

    def write_yaml_files(d: Path):
        target = d / "accounts.yaml"
        if src is None or Path(src).resolve() != target.resolve():
            if src_text is not None:
                target.write_text(src_text)
            else:
                with open(target, "w") as fh:
                    yaml.safe_dump(acc, fh, sort_keys=False)
            written[str(target)] = target
        if not (d / "model.yaml").exists():
            (d / "model.yaml").write_text(default_model_yaml(acc))
            written[str(d / "model.yaml")] = d / "model.yaml"

    if not overwrite:
        targets = []
        if fmt in ("excel", "both"):
            targets.append(out / "template.xlsx")
        if fmt == "csv":
            targets.append(out / "sam.csv")
        if fmt == "both":
            targets.append(out / "csv" / "sam.csv")
        existing = [str(t) for t in targets if t.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite {existing}; pass overwrite=True (CLI: --force)")

    write_yaml_files(out)
    if fmt in ("excel", "both"):
        p = out / "template.xlsx"
        _write_excel(p, acc, sam, tables, structures)
        written["template.xlsx"] = p
    if fmt in ("csv", "both"):
        d = out if fmt == "csv" else out / "csv"
        d.mkdir(parents=True, exist_ok=True)
        if d != out:
            write_yaml_files(d)
        S = sam if sam is not None else pd.DataFrame(0.0, index=labels, columns=labels)
        S.to_csv(d / "sam.csv", float_format="%.17g")
        for name, t in tables.items():
            t.to_csv(d / f"{name}.csv")
        (d / "tax_structure").mkdir(exist_ok=True)
        for k, t in structures.items():
            t.to_csv(d / "tax_structure" / f"{k}.csv")
        written["sam.csv"] = d / "sam.csv"
    return written


def _excel_value(v):
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return None
        if np.isinf(v):
            return "inf"
        return float(v)
    return v


def _write_table(ws, df: pd.DataFrame, index_label: str | None = None, widths: dict | None = None):
    from openpyxl.styles import Alignment, Font, PatternFill
    bold = Font(bold=True)
    head_fill = PatternFill("solid", fgColor="D9E1F2")
    cols = ([index_label or df.index.name or ""] if index_label is not False else []) + [str(c) for c in df.columns]
    ws.append(cols)
    for c in ws[1]:
        c.font = bold
        c.fill = head_fill
        c.alignment = Alignment(wrap_text=True, vertical="top")
    for idx, row in df.iterrows():
        vals = [_excel_value(v) for v in row.to_numpy()]
        ws.append(([idx] if index_label is not False else []) + vals)
    ws.freeze_panes = "B2" if index_label is not False else "A2"
    from openpyxl.utils import get_column_letter
    for j in range(1, len(cols) + 1):
        ws.column_dimensions[get_column_letter(j)].width = (widths or {}).get(j, 16)


def _write_excel(path: Path, acc: dict, sam, tables, structures):
    from openpyxl import Workbook
    from openpyxl.comments import Comment
    from openpyxl.formatting.rule import CellIsRule
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    labels = ordered_accounts(acc)
    roles = account_roles(acc)
    n = len(labels)
    mask = allowed_mask(acc, raw=True).loc[labels, labels]
    names = acc.get("sector_names", {}) or {}
    wb = Workbook()

    # ---------------- Instructions
    ws = wb.active
    ws.title = "Instructions"
    units = acc.get("units", "model units")
    lines = [
        ("regcge SAM template", True),
        (f"Accounts: {len(acc['sectors'])} sectors, {len(acc['households'])} household classes, "
         f"{len(acc['governments'])} governments, {len(acc.get('taxes') or {})} tax instruments. Units: {units} "
         f"(dollar_scale = {acc.get('dollar_scale', 1e9)}).", False),
        ("", False),
        ("How to fill in the SAM sheet", True),
        ("1. Each cell (row r, column c) is a payment RECEIVED by account r FROM account c, at basic prices.", False),
        ("2. Fill the pale-yellow cells only. Grey cells are payments the model does not interpret; leave them empty.", False),
        ("3. Sales and excise taxes are separate payments from the purchaser (column) to the tax account (row).", False),
        ("4. Imports of a commodity are paid by that sector's column to ROW; exports are paid by ROW to the sector.", False),
        ("5. Saving cells (row INV) may be negative. Taxes that do not exist yet keep an all-zero row and column.", False),
        ("6. Every account must balance: the 'Imbalance' column turns red until row total = column total.", False),
        ("   Small residual gaps can be removed with: python -m regcge balance sam.csv sam_balanced.csv", False),
        ("7. The 'Cell guide' sheet explains every block and typical data sources; 'Accounts' lists the roles.", False),
        ("", False),
        ("Parameter sheets", True),
        ("sectors / households / governments / tax_structure_<TAX> hold model parameters, pre-filled with defaults.", False),
        ("Required: households.households (number of households per class). Recommended: jobs, mtr_<TAX> columns.", False),
        ("See 'Parameters guide' for every column. Blank cells take the default.", False),
        ("", False),
        ("Workflow", True),
        ("a. Keep this workbook (template.xlsx) next to accounts.yaml and model.yaml in one directory.", False),
        ("b. python -m regcge validate --data <dir>    (lists every problem at once, plus plausibility ratios)", False),
        ("c. python -m regcge check --data <dir>       (calibration, closures, baseline)", False),
        ("d. python -m regcge run scenarios/<file>.yaml --data <dir>", False),
        ("", False),
        ("Colour legend (SAM headers)", True),
    ]
    for text, b in lines:
        ws.append([text])
        if b:
            ws.cell(ws.max_row, 1).font = Font(bold=True, size=12)
    for role, colr in ROLE_COLORS.items():
        ws.append([role])
        ws.cell(ws.max_row, 1).fill = PatternFill("solid", fgColor=colr)
    ws.append(["input cell"])
    ws.cell(ws.max_row, 1).fill = PatternFill("solid", fgColor=INPUT_FILL)
    ws.append(["not interpreted (leave empty)"])
    ws.cell(ws.max_row, 1).fill = PatternFill("solid", fgColor=GREY_FILL)
    ws.column_dimensions["A"].width = 120

    # ---------------- SAM
    ws = wb.create_sheet("SAM")
    thin = Side(style="thin", color="808080")
    inp = PatternFill("solid", fgColor=INPUT_FILL)
    grey = PatternFill("solid", fgColor=GREY_FILL)
    bold = Font(bold=True)
    ws.cell(1, 1, "receiver \\ payer").font = bold
    group_start = set()
    prev = None
    for i, a in enumerate(labels):
        r = roles[a]
        grp = "factor" if r in ("labor", "capital", "fixed") else r
        if grp != prev:
            group_start.add(i)
        prev = grp
        fill = PatternFill("solid", fgColor=ROLE_COLORS[r])
        desc = role_detail(acc, a)
        if r == "tax":
            desc += f" -> {acc['taxes'][a].get('government', '')}; {acc['taxes'][a].get('description', '')}"
        elif r == "sector":
            desc += f": {names.get(a, '')}"
        for (rr, cc) in ((1, i + 2), (i + 2, 1)):
            cell = ws.cell(rr, cc, a)
            cell.fill = fill
            cell.font = bold
            cell.comment = Comment(desc, "regcge")
        ws.cell(1, i + 2).alignment = Alignment(text_rotation=90, horizontal="center")
    last_col = get_column_letter(n + 1)
    for i, r_lab in enumerate(labels):
        for j, c_lab in enumerate(labels):
            cell = ws.cell(i + 2, j + 2)
            if mask.iloc[i, j]:
                cell.fill = inp
                if sam is not None:
                    v = float(sam.iloc[i, j])
                    cell.value = v if v != 0 else None
            else:
                cell.fill = grey
                if sam is not None and float(sam.iloc[i, j]) != 0:
                    cell.value = float(sam.iloc[i, j])   # keep data, the validator will flag it
            cell.number_format = "#,##0.00"
            cell.border = Border(left=thin if j in group_start else None, top=thin if i in group_start else None)
    tot_c, imb_c = n + 2, n + 3
    ws.cell(1, tot_c, "Row total").font = bold
    ws.cell(1, imb_c, "Imbalance (row - column)").font = bold
    ws.cell(n + 2, 1, "Column total").font = bold
    for i in range(n):
        r = i + 2
        col = get_column_letter(i + 2)
        ws.cell(r, tot_c, f"=SUM(B{r}:{last_col}{r})").number_format = "#,##0.00"
        ws.cell(r, imb_c, f"=SUM(B{r}:{last_col}{r})-SUM({col}2:{col}{n + 1})").number_format = "#,##0.00"
        ws.cell(n + 2, i + 2, f"=SUM({col}2:{col}{n + 1})").number_format = "#,##0.00"
    imb_letter = get_column_letter(imb_c)
    ws.conditional_formatting.add(f"{imb_letter}2:{imb_letter}{n + 1}",
                                  CellIsRule(operator="notBetween", formula=["-0.000001", "0.000001"],
                                             fill=PatternFill("solid", fgColor="FF9999")))
    ws.freeze_panes = "B2"
    ws.column_dimensions["A"].width = 14
    for j in range(2, n + 2):
        ws.column_dimensions[get_column_letter(j)].width = 9
    ws.column_dimensions[get_column_letter(tot_c)].width = 12
    ws.column_dimensions[imb_letter].width = 14
    ws.row_dimensions[1].height = 70

    # ---------------- Cell guide, Accounts, parameters
    g = cell_guide(acc)
    _write_table(wb.create_sheet("Cell guide"), g, index_label=False,
                 widths={1: 12, 2: 12, 3: 30, 4: 30, 5: 70, 6: 7, 7: 60})
    for row in wb["Cell guide"].iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    _write_table(wb.create_sheet("Accounts"), accounts_table(acc), index_label=False,
                 widths={1: 14, 2: 12, 3: 12, 4: 12, 5: 10, 6: 28, 7: 22, 8: 60})
    pg = pd.DataFrame(PARAM_GUIDE, columns=["sheet", "column", "meaning"])
    _write_table(wb.create_sheet("Parameters guide"), pg, index_label=False, widths={1: 22, 2: 26, 3: 100})
    for name, t in tables.items():
        _write_table(wb.create_sheet(name), t)
    req = wb["households"]
    for c in req[1]:
        if c.value == "households":
            c.fill = PatternFill("solid", fgColor="FF9999")
            c.comment = Comment("Required: number of households in each class", "regcge")
    for k, t in structures.items():
        _write_table(wb.create_sheet(structure_sheet_name(k)), t)
    wb.save(path)


def from_dataset(ds, out_dir: str | Path, fmt: str = "excel", overwrite: bool = False) -> dict[str, Path]:
    """Export an existing Dataset (SAM, parameters, tax structures) into an
    annotated template workbook and/or CSV directory."""
    accounts = ds.path / "accounts.yaml" if ds.path is not None and (Path(ds.path) / "accounts.yaml").exists() \
        else ds.accounts
    return write_template(accounts, out_dir, fmt=fmt, sam=ds.sam,
                          params={"sectors": ds.sectors, "households": ds.households, "governments": ds.governments},
                          structures=ds.tax_structure, overwrite=overwrite)
