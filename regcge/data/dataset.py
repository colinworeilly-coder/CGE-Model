"""The Dataset: a SAM plus optional parameter files, with documented defaults.

Directory layout (only ``sam.csv`` and ``accounts.yaml`` are required)::

    accounts.yaml               account roles and tax-instrument definitions
    sam.csv | sam.xlsx          the Social Accounting Matrix
    model.yaml                  model specification (growth, closures, solver)
    sectors.csv                 per-sector elasticities, depreciation, jobs ...
    households.csv              per-class counts, labor-supply/migration params ...
    governments.csv             per-government jobs and transfer rules
    income_elasticities.csv     LES income elasticities (rows: commodities, ROW, SAV)
    capital_coefficients.csv    investment-goods composition by destination sector
    tax_structure/<TAX>.csv     relative rate structure of sales-type taxes

See ``docs/data.md`` for the full description of every column.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .sam import check_balance, read_sam

SECTOR_DEFAULTS = {
    "sigma_va": 0.9,          # CES elasticity between factors (STAMP SIGMA)
    "sigma_top": 0.0,         # VA vs intermediate bundle (0 = Leontief, as STAMP)
    "sigma_armington": 1.5,   # domestic vs imported (STAMP ETAM)
    "cet_elasticity": float("inf"),  # domestic vs export supply (inf = same good)
    "export_elasticity": 1.65,       # export demand price elasticity (STAMP ETAE, abs.)
    "depreciation": 0.06,
    "inv_elasticity": 0.6,    # investment response to relative return (STAMP)
    "inv_share": float("nan"),  # share of investment by destination (nan -> capital income)
    "jobs": float("nan"),     # base-year jobs (thousands); nan -> $1m per job
}

HOUSEHOLD_DEFAULTS = {
    "households": float("nan"),          # count of households (required for per-household output)
    "persons_per_household": 2.5,
    "labor_supply_elasticity": 0.2,      # uncompensated, w.r.t. marginal net wage
    "time_endowment_ratio": 1.75,        # (labor + leisure) / labor at benchmark
    "frisch": -1.0,                      # LES Frisch parameter (-1 = Cobb-Douglas)
    "migration_elasticity": 0.0,         # elasticity of households to relative real income
    "migration_speed": 1.0,              # partial adjustment (1 = instantaneous)
    "salt_deduction_share": 0.0,         # share of deductible taxes deducted at margin
}

GOVERNMENT_DEFAULTS = {
    "jobs": float("nan"),
    "intergov_mode": "share",   # transfers paid to other governments: share of revenue | fixed
}

MODEL_DEFAULTS: dict[str, Any] = {
    "base_year": 2024,
    "periods": 11,
    "growth": {"population": 0.015, "productivity": 0.01},
    "inflation": 0.02,
    "discount_rate": 0.04,
    "numeraire": "row_price",
    "closures": {
        "governments": {},
        "tax_adjusts": {},
        "savings_investment": "open_capital",
        "capital": "sector_specific",
        "labor_supply": "endogenous",
        "migration": True,
        "migration_driver": "real_income",
    },
    "solver": {"method": "auto", "tol": 1e-8},
}


def _deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


SAM_FILES = ("sam.csv", "sam.xlsx", "template.xlsx")


def structure_sheet_name(tax: str) -> str:
    """Worksheet name of a tax-structure table in a template workbook."""
    return f"tax_structure_{tax}"[:31]


def find_sam(directory: Path) -> Path:
    for name in SAM_FILES:
        if (directory / name).exists():
            return directory / name
    raise FileNotFoundError(f"No SAM found in {directory} (expected one of {', '.join(SAM_FILES)})")


def _raw_table(directory: Path, csv_rel: str, workbook: Path | None, sheet: str) -> pd.DataFrame | None:
    """A parameter table from CSV (preferred) or from a template workbook sheet."""
    p = directory / csv_rel
    if p.exists():
        df = pd.read_csv(p, index_col=0, comment="#")
    elif workbook is not None and sheet in pd.ExcelFile(workbook).sheet_names:
        df = pd.read_excel(workbook, sheet_name=sheet, index_col=0)
    else:
        return None
    df.index = df.index.astype(str).str.strip()
    df = df.loc[[i not in ("nan", "") for i in df.index]]
    return df


def _read_table(raw: pd.DataFrame | None, index_name: str, defaults: dict, labels: list[str]) -> pd.DataFrame:
    if raw is not None:
        df = raw.copy()
    else:
        df = pd.DataFrame(index=labels)
    df.index.name = index_name
    missing = [lab for lab in labels if lab not in df.index]
    if missing:
        df = pd.concat([df, pd.DataFrame(index=missing)])
    df = df.loc[labels].copy()
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
        else:
            if isinstance(val, str):
                df[col] = df[col].fillna(val)
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
                if col not in ("inv_share", "jobs", "households"):
                    df[col] = df[col].fillna(val)
    return df


@dataclass
class Dataset:
    sam: pd.DataFrame
    accounts: dict
    model: dict
    sectors: pd.DataFrame
    households: pd.DataFrame
    governments: pd.DataFrame
    tax_structure: dict = field(default_factory=dict)
    income_elasticities: pd.DataFrame | None = None
    capital_coefficients: pd.DataFrame | None = None
    path: Path | None = None

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, directory: str | Path, model_overrides: dict | None = None, check: bool = True) -> "Dataset":
        """Load a dataset directory.  The SAM is read from sam.csv, sam.xlsx or a
        filled template.xlsx (in that order of preference); parameter tables from
        CSV files or, if absent, from the workbook's sheets.  ``check=False``
        skips the balance and account checks (used by the validator)."""
        d = Path(directory)
        with open(d / "accounts.yaml") as fh:
            accounts = yaml.safe_load(fh)
        sam_path = find_sam(d)
        wb = sam_path if sam_path.suffix.lower() in (".xlsx", ".xlsm") else None
        sam = read_sam(sam_path)
        if check:
            check_balance(sam)
        model = MODEL_DEFAULTS
        if (d / "model.yaml").exists():
            with open(d / "model.yaml") as fh:
                model = _deep_update(model, yaml.safe_load(fh) or {})
        model = _deep_update(model, model_overrides or {})
        sectors = _read_table(_raw_table(d, "sectors.csv", wb, "sectors"), "sector", SECTOR_DEFAULTS,
                              list(accounts["sectors"]))
        households = _read_table(_raw_table(d, "households.csv", wb, "households"), "household",
                                 HOUSEHOLD_DEFAULTS, list(accounts["households"]))
        governments = _read_table(_raw_table(d, "governments.csv", wb, "governments"), "government",
                                  GOVERNMENT_DEFAULTS, list(accounts["governments"]))
        tax_structure = {}
        for tax, spec in (accounts.get("taxes") or {}).items():
            if spec.get("base") != "sales":
                continue
            ts = _raw_table(d, spec.get("structure", f"tax_structure/{tax}.csv"), wb, structure_sheet_name(tax))
            if ts is not None:
                tax_structure[tax] = ts
        ie = _raw_table(d, "income_elasticities.csv", wb, "income_elasticities")
        cc = _raw_table(d, "capital_coefficients.csv", wb, "capital_coefficients")
        ds = cls(sam, accounts, model, sectors, households, governments, tax_structure, ie, cc, d)
        if check:
            ds.validate()
        return ds

    def copy(self) -> "Dataset":
        return copy.deepcopy(self)

    # -------------------------------------------------------------- validate
    def validate(self) -> None:
        acc = self.accounts
        declared = list(acc["sectors"]) + list(acc["factors"]) + list(acc["households"]) \
            + list(acc.get("enterprises") or []) + list(acc["governments"]) \
            + list((acc.get("taxes") or {}).keys()) + [acc["investment"], acc["row"]]
        if len(set(declared)) != len(declared):
            raise ValueError("An account is declared twice in accounts.yaml")
        missing = [a for a in declared if a not in self.sam.index]
        extra = [a for a in self.sam.index if a not in declared]
        if missing:
            raise ValueError(f"Accounts declared in accounts.yaml but missing from the SAM: {missing}")
        if extra:
            raise ValueError(f"SAM accounts not declared in accounts.yaml: {extra}")
        ftypes = set(acc["factors"].values())
        if not ftypes <= {"labor", "capital", "fixed"}:
            raise ValueError(f"Factor types must be labor, capital or fixed; got {ftypes}")
        if list(acc["factors"].values()).count("capital") > 1:
            raise ValueError("At most one factor of type 'capital' is supported")
        for tax, spec in (acc.get("taxes") or {}).items():
            b = spec.get("base")
            if b not in {"sales", "output", "factor", "property", "income"}:
                raise ValueError(f"Tax {tax}: unknown base {b!r}")
            if b == "factor" and spec.get("factor") not in acc["factors"]:
                raise ValueError(f"Tax {tax}: base 'factor' needs a valid 'factor' key")

    # ------------------------------------------------------------- overrides
    def with_parameters(self, overrides: dict | None) -> "Dataset":
        """Return a copy with parameter-table overrides applied.

        ``overrides`` maps a table name (``sectors``, ``households``,
        ``governments``, ``model``) to either ``{column: value}`` (all rows) or
        ``{column: {row: value}}``.  Multiplicative changes can be written as
        ``{column: {"*": factor}}``.
        """
        if not overrides:
            return self
        ds = self.copy()
        for table, cols in overrides.items():
            if table == "model":
                ds.model = _deep_update(ds.model, cols)
                continue
            df = getattr(ds, table)
            for col, val in cols.items():
                if col not in df.columns:
                    raise KeyError(f"{table}.csv has no column {col!r}")
                if isinstance(val, dict):
                    for row, v in val.items():
                        if row == "*":
                            df[col] = df[col] * float(v)
                        else:
                            df.loc[row, col] = v
                else:
                    df[col] = val
        return ds
