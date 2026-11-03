"""Reading, writing and validating Social Accounting Matrices.

Convention: a SAM is a square table whose rows and columns carry the same
account labels.  Cell ``(r, c)`` is a payment *received by* row account ``r``
*from* column account ``c``.  Each account's row total (receipts) must equal
its column total (payments).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class SAMError(ValueError):
    pass


def read_sam(path: str | Path, sheet: str | int = 0) -> pd.DataFrame:
    """Read a SAM from CSV or Excel.  The first column holds row labels."""
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        df = pd.read_excel(path, sheet_name=sheet, index_col=0)
    else:
        df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str).str.strip()
    df.columns = df.columns.astype(str).str.strip()
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return canonical(df)


def canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Check that rows and columns carry the same labels; order columns like rows."""
    rows, cols = list(df.index), list(df.columns)
    if len(set(rows)) != len(rows) or len(set(cols)) != len(cols):
        raise SAMError("SAM has duplicate row or column labels")
    if set(rows) != set(cols):
        missing_c = sorted(set(rows) - set(cols))
        missing_r = sorted(set(cols) - set(rows))
        raise SAMError(f"SAM is not square. Rows without columns: {missing_c}; columns without rows: {missing_r}")
    return df.loc[rows, rows].astype(float)


def write_sam(sam: pd.DataFrame, path: str | Path, decimals: int | None = None) -> None:
    path = Path(path)
    out = sam.round(decimals) if decimals is not None else sam
    if path.suffix.lower() in (".xlsx", ".xls"):
        out.to_excel(path)
    else:
        out.to_csv(path, float_format="%.17g")


def imbalances(sam: pd.DataFrame) -> pd.Series:
    """Row total minus column total for each account."""
    return sam.sum(axis=1) - sam.sum(axis=0)


def check_balance(sam: pd.DataFrame, rel_tol: float = 1e-6, raise_error: bool = True) -> float:
    """Return the largest relative imbalance; raise if above ``rel_tol``."""
    tot = np.maximum(sam.sum(axis=1).abs(), sam.sum(axis=0).abs())
    gap = imbalances(sam).abs()
    rel = (gap / tot.where(tot > 0, 1.0)).max()
    if raise_error and rel > rel_tol:
        worst = (gap / tot.where(tot > 0, 1.0)).sort_values(ascending=False).head(5)
        raise SAMError("SAM is not balanced (row total != column total). Worst accounts (relative gap):\n"
                       + worst.to_string() + "\nUse regcge.data.balance to balance it.")
    return float(rel)


def aggregate(sam: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Aggregate accounts with a label -> new-label mapping (unmapped kept)."""
    lab = [mapping.get(a, a) for a in sam.index]
    out = sam.groupby(lab, sort=False).sum()
    out = out.T.groupby(lab, sort=False).sum().T
    return canonical(out)
