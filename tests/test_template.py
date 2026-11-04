import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from openpyxl import load_workbook

from regcge import Dataset, RegionalCGE
from regcge.data.rules import allowed_mask, cell_guide, disallowed_nonzero, ordered_accounts
from regcge.data.template import from_dataset, write_template
from regcge.data.validate import validate_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "illustrative_state"


def test_rules_cover_illustrative_sam(ds):
    M = allowed_mask(ds.accounts).loc[ds.sam.index, ds.sam.columns]
    assert not ((ds.sam.abs() > 0) & ~M).to_numpy().any()
    assert disallowed_nonzero(ds.sam, ds.accounts) == []
    g = cell_guide(ds.accounts)
    assert (g["what goes here"] != "").all()


def test_template_workbook_structure(ds, tmp_path):
    write_template(DATA / "accounts.yaml", tmp_path)
    wb = load_workbook(tmp_path / "template.xlsx")
    for sheet in ("Instructions", "SAM", "Cell guide", "Accounts", "Parameters guide",
                  "sectors", "households", "governments", "tax_structure_ST_SALES"):
        assert sheet in wb.sheetnames
    ws = wb["SAM"]
    labels = ordered_accounts(ds.accounts)
    n = len(labels)
    assert [ws.cell(1, j + 2).value for j in range(n)] == labels
    assert ws.cell(2, n + 2).value.startswith("=SUM(")          # row total
    assert ws.cell(2, n + 3).value.startswith("=SUM(")          # imbalance
    assert ws.cell(n + 2, 2).value.startswith("=SUM(")          # column total
    mask = allowed_mask(ds.accounts)
    grey = "00D0D0D0"
    for i, r in enumerate(labels):
        for j, c in enumerate(labels):
            is_grey = ws.cell(i + 2, j + 2).fill.fgColor.rgb == grey
            assert is_grey == (not mask.loc[r, c])
    # blank template + accounts are a loadable (empty) dataset
    assert (tmp_path / "accounts.yaml").exists() and (tmp_path / "model.yaml").exists()


def test_round_trip_through_workbook(ds, tmp_path):
    from_dataset(ds, tmp_path, fmt="both")
    for d in (tmp_path, tmp_path / "csv"):
        d2 = Dataset.load(d)
        assert np.allclose(d2.sam.to_numpy(), ds.sam.loc[d2.sam.index, d2.sam.columns].to_numpy(), atol=1e-10)
        assert np.allclose(d2.households["households"], ds.households["households"])
        M = RegionalCGE(d2)
        z, p = M.benchmark()
        assert np.abs(M.m.residuals(z, p)).max() < 1e-9
        assert np.allclose(M.c.par["tsh"]["ST_SALES"], RegionalCGE(ds).c.par["tsh"]["ST_SALES"])


def test_blank_template_filled_with_openpyxl(ds, tmp_path):
    write_template(DATA / "accounts.yaml", tmp_path)
    wb = load_workbook(tmp_path / "template.xlsx")
    ws = wb["SAM"]
    labels = ordered_accounts(ds.accounts)
    S = ds.sam.loc[labels, labels]
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = S.iat[i, j]
            if v != 0:
                ws.cell(i + 2, j + 2).value = float(v)
    hh = wb["households"]
    col = [c.value for c in hh[1]].index("households") + 1
    for r in range(2, hh.max_row + 1):
        hh.cell(r, col).value = float(ds.households.loc[hh.cell(r, 1).value, "households"])
    wb.save(tmp_path / "template.xlsx")
    rep = validate_dataset(tmp_path)
    assert rep.ok, rep.to_markdown()
    assert any("benchmark replication" in m and "OK" in m for m in rep.model_check)


def test_validate_illustrative_passes():
    rep = validate_dataset(DATA)
    assert rep.ok and not rep.warnings, rep.to_markdown()
    for t in ("Macro SAM", "Key figures", "Taxes", "Households", "Sectors", "Governments"):
        assert t in rep.tables
    assert "PASSED" in rep.to_markdown()


def test_validate_reports_all_seeded_faults(tmp_path):
    d = tmp_path / "bad"
    shutil.copytree(DATA, d)
    sam = pd.read_csv(d / "sam.csv", index_col=0)
    sam.loc["STATE", "H1"] = 1.0              # household pays a government directly
    sam.loc["FED_PIT", "AGR"] = 2.0           # income tax paid by a sector
    sam.loc["MFG", "TRD"] += 5.0              # unbalances MFG (row) and TRD (column)
    sam.to_csv(d / "sam.csv")
    acc = yaml.safe_load((d / "accounts.yaml").read_text())
    acc["taxes"]["ST_PIT"].pop("government")  # zero-revenue tax without a government
    (d / "accounts.yaml").write_text(yaml.safe_dump(acc, sort_keys=False))
    hh = pd.read_csv(d / "households.csv", index_col=0, comment="#")
    hh.loc["H3", "households"] = np.nan       # missing household count
    hh.to_csv(d / "households.csv")
    rep = validate_dataset(d)
    msgs = " | ".join(i.message for i in rep.errors)
    assert not rep.ok
    assert "do not balance" in msgs and "MFG" in msgs and "TRD" in msgs
    assert "STATE <- H1" in msgs
    assert "FED_PIT <- AGR" in msgs
    assert "ST_PIT has no revenue" in msgs
    assert "H3" in msgs
    # calibration still refuses the same data (it stops at the first problem)
    with pytest.raises(Exception):
        RegionalCGE(Dataset.load(d))


def test_cli_template_and_validate(tmp_path):
    out = tmp_path / "tpl"
    r = subprocess.run([sys.executable, "-m", "regcge", "template", "--out", str(out),
                        "--accounts", str(DATA / "accounts.yaml")], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0 and (out / "template.xlsx").exists()
    r = subprocess.run([sys.executable, "-m", "regcge", "validate", "--data", str(out), "--no-model"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 1 and "the SAM is empty" in r.stdout
    r = subprocess.run([sys.executable, "-m", "regcge", "validate", "--data", str(DATA),
                        "--out", str(tmp_path / "rep.md")], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0 and (tmp_path / "rep.md").exists()


def test_template_in_place_keeps_accounts_and_filled_files(tmp_path):
    d = tmp_path / "state"
    d.mkdir()
    shutil.copy(DATA / "accounts.yaml", d / "accounts.yaml")
    original = (d / "accounts.yaml").read_text()
    write_template(d / "accounts.yaml", d)               # accounts.yaml next to the output
    assert (d / "accounts.yaml").read_text() == original
    with pytest.raises(FileExistsError):
        write_template(d / "accounts.yaml", d)           # would overwrite a (possibly filled) workbook
    write_template(d / "accounts.yaml", d, overwrite=True)
    assert (d / "accounts.yaml").read_text() == original
