import numpy as np

from conftest import walras_ok

from regcge.results.report import Comparison
from regcge.scenarios.scenario import Scenario


def run(sim, baseline, d):
    sc = Scenario.from_dict(d)
    return Comparison(baseline, sim.scenario(baseline, sc), sc)


def test_sales_tax_on_commodity_lowers_its_demand(model, sim, baseline):
    c = run(sim, baseline, {"shocks": [{"tax": "ST_SALES", "purchaser": "household",
                                        "commodities": ["MFG"], "op": "scale", "value": 1.5}]})
    i = model.S["A"].index("MFG")
    assert (c.s.get("QH")[1, i] < c.b.get("QH")[1, i]).all()
    rev = c.revenue(2025)
    assert 0 < rev.loc["ST_SALES", "change"] < rev.loc["ST_SALES", "static_change"]
    assert walras_ok(c.s)


def test_new_income_tax(model, sim, baseline):
    d = {"migration": False, "closure": {"governments": {"STATE": "deficit_adjusts"}},
         "shocks": [{"tax": "ST_PIT", "avg": 0.03, "mtr": 0.045}]}
    c = run(sim, baseline, d)
    # labor supply falls (all elasticities positive), welfare falls
    assert c.s.get("LS")[1].sum() < c.b.get("LS")[1].sum()
    assert c.ev()[1].sum() < 0
    rev = c.revenue(2025)
    assert rev.loc["ST_PIT", "change"] > 0
    # federal offset: state tax is deductible, so federal income tax revenue falls
    assert rev.loc["FED_PIT", "change"] < 0


def test_spending_adjusts_shifts_jobs_to_government(sim, baseline):
    c = run(sim, baseline, {"closure": {"governments": {"STATE": "spending_adjusts"}},
                            "shocks": [{"tax": "ST_SALES", "op": "scale", "value": 1.3}]})
    s = c.summary()
    assert s.loc["Private employment, change (jobs)", 2026] < 0
    assert s.loc["Government employment, change (jobs)", 2026] > 0


def test_equal_yield_swap(model, sim, baseline):
    c = run(sim, baseline, {"closure": {"governments": {"LOCAL": "tax_adjusts"}, "tax_adjusts": {"LOCAL": "LO_SALES"}},
                            "shocks": [{"tax": "LO_PROPB", "op": "scale", "value": 0.75}]})
    i = model.S["G"].index("LOCAL")
    assert np.allclose(c.s.get("QGL")[:, i], c.b.get("QGL")[:, i])
    assert np.allclose(c.s.get("SG")[:, i], c.b.get("SG")[:, i])
    k = model.S["K"].index("LO_SALES")
    assert c.s.get("TSCALE")[2, k] > 1.0


def test_phase_in(model, sim, baseline):
    c = run(sim, baseline, {"shocks": [{"tax": "ST_FRAN", "op": "scale", "value": 2.0,
                                        "phase_in": {2025: 0.5, 2027: 1.0}}]})
    t = c.s.param("tout_ST_FRAN")[:, 0] / c.b.param("tout_ST_FRAN")[:, 0]
    assert np.allclose(t[:5], [1.0, 1.5, 1.5, 2.0, 2.0])


def test_shipped_scenarios_solve(sim, baseline):
    from pathlib import Path
    for f in sorted((Path(__file__).resolve().parents[1] / "scenarios").glob("*.yaml")):
        sc = Scenario.from_yaml(f)
        if sc.parameters:
            continue
        s = sim.scenario(baseline, sc)
        assert all(st.success for st in s.stats)
        assert walras_ok(s)


def test_reports(sim, baseline, tmp_path):
    c = run(sim, baseline, {"name": "t", "shocks": [{"tax": "ST_SALES", "op": "scale", "value": 1.1}]})
    for df in (c.summary(), c.revenue(), c.governments(), c.distribution(), c.sectors(), c.welfare(), c.paths()):
        assert len(df) > 0
    p = c.to_excel(tmp_path / "out.xlsx")
    assert p.exists()
    assert "Summary" in c.to_markdown()
