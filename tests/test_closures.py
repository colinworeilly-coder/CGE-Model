import numpy as np

from conftest import walras_ok
import pytest

from regcge import RegionalCGE
from regcge.core.system import ClosureError, check_square
from regcge.model.dynamics import Simulator
from regcge.scenarios.scenario import Scenario

SHOCK = {"shocks": [{"tax": "ST_SALES", "op": "scale", "value": 1.1}]}


@pytest.mark.parametrize("rule", ["deficit_adjusts", "spending_adjusts", "transfers_adjust", "tax_adjusts"])
def test_government_rules(sim, baseline, rule):
    cl = {"governments": {"STATE": rule}}
    if rule == "tax_adjusts":
        cl["tax_adjusts"] = {"STATE": "ST_FRAN"}
    s = sim.scenario(baseline, Scenario.from_dict({**SHOCK, "closure": cl}))
    assert all(st.success for st in s.stats)
    assert walras_ok(s)


def test_savings_driven_and_cpi_numeraire(sim, baseline):
    cl = {"savings_investment": "savings_driven", "numeraire": "cpi"}
    s = sim.scenario(baseline, Scenario.from_dict({**SHOCK, "closure": cl}))
    assert np.allclose(s.get("CPIAVG"), 1.0)
    assert np.allclose(s.get("SROW"), baseline.get("SROW"))
    assert walras_ok(s)


def test_mobile_capital_and_exogenous_labor(ds):
    d2 = ds.with_parameters({"model": {"closures": {"capital": "mobile", "labor_supply": "exogenous",
                                                    "migration": False}}})
    M = RegionalCGE(d2)
    z, p = M.benchmark()
    assert np.abs(M.m.residuals(z, p)).max() < 1e-9
    sim = Simulator(M, periods=3)
    b = sim.baseline()
    s = sim.scenario(b, Scenario.from_dict(SHOCK))
    assert np.allclose(s.get("LS"), b.get("LS"))                  # labor supply fixed
    assert np.allclose(s.get("K").sum(1), b.get("K").sum(1))       # total capital fixed
    assert not np.allclose(s.get("K"), b.get("K"))                 # but reallocated


def test_bad_closure_is_diagnosed(model):
    fixed = model.fixed_mask()
    v = model.m.vars["QGL"]
    fixed[v.offset:v.offset + v.n] = False      # free every government's spending
    with pytest.raises(ClosureError, match="not square"):
        check_square(model.m, fixed)
    fixed2 = model.fixed_mask()
    # swap: fix the wage, free the exchange rate -> square but singular? fix W, free ER
    w = model.m.vars["W"]
    er = model.m.vars["ER"]
    fixed2[w.offset] = True
    fixed2[er.offset] = False
    assert check_square(model.m, fixed2, raise_error=False) == ""   # a valid alternative closure
    k = model.m.vars["K"]
    fixed3 = model.fixed_mask()
    fixed3[k.offset] = False                      # free AGR capital ...
    t = model.m.vars["TSCALE"]
    fixed3[t.offset] = True                       # ... and keep everything else: not square
    with pytest.raises(ClosureError):
        check_square(model.m, fixed3)
