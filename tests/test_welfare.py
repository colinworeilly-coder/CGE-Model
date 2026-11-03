import numpy as np

from regcge import RegionalCGE
from regcge.results.welfare import direct_utility, equivalent_variation, marginal_cost_of_funds
from regcge.scenarios.scenario import Scenario


def test_utility_consistent_with_demands(model, sim, baseline):
    s = sim.scenario(baseline, Scenario.from_dict({"shocks": [{"tax": "ST_SALES", "op": "scale", "value": 1.5}]}))
    for t in (0, 3, 10):
        assert np.allclose(direct_utility(model, s, t), s.levels[t]["U"], rtol=1e-10)


def test_ev_zero_without_shock(sim, baseline):
    s = sim.scenario(baseline, Scenario.from_dict({"shocks": []}))
    assert np.allclose(equivalent_variation(baseline, s), 0.0)


def test_mcf_above_one_for_labor_tax_when_price_taker(ds):
    # with near price-taking trade and no migration, a distortionary labor tax costs more than $1 per $1
    d2 = ds.with_parameters({"sectors": {"export_elasticity": 30.0, "sigma_armington": 6.0}})
    M = RegionalCGE(d2)
    r = marginal_cost_of_funds(M, "FED_PIT")
    assert r["MCF"] > 1.0
    assert r["d_recycled"] > 0
