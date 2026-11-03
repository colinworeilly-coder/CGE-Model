import numpy as np

from conftest import walras_ok

from regcge.scenarios.scenario import Scenario


def test_baseline_is_balanced_growth(model, baseline):
    g = model.c.par["g"]
    for v in ("QX", "QH", "K", "IND", "LS", "YD", "TAXR"):
        x = baseline.get(v)
        x = x.reshape(len(baseline.years), -1)
        nz = np.abs(x[0]) > 1e-12
        assert np.allclose(x[1:, nz] / x[:-1, nz], 1 + g, rtol=1e-10)
    for v in ("PQ", "W", "RK", "CPI"):
        x = baseline.get(v)
        assert np.allclose(x, x[0], rtol=1e-10)
    assert walras_ok(baseline)
    n = model.c.par["n"]
    N = baseline.get("N")
    assert np.allclose(N[1:] / N[:-1], 1 + n)


def test_zero_shock_scenario_reproduces_baseline(sim, baseline):
    s = sim.scenario(baseline, Scenario.from_dict({"name": "none", "shocks": []}))
    for zb, zs in zip(baseline.z, s.z):
        assert np.allclose(zb, zs, atol=1e-12)


def test_capital_accumulates(model, sim, baseline):
    s = sim.scenario(baseline, Scenario.from_dict({"shocks": [{"tax": "LO_PROPB", "op": "scale", "value": 0.5}]}))
    K, IND = s.get("K"), s.get("IND")
    d = model.c.par["delta"]
    assert np.allclose(K[1:], (1 - d) * K[:-1] + IND[:-1])
    # cutting the business property tax raises investment and the capital stock
    assert s.get("K")[-1].sum() > baseline.get("K")[-1].sum()
    assert walras_ok(s)
