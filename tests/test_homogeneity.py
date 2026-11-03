import numpy as np


def test_homogeneous_of_degree_zero(model):
    z, p = model.benchmark()
    z2 = z.copy()
    model.m.set_level(z2, "ER", 1.1)
    for g, rule in model.closure.governments.items():
        if rule != "deficit_adjusts":
            i = model.S["G"].index(g)
            model.m.set_level(z2, "SG", model.c.bench["SG"][i] * 1.1, g)
    res = model.solve(z2, p)
    assert res.success
    L0, L1 = model.m.levels(z), model.m.levels(res.z)
    for v in ("PQ", "PX", "W", "RK", "CPI"):
        assert np.allclose(L1[v], 1.1 * L0[v], rtol=1e-9)
    for v in ("YH", "YD", "TAXR", "YG"):
        nz = np.abs(L0[v]) > 0
        assert np.allclose(L1[v][nz], 1.1 * L0[v][nz], rtol=1e-9)
    for v in ("QX", "QH", "LS", "QE", "QM", "U", "IND"):
        assert np.allclose(L1[v], L0[v], rtol=1e-9, atol=1e-9)
