import numpy as np

from regcge import RegionalCGE
from regcge.model.calibrate import effective_labor_mtr


def test_benchmark_replication(model):
    z, p = model.benchmark()
    assert np.abs(model.m.residuals(z, p)).max() < 1e-9


def test_solver_returns_benchmark(model):
    z, p = model.benchmark()
    res = model.solve(z, p)
    assert res.success and res.iterations == 0


def test_closures_square(model):
    assert model.check_closure(migration_active=False) == ""
    assert model.check_closure(migration_active=True) == ""


def test_sales_tax_rates_recovered(model):
    # illustrative data: MFG household purchases taxable at 75% x 6.25%
    tsh = model.c.par["tsh"]["ST_SALES"]
    A = model.S["A"].labels
    # (rounding + cross-entropy re-balancing of the SAM perturbs cells slightly)
    assert np.allclose(tsh[A.index("MFG")], 0.75 * 0.0625, rtol=5e-3)
    assert np.allclose(tsh[A.index("FIN")], 0.0)


def test_balanced_growth_capital(model):
    c = model.c
    assert np.allclose(c.bench["IND"], (c.par["g"] + c.par["delta"]) * c.bench["K"])


def test_eles_closes(model):
    c = model.c
    P0, gam, beta = c.par["items_P0"], c.par["gamma_pc"], c.par["beta"]
    assert np.allclose(beta.sum(0), 1.0)
    committed = (P0 * gam * c.bench["N"][None, :]).sum(0)
    assert np.allclose(committed + c.bench["Z"], c.bench["YD"])


def test_effective_mtr_with_deductibility():
    # federal 20% with state tax deductible at 50% of itemizers; state 5%
    K = ["FED", "ST"]
    w = {"FED": {"labor": 1.0}, "ST": {"labor": 1.0}}
    d = {"FED": ["ST"], "ST": []}
    m = {"FED": np.array([0.2]), "ST": np.array([0.05])}
    out = effective_labor_mtr(K, w, d, m, np.array([0.5]), 1)
    assert np.isclose(out[0], 0.05 + 0.2 * (1 - 0.5 * 0.05))


def test_synthetic_dataset_calibrates(ds):
    from regcge.data.synthetic import synthetic_dataset
    big = synthetic_dataset(ds, sector_split=2, household_split=2, noise=0.1, seed=4)
    M = RegionalCGE(big)
    z, p = M.benchmark()
    assert np.abs(M.m.residuals(z, p)).max() < 1e-9
    assert M.check_closure(migration_active=True) == ""


def test_ipopt_and_newton_agree(model):
    from regcge.core.solver import Solver
    z, p = model.benchmark()
    fixed = model.fixed_mask(migration_active=True)
    z2 = z.copy()
    model.m.set_level(z2, "TSCALE", 1.25, "ST_SALES")
    rn = Solver(model.m, fixed, method="newton").solve(z2, p)
    ri = Solver(model.m, fixed, method="ipopt").solve(z2, p)
    assert rn.success and ri.success
    assert np.allclose(rn.z, ri.z, rtol=1e-6, atol=1e-8)
