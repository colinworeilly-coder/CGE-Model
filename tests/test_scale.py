import time

import numpy as np

from conftest import walras_ok

from regcge import RegionalCGE
from regcge.data.synthetic import synthetic_dataset
from regcge.model.dynamics import Simulator
from regcge.scenarios.scenario import Scenario


def test_large_synthetic_model_path(ds):
    big = synthetic_dataset(ds, sector_split=5, household_split=2, noise=0.15, seed=0)   # 55 sectors, 14 classes
    t0 = time.perf_counter()
    M = RegionalCGE(big)
    sim = Simulator(M, periods=11)
    b = sim.baseline()
    s = sim.scenario(b, Scenario.from_dict({"shocks": [{"tax": "ST_SALES", "op": "scale", "value": 1.2}]}))
    elapsed = time.perf_counter() - t0
    assert all(st.success for st in s.stats)
    assert walras_ok(s)
    print(f"55-sector model: build + baseline + 11-year scenario in {elapsed:.1f}s")
    assert elapsed < 120
