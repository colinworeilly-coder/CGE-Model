import warnings
from pathlib import Path

import pytest

from regcge import Dataset, RegionalCGE
from regcge.model.dynamics import Simulator

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "illustrative_state"
SCEN = ROOT / "scenarios"


@pytest.fixture(scope="session")
def ds():
    return Dataset.load(DATA)


@pytest.fixture(scope="session")
def model(ds):
    return RegionalCGE(ds)


@pytest.fixture(scope="session")
def sim(model):
    return Simulator(model)


@pytest.fixture(scope="session")
def baseline(sim):
    return sim.baseline()


def walras_ok(path, rel: float = 1e-8) -> bool:
    """Walras slack relative to the value of investment (solver tolerance is relative)."""
    import numpy as np
    return bool(np.all(np.abs(path.get("WALRAS")) <= rel * np.abs(path.get("INVEST"))))
