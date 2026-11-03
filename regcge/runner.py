"""High-level entry points: run scenarios against a baseline."""

from __future__ import annotations

from pathlib import Path

from .data.dataset import Dataset
from .model.dynamics import Simulator
from .model.model import RegionalCGE
from .results.report import Comparison
from .scenarios.scenario import Scenario


def _as_scenario(s) -> Scenario:
    if isinstance(s, Scenario):
        return s
    if isinstance(s, dict):
        return Scenario.from_dict(s)
    return Scenario.from_yaml(s)


class Runner:
    """Builds (and caches) models and baselines; runs scenarios.

    Scenarios that override parameter tables trigger a separate calibration
    and baseline, so every scenario is compared with a baseline that uses the
    same parameters.
    """

    def __init__(self, data: str | Path | Dataset, periods: int | None = None, verbose: bool = False,
                 model_overrides: dict | None = None):
        self.ds = data if isinstance(data, Dataset) else Dataset.load(data, model_overrides)
        self.periods = periods
        self.verbose = verbose
        self._cache: dict = {}

    def model_and_baseline(self, parameters: dict | None = None):
        key = repr(sorted((parameters or {}).items()))
        if key not in self._cache:
            ds = self.ds.with_parameters(parameters)
            model = RegionalCGE(ds)
            sim = Simulator(model, self.periods, verbose=self.verbose)
            self._cache[key] = (model, sim, sim.baseline())
        return self._cache[key]

    def run(self, scenario) -> Comparison:
        sc = _as_scenario(scenario)
        model, sim, base = self.model_and_baseline(sc.parameters)
        path = sim.scenario(base, sc)
        return Comparison(base, path, sc)


def run_scenario(data, scenario, periods: int | None = None, verbose: bool = False) -> Comparison:
    """One-shot convenience wrapper: ``run_scenario("data/illustrative_state", "scenarios/x.yaml")``."""
    return Runner(data, periods, verbose).run(scenario)
