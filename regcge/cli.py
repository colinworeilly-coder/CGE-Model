"""Command-line interface.

    python -m regcge run scenarios/state_income_tax.yaml [--data DIR] [--out DIR] [--periods N]
    python -m regcge check [--data DIR]
    python -m regcge mcf [--data DIR] [--taxes ST_SALES LO_PROPB ...] [--recycle transfers|spending]
    python -m regcge balance IN.csv OUT.csv [--method cross_entropy|ras]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATA = "data/illustrative_state"


def cmd_run(a):
    from .runner import Runner
    from .scenarios.scenario import Scenario

    runner = Runner(a.data, a.periods, verbose=a.verbose)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in a.scenarios:
        t0 = time.perf_counter()
        sc = Scenario.from_yaml(f)
        comp = runner.run(sc)
        xl = comp.to_excel(out / f"{sc.name}.xlsx")
        md = out / f"{sc.name}.md"
        md.write_text(comp.to_markdown())
        comp.summary().to_csv(out / f"{sc.name}_summary.csv")
        print(f"{sc.name}: solved {len(comp.years)} periods in {time.perf_counter() - t0:.2f}s -> {xl}, {md}")
        if not a.quiet:
            print(comp.summary().round(2).to_string())
            print()


def cmd_check(a):
    from .data.dataset import Dataset
    from .model.dynamics import Simulator
    from .model.model import RegionalCGE

    t0 = time.perf_counter()
    ds = Dataset.load(a.data)
    M = RegionalCGE(ds)
    z, p = M.benchmark()
    r = np.abs(M.m.residuals(z, p)).max()
    print(f"Dataset {a.data}: {len(M.S['A'])} sectors, {len(M.S['H'])} households, "
          f"{len(M.S['G'])} governments, {len(M.S['K'])} tax instruments")
    print(f"Model: {M.m.nz} variables, {M.m.neq} equations, {M.m.np_} parameters "
          f"(built in {time.perf_counter() - t0:.2f}s)")
    print(f"Benchmark replication: max scaled residual {r:.2e}")
    for lab, active in (("baseline", False), ("scenario", True)):
        rep = M.check_closure(migration_active=active)
        print(f"Closure ({lab}): {'OK' if not rep else rep}")
    sim = Simulator(M)
    b = sim.baseline()
    g = M.c.par["g"]
    qx = b.get("QX").sum(1)
    print(f"Baseline {sim.years[0]}-{sim.years[-1]}: output growth {np.mean(qx[1:] / qx[:-1] - 1):.4%} "
          f"(g = {g:.4%}), max |WALRAS| {np.abs(b.get('WALRAS')).max():.1e}, {b.seconds:.2f}s")
    for w in M.c.warnings:
        print("Warning:", w)


def cmd_mcf(a):
    from .data.dataset import Dataset
    from .model.model import RegionalCGE
    from .results.welfare import marginal_cost_of_funds

    M = RegionalCGE(Dataset.load(a.data))
    taxes = a.taxes or [k for k in M.S["K"].labels if M.c.bench["TAXR"][M.S["K"].index(k)] > 0]
    rows = []
    for k in taxes:
        try:
            rows.append(marginal_cost_of_funds(M, k, delta=a.delta, recycle=a.recycle, migration=a.migration))
        except Exception as exc:
            print(f"{k}: {exc}")
    df = pd.DataFrame(rows).set_index("tax")
    print(df.round(3).to_string())


def cmd_balance(a):
    from .data.balance import balance_sam_cross_entropy, balance_sam_ras
    from .data.sam import check_balance, imbalances, read_sam, write_sam

    sam = read_sam(a.input)
    print(f"Input max imbalance: {imbalances(sam).abs().max():.6g}")
    out = balance_sam_ras(sam) if a.method == "ras" else balance_sam_cross_entropy(sam)
    check_balance(out, rel_tol=1e-8)
    write_sam(out, a.output)
    print(f"Balanced SAM written to {a.output}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="regcge", description="Regional tax CGE model")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run one or more scenario YAML files")
    r.add_argument("scenarios", nargs="+")
    r.add_argument("--data", default=DEFAULT_DATA)
    r.add_argument("--out", default="results")
    r.add_argument("--periods", type=int, default=None)
    r.add_argument("--verbose", action="store_true")
    r.add_argument("--quiet", action="store_true")
    r.set_defaults(func=cmd_run)
    c = sub.add_parser("check", help="calibrate, check benchmark replication, closures and baseline")
    c.add_argument("--data", default=DEFAULT_DATA)
    c.set_defaults(func=cmd_check)
    m = sub.add_parser("mcf", help="marginal cost of public funds by tax instrument")
    m.add_argument("--data", default=DEFAULT_DATA)
    m.add_argument("--taxes", nargs="*")
    m.add_argument("--delta", type=float, default=0.01)
    m.add_argument("--recycle", choices=["transfers", "spending"], default="transfers")
    m.add_argument("--migration", action="store_true", help="let population respond (default: fixed)")
    m.set_defaults(func=cmd_mcf)
    b = sub.add_parser("balance", help="balance a SAM")
    b.add_argument("input")
    b.add_argument("output")
    b.add_argument("--method", choices=["cross_entropy", "ras"], default="cross_entropy")
    b.set_defaults(func=cmd_balance)
    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
