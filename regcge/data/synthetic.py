"""Disaggregating SAMs and generating larger synthetic datasets.

:func:`split_accounts` splits accounts of a balanced SAM into sub-accounts
(e.g. MFG -> MFG_1..MFG_k) with random shares.  Splitting rows and columns with
the same shares preserves balance exactly; optional multiplicative noise
differentiates the new accounts, after which the SAM is re-balanced by
cross-entropy.  :func:`synthetic_dataset` applies this to a Dataset, copying
parameter tables to the new accounts.  Used for scale tests, and useful for
prototyping a more detailed model before detailed data are available.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from .balance import balance_sam_cross_entropy
from .dataset import Dataset


def split_accounts(sam: pd.DataFrame, splits: dict[str, int], noise: float = 0.0,
                   seed: int = 0) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    rng = np.random.default_rng(seed)
    labels = list(sam.index)
    new_labels, parent, shares = [], {}, {}
    mapping = {}
    for a in labels:
        k = int(splits.get(a, 1))
        if k <= 1:
            new_labels.append(a)
            parent[a] = a
            shares[a] = 1.0
            mapping[a] = [a]
            continue
        w = rng.dirichlet(np.full(k, 4.0))
        subs = [f"{a}_{i + 1}" for i in range(k)]
        mapping[a] = subs
        for s, wi in zip(subs, w):
            new_labels.append(s)
            parent[s] = a
            shares[s] = wi
    X = sam.to_numpy()
    pos = {a: i for i, a in enumerate(labels)}
    n = len(new_labels)
    pidx = np.array([pos[parent[a]] for a in new_labels])
    w = np.array([shares[a] for a in new_labels])
    Y = X[np.ix_(pidx, pidx)] * np.outer(w, w)   # row r sums to shares[r] x row(parent): balance kept
    out = pd.DataFrame(Y, index=new_labels, columns=new_labels)
    if noise > 0:
        split_set = {s for a, subs in mapping.items() if len(subs) > 1 for s in subs}
        mask = np.array([[(r in split_set) or (c in split_set) for c in new_labels] for r in new_labels])
        fac = np.exp(noise * rng.standard_normal((n, n)))
        Y = np.where(mask & (np.abs(Y) > 0), Y * fac, Y)
        out = balance_sam_cross_entropy(pd.DataFrame(Y, index=new_labels, columns=new_labels))
    return out, mapping


def synthetic_dataset(ds: Dataset, sector_split: int = 5, household_split: int = 2,
                      noise: float = 0.15, seed: int = 0) -> Dataset:
    """A larger dataset obtained by splitting every sector and household account."""
    acc = copy.deepcopy(ds.accounts)
    splits = {a: sector_split for a in acc["sectors"]}
    splits.update({h: household_split for h in acc["households"]})
    sam, mapping = split_accounts(ds.sam, splits, noise=noise, seed=seed)
    acc["sectors"] = [s for a in acc["sectors"] for s in mapping[a]]
    acc["households"] = [s for h in acc["households"] for s in mapping[h]]
    acc["sector_names"] = {s: f"{ds.accounts.get('sector_names', {}).get(a, a)} ({s})"
                           for a in ds.accounts["sectors"] for s in mapping[a]}

    def expand(df, keys, count_col=None):
        rows = []
        for a in keys:
            for s in mapping[a]:
                r = df.loc[a].copy()
                r.name = s
                rows.append(r)
        out = pd.DataFrame(rows)
        out.index.name = df.index.name
        return out

    sectors = expand(ds.sectors, ds.accounts["sectors"])
    sectors["inv_share"] = np.nan
    # jobs follow labor payments
    lab = [f for f, t in acc["factors"].items() if t == "labor"]
    for a in ds.accounts["sectors"]:
        subs = mapping[a]
        pay = sam.loc[lab, subs].sum(axis=0)
        sectors.loc[subs, "jobs"] = ds.sectors.loc[a, "jobs"] * pay / pay.sum()
    households = expand(ds.households, ds.accounts["households"])
    households["households"] = households["households"].astype(float)
    sectors["jobs"] = sectors["jobs"].astype(float)
    for h in ds.accounts["households"]:
        subs = mapping[h]
        inc = sam.loc[subs].sum(axis=1)
        households.loc[subs, "households"] = ds.households.loc[h, "households"] * inc / inc.sum()
    tax_structure = {}
    for k, ts in ds.tax_structure.items():
        rows = []
        for a in ds.accounts["sectors"]:
            for s in mapping[a]:
                r = ts.loc[a].copy() if a in ts.index else pd.Series(0.0, index=ts.columns)
                r.name = s
                rows.append(r)
        tax_structure[k] = pd.DataFrame(rows)
    out = Dataset(sam, acc, copy.deepcopy(ds.model), sectors, households, ds.governments.copy(),
                  tax_structure, None, None, None)
    out.validate()
    return out
