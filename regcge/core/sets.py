"""Named, ordered index sets.

A :class:`Set` is an ordered list of string labels with fast label -> position
lookup.  Sets are used to dimension variables, parameters and equations and to
label results, so every number in the model can be traced back to account names
in the Social Accounting Matrix.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


class Set:
    """An ordered collection of unique string labels."""

    def __init__(self, name: str, labels: Iterable[str]):
        self.name = name
        self.labels: list[str] = [str(x) for x in labels]
        if len(set(self.labels)) != len(self.labels):
            raise ValueError(f"Set {name!r} has duplicate labels: {self.labels}")
        self._pos = {lab: i for i, lab in enumerate(self.labels)}

    # -- container protocol -------------------------------------------------
    def __len__(self) -> int:
        return len(self.labels)

    def __iter__(self):
        return iter(self.labels)

    def __contains__(self, label: str) -> bool:
        return label in self._pos

    def __getitem__(self, i: int) -> str:
        return self.labels[i]

    def __repr__(self) -> str:
        return f"Set({self.name!r}, {self.labels})"

    # -- helpers -------------------------------------------------------------
    def index(self, label: str) -> int:
        try:
            return self._pos[label]
        except KeyError as exc:
            raise KeyError(f"{label!r} is not in set {self.name!r} ({self.labels})") from exc

    def indices(self, labels: Sequence[str] | None) -> list[int]:
        """Positions of ``labels`` (all positions if ``labels`` is None)."""
        if labels is None:
            return list(range(len(self)))
        if isinstance(labels, str):
            labels = [labels]
        return [self.index(lab) for lab in labels]

    def mask(self, labels: Sequence[str] | None) -> np.ndarray:
        """Boolean mask over the set selecting ``labels`` (all if None)."""
        m = np.zeros(len(self), dtype=bool)
        m[self.indices(labels)] = True
        return m

    def subset(self, name: str, labels: Sequence[str]) -> "Set":
        for lab in labels:
            self.index(lab)
        return Set(name, labels)


def product_labels(dims: Sequence[Set]) -> list[tuple[str, ...]]:
    """Row-major (C-order) list of label tuples for a tuple of sets."""
    if len(dims) == 0:
        return [()]
    grids = np.meshgrid(*[np.arange(len(d)) for d in dims], indexing="ij")
    flat = [g.ravel() for g in grids]
    return [tuple(dims[k].labels[flat[k][i]] for k in range(len(dims))) for i in range(flat[0].size)]


def dims_shape(dims: Sequence[Set]) -> tuple[int, ...]:
    return tuple(len(d) for d in dims)
