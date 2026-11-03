"""A thin algebraic modeling layer on top of CasADi.

The layer lets equations be written in vectorised, GAMS-like form while
CasADi provides exact sparse Jacobians/Hessians by automatic differentiation.

Key conventions
---------------
* **Variables** are declared with a benchmark level.  Internally each variable
  element is represented by a scaled symbol ``z`` such that
  ``level = scale * z`` with ``scale = |benchmark level|`` (or 1 when the
  benchmark is zero).  At the benchmark every non-zero variable therefore has
  ``z = 1``, which makes the Newton/IPOPT iterations very well conditioned.
* **Parameters** are CasADi symbols too, so that time-varying exogenous inputs
  (tax rates, capital stocks, growth indices, ...) can change between periods
  and scenarios without rebuilding the model.
* **Equations** are ``lhs == rhs`` pairs.  Each element is scaled by
  ``max(|lhs|, |rhs|)`` evaluated at the benchmark, so residuals are relative.
* Every variable element can be *fixed* (exogenous) or *free* (endogenous).
  A closure is nothing more than a choice of which elements are fixed; the
  system must be square (checked, with diagnostics, in :mod:`regcge.core.system`).

Variables and parameters have up to two dimensions; element order is row-major.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import casadi as ca
import numpy as np

from .sets import Set, dims_shape, product_labels


def _nelem(shape: tuple[int, ...]) -> int:
    return int(np.prod(shape)) if len(shape) else 1


def _shape_expr(flat, dims: Sequence[Set]):
    """Turn a flat (row-major) n x 1 SX into scalar / column / matrix form."""
    if len(dims) <= 1:
        return flat
    if len(dims) == 2:
        n1, n2 = len(dims[0]), len(dims[1])
        return ca.reshape(flat, n2, n1).T
    raise ValueError("At most two dimensions are supported for variables/parameters")


def _flatten(x, shape: tuple[int, ...]):
    """Row-major flatten of an SX/number to an n x 1 SX of the given shape."""
    n = _nelem(shape)
    if not isinstance(x, (ca.SX, ca.MX, ca.DM)):
        arr = np.broadcast_to(np.asarray(x, dtype=float), shape if shape else ())
        return ca.SX(ca.DM(np.ravel(arr)))
    x = ca.SX(x)
    if x.numel() == 1 and n > 1:
        return x * ca.SX.ones(n, 1)
    if x.shape[1] == 1 or x.shape[0] == 1:
        x = ca.vec(x)
        if x.numel() != n:
            raise ValueError(f"Expression has {x.numel()} elements, expected {n} for shape {shape}")
        return x
    # matrix: row-major
    if x.shape != tuple(shape):
        raise ValueError(f"Matrix expression of shape {x.shape} does not match {shape}")
    return ca.vec(x.T)


@dataclass
class VarInfo:
    name: str
    dims: tuple
    shape: tuple
    n: int
    sym: Any
    level0: np.ndarray
    scale: np.ndarray
    lb: np.ndarray | None
    fixed: np.ndarray
    grow: str
    desc: str
    expr: Any = None
    offset: int = 0

    @property
    def labels(self):
        return product_labels(self.dims)


@dataclass
class ParamInfo:
    name: str
    dims: tuple
    shape: tuple
    n: int
    sym: Any
    value0: np.ndarray
    desc: str
    expr: Any = None
    offset: int = 0


@dataclass
class EqInfo:
    name: str
    dims: tuple
    lhs: Any
    rhs: Any
    idx: np.ndarray
    desc: str


@dataclass
class ExprInfo:
    name: str
    dims: tuple
    shape: tuple
    flat: Any
    offset: int = 0


class CoreModel:
    """Registry of variables, parameters, equations and reporting expressions."""

    def __init__(self, name: str = "model"):
        self.name = name
        self.vars: dict[str, VarInfo] = {}
        self.params: dict[str, ParamInfo] = {}
        self.eqs: list[EqInfo] = []
        self.exprs: dict[str, ExprInfo] = {}
        self.compiled = False

    # ------------------------------------------------------------------ build
    def add_var(self, name: str, dims: Sequence[Set] = (), level=None, lb=None,
                grow: str = "p", fixed: bool = False, desc: str = ""):
        """Declare a variable and return its (level-space) expression.

        ``grow`` tags how the element's level evolves along a balanced-growth
        baseline: ``'p'`` (prices / ratios: constant), ``'q'`` (aggregate real
        quantities: grow at the economy's growth rate), ``'n'`` (population).
        """
        if name in self.vars:
            raise ValueError(f"Variable {name!r} declared twice")
        dims = tuple(dims)
        shape = dims_shape(dims)
        n = _nelem(shape)
        lev = np.ones(n) if level is None else np.broadcast_to(np.asarray(level, float), shape).ravel().copy()
        if not np.all(np.isfinite(lev)):
            raise ValueError(f"Non-finite benchmark level for {name}: {lev}")
        scale = np.where(np.abs(lev) > 1e-12, np.abs(lev), 1.0)
        lbv = None if lb is None else np.broadcast_to(np.asarray(lb, float), shape).ravel().copy()
        sym = ca.SX.sym(name, n)
        flat = sym * ca.DM(scale)
        info = VarInfo(name, dims, shape, n, sym, lev, scale, lbv,
                       np.full(n, bool(fixed)), grow, desc)
        info.expr = _shape_expr(flat, dims)
        self.vars[name] = info
        return info.expr

    def add_param(self, name: str, dims: Sequence[Set] = (), value=0.0, desc: str = ""):
        if name in self.params:
            raise ValueError(f"Parameter {name!r} declared twice")
        dims = tuple(dims)
        shape = dims_shape(dims)
        n = _nelem(shape)
        val = np.broadcast_to(np.asarray(value, float), shape).ravel().copy()
        sym = ca.SX.sym(name, n)
        info = ParamInfo(name, dims, shape, n, sym, val, desc)
        info.expr = _shape_expr(sym, dims)
        self.params[name] = info
        return info.expr

    def add_eq(self, name: str, lhs, rhs, dims: Sequence[Set] = (), where=None, desc: str = ""):
        dims = tuple(dims)
        shape = dims_shape(dims)
        n = _nelem(shape)
        L = _flatten(lhs, shape)
        R = _flatten(rhs, shape)
        if where is None:
            idx = np.arange(n)
        else:
            w = np.broadcast_to(np.asarray(where, bool), shape).ravel()
            idx = np.flatnonzero(w)
        if idx.size == 0:
            return
        self.eqs.append(EqInfo(name, dims, L, R, idx, desc))

    def add_expr(self, name: str, ex, dims: Sequence[Set] = ()):
        dims = tuple(dims)
        shape = dims_shape(dims)
        self.exprs[name] = ExprInfo(name, dims, shape, _flatten(ex, shape))

    def V(self, name: str):
        return self.vars[name].expr

    def P(self, name: str):
        return self.params[name].expr

    # -------------------------------------------------------------- selection
    def select(self, name: str, sel=None, kind: str = "var") -> np.ndarray:
        """Flat positions of a selection within a variable or parameter.

        ``sel`` may be None (all), a label or list of labels (1-D), a tuple with
        one entry (label/list/None) per dimension, or a boolean array.
        """
        info = self.vars[name] if kind == "var" else self.params[name]
        n = info.n
        if sel is None:
            return np.arange(n)
        if isinstance(sel, np.ndarray) and sel.dtype == bool:
            return np.flatnonzero(np.broadcast_to(sel, info.shape).ravel())
        dims = info.dims
        if len(dims) == 0:
            return np.arange(1)
        if len(dims) == 1:
            sel = (sel,)
        if not isinstance(sel, tuple) or len(sel) != len(dims):
            raise ValueError(f"Selection {sel!r} does not match dims of {name}")
        masks = [d.mask(None if s is None else ([s] if isinstance(s, str) else list(s)))
                 for d, s in zip(dims, sel)]
        full = masks[0]
        for m in masks[1:]:
            full = np.multiply.outer(full, m)
        return np.flatnonzero(full.ravel())

    def fix(self, name: str, sel=None):
        self.vars[name].fixed[self.select(name, sel)] = True

    def free(self, name: str, sel=None):
        self.vars[name].fixed[self.select(name, sel)] = False

    # ---------------------------------------------------------------- compile
    def compile(self):
        off = 0
        for v in self.vars.values():
            v.offset = off
            off += v.n
        self.nz = off
        off = 0
        for p in self.params.values():
            p.offset = off
            off += p.n
        self.np_ = off
        self.z_sym = ca.vertcat(*[v.sym for v in self.vars.values()])
        self.p_sym = ca.vertcat(*[p.sym for p in self.params.values()]) if self.params else ca.SX(0, 1)

        Ls, Rs, labels, blocks = [], [], [], []
        for e in self.eqs:
            Ls.append(e.lhs[e.idx.tolist()])
            Rs.append(e.rhs[e.idx.tolist()])
            labs = product_labels(e.dims)
            labels.extend(f"{e.name}[{','.join(labs[i])}]" if labs[i] else e.name for i in e.idx)
            blocks.extend([e.name] * e.idx.size)
        L = ca.vertcat(*Ls)
        R = ca.vertcat(*Rs)
        self.eq_labels = labels
        self.eq_blocks = np.array(blocks, dtype=object)
        self.neq = L.numel()

        z0 = self.z_benchmark()
        p0 = self.p_benchmark()
        fLR = ca.Function("LR", [self.z_sym, self.p_sym], [L, R])
        Lv, Rv = (np.array(a).ravel() for a in fLR(z0, p0))
        sc = np.maximum(np.abs(Lv), np.abs(Rv))
        sc = np.where(np.isfinite(sc) & (sc > 1e-9), sc, 1.0)
        self.eq_scale = sc
        self.res_expr = (L - R) / ca.DM(sc)
        self.f_res = ca.Function("res", [self.z_sym, self.p_sym], [self.res_expr])

        off = 0
        for ex in self.exprs.values():
            ex.offset = off
            off += ex.flat.numel()
        allx = ca.vertcat(*[e.flat for e in self.exprs.values()]) if self.exprs else ca.SX(0, 1)
        self.f_expr = ca.Function("expr", [self.z_sym, self.p_sym], [allx])

        self._f_jac = None
        self.var_labels = []
        for v in self.vars.values():
            labs = product_labels(v.dims)
            self.var_labels.extend(f"{v.name}[{','.join(l)}]" if l else v.name for l in labs)
        self.compiled = True

    # --------------------------------------------------------------- jacobian
    def jacobian_function(self):
        """Jacobian of the scaled residuals w.r.t. all variables, built block by
        block (much faster than one global symbolic Jacobian for large models)."""
        if self._f_jac is None:
            blocks = []
            off = 0
            for e in self.eqs:
                n = e.idx.size
                L = (e.lhs[e.idx.tolist()] - e.rhs[e.idx.tolist()]) / ca.DM(self.eq_scale[off:off + n])
                blocks.append(ca.jacobian(L, self.z_sym))
                off += n
            J = ca.vertcat(*blocks)
            self._f_jac = ca.Function("jac_res", [self.z_sym, self.p_sym], [J])
        return self._f_jac

    def jacobian_sparsity(self):
        return self.jacobian_function().sparsity_out(0)

    # ------------------------------------------------------------ vectors I/O
    def z_benchmark(self) -> np.ndarray:
        return np.concatenate([v.level0 / v.scale for v in self.vars.values()])

    def p_benchmark(self) -> np.ndarray:
        if not self.params:
            return np.zeros(0)
        return np.concatenate([p.value0 for p in self.params.values()])

    def fixed_mask(self) -> np.ndarray:
        return np.concatenate([v.fixed for v in self.vars.values()])

    def scale_vector(self) -> np.ndarray:
        return np.concatenate([v.scale for v in self.vars.values()])

    def levels(self, z: np.ndarray) -> dict[str, np.ndarray]:
        """Dict name -> level array (shaped) from a scaled solution vector."""
        out = {}
        for v in self.vars.values():
            seg = z[v.offset:v.offset + v.n] * v.scale
            out[v.name] = seg.reshape(v.shape) if v.shape else seg[0]
        return out

    def set_level(self, z: np.ndarray, name: str, value, sel=None) -> None:
        v = self.vars[name]
        pos = self.select(name, sel)
        val = np.broadcast_to(np.asarray(value, float), (pos.size,)) if np.ndim(value) <= 1 else np.asarray(value, float).ravel()
        z[v.offset + pos] = val / v.scale[pos]

    def get_level(self, z: np.ndarray, name: str):
        v = self.vars[name]
        seg = z[v.offset:v.offset + v.n] * v.scale
        return seg.reshape(v.shape) if v.shape else seg[0]

    def set_param(self, p: np.ndarray, name: str, value, sel=None) -> None:
        info = self.params[name]
        pos = self.select(name, sel, kind="param")
        val = np.asarray(value, float)
        if val.ndim >= 1 and val.size == pos.size:
            val = val.ravel()
        p[info.offset + pos] = val

    def get_param(self, p: np.ndarray, name: str):
        info = self.params[name]
        seg = p[info.offset:info.offset + info.n]
        return seg.reshape(info.shape) if info.shape else seg[0]

    def residuals(self, z: np.ndarray, p: np.ndarray) -> np.ndarray:
        return np.array(self.f_res(z, p)).ravel()

    def evaluate_exprs(self, z: np.ndarray, p: np.ndarray) -> dict[str, Any]:
        vals = np.array(self.f_expr(z, p)).ravel()
        out = {}
        for ex in self.exprs.values():
            seg = vals[ex.offset:ex.offset + ex.flat.numel()]
            out[ex.name] = seg.reshape(ex.shape) if ex.shape else seg[0]
        return out

    def worst_residuals(self, z, p, k: int = 10) -> list[tuple[str, float]]:
        r = self.residuals(z, p)
        order = np.argsort(-np.abs(r))[:k]
        return [(self.eq_labels[i], float(r[i])) for i in order]
