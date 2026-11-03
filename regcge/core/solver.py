"""Solving the square nonlinear system F(x; x_fixed, p) = 0.

Two methods share one block-wise Jacobian of the residuals (built once per
model by :meth:`CoreModel.jacobian_function`):

* ``newton`` -- damped Newton with sparse LU (CasADi/CSparse), backtracking on
  the residual norm and a fraction-to-the-boundary rule that keeps prices and
  quantities positive.  Fast from warm starts (the usual case: each period is
  started from the previous solution).
* ``ipopt`` -- the system is passed to IPOPT (bundled with CasADi) as a
  feasibility problem with a zero objective and positivity bounds, the
  "constrained nonlinear system" approach familiar from GAMS.  For a square
  system the Newton step does not depend on the Hessian, so a limited-memory
  Hessian is used; the exact Jacobian is supplied.

``method='auto'`` (default) tries Newton first and falls back to IPOPT.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import casadi as ca
import numpy as np

from .system import check_square


@dataclass
class SolveResult:
    z: np.ndarray
    success: bool
    status: str
    iterations: int
    max_residual: float
    seconds: float
    method: str


class SolverError(RuntimeError):
    pass


DEFAULT_IPOPT = {
    "ipopt.print_level": 0,
    "ipopt.sb": "yes",
    "print_time": False,
    "ipopt.tol": 1e-10,
    "ipopt.constr_viol_tol": 1e-10,
    "ipopt.acceptable_tol": 1e-8,
    "ipopt.acceptable_constr_viol_tol": 1e-8,
    "ipopt.max_iter": 500,
    "ipopt.mu_strategy": "adaptive",
    "ipopt.nlp_scaling_method": "none",
    "ipopt.hessian_approximation": "limited-memory",
}


try:  # SciPy's SuperLU (with COLAMD ordering) is much faster than CSparse without ordering
    import scipy.sparse as _sp
    import scipy.sparse.linalg as _spla
except ImportError:  # pragma: no cover
    _sp = None


def _sparse_solve(J, rhs: np.ndarray) -> np.ndarray:
    if _sp is not None:
        spJ = J.sparsity()
        A = _sp.csc_matrix((np.array(J.nonzeros(), dtype=float), np.array(spJ.row()), np.array(spJ.colind())),
                           shape=J.shape)
        return _spla.splu(A).solve(rhs)
    return np.array(ca.solve(J, ca.DM(rhs), "csparse")).ravel()


class Solver:
    """Compiled solver for one closure (fixed/free pattern)."""

    def __init__(self, core, fixed: np.ndarray, method: str = "auto", tol: float = 1e-8,
                 ipopt_options: dict | None = None, check: bool = True, max_newton: int = 60):
        if not core.compiled:
            core.compile()
        self.core = core
        self.fixed = fixed.copy()
        self.method = method
        self.tol = tol
        self.max_newton = max_newton
        if check:
            check_square(core, self.fixed)
        N = core.nz
        self.endo = np.flatnonzero(~self.fixed)
        self.fix_idx = np.flatnonzero(self.fixed)
        nE, nX, nP = self.endo.size, self.fix_idx.size, core.np_
        self.nE, self.nX = nE, nX
        Pe = ca.DM(ca.Sparsity.triplet(N, nE, self.endo.tolist(), list(range(nE))), 1.0)
        Pf = ca.DM(ca.Sparsity.triplet(N, nX, self.fix_idx.tolist(), list(range(nX))), 1.0)
        self.Pe = Pe
        X = ca.MX.sym("x", nE)
        PV = ca.MX.sym("pv", nX + nP)
        z = ca.mtimes(Pe, X) + (ca.mtimes(Pf, PV[:nX]) if nX else 0)
        p = PV[nX:]
        g = core.f_res(z, p)
        Jx = ca.mtimes(core.jacobian_function()(z, p), Pe)
        self.g_fun = ca.Function("G", [X, PV], [g])
        self.J_fun = ca.Function("J", [X, PV], [Jx])
        self._X, self._PV, self._g, self._Jx = X, PV, g, Jx
        # bounds on endogenous elements (only where the benchmark level is positive)
        lb = np.full(N, -np.inf)
        for v in core.vars.values():
            if v.lb is not None:
                seg = v.lb / v.scale
                seg = np.where(v.level0 > 0, seg, -np.inf)
                lb[v.offset:v.offset + v.n] = seg
        self.lbx = lb[self.endo]
        self._ipopt = None
        self.ipopt_options = dict(DEFAULT_IPOPT)
        if ipopt_options:
            self.ipopt_options.update(ipopt_options)

    # ----------------------------------------------------------------- IPOPT
    def _get_ipopt(self):
        if self._ipopt is None:
            neq = self.core.neq
            jac = ca.Function("jac_Gnlp", [self._X, self._PV, ca.MX.sym("o", neq)],
                              [self._Jx, ca.MX(neq, self._PV.numel())])
            G = ca.Function("Gnlp", [self._X, self._PV], [self._g],
                            {"custom_jacobian": jac, "jac_penalty": 0})
            nlp = {"x": self._X, "p": self._PV, "f": 0, "g": G(self._X, self._PV)}
            self._ipopt = ca.nlpsol("cge_ipopt", "ipopt", nlp, self.ipopt_options)
        return self._ipopt

    def _solve_ipopt(self, x0, pv):
        S = self._get_ipopt()
        sol = S(x0=x0, p=pv, lbg=0, ubg=0, lbx=self.lbx, ubx=np.inf)
        st = S.stats()
        return np.array(sol["x"]).ravel(), st.get("return_status", "?"), int(st.get("iter_count", -1))

    # ---------------------------------------------------------------- Newton
    def _solve_newton(self, x0, pv):
        x = x0.copy()
        g = np.array(self.g_fun(x, pv)).ravel()
        nrm = np.linalg.norm(g)
        lbx = self.lbx
        has_lb = np.isfinite(lbx)
        for it in range(1, self.max_newton + 1):
            J = self.J_fun(x, pv)
            try:
                dx = _sparse_solve(J, -g)
            except (RuntimeError, ValueError, ArithmeticError):
                return x, "singular Jacobian", it
            if not np.all(np.isfinite(dx)):
                return x, "non-finite Newton step", it
            # fraction to the boundary
            alpha = 1.0
            neg = has_lb & (dx < 0)
            if neg.any():
                room = (x[neg] - lbx[neg]) / (-dx[neg])
                alpha = min(1.0, 0.95 * room.min()) if room.size else 1.0
            # backtracking on the residual norm
            for _ in range(30):
                xt = x + alpha * dx
                gt = np.array(self.g_fun(xt, pv)).ravel()
                nt = np.linalg.norm(gt)
                if np.isfinite(nt) and (nt <= (1 - 1e-4 * alpha) * nrm or nt < self.tol):
                    break
                alpha *= 0.5
            else:
                return x, "line search failed", it
            x, g, nrm = xt, gt, nt
            if np.abs(g).max() < self.tol * 0.1:
                return x, "newton converged", it
        return x, "newton: iteration limit", self.max_newton

    # ----------------------------------------------------------- continuation
    def _solve_continuation(self, x0, pv, pv_start, steps=(2, 4, 8, 16)):
        """Newton along a homotopy from a solved point (pv_start, x0) to pv."""
        for n in steps:
            x = x0.copy()
            ok = True
            it_tot = 0
            for i in range(1, n + 1):
                pv_i = pv_start + (pv - pv_start) * (i / n)
                x, status, it = self._solve_newton(x, pv_i)
                it_tot += it
                if np.abs(np.array(self.g_fun(x, pv_i)).ravel()).max() >= self.tol:
                    ok = False
                    break
            if ok:
                return x, f"continuation ({n} steps)", it_tot
        return x, "continuation failed", it_tot

    # ----------------------------------------------------------------- solve
    def solve(self, z_guess: np.ndarray, p: np.ndarray, z_start: np.ndarray | None = None,
              p_start: np.ndarray | None = None) -> SolveResult:
        """Solve from the initial guess ``z_guess`` (whose fixed entries are the
        exogenous values).  ``z_start``/``p_start`` optionally give a point that
        is known to solve the system; it anchors the continuation fallback."""
        t0 = time.perf_counter()
        x0 = z_guess[self.endo]
        pv = np.concatenate([z_guess[self.fix_idx], p])
        r0 = np.abs(np.array(self.g_fun(x0, pv)).ravel())
        if r0.size == 0 or r0.max() < self.tol * 1e-2:
            return SolveResult(z_guess.copy(), True, "initial point solves system", 0,
                               float(r0.max() if r0.size else 0.0), time.perf_counter() - t0, "none")
        methods = {"auto": ["newton", "continuation", "ipopt"], "ipopt": ["ipopt"],
                   "newton": ["newton", "continuation"]}[self.method]
        if z_start is not None and p_start is not None:
            pv_start = np.concatenate([z_start[self.fix_idx], p_start])
            x_start = z_start[self.endo]
        else:
            pv_start, x_start = None, None
        last = None
        for m in methods:
            try:
                if m == "newton":
                    xs, status, iters = self._solve_newton(x0, pv)
                elif m == "continuation":
                    if pv_start is None:
                        continue
                    xs, status, iters = self._solve_continuation(x_start, pv, pv_start)
                else:
                    xs, status, iters = self._solve_ipopt(x0, pv)
            except RuntimeError as exc:
                last = f"{m}: {exc}"
                continue
            res = np.abs(np.array(self.g_fun(xs, pv)).ravel())
            maxres = float(res.max()) if res.size else 0.0
            if np.isfinite(maxres) and maxres < self.tol:
                z = z_guess.copy()
                z[self.endo] = xs
                return SolveResult(z, True, status, iters, maxres, time.perf_counter() - t0, m)
            last = f"{m}: {status}, max residual {maxres:.3e}"
        return SolveResult(z_guess.copy(), False, str(last), -1, float("nan"), time.perf_counter() - t0, self.method)
