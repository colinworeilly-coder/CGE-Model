"""Square-system diagnostics: counts and structural rank with named culprits.

When a closure is mis-specified the model is either non-square (more free
variables than equations or vice versa) or square but structurally singular
(some subset of equations cannot determine the variables they contain).  Both
situations are reported with the names of the variable and equation blocks
involved, which makes changing closures safe.
"""

from __future__ import annotations

from collections import Counter

import casadi as ca
import numpy as np


class ClosureError(RuntimeError):
    pass


def _max_matching(n_rows: int, n_cols: int, adj: list[list[int]]):
    """Maximum bipartite matching (augmenting paths, iterative DFS).

    ``adj[r]`` lists the columns appearing in row ``r``.  Returns
    ``(match_row, match_col)`` arrays with -1 for unmatched.
    """
    match_row = -np.ones(n_rows, dtype=int)
    match_col = -np.ones(n_cols, dtype=int)
    # greedy initialisation
    for r in range(n_rows):
        for c in adj[r]:
            if match_col[c] < 0:
                match_row[r] = c
                match_col[c] = r
                break
    for r0 in range(n_rows):
        if match_row[r0] >= 0:
            continue
        visited = np.zeros(n_cols, dtype=bool)
        # stack of (row, iterator position); parent pointers for path recovery
        stack = [(r0, 0)]
        parent_col = {}
        found = -1
        while stack:
            r, pos = stack[-1]
            if pos >= len(adj[r]):
                stack.pop()
                continue
            stack[-1] = (r, pos + 1)
            c = adj[r][pos]
            if visited[c]:
                continue
            visited[c] = True
            parent_col[c] = r
            if match_col[c] < 0:
                found = c
                break
            stack.append((match_col[c], 0))
        if found >= 0:
            c = found
            while True:
                r = parent_col[c]
                prev = match_row[r]
                match_row[r] = c
                match_col[c] = r
                if r == r0:
                    break
                c = prev
    return match_row, match_col


def check_square(core, fixed: np.ndarray, raise_error: bool = True) -> str:
    """Check that the system with the given fixed mask is square and
    structurally non-singular.  Returns a report string (empty when fine)."""
    endo = np.flatnonzero(~fixed)
    n_free, n_eq = endo.size, core.neq
    var_blocks = np.array([lab.split("[")[0] for lab in core.var_labels], dtype=object)
    lines = []
    if n_free != n_eq:
        lines.append(f"System is not square: {n_free} free variables vs {n_eq} equations "
                     f"({'too many free variables' if n_free > n_eq else 'too many equations'}).")
        fv = Counter(var_blocks[endo])
        fe = Counter(core.eq_blocks)
        lines.append("Free variable elements by block: " + ", ".join(f"{k}={v}" for k, v in fv.items()))
        lines.append("Equation elements by block: " + ", ".join(f"{k}={v}" for k, v in fe.items()))
    else:
        sp = core.jacobian_sparsity()
        rows, cols = sp.get_triplet()
        rows = np.asarray(rows)
        cols = np.asarray(cols)
        colmap = -np.ones(core.nz, dtype=int)
        colmap[endo] = np.arange(endo.size)
        keep = colmap[cols] >= 0
        rows, cols = rows[keep], colmap[cols[keep]]
        adj = [[] for _ in range(n_eq)]
        for r, c in zip(rows, cols):
            adj[r].append(int(c))
        mr, mc = _max_matching(n_eq, n_free, adj)
        if (mr < 0).any():
            bad_eq = [core.eq_labels[i] for i in np.flatnonzero(mr < 0)]
            bad_var = [core.var_labels[endo[j]] for j in np.flatnonzero(mc < 0)]
            lines.append("System is square but structurally singular.")
            lines.append(f"Unmatched equations ({len(bad_eq)}): " + ", ".join(bad_eq[:25]))
            lines.append(f"Unmatched free variables ({len(bad_var)}): " + ", ".join(bad_var[:25]))
            lines.append("Typically a variable that should be free is fixed (or vice versa) in the closure.")
    report = "\n".join(lines)
    if report and raise_error:
        raise ClosureError(report)
    return report
