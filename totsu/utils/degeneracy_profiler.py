from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from pyomo.core import Var, Constraint, Objective, value
from pyomo.repn.standard_repn import generate_standard_repn

try:
    import scipy.sparse as sp
    import scipy.linalg as spla
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False
    sp = None
    spla = None

if not SCIPY_AVAILABLE:
    print("[DegeneracyProfiler] SciPy not found — using NumPy dense mode.")

# ----------------------------
# Helpers for Pyomo extraction
# ----------------------------

def _is_linear(expr) -> bool:
    # Pyomo linear check (cheap/robust)
    try:
        return expr.polynomial_degree() in (0, 1)
    except Exception:
        return False

@dataclass
class MatrixData:
    A: np.ndarray
    b: np.ndarray
    sense: np.ndarray  # -1: >=, 0: ==, +1: <=
    c: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    var_names: List[str]

def extract_matrix(model) -> MatrixData:
    """
    Linear extraction using Pyomo's standard representation.
    No SciPy, no PyomoNLP. Works for linear models.
    """
    # ---- Variables in a fixed order ----
    vars_list = [v for v in model.component_data_objects(Var, active=True)]
    var_names = [v.name for v in vars_list]
    n = len(vars_list)
    var_index: Dict[int, int] = {id(v): i for i, v in enumerate(vars_list)}

    # Bounds
    lb = np.array([(-np.inf if v.lb is None else float(value(v.lb))) for v in vars_list], dtype=float)
    ub = np.array([( np.inf if v.ub is None else float(value(v.ub))) for v in vars_list], dtype=float)

    # ---- Objective c ----
    objs = list(model.component_data_objects(Objective, active=True))
    if len(objs) != 1:
        raise RuntimeError(f"Expected exactly one active Objective, got {len(objs)}.")
    obj = objs[0]

    repn = generate_standard_repn(obj.expr, compute_values=True)
    if not repn.is_linear():
        raise RuntimeError("Objective is not linear.")

    c = np.zeros(n, dtype=float)
    if repn.linear_vars is not None:
        for coef, var in zip(repn.linear_coefs, repn.linear_vars):
            c[var_index[id(var)]] += float(coef)

    # ---- Constraints A, b, sense ----
    rows: List[Dict[int, float]] = []
    sense: List[int] = []   # +1 <=, 0 ==, -1 >=
    b: List[float] = []

    def push_row(coef_map: Dict[int, float], sgn: int, rhs: float):
        rows.append(coef_map)
        sense.append(sgn)
        b.append(rhs)

    def lin_row(expr) -> Tuple[float, Dict[int, float]]:
        rep = generate_standard_repn(expr, compute_values=True)
        if not rep.is_linear():
            raise RuntimeError("Non-linear constraint detected.")
        coef_map: Dict[int, float] = {}
        if rep.linear_vars is not None:
            for coef, var in zip(rep.linear_coefs, rep.linear_vars):
                coef_map[var_index[id(var)]] = coef_map.get(var_index[id(var)], 0.0) + float(coef)
        const = float(rep.constant if rep.constant is not None else 0.0)
        return const, coef_map

    for con in model.component_data_objects(Constraint, active=True):
        # Equality: body == lower
        if con.equality:
            const, coef_map = lin_row(con.body - con.lower)
            # (body - lower) == 0  ->  treat as == with rhs 0
            push_row(coef_map, 0, -const)
        else:
            # body <= ub  -> (body - ub) <= 0
            if con.has_ub():
                const, coef_map = lin_row(con.body - con.upper)
                push_row(coef_map, +1, -const)
            # body >= lb  -> (body - lb) >= 0
            if con.has_lb():
                const, coef_map = lin_row(con.body - con.lower)
                push_row(coef_map, -1, -const)

    m = len(rows)
    A = np.zeros((m, n), dtype=float)
    for i, rowmap in enumerate(rows):
        for j, v in rowmap.items():
            A[i, j] = float(v)

    return MatrixData(
        A=A,
        b=np.array(b, dtype=float),
        sense=np.array(sense, dtype=int),
        c=c,
        lb=lb,
        ub=ub,
        var_names=var_names,
    )

# ----------------------------
# Degeneracy profiler
# ----------------------------

@dataclass
class StructureStats:
    m: int
    n: int
    rank: int
    rank_deficiency: int
    duplicate_rows: int
    duplicate_cols: int
    near_duplicate_cols: int
    col_norm_spread: Tuple[float, float, float]  # (min, median, max)


@dataclass
class SolutionStats:
    tight_constraints: int
    tight_ratio: float
    zero_basic_count: Optional[int] = None
    reduced_cost_ties: Optional[int] = None
    min_reduced_cost: Optional[float] = None
    median_abs_reduced_cost: Optional[float] = None


class DegeneracyProfiler:
    def __init__(self, md: MatrixData, tol: float = 1e-9):
        self.md = md
        self.tol = tol

    # ---------- Static analysis ----------
    def analyze_structure(self) -> StructureStats:
        A = self.md.A
        m, n = (A.shape if SCIPY_AVAILABLE else A.shape)

        # Rank (use dense SVD when small; fallback to approx)
        if SCIPY_AVAILABLE and (max(m, n) > 1500):
            # approximate rank via svds
            k = min(50, min(m, n)-1) if min(m, n) > 1 else 0
            if k >= 1:
                u, s, vt = sp.linalg.svds(A.astype(float), k=k)  # smallest k singulars
                # We don't have the largest, so conservative: assume remaining are > tol
                # Build a heuristic cut
                rank = int(np.sum(s > self.tol))
            else:
                rank = int(min(m, n))  # tiny case
        else:
            # dense route
            if SCIPY_AVAILABLE:
                A_dense = A.toarray()
            else:
                A_dense = A
            s = np.linalg.svd(A_dense, compute_uv=False)
            rank = int(np.sum(s > self.tol))

        # Duplicate rows/cols: exact hash on sparse pattern + values
        dup_rows = self._count_duplicate_rows()
        dup_cols, near_dup_cols = self._count_duplicate_columns(near=True)

        # Column norm spread (scaling indicator)
        col_norms = self._column_norms()
        col_norm_spread = (float(np.min(col_norms)),
                           float(np.median(col_norms)),
                           float(np.max(col_norms)))

        return StructureStats(
            m=m,
            n=n,
            rank=rank,
            rank_deficiency=m - rank,
            duplicate_rows=dup_rows,
            duplicate_cols=dup_cols,
            near_duplicate_cols=near_dup_cols,
            col_norm_spread=col_norm_spread,
        )

    def _column_norms(self) -> np.ndarray:
        A = self.md.A
        if SCIPY_AVAILABLE and sp.issparse(A):
            return np.sqrt(A.power(2).sum(axis=0)).A1 + 0.0
        else:
            return np.sqrt((A * A).sum(axis=0))

    def _count_duplicate_rows(self) -> int:
        A = self.md.A
        if SCIPY_AVAILABLE and sp.issparse(A):
            A_coo = A.tocoo()
            buckets: Dict[str, int] = {}
            for i in range(A.shape[0]):
                row = A.getrow(i)
                key = (tuple(row.indices.tolist()), tuple(np.round(row.data, 12).tolist()))
                buckets[str(key)] = buckets.get(str(key), 0) + 1
            return int(sum(max(0, c-1) for c in buckets.values()))
        else:
            rows = [tuple(np.round(A[i, :], 12).tolist()) for i in range(A.shape[0])]
            from collections import Counter
            cnt = Counter(rows)
            return int(sum(max(0, c-1) for c in cnt.values()))

    def _count_duplicate_columns(self, near=False) -> Tuple[int, int]:
        A = self.md.A
        if SCIPY_AVAILABLE and sp.issparse(A):
            A_csc = A.tocsc()
            keys = {}
            for j in range(A.shape[1]):
                col = A_csc.getcol(j)
                key = (tuple(col.indices.tolist()), tuple(np.round(col.data, 12).tolist()))
                keys.setdefault(str(key), []).append(j)
            dup_exact = sum(max(0, len(v)-1) for v in keys.values())

            near_dups = 0
            if near:
                # normalize columns by L2 and round to group near-identical up to scale
                norms = self._column_norms()
                sig = {}
                for j in range(A.shape[1]):
                    col = A_csc.getcol(j)
                    if norms[j] < 1e-15:
                        key = ("zero",)
                    else:
                        vals = np.round(col.data / norms[j], 6)  # 1e-6 tolerance
                        key = (tuple(col.indices.tolist()), tuple(vals.tolist()))
                    sig.setdefault(str(key), []).append(j)
                near_dups = sum(max(0, len(v)-1) for v in sig.values())
            return int(dup_exact), int(near_dups)
        else:
            cols = [tuple(np.round(A[:, j], 12).tolist()) for j in range(A.shape[1])]
            from collections import Counter
            cnt = Counter(cols)
            dup_exact = sum(max(0, c-1) for c in cnt.values())
            # near-dup detection on dense (normalized)
            norms = np.linalg.norm(A, axis=0) + 0.0
            cols_norm = []
            for j in range(A.shape[1]):
                if norms[j] < 1e-15:
                    cols_norm.append(("zero",))
                else:
                    cols_norm.append(tuple(np.round(A[:, j] / norms[j], 6).tolist()))
            cnt2 = Counter(cols_norm)
            near_dups = sum(max(0, c-1) for c in cnt2.values())
            return int(dup_exact), int(near_dups)

    # ---------- Dynamic analysis ----------
    def analyze_solution(
        self,
        x: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        basis: Optional[Sequence[int]] = None,
        reduced_costs: Optional[np.ndarray] = None,
    ) -> SolutionStats:
        """
        x: primal point (np.ndarray of length n). If None, tight constraints use x=0 for a quick check against lb/ub.
        y: duals for rows (same order as A rows). If provided and reduced_costs is None, rc = c - A^T y.
        basis: indices of basic variables (for zero-basic count).
        reduced_costs: direct rc vector (overrides y if given).
        """
        A, b, sense, c = self.md.A, self.md.b, self.md.sense, self.md.c
        n = c.shape[0]

        # Tight constraints at x
        if x is None:
            x = np.zeros(n)
        if SCIPY_AVAILABLE and sp.issparse(A):
            Ax = A @ x
        else:
            Ax = A.dot(x)

        # Slack definition depends on sense:
        #   <=: b - Ax >= 0; tight if ≈ 0
        #   ==: |b - Ax| tight if ≈ 0
        #   >=: Ax - b >= 0; tight if ≈ 0
        slack = np.empty_like(b)
        for i, s in enumerate(sense):
            if s > 0:    # <=
                slack[i] = b[i] - Ax[i]
            elif s == 0:  # ==
                slack[i] = abs(b[i] - Ax[i])
            else:        # >=
                slack[i] = Ax[i] - b[i]

        tight_mask = np.isfinite(slack) & (np.abs(slack) <= 1e-8)
        tight_constraints = int(np.sum(tight_mask))
        tight_ratio = float(tight_constraints / len(b)) if len(b) else 0.0

        # Zero-valued basics
        zero_basic_count = None
        if basis is not None:
            zero_basic_count = int(sum(abs(x[j]) <= 1e-12 for j in basis))

        # Reduced costs
        rc_vec = None
        if reduced_costs is not None:
            rc_vec = np.array(reduced_costs, dtype=float)
        elif y is not None:
            if SCIPY_AVAILABLE and sp.issparse(A):
                ATy = A.T @ y
            else:
                ATy = A.T.dot(y)
            rc_vec = c - ATy

        reduced_cost_ties = None
        min_rc = None
        median_abs_rc = None
        if rc_vec is not None:
            # How many are ~0?
            reduced_cost_ties = int(np.sum(np.abs(rc_vec) <= 1e-10))
            min_rc = float(np.min(rc_vec))
            median_abs_rc = float(np.median(np.abs(rc_vec)))

        return SolutionStats(
            tight_constraints=tight_constraints,
            tight_ratio=tight_ratio,
            zero_basic_count=zero_basic_count,
            reduced_cost_ties=reduced_cost_ties,
            min_reduced_cost=min_rc,
            median_abs_reduced_cost=median_abs_rc,
        )

    # ---------- Report ----------
    def report(self, struct: StructureStats, sol: Optional[SolutionStats] = None) -> str:
        lines = []
        lines.append("=== Degeneracy Profile ===")
        lines.append(f"Size: m={struct.m} rows, n={struct.n} cols")
        lines.append(f"Rank: {struct.rank} (deficiency: {struct.rank_deficiency})")
        lines.append(f"Duplicate rows: {struct.duplicate_rows}")
        lines.append(f"Duplicate columns (exact): {struct.duplicate_cols}")
        lines.append(f"Near-duplicate columns (normalized): {struct.near_duplicate_cols}")
        mn, md, mx = struct.col_norm_spread
        lines.append(f"Column L2 norms: min={mn:.3e}, median={md:.3e}, max={mx:.3e}")
        if sol is not None:
            lines.append("--- Solution snapshot ---")
            lines.append(f"Tight constraints: {sol.tight_constraints} ({100*sol.tight_ratio:.1f}%)")
            if sol.zero_basic_count is not None:
                lines.append(f"Zero-valued basics: {sol.zero_basic_count}")
            if sol.reduced_cost_ties is not None:
                lines.append(f"Reduced-cost ~0 ties: {sol.reduced_cost_ties}")
            if sol.min_reduced_cost is not None:
                lines.append(f"Min reduced cost: {sol.min_reduced_cost:.3e}")
            if sol.median_abs_reduced_cost is not None:
                lines.append(f"Median |reduced cost|: {sol.median_abs_reduced_cost:.3e}")
        # Guidance
        lines.append("--- Guidance ---")
        if struct.duplicate_cols + struct.near_duplicate_cols > 0:
            lines.append("• Many identical/near-identical columns → expect reduced-cost ties; consider small cost perturbation ε, Devex/steepest-edge pricing, or symmetry-breaking.")
        if struct.rank_deficiency > 0 or (sol and sol.tight_ratio > 0.2):
            lines.append("• Rank deficiency or many tight constraints → geometric degeneracy; lexicographic ratio test + scaling can help.")
        if struct.col_norm_spread[2] / max(struct.col_norm_spread[0], 1e-12) > 1e6:
            lines.append("• Huge column-norm spread → scaling strongly recommended.")
        if sol and sol.reduced_cost_ties and sol.reduced_cost_ties > 0.05 * struct.n:
            lines.append("• Many ~0 reduced costs → plateau; try ε-perturbation of costs or alternate pricing.")
        return "\n".join(lines)
