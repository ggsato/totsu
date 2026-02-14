# demo_4way_assignment_scaled.py
# Scalable 4-way assignment (workers × customers × days × skills)
# - build_toy_model()      : tiny
# - build_large_model()    : medium/large
# - build_massive_model()  : bigger (still demo-friendly)
#
# You need the AdvancedBranchAndBound class available on PYTHONPATH.
# If not, paste the class into this file above the imports below.

import argparse
from collections import defaultdict
import random

from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Binary, Objective, Constraint, value, minimize,
    Reals
)
from pyomo.repn.standard_repn import generate_standard_repn

# --- your advanced B&B (make sure this import path matches your project) ---
from totsu.core.advanced_branch_and_bound_solver import AdvancedBranchAndBound, Families
# If you don’t have ModelProcessor in that module, your AdvancedB&B should not require it;
# or modify AdvancedB&B to not depend on it. Many users wire directly via Pyomo names.


# ----------------------------
# Utilities
# ----------------------------
def make_costs(W, C, D, S, seed=0, noise=3, day_trend=True, worker_affinity=True):
    """
    Structured random costs:
      - base = 10
      - day_trend: later days slightly more expensive
      - worker_affinity: each worker is cheaper for a subrange of customers
      - plus small random noise in [0, noise]
    """
    rng = random.Random(seed)
    costs = {}
    base = 10
    # affinity: partition customers for each worker
    C_list = list(C)
    W_list = list(W)
    D_list = list(D)
    S_list = list(S)
    chunk = max(1, len(C_list) // max(1, len(W_list)))
    worker_block = {w: set(C_list[i*chunk:(i+1)*chunk]) for i, w in enumerate(W_list)}
    last_day_index = len(D_list) - 1

    for w in W_list:
        for c in C_list:
            for di, d in enumerate(D_list):
                for s in S_list:
                    val = base
                    if day_trend:
                        # later days a bit more expensive
                        val += 0.5 * di
                    if worker_affinity and c in worker_block.get(w, set()):
                        # cheaper if in worker's "affinity" block
                        val -= 2.5
                    val += rng.uniform(0, noise)
                    costs[(w, c, d, s)] = float(val)
    return costs


def build_exact_cover_constraints(m):
    # Each (c,d,s) must be served by exactly one worker
    def exactly_one_worker(m, c, d, s):
        return sum(m.x[w, c, d, s] for w in m.W) == 1
    m.customer_day_skill_eq = Constraint(m.C, m.D, m.S, rule=exactly_one_worker)

    # Per (w,d): at most one assignment across all c,s  (tightens LP a lot)
    def worker_day_cap(m, w, d):
        return sum(m.x[w, c, d, s] for c in m.C for s in m.S) <= 1
    m.worker_day_cap = Constraint(m.W, m.D, rule=worker_day_cap)


def build_soft_cover_constraints(m, penalty=1000.0):
    # y[c,d,s] == 1 if unserved; pay penalty
    m.y = Var(m.C, m.D, m.S, domain=Binary)

    def cover_or_pay(m, c, d, s):
        return sum(m.x[w, c, d, s] for w in m.W) + m.y[c, d, s] == 1
    m.cover_or_pay = Constraint(m.C, m.D, m.S, rule=cover_or_pay)

    # Per (w,d): at most one assignment across all c,s
    def worker_day_cap(m, w, d):
        return sum(m.x[w, c, d, s] for c in m.C for s in m.S) <= 1
    m.worker_day_cap = Constraint(m.W, m.D, rule=worker_day_cap)

    def obj_rule(m):
        lin = sum(m.cost[w, c, d, s] * m.x[w, c, d, s] for (w, c, d, s) in m.x_index)
        pen = penalty * sum(m.y[c, d, s] for c in m.C for d in m.D for s in m.S)
        return lin + pen
    # remove any existing objective first
    if hasattr(m, "obj"):
        m.del_component(m.obj)
    m.obj = Objective(rule=obj_rule, sense=minimize)


def build_families_worker_day(m):
    """
    Families keyed by (w,d) across ALL c,s  → choose ≤ 1.
    This makes the incumbent rounding/repair much stronger than (w,d,s) families.
    """
    groups = defaultdict(list)
    for w in m.W:
        for d in m.D:
            key = ("wd", w, d)
            for c in m.C:
                for s in m.S:
                    groups[key].append(f"x[{w},{c},{d},{s}]")
    return Families(groups=dict(groups), equals_one=False)


# ----------------------------
# Models
# ----------------------------
def build_toy_model(seed=0, exact_cover=True):
    m = ConcreteModel()
    m.W = Set(initialize=["w1", "w2"]) # w >= c * s for exact_cover
    m.C = Set(initialize=["c1", "c2"])
    m.D = Set(initialize=["d1", "d2"])
    m.S = Set(initialize=["s1"])

    cost = make_costs(m.W, m.C, m.D, m.S, seed=seed)
    m.cost = Param(m.W, m.C, m.D, m.S, initialize=cost, within=Reals)

    m.x_index = m.W * m.C * m.D * m.S
    m.x = Var(m.x_index, domain=Binary)

    # base objective (may be overridden in soft-cover)
    def obj_rule(m):
        return sum(m.cost[w, c, d, s] * m.x[w, c, d, s] for (w, c, d, s) in m.x_index)
    m.obj = Objective(rule=obj_rule, sense=minimize)

    if exact_cover:
        build_exact_cover_constraints(m)
    else:
        build_soft_cover_constraints(m, penalty=1000.0)

    return m

def build_medium_model(seed=0, exact_cover=True, W=18, C=9, D=4, S=2):
    """
    About 18*9*4*2 = 1,296 binaries. 2^1,296 possible solutions!
    ✔ Scale intuition
    Atoms in the observable universe: 10⁸⁰
    Medium model combinations: 10³⁹⁰
    """
    m = ConcreteModel()
    m.W = Set(initialize=[f"w{i+1}" for i in range(W)])  # w >= c * s for exact_cover
    m.C = Set(initialize=[f"c{i+1}" for i in range(C)])
    m.D = Set(initialize=[f"d{i+1}" for i in range(D)])
    m.S = Set(initialize=[f"s{i+1}" for i in range(S)])

    cost = make_costs(m.W, m.C, m.D, m.S, seed=seed, noise=2.5, day_trend=True, worker_affinity=True)
    m.cost = Param(m.W, m.C, m.D, m.S, initialize=cost, within=Reals)

    m.x_index = m.W * m.C * m.D * m.S
    m.x = Var(m.x_index, domain=Binary)

    def obj_rule(m):
        return sum(m.cost[w, c, d, s] * m.x[w, c, d, s] for (w, c, d, s) in m.x_index)
    m.obj = Objective(rule=obj_rule, sense=minimize)

    if exact_cover:
        build_exact_cover_constraints(m)
    else:
        build_soft_cover_constraints(m, penalty=900.0)  # slightly smaller penalty for scale

    return m


def build_large_model(seed=0, exact_cover=True, W=54, C=18, D=8, S=3):
    """
    About 54*18*8*3 = 23,328 binaries. => 2^23,328 possible solutions!
    ✔ Scale intuition
    Earth's sand grains: 10¹⁹
    Observable universe atoms: 10⁸⁰
    Large model: 10⁷⁰²⁴
    """
    m = ConcreteModel()
    m.W = Set(initialize=[f"w{i+1}" for i in range(W)]) # w >= c * s for exact_cover
    m.C = Set(initialize=[f"c{i+1}" for i in range(C)])
    m.D = Set(initialize=[f"d{i+1}" for i in range(D)])
    m.S = Set(initialize=[f"s{i+1}" for i in range(S)])

    cost = make_costs(m.W, m.C, m.D, m.S, seed=seed, noise=2.5, day_trend=True, worker_affinity=True)
    m.cost = Param(m.W, m.C, m.D, m.S, initialize=cost, within=Reals)

    m.x_index = m.W * m.C * m.D * m.S
    m.x = Var(m.x_index, domain=Binary)

    def obj_rule(m):
        return sum(m.cost[w, c, d, s] * m.x[w, c, d, s] for (w, c, d, s) in m.x_index)
    m.obj = Objective(rule=obj_rule, sense=minimize)

    if exact_cover:
        build_exact_cover_constraints(m)
    else:
        build_soft_cover_constraints(m, penalty=800.0)  # slightly smaller penalty for scale

    return m


def build_massive_model(seed=0, exact_cover=False, W=90, C=30, D=12, S=3):
    """
    About 90*30*12*3 = 97,200 binaries. => 2^97,200 possible solutions!
    This number is so large that:
    * There aren’t enough particles in any reachable region of spacetime to even represent one index into the search tree.
    * Not even hypothetical “planet-sized quantum computers” can brute-force it.
    This is far beyond any combinatorial space used in cryptography.
    For reference:
    * Breaking 256-bit AES requires ~10⁷⁷ operations.
    * This “massive” model requires ~10²⁹,²⁷⁰ operations.
      (~10²⁹,²⁰⁰ times harder.)
    """
    m = ConcreteModel()
    m.W = Set(initialize=[f"w{i+1}" for i in range(W)]) # w >= c * s for exact_cover
    m.C = Set(initialize=[f"c{i+1}" for i in range(C)])
    m.D = Set(initialize=[f"d{i+1}" for i in range(D)])
    m.S = Set(initialize=[f"s{i+1}" for i in range(S)])

    cost = make_costs(m.W, m.C, m.D, m.S, seed=seed, noise=2.0, day_trend=True, worker_affinity=True)
    m.cost = Param(m.W, m.C, m.D, m.S, initialize=cost, within=Reals)

    m.x_index = m.W * m.C * m.D * m.S
    m.x = Var(m.x_index, domain=Binary)

    def obj_rule(m):
        return sum(m.cost[w, c, d, s] * m.x[w, c, d, s] for (w, c, d, s) in m.x_index)
    m.obj = Objective(rule=obj_rule, sense=minimize)

    if exact_cover:
        build_exact_cover_constraints(m)
    else:
        build_soft_cover_constraints(m, penalty=600.0)

    return m


# ----------------------------
# Runner
# ----------------------------
def run(size: str, exact_cover: bool, seed: int, time_limit: int, gap: float,
        strong_k: int, sb_depth: int, simplex_max_itr: int):
    if size == "toy":
        model = build_toy_model(seed=seed, exact_cover=exact_cover)
    elif size == "medium":
        model = build_medium_model(seed=seed, exact_cover=exact_cover)
    elif size == "large":
        model = build_large_model(seed=seed, exact_cover=exact_cover)
    elif size == "massive":
        model = build_massive_model(seed=seed, exact_cover=exact_cover)
    else:
        raise ValueError("size must be one of: toy | large | massive")

    const = objective_constant(model)
    print(f"Objective constant term (pre-solve): {const:.6f}")  

    # Families: (w,d) groups across all c,s (≤1)
    fam = build_families_worker_day(model)

    solver = AdvancedBranchAndBound(
        strong_branch_k=strong_k,        # try 6–16
        strong_branch_depth=sb_depth,    # 1–2 is usually enough
        node_selection="best_bound",
        mip_gap=gap,                     # ABSOLUTE gap. E.g., 5.0 or 10.0 on larger models
        time_limit_s=time_limit,
        families=fam,
        branch_priority=None,
        log_every=500,
        simplex_max_itr=simplex_max_itr,          # NEW
        simplex_backoff=[simplex_max_itr, 5000, 12000],  # NEW
    )

    sol, obj = solver.solve(model)
    print("\n--- Progress (time, bestBound, incumbent) ---")
    def fmt(v):
        return f"{v:.3f}" if v is not None else "---"
    for t, bnd, inc in solver.history[:25]:  # first few points
        print(f"{t:.1f}s  bound={fmt(bnd)}  inc={fmt(inc)}")

    print("... total points:", len(solver.history))

    print("Objective (evaluated on model):", value(model.obj))
    print("Objective (reported by B&B):   ", obj)

    print("\n=== Best objective ===")
    print(obj)
    print("\n=== Some nonzero assignments (x=1) ===")
    if sol is None:
        print("No incumbent found.")
        return

    shown = 0
    for (w, c, d, s) in model.x_index:
        name = f"x[{w},{c},{d},{s}]"
        if sol.get(name, 0.0) > 0.5:
            print(f"{name} = 1  (cost={value(model.cost[w,c,d,s])})")
            shown += 1
            if shown >= 40:  # avoid flooding output
                print("... (truncated)")
                break

    # 1) Count how many binaries are 1
    num_ones = sum(1 for k in model.x if value(model.x[k]) > 0.5)

    # 2) Recompute the objective directly from x and *the same* model.cost
    sum_cost = sum(value(model.cost[k]) * value(model.x[k]) for k in model.x)

    print(f"Chosen x=1 count: {num_ones}")
    print(f"Sum of printed costs over x=1: {sum_cost:.6f}")
    print(f"Objective from model          : {value(model.obj):.6f}")

    # 3) If there are cover penalties in obj, print them (should be 0 with exact_cover=True)
    penalty_terms = []
    for comp in model.component_objects():
        if comp.name.lower().startswith("penalty") or comp.name.lower().endswith("penalty"):
            try:
                penalty_val = value(comp)
                penalty_terms.append((comp.name, penalty_val))
            except:
                pass
    if penalty_terms:
        print("Penalty-like components:", penalty_terms)

    # 4) Check cover constraints (adjust to how you built them)
    def max_violation():
        viol = 0.0
        for c in model.component_data_objects(ctype=type(next(iter(model.component_data_objects()))).__class__, active=True):
            try:
                # generic: works if ‘body <=/==/>= bound’ shaped
                if hasattr(c, "lower") and c.lower is not None:
                    viol = max(viol, float(max(0.0, abs(value(c.lower) - value(c.body)))))
                if hasattr(c, "upper") and c.upper is not None:
                    viol = max(viol, float(max(0.0, abs(value(c.body) - value(c.upper)))))
            except:
                pass
        return viol
    print(f"Max constraint violation: {max_violation():.3e}")

    const = objective_constant(model)
    lin   = objective_linear_value(model)
    print(f"Objective constant: {const:.6f}")
    print(f"Linear part @ solution: {lin:.6f}")
    print(f"Const + Linear: {const + lin:.6f}  vs  value(m.obj)={value(model.obj):.6f}")

    objective_breakdown(model)

def objective_constant(model) -> float:
    """Safe before/after solve: returns the constant term only."""
    repn = generate_standard_repn(model.obj.expr, compute_values=True)
    return float(repn.constant or 0.0)

def objective_linear_value(model) -> float:
    """Use only AFTER solve: evaluates the linear part at current var values."""
    repn = generate_standard_repn(model.obj.expr, compute_values=True)
    if not repn.is_linear():
        raise RuntimeError("Objective is not linear.")
    lin = 0.0
    if repn.linear_vars is not None:
        for coef, var in zip(repn.linear_coefs, repn.linear_vars):
            # safe now because variables have values after solve
            lin += float(coef) * float(value(var))
    return lin

def objective_breakdown(model, top=20):
    repn = generate_standard_repn(model.obj.expr, compute_values=True)
    fam = defaultdict(lambda: {"terms": 0, "contrib": 0.0})

    linear_vars = repn.linear_vars or []
    linear_coefs = repn.linear_coefs or []
    contribs = []

    for coef, var in zip(linear_coefs, linear_vars):
        try:
            val = float(value(var))
        except Exception:
            val = 0.0
        c = float(coef) * val
        contribs.append((c, coef, var))

        base = var.parent_component().name  # e.g., "x", "y_cover", "slack_cover"
        fam[base]["terms"] += 1
        fam[base]["contrib"] += c

    total = sum(c for c,_,_ in contribs)

    print("\n=== Objective contribution by variable family ===")
    for name, info in sorted(fam.items(), key=lambda kv: -kv[1]["contrib"]):
        print(f"{name:25s}  contrib={info['contrib']:.6f}  terms={info['terms']}")

    print("\n=== Top individual contributors (coef*value) ===")
    for c, coef, var in sorted(contribs, key=lambda t: -t[0])[:top]:
        try:
            vval = float(value(var))
        except Exception:
            vval = float('nan')
        print(f"{var.name:40s}  coef={float(coef):.6f}  val={vval:.6f}  contrib={c:.6f}")

    print(f"\nTotal objective (linear part @ solution): {total:.6f}")

def main():
    """
    # Large, soft cover (default), 120s, abs gap 5.0
    python demo_4way_assignment_scaled.py --size large

    # Large + exact cover (tighter but may be harder)
    python demo_4way_assignment_scaled.py --size large --exact-cover --time 300 --gap 5

    # Massive (25,920 binaries), soft cover; give it more time and a looser gap
    python demo_4way_assignment_scaled.py --size massive --time 600 --gap 15 --strong-k 8 --sb-depth 1
    """

    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=["toy", "medium", "large", "massive"], default="medium")
    ap.add_argument("--exact-cover", action="store_true",
                    help="Use exact cover on (c,d,s)==1; else soft cover with penalty.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time", type=int, default=120, help="Time limit (seconds).")
    ap.add_argument("--gap", type=float, default=5.0,
                    help="ABSOLUTE mipgap for early stop (e.g., 5.0). Use 0 for no gap stop.")
    ap.add_argument("--strong-k", type=int, default=10, help="#candidates for strong-branching-lite.")
    ap.add_argument("--sb-depth", type=int, default=2, help="Depth at which to use strong-branching-lite.")
    ap.add_argument("--simplex-iters", type=int, default=2000, help="Initial simplex max iterations.")

    args = ap.parse_args()

    run(
        size=args.size,
        exact_cover=args.exact_cover,
        seed=args.seed,
        time_limit=args.time,
        gap=args.gap,
        strong_k=args.strong_k,
        sb_depth=args.sb_depth,
        simplex_max_itr=args.simplex_iters,
    )

if __name__ == "__main__":
    main()
