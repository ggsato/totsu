# demo_4way_assignment.py
# A tiny 4-way assignment (workers × customers × days × skills)
# Uses AdvancedBranchAndBound with family-aware rounding and best-bound search.

from collections import defaultdict
from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Binary, Objective, Constraint, summation, value, minimize, Reals
)

from totsu.core.advanced_branch_and_bound_solver import AdvancedBranchAndBound, Families
from totsu.utils.model_processor import ModelProcessor  # your helper used by the B&B class

def build_toy_model():
    m = ConcreteModel()

    m.W = Set(initialize=["w1", "w2"])          # workers
    m.C = Set(initialize=["c1", "c2"])          # customers
    m.D = Set(initialize=["d1", "d2"])          # days
    m.S = Set(initialize=["s1"])                # skills

    base_cost = {
        ("w1","c1","d1","s1"): 7,
        ("w1","c1","d2","s1"): 6,
        ("w1","c2","d1","s1"): 5,
        ("w1","c2","d2","s1"): 8,
        ("w2","c1","d1","s1"): 6,
        ("w2","c1","d2","s1"): 5,
        ("w2","c2","d1","s1"): 7,
        ("w2","c2","d2","s1"): 4,
    }
    m.cost = Param(m.W, m.C, m.D, m.S, initialize=base_cost, within=Reals)

    m.x_index = m.W * m.C * m.D * m.S
    m.x = Var(m.x_index, domain=Binary)

    def obj_rule(m):
        return sum(m.cost[w,c,d,s] * m.x[w,c,d,s] for (w,c,d,s) in m.x_index)
    m.obj = Objective(rule=obj_rule, sense=minimize)

    def at_most_one_customer(m, w, d, s):
        return sum(m.x[w,c,d,s] for c in m.C) == 1
    m.worker_day_skill_cap = Constraint(m.W, m.D, m.S, rule=at_most_one_customer)

    def at_most_one_worker(m, c, d, s):
        return sum(m.x[w,c,d,s] for w in m.W) <= 1
    m.customer_day_skill_cap = Constraint(m.C, m.D, m.S, rule=at_most_one_worker)

    return m

def build_families(m):
    """
    Families tell the solver which binaries form 'choose ≤ 1' or '= 1' groups.
    Here: for each (w,d,s), choose at most one customer c  ⇒ a ≤1 family.
    """
    groups = defaultdict(list)
    for w in m.W:
        for d in m.D:
            for s in m.S:
                key = ("wds", w, d, s)
                for c in m.C:
                    var_name = f"x[{w},{c},{d},{s}]"
                    groups[key].append(var_name)

    # equals_one=False → these families are '≤ 1' groups
    return Families(groups=dict(groups), equals_one=False)

def main():
    model = build_toy_model()

    # Families enable the fast incumbent heuristic
    fam = build_families(model)

    # Create the solver with helpful defaults
    solver = AdvancedBranchAndBound(
        strong_branch_k=6,
        strong_branch_depth=2,
        node_selection="best_bound",
        mip_gap=0.0,             # absolute gap; try 1.0 if your model uses large costs
        time_limit_s=30,         # keep short for demo; increase for real runs
        node_limit=None,
        families=fam,
        branch_priority=None,    # you can pass a dict: {"x[w1,c2,d1,s1]": 10, ...}
        log_every=200
    )

    # Solve
    sol, obj = solver.solve(model)

    # Report
    print("\n=== Best objective ===")
    print(obj)
    print("\n=== Nonzero assignments (x=1) ===")
    if sol is None:
        print("No incumbent found.")
        return

    # Pretty print decisions
    for (w,c,d,s) in model.x_index:
        name = f"x[{w},{c},{d},{s}]"
        if sol.get(name, 0.0) > 0.5:
            print(f"{name} = 1  (cost={value(model.cost[w,c,d,s])})")

if __name__ == "__main__":
    main()
