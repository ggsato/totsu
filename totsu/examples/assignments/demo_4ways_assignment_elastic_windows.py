from pyomo.environ import *
from typing import Callable
import logging

from totsu.core.advanced_branch_and_bound_solver import AdvancedBranchAndBound
from totsu.utils.elastic_feasibility_tool import ElasticFeasibilityTool, ElasticResult
from totsu.utils.logger import totsu_logger

def build_model(with_windows: bool = False) -> ConcreteModel:
    m = ConcreteModel()

    # Sets
    m.W = Set(initialize=["w1", "w2"])
    m.C = Set(initialize=["c1", "c2"])
    m.D = Set(initialize=[1,2,3])
    m.S = Set(initialize=[1,2])

    # Availability windows
    m.start = Param(m.C, initialize={"c1": 1, "c2": 2})
    m.end   = Param(m.C, initialize={"c1": 3, "c2": 3})

    # Forbidden assignment (public example)
    m.forbidden_pairs = Set(within=m.W * m.C,
        initialize=[("w1", "c2")]
    )

    # Decision variable
    m.x = Var(m.W, m.C, m.D, m.S, within=Binary)

    # Client must be served exactly once per day
    def client_demand_rule(m, c, d):
        return sum(m.x[w,c,d,s] for w in m.W for s in m.S) == 1

    m.client_demand = Constraint(m.C, m.D, rule=client_demand_rule)

    # Worker can do at most 1 shift per day
    worker_capacity = {"w1": 1, "w2": 1}
    def worker_daily_rule(m, w, d):
        return sum(m.x[w,c,d,s] for c in m.C for s in m.S) <= worker_capacity[w]
    m.worker_daily = Constraint(m.W, m.D, rule=worker_daily_rule)

    # Forbidden assignment rule (hard before elasticity)
    m.forbidden_constraint = ConstraintList()
    for (w,c) in m.forbidden_pairs:
        for d in m.D:
            for s in m.S:
                m.forbidden_constraint.add(m.x[w,c,d,s] == 0)

    if with_windows:
        def window_early_rule(m, c, d):
            # If day d is before the customer's window, forbid service.
            if d < m.start[c]:
                assigned = sum(m.x[w, c, d, s] for w in m.W for s in m.S)
                return assigned <= 0
            # Otherwise, nothing to enforce for "early".
            return Constraint.Skip

        def window_late_rule(m, c, d):
            # If day d is after the customer's window, forbid service.
            if d > m.end[c]:
                assigned = sum(m.x[w, c, d, s] for w in m.W for s in m.S)
                return assigned <= 0
            # Otherwise, nothing to enforce for "late".
            return Constraint.Skip

        m.window_early = Constraint(m.C, m.D, rule=window_early_rule)
        m.window_late  = Constraint(m.C, m.D, rule=window_late_rule)

    return m


def apply_elasticity(m, penalties):
    """
    penalties = dict with penalty multipliers:
        {
          "shortfall": 1000,
          "overtime": 100,
          "forbidden": 5000,
          "early": 800,
          "late": 800
        }
    """

    # New deviation variables
    m.s_short = Var(m.C, m.D, within=NonNegativeReals)
    m.e_ot = Var(m.W, m.D, within=NonNegativeReals)
    m.e_forbidden = Var(m.W, m.C, m.D, m.S, within=NonNegativeReals)

    # Early/Late window violations
    m.e_early = Var(m.C, m.D, within=NonNegativeReals)
    m.e_late  = Var(m.C, m.D, within=NonNegativeReals)

    # ========== Replace constraints with elastic versions ==========

    # Elastic client demand
    m.client_demand_elastic = ConstraintList()
    for c in m.C:
        for d in m.D:
            m.client_demand_elastic.add(
                sum(m.x[w,c,d,s] for w in m.W for s in m.S)
                + m.s_short[c,d] == 1
            )

    # Elastic worker capacity
    worker_capacity = {"w1": 1, "w2": 1}
    m.worker_daily_elastic = ConstraintList()
    for w in m.W:
        for d in m.D:
            m.worker_daily_elastic.add(
                sum(m.x[w,c,d,s] for c in m.C for s in m.S)
                - m.e_ot[w,d] <= worker_capacity[w]
            )

    # Elastic forbidden assignments
    m.forbidden_elastic = ConstraintList()
    for (w,c) in m.forbidden_pairs:
        for d in m.D:
            for s in m.S:
                m.forbidden_elastic.add(
                    m.x[w,c,d,s] <= m.e_forbidden[w,c,d,s]
                )

    # ========== Elastic start/end window constraints ==========

    m.window_elastic = ConstraintList()

    for c in m.C:
        for d in m.D:
            assigned = sum(m.x[w,c,d,s] for w in m.W for s in m.S)

            # Early service
            if d < m.start[c]:
                m.window_elastic.add(assigned <= m.e_early[c,d])

            # Late service
            if d > m.end[c]:
                m.window_elastic.add(assigned <= m.e_late[c,d])

    # ========== Elastic objective ==========
    m.obj_elastic = Objective(
        expr =
            penalties["shortfall"] * sum(m.s_short[c,d] for c in m.C for d in m.D)
          + penalties["overtime"] * sum(m.e_ot[w,d] for w in m.W for d in m.D)
          + penalties["forbidden"] * sum(
                m.e_forbidden[w,c,d,s]
                for w in m.W for c in m.C for d in m.D for s in m.S
            )
          + penalties["early"] * sum(m.e_early[c,d] for c in m.C for d in m.D)
          + penalties["late"]  * sum(m.e_late[c,d]  for c in m.C for d in m.D),
        sense=minimize
    )

    return m

def report_elastic_infeasibility_for_4way(
    model,
    result: ElasticResult,
    tol: float = 1e-6,
    printer: Callable[[str], None] = print,
) -> None:
    """
    Summarize infeasibility for the 4-way assignment (W–C–D–S) model.

    Answers:
      * Which customer's demand is unmet?
      * Which worker's availability is violated?
      * Which customers' windows are violated (by day)?

    It looks at dev.component_name and dev.index to categorize deviations.
    """

    def build_display_label(component_name, index):
        idx = tuple(index) if index is not None else ()

        if component_name in ("window_early", "window_late"):
            if len(idx) >= 2:
                return f"{component_name}[{idx[0]}, day {idx[1]}]"
            return component_name

        if component_name == "client_demand":
            if len(idx) >= 1:
                return f"demand_constraint[{idx[0]}]"
            return "demand_constraint"

        if idx:
            return f"{component_name}[{', '.join(str(v) for v in idx)}]"
        return component_name

    def build_category(component_name):
        if component_name in ("window_early", "window_late"):
            return "delivery windows"
        if component_name == "client_demand":
            return "demand shortfalls"
        if component_name in ("worker_daily", "worker_total_days", "worker_availability"):
            return "worker availability"
        return "other"

    rows = []
    for dev in result.deviations:
        deviation = dev.var.value
        if deviation is None or deviation <= tol:
            continue

        cost = dev.penalty * deviation
        rows.append(
            {
                "display_label": build_display_label(dev.component_name, dev.index),
                "deviation": deviation,
                "penalty": dev.penalty,
                "cost": cost,
                "category": build_category(dev.component_name),
            }
        )

    rows.sort(key=lambda row: (-row["cost"], row["display_label"]))
    total_violation_cost = sum(row["cost"] for row in rows)

    printer("=== Structural Diagnosis Summary ===")
    printer("")
    printer(f"Total violation cost: {total_violation_cost:g}")
    printer("")
    printer("Top structural tensions:")

    top_rows = rows[:5]
    if not top_rows:
        printer("  - none (all deviations within tolerance)")
    else:
        for row in top_rows:
            printer(
                "  - "
                f"{row['display_label']}: "
                f"deviation={row['deviation']:.1f}, "
                f"penalty={row['penalty']:g}, "
                f"cost={row['cost']:g}"
            )

    category_cost = {}
    for row in rows:
        cat = row["category"]
        category_cost[cat] = category_cost.get(cat, 0.0) + row["cost"]

    category_priority = {
        "demand shortfalls": 0,
        "delivery windows": 1,
        "worker availability": 2,
        "other": 3,
    }
    dominant_category = (
        sorted(
            category_cost.items(),
            key=lambda kv: (-kv[1], category_priority.get(kv[0], 99), kv[0]),
        )[0][0]
        if category_cost
        else "other"
    )

    if dominant_category == "demand shortfalls":
        question_subject = "demand"
    else:
        question_subject = dominant_category

    printer("")
    printer("Interpretation:")
    printer(f"The model absorbs infeasibility primarily through {dominant_category}.")
    printer(f"Are {question_subject} constraints negotiable?")
    printer("")


if __name__ == "__main__":
    # Reduce solver/demo logger noise for clearer diagnosis output.
    totsu_logger.setLevel("ERROR")
    logging.getLogger("pyomo").setLevel(logging.ERROR)

    penalties = {
        "shortfall": 1000,
        "overtime": 100,
        "forbidden": 5000,
        "early": 800,
        "late": 800,
    }

    m = build_model()

    print("Applying elasticity to the model...")
    m = apply_elasticity(m, penalties)

    solver = AdvancedBranchAndBound()
    results = solver.solve(m)

    print("=== Solution ===")
    for w in m.W:
        for c in m.C:
            for d in m.D:
                for s in m.S:
                    if m.x[w,c,d,s].value > 0.5:
                        print(f"x[{w},{c},d{d},s{s}] = 1")

    print("\n=== Early/Late Violations ===")
    for c in m.C:
        for d in m.D:
            if m.e_early[c,d].value > 1e-6:
                print(f"Early: {c} on day {d} => {m.e_early[c,d].value}")
            if m.e_late[c,d].value > 1e-6:
                print(f"Late: {c} on day {d} => {m.e_late[c,d].value}")

    print("\n=== Elastic Feasibility Tool ===")
    # 1. Build base (hard) model
    m_with_w = build_model(with_windows=True)

    tool = ElasticFeasibilityTool(default_penalty=1.0)

    # 2. Domain-level penalties and constraints to elasticize
    constraints_to_elasticize = [
        "client_demand",     # unmet demand
        "worker_daily",      # or "worker_total_days" – match your component name
        "window_early",      # before start[c]
        "window_late",       # after end[c]
    ]

    
    # Map each component name to a penalty
    penalty_map = {
        "client_demand": penalties["shortfall"],
        "worker_daily":  penalties["overtime"],     # or "worker_total_days"
        "window_early":  penalties["early"],
        "window_late":   penalties["late"],
    }

    # 3. Apply generic elastic feasibility tool
    result = tool.apply(
        m_with_w,
        constraints=constraints_to_elasticize,
        penalty_map=penalty_map,
        deactivate_original_objective=True,  # replace with "min total violation"
        clone=False,                         # modify m in-place
    )

    # 4. Solve the elastic model with advanced B&B
    solver = AdvancedBranchAndBound()
    solver.solve(m_with_w)

    # 5. Print core assignments
    print("\n=== Solution ===")
    for w in m_with_w.W:
        for c in m_with_w.C:
            for d in m_with_w.D:
                for s in m_with_w.S:
                    if m_with_w.x[w, c, d, s].value is not None and m_with_w.x[w, c, d, s].value > 0.5:
                        print(f"x[{w},{c},d{d},s{s}] = 1")

    # 6. Print elastic infeasibility summary (customers / workers / windows)
    print()
    report_elastic_infeasibility_for_4way(m_with_w, result)
