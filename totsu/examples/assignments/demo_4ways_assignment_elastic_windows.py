from pyomo.environ import *
from typing import Callable

from totsu.core.advanced_branch_and_bound_solver import AdvancedBranchAndBound
from totsu.utils.elastic_feasibility_tool import ElasticFeasibilityTool, ElasticResult

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

    demand_violation: Dict[Any, float] = {}
    worker_violation: Dict[Any, float] = {}
    window_violation: Dict[Tuple[Any, Any], float] = {}

    for dev in result.deviations:
        v = dev.var.value
        if v is None or v <= tol:
            continue  # ignore tiny violations / numerical noise

        cname = dev.component_name

        # --- Client demand unmet: group by customer c ---
        if cname in ("client_demand", "client_demand_elastic"):
            if len(dev.index) >= 1:
                c = dev.index[0]
            else:
                c = "?"
            demand_violation[c] = demand_violation.get(c, 0.0) + v

        # --- Worker availability / total days: group by worker w ---
        elif cname in ("worker_total_days", "worker_daily", "worker_daily_elastic"):
            if len(dev.index) >= 1:
                w = dev.index[0]
            else:
                w = "?"
            worker_violation[w] = worker_violation.get(w, 0.0) + v

        # --- Time window violations: group by (customer, day) ---
        elif cname in ("window", "window_elastic"):
            if len(dev.index) >= 2:
                c, d = dev.index[0], dev.index[1]
            elif len(dev.index) == 1:
                c, d = dev.index[0], "?"
            else:
                c, d = "?", "?"
            window_violation[(c, d)] = window_violation.get((c, d), 0.0) + v

        # else: other components (forbidden, etc.) can be added as needed

    printer("=== Elastic infeasibility summary (4-way W–C–D–S) ===")

    # 1. Customers whose demand is not fully met
    if demand_violation:
        printer("\nCustomers with unmet demand (by total shortfall):")
        for c, v in sorted(demand_violation.items(), key=lambda kv: -kv[1]):
            printer(f"  - customer {c}: shortfall ≈ {v:.3f}")
    else:
        printer("\nNo client-demand violations detected (within tolerance).")

    # 2. Workers whose availability / capacity is violated
    if worker_violation:
        printer("\nWorkers with availability violations (overtime / overload):")
        for w, v in sorted(worker_violation.items(), key=lambda kv: -kv[1]):
            printer(f"  - worker {w}: violation ≈ {v:.3f}")
    else:
        printer("\nNo worker-availability violations detected (within tolerance).")

    # 3. Customer windows violated (early/late service)
    if window_violation:
        printer("\nCustomer window violations (by customer, day):")
        for (c, d), v in sorted(window_violation.items(), key=lambda kv: -kv[1]):
            printer(f"  - customer {c}, day {d}: window violation ≈ {v:.3f}")
    else:
        printer("\nNo time-window violations detected (within tolerance).")

    printer("")  # final newline


if __name__ == "__main__":
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