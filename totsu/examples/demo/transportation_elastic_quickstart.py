"""3-minute quickstart for Totsu Elastic Tool.

This example deliberately builds an *infeasible* transportation LP and then
uses :class:`totsu.utils.elastic_feasibility_tool.ElasticFeasibilityTool` to
measure the minimal relaxations needed to restore feasibility.

Run:

    python -m totsu.examples.elastic.transportation_elastic_quickstart --solver glpk

If you already have Pyomo and a solver installed, this should work out of the box.
"""

from __future__ import annotations

import argparse
from typing import Iterable

from pyomo.environ import (
    ConcreteModel,
    Set,
    Param,
    Var,
    NonNegativeReals,
    Constraint,
    Objective,
    minimize,
    SolverFactory,
    value,
)

from totsu.utils.elastic_feasibility_tool import ElasticFeasibilityTool


def build_infeasible_transportation_model() -> ConcreteModel:
    """Classic transportation LP with *supply < demand* to force infeasibility."""
    m = ConcreteModel()

    m.S = Set(initialize=["S1", "S2", "S3"])
    m.D = Set(initialize=["D1", "D2", "D3"])

    # Total supply = 200
    supply = {"S1": 50, "S2": 60, "S3": 90}
    # Total demand = 275  (intentionally infeasible)
    demand = {"D1": 80, "D2": 95, "D3": 100}

    m.supply = Param(m.S, initialize=supply)
    m.demand = Param(m.D, initialize=demand)

    # Costs (arbitrary but structured)
    cost = {
        ("S1", "D1"): 4,
        ("S1", "D2"): 6,
        ("S1", "D3"): 9,
        ("S2", "D1"): 5,
        ("S2", "D2"): 4,
        ("S2", "D3"): 7,
        ("S3", "D1"): 6,
        ("S3", "D2"): 3,
        ("S3", "D3"): 4,
    }
    m.cost = Param(m.S, m.D, initialize=cost)

    m.x = Var(m.S, m.D, within=NonNegativeReals)

    def supply_rule(m, s):
        return sum(m.x[s, d] for d in m.D) <= m.supply[s]

    def demand_rule(m, d):
        return sum(m.x[s, d] for s in m.S) >= m.demand[d]

    m.supply_con = Constraint(m.S, rule=supply_rule)
    m.demand_con = Constraint(m.D, rule=demand_rule)

    m.obj = Objective(
        expr=sum(m.cost[s, d] * m.x[s, d] for s in m.S for d in m.D),
        sense=minimize,
    )

    return m


def _format_rows(rows: Iterable[dict], max_rows: int = 10) -> str:
    rows = list(rows)
    lines = []
    for i, row in enumerate(rows[:max_rows]):
        lines.append(
            "  - "
            f"{row['constraint_name']}: "
            f"deviation={row['deviation']:.3g}, "
            f"penalty={row['penalty']:.3g}, "
            f"cost={row['cost']:.3g}"
        )
    if len(rows) > max_rows:
        lines.append(f"  ... ({len(rows) - max_rows} more)")
    return "\n".join(lines) if lines else "  - none"


def main() -> None:
    parser = argparse.ArgumentParser(description="Totsu Elastic Tool 3-minute quickstart")
    parser.add_argument(
        "--solver",
        default="glpk",
        help="Pyomo solver name (e.g., glpk, cbc, highs). Default: glpk",
    )
    parser.add_argument(
        "--mode",
        default="violation_only",
        choices=["violation_only", "original_plus_violation"],
        help="Elastic objective mode. Default: violation_only",
    )
    parser.add_argument(
        "--penalty",
        type=float,
        default=10.0,
        help="Penalty multiplier used in original_plus_violation mode. Default: 10.0",
    )
    args = parser.parse_args()

    m = build_infeasible_transportation_model()

    # 1) Show that the base model is infeasible
    solver = SolverFactory(args.solver)
    if solver is None or not solver.available(exception_flag=False):
        raise SystemExit(
            "Solver is not available via Pyomo: "
            f"{args.solver!r}. "
            "Install a solver (e.g., glpk/cbc/highs) and try again, or pass --solver <name>."
        )

    base_res = solver.solve(m, tee=False)
    base_term = getattr(base_res.solver, "termination_condition", None)
    print("=== Base model ===")
    print(f"Termination condition: {base_term}")
    print("Expected: infeasible (because total demand > total supply).")
    print()

    # 2) Apply Elastic Tool
    tool = ElasticFeasibilityTool(default_penalty=1.0)

    # Elasticize supply and demand constraints only.
    # - supply: allow 'excess' beyond supply
    # - demand: allow 'slack' below demand
    penalty_map = {
        "supply_con": 1.0,
        "demand_con": 1.0,
    }

    objective_mode = args.mode
    deactivate_original_objective = objective_mode == "violation_only"
    original_weight = 1.0
    if objective_mode == "original_plus_violation":
        original_weight = 1.0

    result = tool.apply(
        m,
        constraints=["supply_con", "demand_con"],
        penalty_map=penalty_map,
        deactivate_original_objective=deactivate_original_objective,
        objective_mode=objective_mode,
        original_objective_weight=original_weight,
        clone=True,
    )

    # If we combine original objective, scale violation penalties.
    # (This keeps the example simple and makes the effect of --penalty visible.)
    if objective_mode == "original_plus_violation":
        # Multiply all deviation penalties by args.penalty
        for dev in result.deviations:
            dev.penalty *= float(args.penalty)

    elastic_model = result.model
    elastic_res = solver.solve(elastic_model, tee=False)
    elastic_term = getattr(elastic_res.solver, "termination_condition", None)

    # 3) Summarize
    ElasticFeasibilityTool.populate_violation_summary(result, tol=1e-9)

    print("=== Elastic analysis ===")
    print(f"Termination condition: {elastic_term}")
    print(f"Objective mode: {result.objective_mode}")
    if result.original_objective_value is not None:
        print(f"Original objective value (transport cost): {result.original_objective_value:.6g}")
    print(f"Total violation cost: {result.total_violation_cost:.6g}")
    print()
    print("Top relaxations (highest cost first):")
    print(_format_rows(result.violation_breakdown[:10], max_rows=10))
    print()

    # Add a simple human-level interpretation for this model family.
    total_supply = sum(value(elastic_model.supply[s]) for s in elastic_model.S)
    total_demand = sum(value(elastic_model.demand[d]) for d in elastic_model.D)
    print("Interpretation:")
    print(f"- Total supply: {total_supply:g}")
    print(f"- Total demand: {total_demand:g}")
    print(f"- Structural deficit (demand - supply): {total_demand - total_supply:g}")
    print("The deviations above tell you *where* to relax (which constraints) and *by how much*. ")


if __name__ == "__main__":
    main()
