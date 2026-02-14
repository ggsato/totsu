import pytest

from totsu.core.advanced_branch_and_bound_solver import AdvancedBranchAndBound
from totsu.examples.assignments.demo_4ways_assignment_elastic_windows import (
    build_model,
    report_elastic_infeasibility_for_4way,
)
from totsu.utils.elastic_feasibility_tool import ElasticFeasibilityTool


def test_demo_4way_elastic_windows_regression(capsys):
    model = build_model(with_windows=True)

    tool = ElasticFeasibilityTool(default_penalty=1.0)
    penalty_map = {
        "client_demand": 1000.0,
        "worker_daily": 2000.0,
        "window_early": 100.0,
        "window_late": 100.0,
    }

    result = tool.apply(
        model,
        constraints=["client_demand", "worker_daily", "window_early", "window_late"],
        penalty_map=penalty_map,
        objective_mode="violation_only",
        clone=False,
    )

    solver = AdvancedBranchAndBound()
    solver.solve(model)
    tool.populate_violation_summary(result, tol=1e-8)

    expected_total_violation_cost = penalty_map["window_early"] * 1.0
    assert result.total_violation_cost == pytest.approx(expected_total_violation_cost, abs=1e-6)
    assert result.violation_breakdown, "Expected non-empty violation breakdown"

    top = result.violation_breakdown[0]
    assert top["component_name"] in ("window_early", "window_late")

    c2_day1_early = [
        dev for dev in result.deviations
        if dev.component_name == "window_early"
        and tuple(dev.index) == ("c2", 1)
        and dev.var.value is not None
        and dev.var.value > 1e-8
    ]
    assert c2_day1_early, "Expected positive window_early deviation at (c2, 1)"

    window_cost = sum(
        row["cost"] for row in result.violation_breakdown
        if row["component_name"] in ("window_early", "window_late")
    )
    demand_costs = [
        row["cost"] for row in result.violation_breakdown if row["component_name"] == "client_demand"
    ]
    demand_cost = max(demand_costs) if demand_costs else 0.0
    assert demand_cost == pytest.approx(0.0)
    assert window_cost > demand_cost

    report_elastic_infeasibility_for_4way(model, result, tol=1e-8)
    output = capsys.readouterr().out
    assert "Interpretation:" in output
    assert "delivery windows" in output
