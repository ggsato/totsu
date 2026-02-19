import ast
import re
from contextlib import contextmanager

import pytest
from pyomo.environ import SolverFactory, value
from pyomo.opt import TerminationCondition

from totsu.core.super_simplex_solver import SuperSimplexSolver
from totsu.elastic import analyze_infeasibility
from totsu.examples.model_building_imp.examples.chp5_2_transportation import (
    transportation as transportation_mod,
    transportation_elastic_api as transportation_api_mod,
)
from totsu.utils.elastic_feasibility_tool import ElasticFeasibilityTool


REFERENCE_SOLVERS = ("highs", "cbc", "glpk")
SELECTED_ARCS = (("S1", "T3"), ("S1", "T4"), ("S2", "T1"), ("S3", "T2"))


def _is_feasible_termination(term):
    return term in {
        TerminationCondition.optimal,
        TerminationCondition.feasible,
        TerminationCondition.locallyOptimal,
        TerminationCondition.globallyOptimal,
    }


def _detect_reference_solver():
    for name in REFERENCE_SOLVERS:
        try:
            solver = SolverFactory(name)
        except Exception:
            continue
        if solver is None:
            continue
        try:
            if solver.available(exception_flag=False):
                return name
        except TypeError:
            if solver.available(False):
                return name
        except Exception:
            continue
    return None


def _solve_model_with_solver(model, solver_name):
    solver = SolverFactory(solver_name)
    if solver is None or not solver.available(exception_flag=False):
        pytest.skip(f"Solver '{solver_name}' is not available in this environment.")
    results = solver.solve(model)
    term = getattr(results.solver, "termination_condition", None)
    assert _is_feasible_termination(term), f"Expected feasible solve, got termination={term!r}"
    return results


def _extract_transport_solution(model, objective_value=None):
    if objective_value is None:
        objective_value = value(model.obj)
    arc_values = {
        (s, t): float(model.x[s, t].value or 0.0)
        for (s, t) in SELECTED_ARCS
    }
    by_source = {
        s: float(sum((model.x[s, t].value or 0.0) for t in model.T))
        for s in model.S
    }
    by_destination = {
        t: float(sum((model.x[s, t].value or 0.0) for s in model.S))
        for t in model.T
    }
    return {
        "objective": float(objective_value),
        "arc_values": arc_values,
        "by_source": by_source,
        "by_destination": by_destination,
    }


def _normalize_top_relaxations(payload):
    if hasattr(payload, "to_dict"):
        data = payload.to_dict()
        return list(data.get("top_relaxations", []))
    if isinstance(payload, dict):
        return list(payload.get("top_relaxations", []))
    if isinstance(payload, str):
        rows = []
        pattern = re.compile(
            r"-\s+([^:]+):\s+violation=([0-9eE+\-.]+),\s+cost=([0-9eE+\-.]+)"
        )
        for match in pattern.finditer(payload):
            rows.append(
                {
                    "constraint_name": match.group(1).strip(),
                    "violation": float(match.group(2)),
                    "cost": float(match.group(3)),
                }
            )
        return rows
    return []


def _normalize_violation_rows(payload):
    if hasattr(payload, "violation_breakdown"):
        return list(payload.violation_breakdown)
    if isinstance(payload, dict):
        return list(payload.get("violation_breakdown", []))
    if isinstance(payload, str):
        rows = []
        for line in payload.splitlines():
            line = line.strip()
            if line.startswith("{") and "constraint_name" in line and "violation" in line:
                try:
                    row = ast.literal_eval(line)
                except Exception:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
        return rows
    return []


@contextmanager
def _temporary_transport_requirements(t2_multiplier=1.0):
    original_requirements = dict(transportation_mod.requirements)
    try:
        transportation_mod.requirements.clear()
        transportation_mod.requirements.update(original_requirements)
        transportation_mod.requirements["T2"] = (
            original_requirements["T2"] * float(t2_multiplier)
        )
        yield
    finally:
        transportation_mod.requirements.clear()
        transportation_mod.requirements.update(original_requirements)


def test_super_simplex_matches_reference_on_feasible_transportation_baseline():
    reference_solver = _detect_reference_solver()
    if reference_solver is None:
        pytest.skip("No reference solver available (need one of: highs/cbc/glpk).")

    with _temporary_transport_requirements(t2_multiplier=1.0):
        model_super = transportation_mod.create_model()
        super_solver = SuperSimplexSolver()
        super_solver.solve(model_super)
        super_solution = _extract_transport_solution(
            model_super,
            objective_value=super_solver.get_current_objective_value(),
        )

    with _temporary_transport_requirements(t2_multiplier=1.0):
        model_ref = transportation_mod.create_model()
        _solve_model_with_solver(model_ref, reference_solver)
        reference_solution = _extract_transport_solution(model_ref, objective_value=value(model_ref.obj))

    obj_tol = max(1e-4, 1e-6 * max(1.0, abs(reference_solution["objective"])))
    assert abs(super_solution["objective"] - reference_solution["objective"]) <= obj_tol

    for arc in SELECTED_ARCS:
        assert super_solution["arc_values"][arc] == pytest.approx(
            reference_solution["arc_values"][arc], abs=1e-4
        )


def test_transportation_elastic_api_allows_demand_relaxation():
    reference_solver = _detect_reference_solver()
    if reference_solver is None:
        pytest.skip("No LP solver available for API elasticity regression (highs/cbc/glpk).")

    with _temporary_transport_requirements(t2_multiplier=2.0):
        model = transportation_mod.create_model()
        pretty_name = transportation_api_mod._build_pretty_name(model)
        result = analyze_infeasibility(
            model,
            solver=reference_solver,
            violation_only=False,
            default_penalty=1000.0,
            pretty_name=pretty_name,
            max_items=10,
        )

    assert result.is_feasible_original is False
    assert result.is_feasible_elastic is True

    top_relaxations = _normalize_top_relaxations(result)
    assert top_relaxations, "Expected non-empty Top relaxations."
    assert any(
        "demand_constraints" in str(row.get("constraint_name", ""))
        for row in top_relaxations
    ), "Expected API path to allow demand constraint relaxations."
    for row in top_relaxations:
        assert "constraint_name" in row
        assert "violation" in row
        assert "cost" in row


@pytest.mark.parametrize("objective_mode", ["violation_only", "original_plus_violation"])
def test_transportation_tool_policy_keeps_demand_hard_and_relaxes_supply(objective_mode):
    reference_solver = _detect_reference_solver()
    if reference_solver is None:
        pytest.skip("No LP solver available for tool-level elasticity regression (highs/cbc/glpk).")

    with _temporary_transport_requirements(t2_multiplier=2.0):
        model = transportation_mod.create_model()
        tool = ElasticFeasibilityTool(default_penalty=1.0, tol=1e-8)
        result = tool.apply(
            model,
            constraints=["supply_constraints"],
            penalty_map={"supply_constraints": 10.0},
            objective_mode=objective_mode,
            clone=True,
        )
        solve_results = _solve_model_with_solver(result.model, reference_solver)
        term = getattr(solve_results.solver, "termination_condition", None)
        assert _is_feasible_termination(term)
        ElasticFeasibilityTool.populate_violation_summary(result, tol=1e-8)

    rows = _normalize_violation_rows(result)
    assert rows, "Expected non-empty violation diagnostics for infeasible transportation case."
    assert result.total_violation_cost >= 0.0
    assert all(
        "demand_constraints" not in str(row.get("constraint_name", ""))
        for row in rows
    ), "Demand constraints should remain hard in tool-level policy run."
    assert any(
        "supply_constraints" in str(row.get("constraint_name", ""))
        for row in rows
    ), "Expected at least one violated supply constraint."

    if objective_mode == "original_plus_violation":
        assert result.original_objective_value is not None
        assert result.violation_objective_value is not None
        assert result.combined_objective_value is not None
        assert result.combined_objective_value == pytest.approx(
            result.original_objective_value + result.violation_objective_value,
            abs=1e-6,
        )
