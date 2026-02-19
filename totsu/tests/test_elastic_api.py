from types import SimpleNamespace

from pyomo.environ import ConcreteModel, Constraint, NonNegativeReals, Objective, Var, minimize
from pyomo.opt import TerminationCondition

import totsu.elastic as elastic_api


def _build_infeasible_model():
    m = ConcreteModel()
    m.x = Var(within=NonNegativeReals)
    m.c_ge = Constraint(expr=m.x >= 5)
    m.c_le = Constraint(expr=m.x <= 3)
    m.obj = Objective(expr=m.x, sense=minimize)
    return m


class _FakeSolver:
    def available(self, exception_flag=False):
        return True

    def solve(self, model, tee=False):
        if hasattr(model, "elastic"):
            for v in model.elastic.component_data_objects(Var, descend_into=False):
                if "elastic_dev_" in v.name:
                    v.set_value(1.0)
            term = TerminationCondition.optimal
        else:
            term = TerminationCondition.infeasible
        return SimpleNamespace(solver=SimpleNamespace(termination_condition=term))


def test_analyze_infeasibility_runs_elastic_and_returns_top_relaxations(monkeypatch):
    monkeypatch.setattr(elastic_api, "SolverFactory", lambda name: _FakeSolver())

    model = _build_infeasible_model()
    result = elastic_api.analyze_infeasibility(
        model,
        solver="auto",
        max_items=1,
        pretty_name=lambda con: f"pretty:{con.name}",
    )

    assert result.solver_name == "highs"
    assert result.is_feasible_original is False
    assert result.is_feasible_elastic is True
    assert result.original_results is not None
    assert result.elastic_solve_results is not None
    assert len(result.top_relaxations) == 1
    assert "constraint_name" in result.top_relaxations[0]
    assert "cost" in result.top_relaxations[0]
    assert "direction" in result.top_relaxations[0]
    assert result.top_relaxations[0]["pretty_name"].startswith("pretty:")

    as_dict = result.to_dict()
    assert as_dict["is_feasible_original"] is False
    assert as_dict["is_feasible_elastic"] is True
    assert as_dict["original_results"] is result.original_results
    assert as_dict["elastic_solve_results"] is result.elastic_solve_results


def test_analyze_infeasibility_respects_default_penalty(monkeypatch):
    monkeypatch.setattr(elastic_api, "SolverFactory", lambda name: _FakeSolver())

    model = _build_infeasible_model()
    result = elastic_api.analyze_infeasibility(
        model,
        solver="auto",
        default_penalty=7.0,
        max_items=1,
    )

    assert len(result.top_relaxations) == 1
    assert result.top_relaxations[0]["cost"] == 7.0
