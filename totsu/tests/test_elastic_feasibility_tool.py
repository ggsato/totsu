import pytest
from pyomo.environ import Constraint, ConcreteModel, NonNegativeReals, Objective, Set, Var, minimize
from pyomo.repn import generate_standard_repn

from totsu.core.totsu_simplex_solver import TotsuSimplexSolver
from totsu.utils.elastic_feasibility_tool import ElasticFeasibilityTool


def _build_infeasible_model() -> ConcreteModel:
    m = ConcreteModel()
    m.x = Var(within=NonNegativeReals)
    m.c_ge = Constraint(expr=m.x >= 5)
    m.c_le = Constraint(expr=m.x <= 3)
    m.obj = Objective(expr=m.x, sense=minimize)
    return m


def test_violation_only_creates_deviations_and_elastic_objective():
    model = _build_infeasible_model()
    tool = ElasticFeasibilityTool(default_penalty=2.0)

    result = tool.apply(model, constraints=["c_ge", "c_le"], objective_mode="violation_only", clone=True)

    assert len(result.deviations) == 2
    assert hasattr(result.model, "elastic")
    assert hasattr(result.model.elastic, "elastic_obj")
    assert not result.model.obj.active
    assert result.model.elastic.elastic_obj.active

    repn = generate_standard_repn(result.model.elastic.elastic_obj.expr, compute_values=False)
    coef_by_name = {var.name: coef for var, coef in zip(repn.linear_vars, repn.linear_coefs)}
    for dev in result.deviations:
        assert dev.var.name in coef_by_name
        assert coef_by_name[dev.var.name] == pytest.approx(2.0)
        assert dev.generated_constraint_name.startswith("elastic.elastic_con_")


def test_original_plus_violation_contains_original_objective_term():
    model = _build_infeasible_model()
    tool = ElasticFeasibilityTool(default_penalty=4.0)

    result = tool.apply(
        model,
        constraints=["c_ge", "c_le"],
        objective_mode="original_plus_violation",
        original_objective_weight=3.0,
        clone=True,
    )

    assert not result.model.obj.active
    assert result.model.elastic.elastic_obj.active

    repn = generate_standard_repn(result.model.elastic.elastic_obj.expr, compute_values=False)
    coef_by_name = {var.name: coef for var, coef in zip(repn.linear_vars, repn.linear_coefs)}

    assert coef_by_name[result.model.x.name] == pytest.approx(3.0)
    for dev in result.deviations:
        assert coef_by_name[dev.var.name] == pytest.approx(4.0)


def test_generated_components_are_under_elastic_block():
    model = _build_infeasible_model()
    tool = ElasticFeasibilityTool()

    result = tool.apply(model, constraints=["c_ge", "c_le"], objective_mode="violation_only", clone=True)

    root_elastic_dev_vars = [
        v for v in result.model.component_objects(Var, descend_into=False)
        if v.name.startswith("elastic_dev_")
    ]
    assert root_elastic_dev_vars == []

    elastic_block_vars = list(result.model.elastic.component_data_objects(Var, descend_into=False))
    assert len(elastic_block_vars) == 2
    assert all(var.name.startswith("elastic.elastic_dev_") for var in elastic_block_vars)

    elastic_constraints = list(result.model.elastic.component_data_objects(Constraint, descend_into=False))
    assert len(elastic_constraints) == 2
    assert all(con.name.startswith("elastic.elastic_con_") for con in elastic_constraints)


def test_constraint_component_name_selection_still_works():
    model = ConcreteModel()
    model.I = Set(initialize=[1, 2, 3])
    model.x = Var(model.I, within=NonNegativeReals)
    model.target = Constraint(model.I, rule=lambda m, i: m.x[i] >= i)
    model.other = Constraint(expr=sum(model.x[i] for i in model.I) <= 100)
    model.obj = Objective(expr=sum(model.x[i] for i in model.I), sense=minimize)

    tool = ElasticFeasibilityTool()
    result = tool.apply(model, constraints=["target"], objective_mode="keep_original", clone=True)

    assert len(result.deviations) == 3
    assert all(dev.component_name == "target" for dev in result.deviations)
    assert not any(dev.component_name == "other" for dev in result.deviations)

    assert all(not result.model.target[i].active for i in result.model.I)
    assert result.model.other.active
    assert result.model.obj.active


def test_deactivate_original_objective_false_keeps_original_objective():
    model = _build_infeasible_model()
    tool = ElasticFeasibilityTool()

    result = tool.apply(
        model,
        constraints=["c_ge", "c_le"],
        deactivate_original_objective=False,
        clone=True,
    )

    assert result.model.obj.active
    assert not hasattr(result.model.elastic, "elastic_obj")


def test_populate_violation_summary_computes_sorted_costs():
    model = _build_infeasible_model()
    tool = ElasticFeasibilityTool(default_penalty=1.0)

    result = tool.apply(
        model,
        constraints=["c_ge", "c_le"],
        objective_mode="keep_original",
        clone=True,
    )

    # Deterministic values for summary verification
    for dev in result.deviations:
        if dev.component_name == "c_ge":
            dev.var.set_value(2.5)  # cost = 2.5
            dev.penalty = 1.0
        elif dev.component_name == "c_le":
            dev.var.set_value(1.0)  # cost = 10.0
            dev.penalty = 10.0

    tool.populate_violation_summary(result, tol=1e-6)

    assert result.total_violation_cost == pytest.approx(12.5)
    assert len(result.violation_breakdown) == 2
    assert result.violation_breakdown[0]["component_name"] == "c_le"
    assert result.violation_breakdown[0]["cost"] == pytest.approx(10.0)
    assert result.violation_breakdown[1]["component_name"] == "c_ge"
    assert result.violation_breakdown[1]["cost"] == pytest.approx(2.5)


def test_le_constraint_violation_creates_positive_excess_and_expected_cost():
    model = ConcreteModel()
    model.assigned = Var(within=NonNegativeReals)
    model.must_assign = Constraint(expr=model.assigned == 1)
    model.window_late = Constraint(expr=model.assigned <= 0)
    model.obj = Objective(expr=0.0, sense=minimize)

    tool = ElasticFeasibilityTool(default_penalty=1.0)
    result = tool.apply(
        model,
        constraints=["window_late"],
        penalty_map={"window_late": 100.0},
        objective_mode="violation_only",
        clone=False,
    )

    solver = TotsuSimplexSolver()
    solver.solve(model)
    tool.populate_violation_summary(result, tol=1e-8)

    window_dev = [dev for dev in result.deviations if dev.component_name == "window_late"]
    assert len(window_dev) == 1
    assert window_dev[0].var.value == pytest.approx(1.0)
    assert window_dev[0].var.value > 0.0
    assert result.total_violation_cost == pytest.approx(100.0)
