import pytest
from pyomo.environ import Constraint, ConcreteModel, NonNegativeReals, Objective, Set, Var, minimize
from pyomo.repn import generate_standard_repn

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
