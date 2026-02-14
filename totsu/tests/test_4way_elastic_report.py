from pyomo.environ import ConcreteModel, Var

from totsu.examples.assignments.demo_4ways_assignment_elastic_windows import (
    report_elastic_infeasibility_for_4way,
)
from totsu.utils.elastic_feasibility_tool import ElasticDeviation, ElasticResult


def test_structural_diagnosis_report_labels_and_interpretation(capsys):
    m = ConcreteModel()
    m.dev_window = Var(initialize=1.0)
    m.dev_demand = Var(initialize=1.0)

    deviations = [
        ElasticDeviation(
            var=m.dev_window,
            penalty=100.0,
            kind="slack",
            component_name="window_late",
            index=("c2", 1),
            original_name="window_late[c2,1]",
            sense="LE",
            bound=0.0,
            generated_constraint_name="",
        ),
        ElasticDeviation(
            var=m.dev_demand,
            penalty=100.0,
            kind="slack",
            component_name="client_demand",
            index=("c1",),
            original_name="client_demand[c1]",
            sense="EQ",
            bound=1.0,
            generated_constraint_name="",
        ),
    ]
    result = ElasticResult(model=m, deviations=deviations)

    report_elastic_infeasibility_for_4way(m, result, tol=1e-9)
    output = capsys.readouterr().out

    assert "=== Structural Diagnosis Summary ===" in output
    assert "Total violation cost: 200" in output
    assert "window_late[c2, day 1]" in output
    assert "demand_constraint[c1]" in output
    assert "Interpretation:" in output
    assert "Are demand constraints negotiable?" in output
