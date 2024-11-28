# test_totsu_simplex_solver.py

import pytest
from pyomo.environ import *
from pyomo.opt import SolverStatus, TerminationCondition
from totsu.core.totsu_simplex_solver import TotsuSimplexSolver

# Fixture to initialize the Simplex solver
@pytest.fixture
def solver():
    """Fixture to create and return a SimplexSolver instance."""
    return TotsuSimplexSolver()

# Fixtures for various test models
@pytest.fixture
def basic_lp_model():
    """Basic LP model used for multiple tests (feasibility, optimality, shadow prices)."""
    model = ConcreteModel()
    model.x1 = Var(within=NonNegativeReals)
    model.x2 = Var(within=NonNegativeReals)
    model.c1 = Constraint(expr=model.x1 + model.x2 <= 4)
    model.c2 = Constraint(expr=2 * model.x1 + model.x2 <= 5)
    model.objective = Objective(expr=3 * model.x1 + 2 * model.x2, sense=maximize)
    return model

@pytest.fixture
def unbounded_model():
    """Model representing an unbounded LP problem."""
    model = ConcreteModel()
    model.x1 = Var(within=NonNegativeReals)
    model.x2 = Var()
    model.constraints = ConstraintList()
    model.constraints.add(model.x1 - model.x2 <= 4)
    model.objective = Objective(expr=2 * model.x1 + model.x2, sense=maximize)
    return model

@pytest.fixture
def infeasible_model():
    """Model representing an infeasible LP problem due to conflicting constraints."""
    model = ConcreteModel()
    model.x1 = Var(within=NonNegativeReals)
    model.x2 = Var(within=NonNegativeReals)
    model.constraints = ConstraintList()
    model.constraints.add(model.x1 + model.x2 <= 4)
    model.constraints.add(model.x1 + model.x2 >= 6)
    model.objective = Objective(expr=3 * model.x1 + 2 * model.x2, sense=maximize)
    return model

@pytest.fixture
def mixed_constraints_model():
    """Model containing both equality and inequality constraints."""
    model = ConcreteModel()
    model.x1 = Var(within=NonNegativeReals)
    model.x2 = Var(within=NonNegativeReals)
    model.x3 = Var(within=NonNegativeReals)
    model.constraints = ConstraintList()
    # Equality constraint
    model.constraints.add(model.x1 + 2 * model.x2 + model.x3 == 4)
    # Inequality constraints
    model.constraints.add(model.x1 + model.x2 >= 1)
    model.constraints.add(model.x2 + model.x3 <= 2)
    model.objective = Objective(expr=2 * model.x1 + model.x2 + 3 * model.x3, sense=maximize)
    return model

# Test methods using the fixtures
def test_feasibility(solver, basic_lp_model):
    """Test that solutions satisfy all constraints."""
    results = solver.solve(basic_lp_model)
    assert results.solver.status == SolverStatus.ok, "Expected ok status, but not"

def test_unbounded_problem(solver, unbounded_model):
    """Test that the solver identifies an unbounded problem."""
    results = solver.solve(unbounded_model)
    assert results.solver.termination_condition == TerminationCondition.unbounded, "Expected unbounded condition, but not"

def test_optimality(solver, basic_lp_model):
    """Test that the solver finds the optimal solution."""
    results = solver.solve(basic_lp_model)
    assert results.solver.termination_condition == TerminationCondition.optimal, "Expected optimal condition, but not"

def test_standard_lp(solver, basic_lp_model):
    """Test a standard LP problem to verify correctness of the solution."""
    # Access dual variables from the model
    model = basic_lp_model
    if not hasattr(model, 'dual'):
        model.dual = Suffix(direction=Suffix.IMPORT)
    if not hasattr(model, 'rc'):
        model.rc = Suffix(direction=Suffix.IMPORT)
        
    results = solver.solve(model)
    x1 = model.x1.value
    x2 = model.x2.value
    assert abs(x1 - 1) <= 1e-6, f"Expected x1 = 1, got {x1}"
    assert abs(x2 - 3) <= 1e-6, f"Expected x2 = 3, got {x2}"

    # Expected dual variables and reduced costs
    expected_duals = {'c1': 1.0, 'c2': 1.0}
    expected_rc = {'x1': 0.0, 'x2': 0.0}

    tolerance = 1e-6

    # Check dual variables
    for con in [model.c1, model.c2]:
        computed_dual = model.dual[con]
        expected_dual = expected_duals[con.name]
        assert abs(computed_dual - expected_dual) <= tolerance, \
            f"Dual variable for {con.name} incorrect: expected {expected_dual}, got {computed_dual}"

    # Check reduced costs
    for var in [model.x1, model.x2]:
        computed_rc = model.rc[var]
        expected_reduced_cost = expected_rc[var.name]
        assert abs(computed_rc - expected_reduced_cost) <= tolerance, \
            f"Reduced cost for {var.name} incorrect: expected {expected_reduced_cost}, got {computed_rc}"

def test_infeasible_problem(solver, infeasible_model):
    """Test the solver's ability to detect infeasibility."""
    results = solver.solve(infeasible_model)
    assert results.solver.termination_condition == TerminationCondition.infeasible, "Expected infeasible condition, but not"

def test_mixed_constraints(solver, mixed_constraints_model):
    """Test solving a problem with mixed equality and inequality constraints."""
    # Access dual variables from the model
    model = mixed_constraints_model
    if not hasattr(model, 'dual'):
        model.dual = Suffix(direction=Suffix.IMPORT)
    if not hasattr(model, 'rc'):
        model.rc = Suffix(direction=Suffix.IMPORT)

    results = solver.solve(model)
    assert results.solver.status == SolverStatus.ok, "Expected ok status, but not"

    model = model
    x1 = model.x1.value
    x2 = model.x2.value
    x3 = model.x3.value

    # Validate constraints are satisfied
    assert abs(x1 + 2 * x2 + x3 - 4) <= 1e-6, f"Equality constraint violated: {x1 + 2 * x2 + x3} != 4"
    assert x1 + x2 >= 1 - 1e-6, f"Inequality constraint x1 + x2 >= 1 violated: {x1 + x2}"
    assert x2 + x3 <= 2 + 1e-6, f"Inequality constraint x2 + x3 <= 2 violated: {x2 + x3}"

    # Validate variable bounds
    assert x1 >= 0, f"x1 should be non-negative, got {x1}"
    assert x2 >= 0, f"x2 should be non-negative, got {x2}"
    assert x3 >= 0, f"x3 should be non-negative, got {x3}"

    # Validate objective value
    expected_objective = 2 * x1 + x2 + 3 * x3
    computed_objective = model.objective.expr()
    assert abs(expected_objective - computed_objective) <= 1e-6, "Objective value mismatch"

    # Check dual variables and reduced costs
    tolerance = 1e-6
    for index in model.constraints:
        con = model.constraints[index]
        assert con in model.dual, f"Dual variable for {con.name} not found."
        dual_value = model.dual[con]
        print(f"Dual variable for {con.name}: {dual_value}")

    for var in [model.x1, model.x2, model.x3]:
        assert var in model.rc, f"Reduced cost for {var.name} not found."
        rc_value = model.rc[var]
        print(f"Reduced cost for {var.name}: {rc_value}")

def test_dual_variables_and_reduced_costs(solver, basic_lp_model):
    """Test the correctness of dual variables and reduced costs."""
    # Access dual variables from the model
    model = basic_lp_model
    if not hasattr(model, 'dual'):
        model.dual = Suffix(direction=Suffix.IMPORT)
    if not hasattr(model, 'rc'):
        model.rc = Suffix(direction=Suffix.IMPORT)

    results = solver.solve(model)
    assert results.solver.status == SolverStatus.ok, "Expected solver status to be ok."

    for con in model.dual:
        print(f"{con} is available in dual model")

    # Expected dual variables
    expected_duals = {
        'c1': 1.0,
        'c2': 1.0
    }

    # Expected reduced costs (for x1 and x2, should be zero)
    expected_rc = {
        'x1': 0.0,
        'x2': 0.0
    }

    tolerance = 1e-6

    # Check dual variables
    constraints = [model.c1, model.c2]
    for con in constraints:
        assert con in constraints, f"{con} instance is not available in dual variables"
        computed_dual = model.dual[con]
        expected_dual = expected_duals[con.name]
        assert abs(computed_dual - expected_dual) <= tolerance, \
            f"Dual variable for {con.name} incorrect: expected {expected_dual}, got {computed_dual}"

    # Check reduced costs
    variables = [model.x1, model.x2]
    for var in variables:
        computed_rc = model.rc[var]
        expected_reduced_cost = expected_rc[var.name]
        assert abs(computed_rc - expected_reduced_cost) <= tolerance, \
            f"Reduced cost for {var.name} incorrect: expected {expected_reduced_cost}, got {computed_rc}"
