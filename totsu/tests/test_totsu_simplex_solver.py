# test_simplex_solver.py

import pytest
from pyomo.environ import *
from pyomo.opt import SolverStatus, TerminationCondition
from totsu.core.totsu_simplex_solver import TotsuSimplexSolver

"""
1. **Feasibility and Optimality**:
   - `test_feasibility`: Tests that solutions satisfy all constraints.
   - `test_optimality`: Checks if the solver finds the optimal solution.

2. **Unbounded and Infeasible Problems**:
   - `test_unbounded_problem`: Tests identification of unbounded problems.
   - `test_infeasible_problem`: Tests detection of infeasible problems.

3. **Objective and others**:
   - `test_mixed_constraints`: Tests mixed equality and inequality constraints.
"""

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
    model.x2 = Var(within=NonNegativeReals)
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
    results = solver.solve(basic_lp_model)
    assert abs(results.solver.variables['x1'] - 1) <= 1e-6, f"Expected x1 = 1, got {results.solver.variables['x1']}"
    assert abs(results.solver.variables['x2'] - 3) <= 1e-6, f"Expected x2 = 3, got {results.solver.variables['x2']}"

def test_infeasible_problem(solver, infeasible_model):
    """Test the solver's ability to detect infeasibility."""
    results = solver.solve(infeasible_model)
    assert results.solver.termination_condition == TerminationCondition.infeasible, "Expected infeasible condition, but not"

def test_mixed_constraints(solver, mixed_constraints_model):
    """Test solving a problem with mixed equality and inequality constraints."""
    results = solver.solve(mixed_constraints_model)
    assert results.solver.status == SolverStatus.ok, "Expected ok status, but not"

    x1 = results.solver.variables['x1']
    x2 = results.solver.variables['x2']
    x3 = results.solver.variables['x3']

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
    assert abs(expected_objective - results.solver.objective) <= 1e-6, "Objective value mismatch"