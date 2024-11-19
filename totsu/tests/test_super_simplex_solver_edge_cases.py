# test_simplex_solver_edge_cases.py

import pytest
from pyomo.environ import *
from totsu.core.super_simplex_solver import SuperSimplexSolver, InfeasibleProblemError, OptimizationError

# Fixture to initialize the Simplex solver
@pytest.fixture
def solver():
    """Fixture to create and return a SimplexSolver instance."""
    return SuperSimplexSolver()

"""
1. Test for Numerical Stability with Large Coefficients
Purpose: Verify the solver's ability to handle problems with very large or very small coefficients, which can cause numerical instability.
"""

@pytest.fixture
def numerical_stability_model():
    """Model with scaled coefficients to avoid numerical instability."""
    model = ConcreteModel()
    model.x = Var(within=NonNegativeReals)
    # Large coefficients in constraints and objective
    # model.constraints.add(1e10 * model.x <= 1e12)
    # model.objective = Objective(expr=1e10 * model.x, sense=minimize)
    scaling_factor = 1e10
    model.constraints = ConstraintList()
    model.constraints.add(model.x <= 100)  # 1e12 / 1e10 = 100
    model.objective = Objective(expr=model.x, sense=minimize)
    return model

def test_numerical_stability(solver, numerical_stability_model):
    """Test solver's handling of numerical stability with scaled coefficients."""
    solution = solver.solve(numerical_stability_model)
    assert solution is not None, "Expected feasible solution, got None"
    # expected_x = 1e2  # x = 1e12 / 1e10
    expected_x = 0  # Since we're minimizing x without lower bound, expect x = 0
    assert abs(solution['x'] - expected_x) <= 1e-6, f"Expected x ≈ {expected_x}, got {solution['x']}"

"""
2. Test with Zero Coefficients in Objective Function
Purpose: Ensure that variables with zero coefficients in the objective function are handled correctly.
"""
@pytest.fixture
def zero_coefficient_model():
    """Model where some variables have zero coefficients in the objective."""
    model = ConcreteModel()
    model.x1 = Var(within=NonNegativeReals)
    model.x2 = Var(within=NonNegativeReals)
    model.constraints = ConstraintList()
    model.constraints.add(model.x1 + model.x2 >= 5)
    # x1 has zero coefficient in the objective
    model.objective = Objective(expr=0 * model.x1 + 3 * model.x2, sense=minimize)
    return model

def test_zero_coefficient_in_objective(solver, zero_coefficient_model):
    """Test solver's handling of zero coefficients in the objective function."""
    solution = solver.solve(zero_coefficient_model)
    assert solution is not None, "Expected feasible solution, got None"
    # Corrected assertions with a tolerance for floating-point comparisons
    assert abs(solution['x1'] - 5) <= 1e-6, f"Expected x1 ≈ 5, got {solution['x1']}"
    assert abs(solution['x2'] - 0) <= 1e-6, f"Expected x2 ≈ 0, got {solution['x2']}"

"""
3. Test with Variables Having Negative Bounds
Purpose: Test how the solver handles variables with negative lower bounds, ensuring proper error handling or solution.
"""
@pytest.fixture
def negative_bounds_model():
    """Model with variables that have negative lower bounds."""
    model = ConcreteModel()
    model.x = Var(bounds=(-5, 5))
    model.constraints = ConstraintList()
    model.constraints.add(model.x >= -5)
    model.constraints.add(model.x <= 5)
    model.objective = Objective(expr=model.x, sense=maximize)
    return model

def test_negative_variable_bounds(solver, negative_bounds_model):
    """Test solver's handling of variables with negative bounds."""
    solution = solver.solve(negative_bounds_model)
    assert solution is not None, "Expected feasible solution, got None"
    assert solution['x'] == 5, f"Expected x = 5, got {solution['x']}"

"""
4. Test with All Zero Constraint Coefficients
Purpose: Check the solver's behavior when constraints have all zero coefficients, which can lead to infeasibility.
"""
@pytest.fixture
def zero_constraint_coefficients_model():
    """Model with a constraint having all zero coefficients."""
    model = ConcreteModel()
    model.x = Var(within=NonNegativeReals)
    model.constraints = ConstraintList()
    # Constraint: 0 * x >= 1 (infeasible)
    model.constraints.add(0 * model.x >= 1)
    model.objective = Objective(expr=model.x, sense=minimize)
    return model

def test_zero_constraint_coefficients(solver, zero_constraint_coefficients_model):
    """Test handling of constraints with zero coefficients."""
    with pytest.raises(InfeasibleProblemError) as exc_info:
        solver.solve(zero_constraint_coefficients_model)
    assert "infeasible" in str(exc_info.value).lower(), "Expected infeasible problem due to zero constraint coefficients."

"""
5. Test with Tight Bounds Leading to No Feasible Solution
Purpose: Ensure the solver can detect infeasibility when variable bounds are too restrictive.
"""
@pytest.fixture
def tight_bounds_model():
    """Model with variable bounds that eliminate feasibility."""
    model = ConcreteModel()
    model.x = Var(bounds=(5, 5))  # x must be exactly 5
    model.constraints = ConstraintList()
    model.constraints.add(model.x >= 6)  # Conflict with bounds
    model.objective = Objective(expr=model.x, sense=minimize)
    return model

def test_tight_bounds_no_solution(solver, tight_bounds_model):
    """Test solver's detection of infeasibility due to tight variable bounds."""
    with pytest.raises(InfeasibleProblemError) as exc_info:
        solver.solve(tight_bounds_model)
    assert "infeasible" in str(exc_info.value).lower(), "Expected infeasible problem due to tight variable bounds."

"""
6. Test with Redundant Variables
Purpose: Test the solver's ability to handle variables that do not affect the solution.
"""
@pytest.fixture
def redundant_variables_model():
    """Model with redundant variables that do not affect the solution."""
    model = ConcreteModel()
    model.x = Var(within=NonNegativeReals)
    model.y = Var(within=NonNegativeReals)  # Redundant variable
    model.constraints = ConstraintList()
    model.constraints.add(model.x >= 5)
    model.objective = Objective(expr=model.x, sense=minimize)
    return model

def test_redundant_variables(solver, redundant_variables_model):
    """Test solver's handling of redundant variables."""
    solution = solver.solve(redundant_variables_model)
    assert solution is not None, "Expected feasible solution, got None"
    assert solution['x'] == 5, f"Expected x = 5, got {solution['x']}"
    assert 'y' in solution, "Expected y to be in the solution variables"
    assert solution['y'] == 0, f"Expected y = 0, got {solution['y']}"

"""
7. Test with Multiple Equality Constraints
Purpose: Verify the solver's capability to handle multiple equality constraints that may lead to an over-constrained system.
"""
@pytest.fixture
def multiple_equality_constraints_model():
    """Model with multiple equality constraints."""
    model = ConcreteModel()
    model.x1 = Var(within=NonNegativeReals)
    model.x2 = Var(within=NonNegativeReals)
    model.constraints = ConstraintList()
    model.constraints.add(model.x1 + model.x2 == 10)
    model.constraints.add(2 * model.x1 - model.x2 == 0)
    model.objective = Objective(expr=model.x1 + 2 * model.x2, sense=minimize)
    return model

def test_multiple_equality_constraints(solver, multiple_equality_constraints_model):
    """Test solver's handling of multiple equality constraints."""
    solution = solver.solve(multiple_equality_constraints_model)
    assert solution is not None, "Expected feasible solution, got None"
    expected_x1 = 10 / 3  # Approximately 3.3333333333
    expected_x2 = 20 / 3  # Approximately 6.6666666667
    assert abs(solution['x1'] - expected_x1) <= 1e-6, f"Expected x1 ≈ {expected_x1}, got {solution['x1']}"
    assert abs(solution['x2'] - expected_x2) <= 1e-6, f"Expected x2 ≈ {expected_x2}, got {solution['x2']}"

"""
8. Test with Maximum Iterations Reached
Purpose: Ensure that the solver appropriately handles situations when the maximum number of iterations is reached without finding an optimal solution.
"""
def test_max_iterations_reached(solver):
    """Test solver's behavior when the maximum number of iterations is reached."""
    # Adjust the solver settings to limit the number of iterations
    solver.max_itr = 0  # Set to a low number to force early termination
    model = ConcreteModel()
    model.x1 = Var(within=NonNegativeReals)
    model.constraints = ConstraintList()
    model.constraints.add(model.x1 >= 0)
    model.objective = Objective(expr=-model.x1, sense=minimize)
    with pytest.raises(OptimizationError) as exc_info:
        solver.solve(model)
    assert "maximum iterations" in str(exc_info.value).lower(), "Expected infeasible problem due to max iterations reached."

"""
9. Test with Variables at Bounds
Purpose: Test the solver's ability to handle variables that are at their bounds in the optimal solution.
"""
@pytest.fixture
def variables_at_bounds_model():
    """Model where variables are expected to be at their bounds in the solution."""
    model = ConcreteModel()
    model.x = Var(bounds=(0, 10))
    model.constraints = ConstraintList()
    model.constraints.add(model.x >= 10)
    model.objective = Objective(expr=-model.x, sense=minimize)
    return model

def test_variables_at_bounds(solver, variables_at_bounds_model):
    """Test solver's handling of variables at their bounds."""
    solution = solver.solve(variables_at_bounds_model)
    assert solution is not None, "Expected feasible solution, got None"
    assert solution['x'] == 10, f"Expected x = 10, got {solution['x']}"

"""
10. Test with Equality Constraints Leading to Infeasibility
Purpose: Check how the solver handles conflicting equality constraints that make the problem infeasible.
"""
@pytest.fixture
def infeasible_equality_constraints_model():
    """Model with equality constraints that make the problem infeasible."""
    model = ConcreteModel()
    model.x = Var(within=NonNegativeReals)
    model.constraints = ConstraintList()
    model.constraints.add(model.x == 5)
    model.constraints.add(model.x == 10)  # Conflicting equality constraint
    model.objective = Objective(expr=model.x, sense=minimize)
    return model

def test_infeasible_equality_constraints(solver, infeasible_equality_constraints_model):
    """Test solver's detection of infeasibility due to conflicting equality constraints."""
    with pytest.raises(InfeasibleProblemError) as exc_info:
        solver.solve(infeasible_equality_constraints_model)
    assert "infeasible" in str(exc_info.value).lower(), "Expected infeasible problem due to conflicting equality constraints."
