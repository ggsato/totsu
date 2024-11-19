# test_simplex_solver.py

import pytest
from pyomo.environ import *
from totsu.core.super_simplex_solver import SuperSimplexSolver, InfeasibleProblemError, UnboundedProblemError

"""
1. **Feasibility and Optimality**:
   - `test_feasibility`: Tests that solutions satisfy all constraints.
   - `test_optimality`: Checks if the solver finds the optimal solution.

2. **Unbounded and Infeasible Problems**:
   - `test_unbounded_problem`: Tests identification of unbounded problems.
   - `test_infeasible_problem`: Tests detection of infeasible problems.
   - `test_problem_with_no_feasible_solution`: Another test for infeasibility.

3. **Degeneracy and Multiple Optimal Solutions**:
   - `test_degenerate_solution`: Tests handling of degeneracy.
   - `test_alternate_optimal_solutions`: Tests for multiple optimal solutions.

4. **Variable Bounds and Non-Negativity**:
   - `test_variable_upper_bounds`: Tests variables with upper bounds.
   - `test_non_negativity_violations`: Tests handling of non-negativity constraints.
   - `test_upper_bound_variable`: Tests a variable with an upper bound.

5. **Redundant Constraints and Mixed Constraints**:
   - `test_redundant_constraints`: Tests impact of redundant constraints.
   - `test_mixed_constraints`: Tests mixed equality and inequality constraints.

6. **Objective Function Variations**:
   - `test_negative_right_hand_side`: Tests handling of negative RHS values.
   - `test_negative_objective_coefficients`: Tests negative coefficients in the objective function.
   - `test_fractional_coefficients`: Tests fractional coefficients in constraints and objective.
   - `test_problem_with_zero_objective`: Tests a zero objective function.
   - `test_minimization_problem`: Test solving a standard minimization problem.

7. **Scaling and Performance**:
   - `test_large_scale_problem`: Tests solver performance on a larger problem.

8. **Special Cases and Edge Conditions**:
   - `test_cycling_prevention`: Ensures the solver handles potential cycling scenarios.
   - `test_convexity`: Tests that multiple feasible starting points lead to the same optimal solution.
   - `test_basic_non_basic_partition`: Verifies basic and non-basic variable partitioning.
"""

# Fixture to initialize the Simplex solver
@pytest.fixture
def solver():
    """Fixture to create and return a SimplexSolver instance."""
    return SuperSimplexSolver()

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
def basic_lp_model_reordered():
    # Create a new model with the same constraints but in different order
    model_reordered = ConcreteModel()
    model_reordered.x1 = Var(within=NonNegativeReals)
    model_reordered.x2 = Var(within=NonNegativeReals)
    # Add constraints in reversed order
    model_reordered.c2 = Constraint(expr=2 * model_reordered.x1 + model_reordered.x2 <= 5)
    model_reordered.c1 = Constraint(expr=model_reordered.x1 + model_reordered.x2 <= 4)
    # Same objective function
    model_reordered.objective = Objective(expr=3 * model_reordered.x1 + 2 * model_reordered.x2, sense=maximize)
    return model_reordered

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
def degenerate_model():
    """Model designed to test degeneracy (multiple basic feasible solutions)."""
    model = ConcreteModel()
    model.x1 = Var(within=NonNegativeReals)
    model.x2 = Var(within=NonNegativeReals)
    model.constraints = ConstraintList()
    # Constraints leading to degeneracy
    model.constraints.add(model.x1 + model.x2 <= 10)
    model.constraints.add(2 * model.x1 + 2 * model.x2 <= 20)
    model.constraints.add(model.x1 <= 5)
    model.objective = Objective(expr=3 * model.x1 + 3 * model.x2, sense=maximize)
    return model

@pytest.fixture
def redundant_constraints_model():
    """Model with redundant constraints that do not affect the feasible region."""
    model = ConcreteModel()
    model.x1 = Var(within=NonNegativeReals)
    model.x2 = Var(within=NonNegativeReals)
    model.constraints = ConstraintList()
    model.constraints.add(model.x1 + model.x2 <= 4)
    model.constraints.add(2 * model.x1 + 2 * model.x2 <= 8)  # Redundant constraint
    model.objective = Objective(expr=3 * model.x1 + 2 * model.x2, sense=maximize)
    return model

@pytest.fixture
def non_negativity_violations_model():
    """Model to test handling of non-negativity constraints violations."""
    model = ConcreteModel()
    model.x1 = Var(within=NonNegativeReals)
    model.x2 = Var(within=NonNegativeReals)
    model.constraints = ConstraintList()
    model.constraints.add(model.x1 - model.x2 <= 2)
    model.constraints.add(-model.x1 + model.x2 <= 1)
    model.objective = Objective(expr=model.x1 - model.x2, sense=maximize)
    return model

@pytest.fixture
def alternate_optimal_solutions_model():
    """Model with multiple optimal solutions to test solver's handling."""
    model = ConcreteModel()
    model.x1 = Var(within=NonNegativeReals)
    model.x2 = Var(within=NonNegativeReals)
    model.constraints = ConstraintList()
    model.constraints.add(model.x1 + model.x2 == 4)
    model.constraints.add(model.x1 - model.x2 == 0)
    model.objective = Objective(expr=3 * model.x1 + 3 * model.x2, sense=minimize)
    return model

@pytest.fixture
def negative_rhs_model():
    """Model with a negative right-hand side in the constraint."""
    model = ConcreteModel()
    model.x = Var(within=NonNegativeReals)
    model.constraints = ConstraintList()
    # Constraint: -2x >= -4  => x <= 2
    model.constraints.add(-2 * model.x >= -4)
    model.objective = Objective(expr=model.x, sense=minimize)
    return model

@pytest.fixture
def variable_upper_bounds_model():
    """Model with a variable upper bound other than infinity."""
    model = ConcreteModel()
    # Variable x with upper bound of 10
    model.x = Var(bounds=(0, 10))
    model.constraints = ConstraintList()
    model.constraints.add(model.x >= 5)
    model.objective = Objective(expr=model.x, sense=minimize)
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

@pytest.fixture
def cycling_prevention_model():
    """Model that could potentially cause cycling in the Simplex algorithm."""
    model = ConcreteModel()
    # Variables
    model.x1 = Var(within=NonNegativeReals)
    model.x2 = Var(within=NonNegativeReals)
    model.x3 = Var(within=NonNegativeReals)
    model.x4 = Var(within=NonNegativeReals)
    model.x5 = Var(within=NonNegativeReals)
    model.constraints = ConstraintList()
    # Constraints designed to create cycling scenarios
    model.constraints.add(0.5 * model.x1 + 13 * model.x2 + model.x3 + model.x4 == 15)
    model.constraints.add(0.5 * model.x1 + 5 * model.x2 + model.x5 == 10)
    model.objective = Objective(expr=model.x1, sense=maximize)
    return model

@pytest.fixture
def large_scale_model():
    """Model representing a larger scale problem to test performance."""
    model = ConcreteModel()
    # Ten variables
    model.vars = Var(range(1, 11), within=NonNegativeReals)
    model.constraints = ConstraintList()
    # Multiple constraints to increase problem size
    for i in range(1, 6):
        model.constraints.add(sum(model.vars[j] for j in range(1, 11)) >= i * 10)
    model.objective = Objective(expr=sum(2 * model.vars[j] for j in range(1, 11)), sense=minimize)
    return model

@pytest.fixture
def minimization_problem_model():
    """Standard minimization problem."""
    model = ConcreteModel()
    # Decision variables
    model.x = Var(within=NonNegativeReals)
    model.y = Var(within=NonNegativeReals)
    model.constraints = ConstraintList()
    # Constraints
    model.constraints.add(2 * model.x + model.y >= 20)
    model.constraints.add(model.x + 3 * model.y >= 30)
    model.objective = Objective(expr=3 * model.x + 2 * model.y, sense=minimize)
    return model

@pytest.fixture
def no_feasible_solution_model():
    """Model with conflicting constraints resulting in infeasibility."""
    model = ConcreteModel()
    model.x = Var(within=NonNegativeReals)
    model.constraints = ConstraintList()
    # Conflicting constraints
    model.constraints.add(model.x >= 5)
    model.constraints.add(model.x <= 3)
    model.objective = Objective(expr=model.x, sense=minimize)
    return model

@pytest.fixture
def upper_bound_model():
    """Model with an upper bound on a variable."""
    model = ConcreteModel()
    # Variable with upper bound
    model.x = Var(bounds=(0, 10))
    model.constraints = ConstraintList()
    model.constraints.add(model.x >= 5)
    model.objective = Objective(expr=-model.x, sense=maximize)
    return model

@pytest.fixture
def negative_objective_coefficients_model():
    """Model with negative coefficients in the objective function."""
    model = ConcreteModel()
    # Decision variables
    model.x1 = Var(within=NonNegativeReals)
    model.x2 = Var(within=NonNegativeReals)
    model.constraints = ConstraintList()
    # Constraints
    model.constraints.add(2 * model.x1 + model.x2 <= 10)
    model.constraints.add(model.x1 + 3 * model.x2 <= 15)
    # Objective with negative coefficients
    model.objective = Objective(expr=-4 * model.x1 - 3 * model.x2, sense=minimize)
    return model

@pytest.fixture
def zero_objective_model():
    """Model where the objective function is zero."""
    model = ConcreteModel()
    model.x = Var(within=NonNegativeReals)
    model.constraints = ConstraintList()
    # Simple bounds on x
    model.constraints.add(model.x >= 0)
    model.constraints.add(model.x <= 10)
    model.objective = Objective(expr=0 * model.x, sense=minimize)
    return model

@pytest.fixture
def fractional_coefficients_model():
    """Model with fractional coefficients in constraints and objective."""
    model = ConcreteModel()
    model.x = Var(within=NonNegativeReals)
    model.constraints = ConstraintList()
    # Constraint with fractional coefficient
    model.constraints.add(0.5 * model.x >= 1.5)
    model.objective = Objective(expr=1.5 * model.x, sense=minimize)
    return model

@pytest.fixture
def free_variable_model():
    """Model with variables unrestricted in sign."""
    model = ConcreteModel()
    model.x = Var()  # Unrestricted variable
    model.constraints = ConstraintList()
    model.constraints.add(2 * model.x >= 4)
    model.objective = Objective(expr=model.x, sense=minimize)
    return model



# Test methods using the fixtures
def test_feasibility(solver, basic_lp_model):
    """Test that solutions satisfy all constraints."""
    solution = solver.solve(basic_lp_model)
    assert solution is not None, "Expected feasible solution, got None"


def test_unbounded_problem(solver, unbounded_model):
    """Test that the solver identifies an unbounded problem."""
    with pytest.raises(UnboundedProblemError) as exc_info:
        solver.solve(unbounded_model)
    assert "unbounded" in str(exc_info.value).lower(), "Unboundedness not correctly detected."

def test_optimality(solver, basic_lp_model):
    """Test that the solver finds the optimal solution."""
    solution = solver.solve(basic_lp_model)
    assert solution is not None, "Expected optimal solution, got None"
    assert solver.is_optimal(), "Final tableau does not represent an optimal solution"


def test_convexity(solver, basic_lp_model, basic_lp_model_reordered):
    """Test that multiple feasible starting points lead to the same optimal solution."""
    # First solve with the original model
    solution1 = solver.solve(basic_lp_model)
    
    # Solve with the reordered model
    solution2 = solver.solve(basic_lp_model_reordered)
    
    # Compare the solutions
    tolerance = 1e-6
    assert abs(solution1['x1'] - solution2['x1']) <= tolerance, f"Expected x1 to be the same, got {solution1['x1']} and {solution2['x1']}"
    assert abs(solution1['x2'] - solution2['x2']) <= tolerance, f"Expected x2 to be the same, got {solution1['x2']} and {solution2['x2']}"
    assert abs(solver.get_current_objective_value() - solver.get_current_objective_value()) <= tolerance, "Objective values differ"


def test_basic_non_basic_partition(solver, basic_lp_model):
    """Verify basic and non-basic variable partitioning after each pivot."""
    solver.solve(basic_lp_model)
    basic_vars = solver.basic_variables
    non_basic_vars = solver.non_basic_variables
    assert len(basic_vars) == solver.tableau.shape[0] - 1, "Mismatch in basic variable count"
    assert not set(basic_vars).intersection(non_basic_vars), "Overlap between basic and non-basic variables"

def test_standard_lp(solver, basic_lp_model):
    """Test a standard LP problem to verify correctness of the solution."""
    solution = solver.solve(basic_lp_model)
    assert abs(solution['x1'] - 1) <= 1e-6, f"Expected x1 = 1, got {solution['x1']}"
    assert abs(solution['x2'] - 3) <= 1e-6, f"Expected x2 = 3, got {solution['x2']}"

def test_infeasible_problem(solver, infeasible_model):
    """Test the solver's ability to detect infeasibility."""
    with pytest.raises(InfeasibleProblemError) as exc_info:
        solver.solve(infeasible_model)
    assert "infeasible" in str(exc_info.value).lower(), "Infeasibility not correctly detected."

def test_degenerate_solution(solver, degenerate_model):
    """Test handling of degeneracy in feasible solutions."""
    solution = solver.solve(degenerate_model)
    assert solution is not None, "Expected feasible solution, got None"

    x1 = solution['x1']
    x2 = solution['x2']

    # Validate constraints are satisfied
    assert x1 + x2 <= 10 + 1e-6, f"Constraint x1 + x2 <= 10 violated: {x1 + x2}"
    assert 2 * x1 + 2 * x2 <= 20 + 1e-6, f"Constraint 2x1 + 2x2 <= 20 violated: {2 * x1 + 2 * x2}"
    assert x1 <= 5 + 1e-6, f"Constraint x1 <= 5 violated: {x1}"

    # Validate objective value
    expected_objective = 3 * x1 + 3 * x2
    assert abs(expected_objective - solver.get_current_objective_value()) <= 1e-6, "Objective value mismatch"


def test_redundant_constraints(solver, redundant_constraints_model):
    """Test that redundant constraints do not affect the optimal solution."""
    solution = solver.solve(redundant_constraints_model)
    assert abs(solution['x1'] - 4) <= 1e-6, f"Expected x1 = 4, got {solution['x1']}"
    assert abs(solution['x2']) <= 1e-6, f"Expected x2 = 0, got {solution['x2']}"

def test_non_negativity_violations(solver, non_negativity_violations_model):
    """Test solver's handling when variables could potentially violate non-negativity constraints."""
    solution = solver.solve(non_negativity_violations_model)
    assert solution['x1'] >= -1e-6, f"Expected x1 to be non-negative, got {solution['x1']}"
    assert solution['x2'] >= -1e-6, f"Expected x2 to be non-negative, got {solution['x2']}"

def test_alternate_optimal_solutions(solver, alternate_optimal_solutions_model):
    """Test for multiple optimal solutions."""
    solution = solver.solve(alternate_optimal_solutions_model)
    # Both x1 and x2 should be 2 in this case
    assert abs(solution['x1'] - 2) <= 1e-6, f"Expected x1 ≈ 2, got {solution['x1']}"
    assert abs(solution['x2'] - 2) <= 1e-6, f"Expected x2 ≈ 2, got {solution['x2']}"

def test_negative_right_hand_side(solver, negative_rhs_model):
    """ Test handling of negative right-hand sides in constraints."""
    solution = solver.solve(negative_rhs_model)
    assert abs(solution['x']) <= 1e-6, f"Expected x = 0, got {solution['x']}"

def test_variable_upper_bounds(solver, variable_upper_bounds_model):
    """Test variables with specified upper bounds."""
    solution = solver.solve(variable_upper_bounds_model)
    assert abs(solution['x'] - 5) <= 1e-6, f"Expected x = 5, got {solution['x']}"

def test_cycling_prevention(solver, cycling_prevention_model):
    """Test to ensure the solver handles potential cycling scenarios."""
    solution = solver.solve(cycling_prevention_model)
    assert solution is not None, "Expected feasible solution, got None"

def test_large_scale_problem(solver, large_scale_model):
    """Test a larger problem to assess performance on complex models."""
    solution = solver.solve(large_scale_model)
    assert solution is not None, "Expected feasible solution, got None"
    # Verify total sum meets the constraint
    total = sum(solution[f'vars[{j}]'] for j in range(1, 11))
    assert total >= 50, f"Expected total >= 50, got {total}"

def test_problem_with_no_feasible_solution(solver, no_feasible_solution_model):
    """Test that the solver identifies an infeasible problem."""
    with pytest.raises(InfeasibleProblemError) as exc_info:
        solver.solve(no_feasible_solution_model)
    assert "infeasible" in str(exc_info.value).lower(), "Infeasibility not correctly detected."

def test_problem_with_zero_objective(solver, zero_objective_model):
    """Test a problem where the objective function is zero."""
    solution = solver.solve(zero_objective_model)
    assert abs(solution['x']) <= 1e-6, f"Expected x = 0, got {solution['x']}"

def test_fractional_coefficients(solver, fractional_coefficients_model):
    """Test a problem with fractional coefficients."""
    solution = solver.solve(fractional_coefficients_model)
    assert abs(solution['x'] - 3) <= 1e-6, f"Expected x = 3, got {solution['x']}"

def test_mixed_constraints(solver, mixed_constraints_model):
    """Test solving a problem with mixed equality and inequality constraints."""
    solution = solver.solve(mixed_constraints_model)
    assert solution is not None, "Expected feasible solution, got None"

    x1 = solution['x1']
    x2 = solution['x2']
    x3 = solution['x3']

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
    assert abs(expected_objective - solver.get_current_objective_value()) <= 1e-6, "Objective value mismatch"

def test_negative_objective_coefficients(solver, negative_objective_coefficients_model):
    """Test handling of negative coefficients in the objective function."""
    solution = solver.solve(negative_objective_coefficients_model)
    assert solution is not None, "Expected feasible solution, got None"

    x1 = solution['x1']
    x2 = solution['x2']

    # Validate constraints are satisfied
    assert 2 * x1 + x2 <= 10 + 1e-6, f"Constraint 2x1 + x2 <= 10 violated: {2 * x1 + x2}"
    assert x1 + 3 * x2 <= 15 + 1e-6, f"Constraint x1 + 3x2 <= 15 violated: {x1 + 3 * x2}"

    # Validate variable bounds
    assert x1 >= 0, f"x1 should be non-negative, got {x1}"
    assert x2 >= 0, f"x2 should be non-negative, got {x2}"

    # Since the objective has negative coefficients and we're minimizing, the solver should maximize x1 and x2 within constraints
    expected_objective = -4 * x1 - 3 * x2
    assert abs(expected_objective - solver.get_current_objective_value()) <= 1e-6, "Objective value mismatch"

def test_minimization_problem(solver, minimization_problem_model):
    """Test solving a standard minimization problem."""
    solution = solver.solve(minimization_problem_model)
    assert solution is not None, "Expected feasible solution, got None"

    x_val = solution['x']
    y_val = solution['y']

    # Validate constraints are satisfied
    assert 2 * x_val + y_val >= 20 - 1e-6, f"Constraint 2x + y >= 20 violated: {2 * x_val + y_val}"
    assert x_val + 3 * y_val >= 30 - 1e-6, f"Constraint x + 3y >= 30 violated: {x_val + 3 * y_val}"

    # Validate variable bounds
    assert x_val >= 0, f"x should be non-negative, got {x_val}"
    assert y_val >= 0, f"y should be non-negative, got {y_val}"

    # Validate objective value
    expected_objective = 3 * x_val + 2 * y_val
    assert abs(expected_objective - solver.get_current_objective_value()) <= 1e-6, "Objective value mismatch"

def test_upper_bound_variable(solver, upper_bound_model):
    """Test a problem with variable upper bounds in a maximization context."""
    solution = solver.solve(upper_bound_model)
    assert solution is not None, "Expected feasible solution, got None"

    x = solution['x']

    # Validate constraints are satisfied
    assert x >= 5 - 1e-6, f"Constraint x >= 5 violated: x = {x}"
    assert x <= 10 + 1e-6, f"Upper bound x <= 10 violated: x = {x}"

    # Objective is to maximize -x, so we should minimize x within bounds
    # Expected optimal x is 5
    assert abs(x - 5) <= 1e-6, f"Expected x = 5, got {x}"

    expected_objective = -x
    assert abs(expected_objective - solver.get_current_objective_value()) <= 1e-6, "Objective value mismatch"

def test_free_variable(solver, free_variable_model):
    """Test handling of variables unrestricted in sign."""
    solution = solver.solve(free_variable_model)
    assert solution is not None, "Expected feasible solution, got None"

    x_val = solution['x']

    # Validate constraints are satisfied
    assert 2 * x_val >= 4 - 1e-6, f"Constraint 2x >= 4 violated: {2 * x_val}"

    # Since we're minimizing x, and x can be negative, the solver should find the smallest x satisfying the constraint
    expected_x = 2  # Since 2 * x >= 4 => x >= 2
    assert x_val >= expected_x - 1e-6, f"Expected x >= {expected_x}, got {x_val}"
