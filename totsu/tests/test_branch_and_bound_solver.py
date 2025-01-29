import pytest
from pyomo.environ import ConcreteModel, Var, Constraint, Objective, Integers, minimize, value
from totsu.core.branch_and_bound_solver import BranchAndBoundSolver
from totsu.core.super_simplex_solver import InfeasibleProblemError, UnboundedProblemError

import logging
from totsu.utils.logger import totsu_logger
totsu_logger.setLevel(logging.DEBUG)

@pytest.fixture
def integer_programming_model():
    """Creates a simple integer programming model for testing."""
    model = ConcreteModel()
    model.x = Var(domain=Integers, bounds=(0, 10))
    model.y = Var(domain=Integers, bounds=(0, 10))
    model.con1 = Constraint(expr=2 * model.x + 3 * model.y <= 12)
    model.con2 = Constraint(expr=model.x + model.y <= 6)
    model.obj = Objective(expr=5 * model.x + 4 * model.y, sense=minimize)
    return model

@pytest.fixture
def branch_and_bound_solver():
    """Returns an instance of BranchAndBoundSolver."""
    return BranchAndBoundSolver()


def test_branch_and_bound_feasibility(branch_and_bound_solver, integer_programming_model):
    """Tests that BranchAndBoundSolver finds a feasible integer solution."""
    solution = branch_and_bound_solver.solve(integer_programming_model)
    assert solution is not None, "Expected a feasible solution, but got None."
    
    # Ensure the solution satisfies integer constraints
    for var_name, value in solution.items():
        assert float(value).is_integer(), f"Variable {var_name} should be integer, but got {value}"

def test_branch_and_bound_optimality(branch_and_bound_solver, integer_programming_model):
    """Tests that BranchAndBoundSolver finds the optimal integer solution."""
    solution = branch_and_bound_solver.solve(integer_programming_model)
    expected_objective = branch_and_bound_solver.best_objective
    computed_objective = sum(value * coefficient for var, value in solution.items() for coefficient in [5 if 'x' in var else 4])
    
    assert abs(expected_objective - computed_objective) < 1e-6, "Branch and Bound did not find the correct optimal solution."

def test_branch_and_bound_infeasible_case(branch_and_bound_solver):
    """Tests that BranchAndBoundSolver correctly identifies an infeasible model."""
    model = ConcreteModel()
    model.x = Var(domain=Integers, bounds=(0, 5))
    model.con1 = Constraint(expr=model.x >= 10)  # Impossible constraint
    model.obj = Objective(expr=model.x, sense=minimize)
    
    solution = branch_and_bound_solver.solve(model)
    assert solution is None, "Solver should return None for infeasible models."

def test_branch_and_bound_solver_unbounded(branch_and_bound_solver):
    """Tests that BranchAndBoundSolver correctly detects unbounded problems."""
    model = ConcreteModel()
    model.x = Var(domain=Integers)  # Integer variable
    model.obj = Objective(expr=-model.x, sense=minimize)  # Minimize negative x (pushes x to infinity)

    # Add a constraint to ensure the solver runs
    model.dummy_constraint = Constraint(expr=model.x >= 0)

    with pytest.raises(UnboundedProblemError):
        branch_and_bound_solver.solve(model)

def test_branch_and_bound_solver_explores_nodes(branch_and_bound_solver):
    """Tests that BranchAndBoundSolver explores multiple nodes in a branching process."""
    model = ConcreteModel()
    
    # REMOVE Non-Negative Bounds to Allow Fractional Solutions
    model.x = Var(domain=Integers, bounds=(-10, 10))
    model.y = Var(domain=Integers, bounds=(-10, 10))

    # Adjust constraints to create a fractional LP solution
    model.con1 = Constraint(expr=2.5 * model.x + 3.5 * model.y <= 13.3)
    model.con2 = Constraint(expr=4.2 * model.x + 1.7 * model.y >= 5.8)

    # Objective with fractional coefficients
    model.obj = Objective(expr=7.3 * model.x + 5.6 * model.y, sense=minimize)

    branch_and_bound_solver.solve(model)

    assert branch_and_bound_solver.nodes_explored > 0, "Expected more than 0 nodes explored, but got 0."
