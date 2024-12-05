# test_tableau.py

import pytest
from pyomo.environ import ConcreteModel, Var, Constraint, Objective, NonNegativeReals, minimize, value
from totsu.core.modelstandardizer import ModelStandardizer
from totsu.core.tableau import Tableau
from totsu.utils.model_processor import ModelProcessor
from totsu.utils.model_builder import ModelBuilder

def test_tableau_initialization():
    # Create a simple LP model
    model = ModelBuilder.build_model_by_name('simple')

    # Standardize the model
    standardizer = ModelStandardizer(model)
    standardizer.standardize_model()

    # Initialize the tableau
    tableau = Tableau(standardizer)
    tableau.initialize_tableau()

    # Verify the size of the tableau
    num_constraints = len(standardizer.constraints)
    num_variables = len(standardizer.variables)
    assert tableau.tableau.shape == (num_constraints + 1, num_variables + 1), "Tableau dimensions are incorrect."

    # Verify that the tableau coefficients match the constraints and objective
    # Since this is a basic test, you can print the tableau for manual verification or compare expected values

    # Example: Check a specific coefficient
    var_name_to_index = tableau.var_name_to_index()
    x_idx = var_name_to_index['x']
    y_idx = var_name_to_index['y']

    # Check coefficients in the first constraint row
    assert tableau.tableau[0, x_idx] == 2, "Coefficient for x in first constraint is incorrect."
    assert tableau.tableau[0, y_idx] == 1, "Coefficient for y in first constraint is incorrect."

def test_pivot_operation():
    # Create a model that will require pivoting
    model = ModelBuilder.build_model_by_name('feasible')

    # Standardize the model
    standardizer = ModelStandardizer(model)
    standardizer.standardize_model()

    # Initialize the tableau
    tableau = Tableau(standardizer)
    tableau.initialize_tableau()

    # Perform one pivot operation manually
    # For this test, we'll simulate selecting pivot column and row
    pivot_col = tableau.select_pivot_column(phase=1)
    assert pivot_col is not None, "Pivot column should not be None."

    pivot_row = tableau.select_pivot_row(pivot_col)
    assert pivot_row is not None, "Pivot row should not be None."

    # Store the old basis variable
    leaving_var_idx = tableau.basis_vars[pivot_row]

    # Perform the pivot
    tableau.pivot_operation(pivot_row, pivot_col)

    # Check that the basis has been updated
    entering_var_idx = pivot_col
    assert tableau.basis_vars[pivot_row] == entering_var_idx, "Basis variable was not updated correctly."

    # Check that the tableau has been updated
    # For a more thorough test, compare the new tableau to expected values

    # Ensure that the leaving variable is now in non_basis_vars
    assert leaving_var_idx in tableau.non_basis_vars, "Leaving variable not in non_basis_vars."

def test_is_optimal():
    # Create the simple optimal model
    model = ModelBuilder.build_model_by_name('optimal')

    # Standardize and initialize the tableau
    standardizer = ModelStandardizer(model)
    standardizer.standardize_model()
    tableau = Tableau(standardizer)
    tableau.initialize_tableau()

    # Perform Phase I iterations
    while not tableau.is_optimal():
        pivot_col = tableau.select_pivot_column(phase=1)
        if pivot_col is None:
            break
        pivot_row = tableau.select_pivot_row(pivot_col)
        if pivot_row is None:
            break
        tableau.pivot_operation(pivot_row, pivot_col)

    # Now check that the tableau is optimal
    assert tableau.is_optimal(), "Tableau should be optimal after Phase I."

def test_is_feasible_after_phase1():
    # Create an infeasible model
    model = ModelBuilder.build_model_by_name('infeasible')

    # Standardize the model
    standardizer = ModelStandardizer(model)
    try:
        standardizer.standardize_model()
    except ValueError:
        # The model is infeasible during standardization
        pytest.skip("Model is infeasible during standardization.")
        return

    # Initialize the tableau
    tableau = Tableau(standardizer)
    tableau.initialize_tableau()

    # Perform Phase I iterations
    while not tableau.is_optimal():
        pivot_col = tableau.select_pivot_column(phase=1)
        if pivot_col is None:
            break
        pivot_row = tableau.select_pivot_row(pivot_col)
        if pivot_row is None:
            break
        tableau.pivot_operation(pivot_row, pivot_col)

    # Check feasibility after Phase I
    assert not tableau.is_feasible(), "Tableau incorrectly marked as feasible after Phase I."

def test_extract_solution():
    # Create a simple feasible model
    model = ModelBuilder.build_model_by_name("feasible")

    # Standardize the model
    standardizer = ModelStandardizer(model)
    standardizer.standardize_model()

    # Initialize the tableau
    tableau = Tableau(standardizer)
    tableau.initialize_tableau()

    # Since it's an equality constraint, an artificial variable is added
    # Perform Phase I iterations
    while not tableau.is_optimal():
        pivot_col = tableau.select_pivot_column(phase=1)
        if pivot_col is None:
            break
        pivot_row = tableau.select_pivot_row(pivot_col)
        if pivot_row is None:
            break
        tableau.pivot_operation(pivot_row, pivot_col)

    # Check feasibility
    assert tableau.is_feasible(), "Tableau should be feasible after Phase I."

    # Set Phase II objective
    tableau.set_phase2_objective()

    # Perform Phase II iterations
    while not tableau.is_optimal():
        pivot_col = tableau.select_pivot_column(phase=2)
        if pivot_col is None:
            break
        pivot_row = tableau.select_pivot_row(pivot_col)
        if pivot_row is None:
            break
        tableau.pivot_operation(pivot_row, pivot_col, 2)

    # Extract solution
    solution = tableau.extract_solution()

    # Check that the solution is correct
    assert abs(solution['x'] + solution['y'] - 10) <= 1e-6, "Solution does not satisfy the constraint."
    assert abs(solution['x'] + 2 * solution['y'] - tableau.get_current_objective_value()) <= 1e-6, "Objective value is incorrect."

def test_blands_rule():
    # Create a model that could potentially cycle without Bland's Rule
    model = ModelBuilder.build_model_by_name("cyclic")

    # Standardize the model
    standardizer = ModelStandardizer(model)
    standardizer.standardize_model()

    # Initialize the tableau
    tableau = Tableau(standardizer)
    tableau.initialize_tableau()

    # Perform Phase I iterations
    while not tableau.is_optimal():
        pivot_col = tableau.select_pivot_column(phase=1)
        if pivot_col is None:
            break
        pivot_row = tableau.select_pivot_row(pivot_col)
        if pivot_row is None:
            break
        tableau.pivot_operation(pivot_row, pivot_col)

    # Check feasibility
    assert tableau.is_feasible(), "Tableau should be feasible after Phase I."

    # Set Phase II objective
    tableau.set_phase2_objective()

    # Perform Phase II iterations
    iteration_count = 0
    max_iterations = 100  # Set a reasonable limit to prevent infinite loops
    while not tableau.is_optimal() and iteration_count < max_iterations:
        pivot_col = tableau.select_pivot_column(phase=2)
        if pivot_col is None:
            break
        pivot_row = tableau.select_pivot_row(pivot_col)
        if pivot_row is None:
            break
        tableau.pivot_operation(pivot_row, pivot_col)
        iteration_count += 1

    assert iteration_count < max_iterations, "Bland's Rule failed to prevent cycling."

def test_negative_rhs_feasible():
    model = ModelBuilder.build_model_by_name("negative_rhs")

    # Standardize the model
    standardizer = ModelStandardizer(model)
    standardizer.standardize_model()

    # Initialize the tableau
    tableau = Tableau(standardizer)
    tableau.initialize_tableau()

    # Perform Phase I iterations
    while not tableau.is_optimal():
        pivot_col = tableau.select_pivot_column(phase=1)
        if pivot_col is None:
            break
        pivot_row = tableau.select_pivot_row(pivot_col)
        if pivot_row is None:
            break
        tableau.pivot_operation(pivot_row, pivot_col)

    # Check feasibility
    assert tableau.is_feasible(), "Tableau should be feasible after Phase I."

    # Set Phase II objective
    tableau.set_phase2_objective()

    # Perform Phase II iterations
    while not tableau.is_optimal():
        pivot_col = tableau.select_pivot_column(phase=2)
        if pivot_col is None:
            break
        pivot_row = tableau.select_pivot_row(pivot_col)
        if pivot_row is None:
            break
        tableau.pivot_operation(pivot_row, pivot_col)

    # Extract solution
    solution = tableau.extract_solution()

    # Verify the solution satisfies the adjusted constraints
    x_val = solution['x']
    y_val = solution['y']
    assert 2 * x_val + y_val >= -4 - 1e-6, "Solution does not satisfy constraint 1."
    assert x_val + 2 * y_val <= 6 + 1e-6, "Solution does not satisfy constraint 2."
    assert abs(x_val + y_val - tableau.get_current_objective_value()) <= 1e-6, "Objective value is incorrect."
    
