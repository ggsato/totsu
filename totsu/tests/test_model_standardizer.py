# test_model_standardizer.py

import pytest
from pyomo.environ import ConcreteModel, Var, Constraint, Objective, NonNegativeReals, Reals, Integers, minimize, maximize, value
from pyomo.repn import generate_standard_repn

from totsu.core.modelstandardizer import ModelStandardizer

def test_less_than_constraints():
    # Create a simple model with less-than constraints
    model = ConcreteModel()
    model.x = Var(domain=NonNegativeReals)
    model.y = Var(domain=NonNegativeReals)
    model.con1 = Constraint(expr=2 * model.x + 3 * model.y <= 6)
    model.obj = Objective(expr=model.x + model.y, sense=minimize)

    # Standardize the model
    standardizer = ModelStandardizer(model)
    standard_model = standardizer.standardize_model()

    # Check that a slack variable was added
    slack_vars = [var for var in standard_model.component_objects(Var) if 'slack' in var.name]
    assert len(slack_vars) == 1, "Slack variable not added for less-than constraint."

    # Check that the constraint was converted to an equality
    constraints = list(standard_model.component_data_objects(Constraint, active=True))
    assert len(constraints) == 1, "There should be one constraint after standardization."
    con = constraints[0]
    assert con.lower == con.upper, "Constraint should be an equality after adding slack variable."

    # Verify that the slack variable is in the constraint
    slack_var = slack_vars[0]
    repn = generate_standard_repn(con.body)
    var_names = [var.name for var in repn.linear_vars]
    assert slack_var.name in var_names, "Slack variable not present in standardized constraint."

def test_greater_than_constraints():
    # Create a model with a greater-than constraint
    model = ConcreteModel()
    model.x = Var(domain=NonNegativeReals)
    model.con1 = Constraint(expr=4 * model.x >= 8)
    model.obj = Objective(expr=model.x, sense=minimize)

    # Standardize the model
    standardizer = ModelStandardizer(model)
    standard_model = standardizer.standardize_model()

    # Check that surplus and artificial variables were added
    surplus_vars = [var for var in standard_model.component_objects(Var) if 'surplus' in var.name]
    artificial_vars = [var for var in standard_model.component_objects(Var) if 'artificial' in var.name]
    assert len(surplus_vars) == 1, "Surplus variable not added for greater-than constraint."
    assert len(artificial_vars) == 1, "Artificial variable not added for greater-than constraint."

    # Check that the constraint was converted to an equality
    constraints = list(standard_model.component_data_objects(Constraint, active=True))
    assert len(constraints) == 1, "There should be one constraint after standardization."
    con = constraints[0]
    assert con.lower == con.upper, "Constraint should be an equality after adding surplus and artificial variables."

    # Verify that the surplus and artificial variables are in the constraint
    surplus_var = surplus_vars[0]
    artificial_var = artificial_vars[0]
    repn = generate_standard_repn(con.body)
    var_names = [var.name for var in repn.linear_vars]
    assert surplus_var.name in var_names, "Surplus variable not present in standardized constraint."
    assert artificial_var.name in var_names, "Artificial variable not present in standardized constraint."

def test_equality_constraints():
    # Create a model with an equality constraint
    model = ConcreteModel()
    model.x = Var(domain=NonNegativeReals)
    model.con1 = Constraint(expr=5 * model.x == 10)
    model.obj = Objective(expr=model.x, sense=minimize)

    # Standardize the model
    standardizer = ModelStandardizer(model)
    standard_model = standardizer.standardize_model()

    # Check that an artificial variable was added
    artificial_vars = [var for var in standard_model.component_objects(Var) if 'artificial' in var.name]
    assert len(artificial_vars) == 1, "Artificial variable not added for equality constraint."

    # Check that the constraint remains an equality
    constraints = list(standard_model.component_data_objects(Constraint, active=True))
    assert len(constraints) == 1, "There should be one constraint after standardization."
    con = constraints[0]
    assert con.lower == con.upper, "Constraint should remain an equality."

    # Verify that the artificial variable is in the constraint
    artificial_var = artificial_vars[0]
    repn = generate_standard_repn(con.body)
    var_names = [var.name for var in repn.linear_vars]
    assert artificial_var.name in var_names, "Artificial variable not present in standardized constraint."

def test_mixed_constraints():
    # Create a model with mixed constraint types
    model = ConcreteModel()
    model.x = Var(domain=NonNegativeReals)
    model.y = Var(domain=NonNegativeReals)
    model.con1 = Constraint(expr=model.x + model.y <= 5)
    model.con2 = Constraint(expr=2 * model.x - model.y >= 3)
    model.con3 = Constraint(expr=model.x - 2 * model.y == 0)
    model.obj = Objective(expr=model.x - model.y, sense=maximize)

    # Standardize the model
    standardizer = ModelStandardizer(model)
    standard_model = standardizer.standardize_model()

    # Check that slack, surplus, and artificial variables were added appropriately
    slack_vars = [var for var in standard_model.component_objects(Var) if 'slack' in var.name]
    surplus_vars = [var for var in standard_model.component_objects(Var) if 'surplus' in var.name]
    artificial_vars = [var for var in standard_model.component_objects(Var) if 'artificial' in var.name]

    assert len(slack_vars) == 1, "One slack variable should be added."
    assert len(surplus_vars) == 1, "One surplus variable should be added."
    assert len(artificial_vars) == 2, "Two artificial variables should be added."

    # Check that all constraints are converted to equalities
    constraints = list(standard_model.component_data_objects(Constraint, active=True))
    assert len(constraints) == 3, "There should be three constraints after standardization."
    for con in constraints:
        assert con.lower == con.upper, "All constraints should be equalities after standardization."

def test_infeasible_variable_bounds():
    # Create a model with infeasible variable bounds
    model = ConcreteModel()
    model.x = Var(bounds=(5, 3))  # Lower bound > Upper bound
    model.obj = Objective(expr=model.x, sense=minimize)

    # Standardize the model and expect it to raise a ValueError
    with pytest.raises(ValueError, match="Model is infeasible during preprocessing."):
        standardizer = ModelStandardizer(model)
        standardizer.standardize_model()

def test_fixed_variables():
    # Create a model with fixed variables
    model = ConcreteModel()
    model.x = Var(bounds=(2, 2))  # Fixed variable
    model.y = Var(domain=NonNegativeReals)
    model.con1 = Constraint(expr=model.x + model.y >= 4)
    model.obj = Objective(expr=model.y, sense=minimize)

    # Standardize the model
    standardizer = ModelStandardizer(model)
    standard_model = standardizer.standardize_model()

    # Check that the fixed variable is handled correctly
    fixed_vars = [var for var in standard_model.component_data_objects(Var) if var.fixed]
    assert len(fixed_vars) == 1, "Fixed variable should be identified and handled."
    assert fixed_vars[0].name == 'x', "Fixed variable should be 'x'."

    # Check that the non-fixed variables include y and any added variables
    variables = standardizer.variables
    non_fixed_vars = [var for var in variables if not var.fixed]

    # Expect y and any slack/artificial variables
    expected_non_fixed_var_names = {'y'}
    expected_non_fixed_var_names.update(var.name for var in standardizer.artificial_vars)
    expected_non_fixed_var_names.update(var.name for var in variables if 'slack' in var.name or 'surplus' in var.name)

    actual_non_fixed_var_names = {var.name for var in non_fixed_vars}

    assert actual_non_fixed_var_names == expected_non_fixed_var_names, (
        f"Non-fixed variables do not match expected.\n"
        f"Expected: {expected_non_fixed_var_names}\n"
        f"Found: {actual_non_fixed_var_names}"
    )

def test_objective_adjustment_for_phase1():
    # Create a model that requires artificial variables
    model = ConcreteModel()
    model.x = Var(domain=NonNegativeReals)
    model.con1 = Constraint(expr=model.x >= 5)
    model.obj = Objective(expr=model.x, sense=minimize)

    # Standardize the model
    standardizer = ModelStandardizer(model)
    standard_model = standardizer.standardize_model()

    # Check that the original objective is deactivated
    original_objectives = [obj for obj in standard_model.component_data_objects(Objective, active=True) if obj.name == 'obj']
    assert len(original_objectives) == 0, "Original objective should be deactivated."

    # Check that the phase 1 objective is added and active
    phase1_objectives = [obj for obj in standard_model.component_data_objects(Objective, active=True) if 'obj_phase1' in obj.name]
    assert len(phase1_objectives) == 1, "Phase 1 objective should be active."

def test_no_artificial_variables_needed():
    # Create a model where no artificial variables are needed
    model = ConcreteModel()
    model.x = Var(domain=NonNegativeReals)
    model.con1 = Constraint(expr=2 * model.x <= 10)
    model.obj = Objective(expr=model.x, sense=minimize)

    # Standardize the model
    standardizer = ModelStandardizer(model)
    standard_model = standardizer.standardize_model()

    # Check that no artificial variables were added
    artificial_vars = [var for var in standard_model.component_objects(Var) if 'artificial' in var.name]
    assert len(artificial_vars) == 0, "No artificial variables should be added."

    # Check that a slack variable was added
    slack_vars = [var for var in standard_model.component_objects(Var) if 'slack' in var.name]
    assert len(slack_vars) == 1, "One slack variable should be added."

    # Check that the original objective is still active
    original_objectives = [obj for obj in standard_model.component_data_objects(Objective, active=True) if obj.name == 'obj']
    assert len(original_objectives) == 1, "Original objective should remain active."

def test_infeasibility_during_standardization():
    # Create a model that is infeasible during standardization
    model = ConcreteModel()
    model.x = Var(domain=NonNegativeReals)

    # Contradictory constraints
    model.constraint1 = Constraint(expr=model.x >= 5)
    model.constraint2 = Constraint(expr=model.x <= 3)
    
    # Objective function
    model.objective = Objective(expr=model.x, sense=minimize)
    
    # Attempt to standardize the model
    with pytest.raises(ValueError) as exc_info:
        standardizer = ModelStandardizer(model)
        standardizer.standardize_model()
    
    # Verify that the correct exception is raised
    assert "infeasible" in str(exc_info.value).lower(), "Infeasibility not detected during standardization."

def test_integer_variable_preservation():
    # Create a model with integer variables
    model = ConcreteModel()
    model.x = Var(domain=Integers)
    model.y = Var(domain=NonNegativeReals)
    model.con1 = Constraint(expr=2 * model.x + model.y <= 10)
    model.obj = Objective(expr=model.x + model.y, sense=minimize)

    # Standardize the model
    standardizer = ModelStandardizer(model)
    standard_model = standardizer.standardize_model()

    # Ensure integer variables are still integer
    for var in standardizer.integer_variables:
        assert var.domain is Integers, f"Variable {var.name} should remain integer, but it is {var.domain}"
