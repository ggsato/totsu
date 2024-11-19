from typing import List, Tuple, Optional
from pyomo.environ import Var, Constraint, Objective, ConcreteModel, value
from pyomo.repn import generate_standard_repn
from pyomo.common.collections import ComponentMap
from pyomo.core.expr.compare import compare_expressions
import math

def compare_coefficient_dicts(coefs1, coefs2, tol=1e-6):
    if set(coefs1.keys()) != set(coefs2.keys()):
        return False
    for var_name in coefs1:
        coef1 = coefs1[var_name]
        coef2 = coefs2[var_name]
        if not math.isclose(coef1, coef2, abs_tol=tol):
            return False
    return True

class ModelProcessor:
    @staticmethod
    def get_variables(model: ConcreteModel) -> List[Var]:
        """Extract all variables from the model."""
        return list(model.component_data_objects(Var, descend_into=True))

    @staticmethod
    def get_constraints(model: ConcreteModel) -> List[Constraint]:
        """Extract all active constraints from the model."""
        return list(model.component_data_objects(Constraint, active=True))

    @staticmethod
    def get_active_objective(model: ConcreteModel) -> Objective:
        """Return the single active objective of the model."""
        objectives = list(model.component_data_objects(Objective, active=True))
        if not objectives:
            raise ValueError("No active objective function found.")
        if len(objectives) > 1:
            raise ValueError("Multiple active objective functions found.")
        return objectives[0]

    @staticmethod
    def get_bounds(var: Var) -> Tuple[Optional[float], Optional[float]]:
        """Get the lower and upper bounds of a variable."""
        return var.lb, var.ub

    @staticmethod
    def get_coefficient(constraint: Constraint, variable: Var) -> float:
        """Get the coefficient of a variable in a linear constraint."""
        degree = constraint.body.polynomial_degree()
        # Handle constant constraints (degree 0)
        if degree == 0:
            # There are no variables in the constraint
            return 0.0
        elif degree == 1:
            # Proceed to extract the coefficient
            pass
        else:
            # Ensure the constraint expression is linear
            raise ValueError(f"The constraint '{constraint.name}' is not linear.")

        # Generate the standard representation of the constraint body
        repn = generate_standard_repn(constraint.body, compute_values=False)

        if not repn.is_linear():
            raise ValueError(f"The constraint '{constraint.name}' is not linear.")

        # Map variables to their coefficients
        coef_map = ComponentMap(zip(repn.linear_vars, repn.linear_coefs))

        # Return the coefficient for the requested variable
        return coef_map.get(variable, 0.0)

    @staticmethod
    def display_model_info(model: ConcreteModel):
        """Display key information about the model."""
        variables = ModelProcessor.get_variables(model)
        constraints = ModelProcessor.get_constraints(model)
        objective = ModelProcessor.get_active_objective(model)

        print(f"Objective: {objective.expr}")
        print("Variables:")
        for var in variables:
            lb, ub = ModelProcessor.get_bounds(var)
            print(f" - {var.name}: LB = {lb}, UB = {ub}")

        print("Constraints:")
        for con in constraints:
            lb = con.lower
            ub = con.upper
            body = con.body
            if lb is not None and ub is not None and lb == ub:
                # Equality constraint
                print(f" - {con.name}: {body} == {lb}")
            else:
                constraint_str = f" - {con.name}: "
                if lb is not None:
                    constraint_str += f"{body} >= {lb}"
                if ub is not None:
                    if lb is not None:
                        constraint_str += ", "
                    constraint_str += f"{body} <= {ub}"
                print(constraint_str)

    @staticmethod
    def compare_models(model1: ConcreteModel, model2: ConcreteModel) -> List[str]:
        """Compare two models and report differences."""
        differences = []

        # Compare objectives
        try:
            obj1 = ModelProcessor.get_active_objective(model1)
            obj2 = ModelProcessor.get_active_objective(model2)
        except ValueError as e:
            differences.append(str(e))
            return differences

        # Generate standard representations
        repn1 = generate_standard_repn(obj1.expr, compute_values=False)
        repn2 = generate_standard_repn(obj2.expr, compute_values=False)

        # Map variable names to coefficients
        obj_coefs1 = {var.name: coef for var, coef in zip(repn1.linear_vars, repn1.linear_coefs)}
        obj_coefs2 = {var.name: coef for var, coef in zip(repn2.linear_vars, repn2.linear_coefs)}

        # Compare objective coefficients and constants
        if not compare_coefficient_dicts(obj_coefs1, obj_coefs2) or not math.isclose(value(repn1.constant), value(repn2.constant), abs_tol=1e-6):
            differences.append(f"Objective mismatch:\nModel 1: {obj1.expr}\nModel 2: {obj2.expr}")

        # Compare variables
        vars1 = {var.name: var.value for var in ModelProcessor.get_variables(model1)}
        vars2 = {var.name: var.value for var in ModelProcessor.get_variables(model2)}

        for var_name in set(vars1) | set(vars2):  # Union of both variable sets
            val1 = vars1.get(var_name)
            val2 = vars2.get(var_name)
            if val1 is None or val2 is None or not math.isclose(val1, val2, abs_tol=1e-6):
                differences.append(f"Variable {var_name} value mismatch:\nModel 1: {val1}\nModel 2: {val2}")

        # Compare constraints
        cons1 = {con.name: con for con in ModelProcessor.get_constraints(model1)}
        cons2 = {con.name: con for con in ModelProcessor.get_constraints(model2)}

        for con_name in set(cons1) | set(cons2):  # Union of constraint names
            con1 = cons1.get(con_name)
            con2 = cons2.get(con_name)
            if con1 is None or con2 is None:
                differences.append(f"Constraint {con_name} exists only in one model.")
                continue

            # Generate standard representations
            repn1 = generate_standard_repn(con1.body, compute_values=False)
            repn2 = generate_standard_repn(con2.body, compute_values=False)

            # Map variable names to coefficients
            coefs1 = {var.name: coef for var, coef in zip(repn1.linear_vars, repn1.linear_coefs)}
            coefs2 = {var.name: coef for var, coef in zip(repn2.linear_vars, repn2.linear_coefs)}

            # Compare coefficients and constants
            if not compare_coefficient_dicts(coefs1, coefs2) or not math.isclose(value(repn1.constant), value(repn2.constant), abs_tol=1e-6):
                differences.append(f"Constraint {con_name} body mismatch.")

            # Compare constraint bounds
            lb1 = value(con1.lower) if con1.has_lb() else None
            ub1 = value(con1.upper) if con1.has_ub() else None
            lb2 = value(con2.lower) if con2.has_lb() else None
            ub2 = value(con2.upper) if con2.has_ub() else None

            if ((lb1 is not None and lb2 is not None and not math.isclose(lb1, lb2, abs_tol=1e-6)) or
                (ub1 is not None and ub2 is not None and not math.isclose(ub1, ub2, abs_tol=1e-6)) or
                (lb1 is None) != (lb2 is None) or (ub1 is None) != (ub2 is None)):
                differences.append(f"Constraint {con_name} bounds mismatch.")

        # Compare variable bounds
        vars1_full = {var.name: var for var in ModelProcessor.get_variables(model1)}
        vars2_full = {var.name: var for var in ModelProcessor.get_variables(model2)}

        for var_name in set(vars1_full) | set(vars2_full):  # Union of variable names
            var1 = vars1_full.get(var_name)
            var2 = vars2_full.get(var_name)
            if var1 is None or var2 is None:
                differences.append(f"Variable {var_name} exists only in one model.")
                continue
            lb1, ub1 = ModelProcessor.get_bounds(var1)
            lb2, ub2 = ModelProcessor.get_bounds(var2)
            if ((lb1 is not None and lb2 is not None and not math.isclose(lb1, lb2, abs_tol=1e-6)) or
                (ub1 is not None and ub2 is not None and not math.isclose(ub1, ub2, abs_tol=1e-6)) or
                (lb1 is None) != (lb2 is None) or (ub1 is None) != (ub2 is None)):
                differences.append(f"Bounds for variable {var_name} mismatch:\n"
                                   f"Model 1: LB = {lb1}, UB = {ub1}\nModel 2: LB = {lb2}, UB = {ub2}")

        if not differences:
            print("Models are identical.")
        else:
            print("Differences between models:")
            for diff in differences:
                print(diff)

        return differences

    @staticmethod
    def set_variable_values(model: ConcreteModel, values: dict):
        """Set variable values from a dictionary."""
        for var in ModelProcessor.get_variables(model):
            if var.name in values:
                var.set_value(values[var.name])

    @staticmethod
    def get_variable_values(model: ConcreteModel) -> dict:
        """Get the values of all variables in the model."""
        return {var.name: var.value for var in ModelProcessor.get_variables(model)}
