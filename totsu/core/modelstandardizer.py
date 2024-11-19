from pyomo.environ import (
    value, Var, Constraint, NonNegativeReals, Objective, minimize
)
from pyomo.repn import generate_standard_repn

from ..utils.model_processor import ModelProcessor

class ModelStandardizer:
    def __init__(self, model):
        self.original_model = model
        self.standard_model = None
        self.variables = None
        self.constraints = None
        self.objective = None
        self.original_objective = None
        self.original_constraints = None
        self.artificial_vars = []

    def standardize_model(self):
        # Clone the original model to avoid modifying it
        self.standard_model = self.original_model.clone()
        # Early infeasibility detection and preprocessing
        if not self.preprocess_model():
            raise ValueError("Model is infeasible during preprocessing.")
        if self.all_variables_fixed():
            print("All variables are fixed.")
        else:
            # Convert constraints to standard form
            self.convert_constraints()
        return self.standard_model

    def preprocess_model(self):
        self.variables = list(ModelProcessor.get_variables(self.standard_model))
        self.constraints = list(ModelProcessor.get_constraints(self.standard_model))
        self.original_objective = ModelProcessor.get_active_objective(self.standard_model)

        for var in self.variables:
            print(f"Variable '{var.name}' bounds: [{var.lb}, {var.ub}]")

        # Tighten variable bounds based on constraints
        if not self.tighten_variable_bounds():
            return False  # Problem is infeasible

        # Preprocess variable bounds
        for var in self.variables:
            lb, ub = var.bounds
            if lb is not None and ub is not None and lb > ub:
                print(f"Infeasible variable bounds detected for variable '{var.name}': lower bound ({lb}) > upper bound ({ub})")
                return False  # Problem is infeasible due to inconsistent bounds

            # Handle fixed variables (lb == ub)
            if lb is not None and ub is not None and abs(lb - ub) <= 1e-8:
                var.fix(lb)
                print(f"Variable '{var.name}' is fixed at {lb}")

        return True
    
    def all_variables_fixed(self):
        return len([var for var in self.variables if not var.fixed]) == 0
    
    def check_constraints(self):
        """Check if constraints are satisfied."""
        # Check constraints
        for con in self.constraints:
            body_value = value(con.body)
            lower_bound = value(con.lower) if con.has_lb() else None
            upper_bound = value(con.upper) if con.has_ub() else None

            if lower_bound is not None and body_value < lower_bound - 1e-8:
                print(f"Constraint '{con.name}' violated: {body_value} < {lower_bound}")
                return False
            if upper_bound is not None and body_value > upper_bound + 1e-8:
                print(f"Constraint '{con.name}' violated: {body_value} > {upper_bound}")
                return False

        # Variable bounds are already checked during variable value assignment
        return True

    def tighten_variable_bounds(self):
        """Tighten variable bounds based on the constraints."""
        var_bounds = {var.name: [var.lb, var.ub] for var in self.variables}
        for con in self.constraints:
            # Only process constraints that are simple bounds on variables
            repn = generate_standard_repn(con.body)
            if len(repn.linear_vars) == 1 and repn.is_linear() and repn.constant == 0:
                var = repn.linear_vars[0]
                coef = repn.linear_coefs[0]
                var_name = var.name
                if var_name in var_bounds:
                    if coef == 1:
                        if con.has_lb():
                            lb = value(con.lower)
                            if var_bounds[var_name][0] is None or lb > var_bounds[var_name][0]:
                                var_bounds[var_name][0] = lb
                        if con.has_ub():
                            ub = value(con.upper)
                            if var_bounds[var_name][1] is None or ub < var_bounds[var_name][1]:
                                var_bounds[var_name][1] = ub
                    elif coef == -1:
                        if con.has_lb():
                            ub = -value(con.lower)
                            if var_bounds[var_name][1] is None or ub < var_bounds[var_name][1]:
                                var_bounds[var_name][1] = ub
                        if con.has_ub():
                            lb = -value(con.upper)
                            if var_bounds[var_name][0] is None or lb > var_bounds[var_name][0]:
                                var_bounds[var_name][0] = lb
        # Update variable bounds
        for var in self.variables:
            lb, ub = var_bounds[var.name]
            var.setlb(lb)
            var.setub(ub)
            if lb is not None and ub is not None and lb > ub:
                print(f"Infeasible variable bounds detected for variable '{var.name}': lower bound ({lb}) > upper bound ({ub})")
                return False
        return True
    
    def convert_constraints(self):
        # Convert all constraints to standard form and add slack/artificial variables as needed
        model = self.standard_model
        self.artificial_vars = []
        
        # Keep a list of new constraints
        new_constraints = []
        self.original_constraints = self.constraints
        
        for con in self.constraints:
            repn = generate_standard_repn(con.body)
            vars_in_con = repn.linear_vars
            coefs_in_con = repn.linear_coefs
            constant = value(repn.constant)
            
            lb = value(con.lower) - constant if con.has_lb() else None
            ub = value(con.upper) - constant if con.has_ub() else None
            
            # Remove the original constraint
            con.deactivate()
            
            # Construct the left-hand side expression
            expr = sum(coef * var for coef, var in zip(coefs_in_con, vars_in_con))

            # Adjust for negative rhs
            expr, lb, ub = self.adjust_for_negative_rhs(expr, lb, ub)
            
            # Handle the constraint based on its type
            if lb is not None and ub is not None and abs(lb - ub) <= 1e-8:
                # Equality constraint
                rhs = lb
                new_con, artificial_var = self.handle_equality_constraint(expr, rhs)
            elif ub is not None:
                # Less-than-or-equal-to constraint
                rhs = ub
                new_con = self.handle_le_constraint(expr, rhs)
            elif lb is not None:
                # Greater-than-or-equal-to constraint
                rhs = lb
                new_con, artificial_var = self.handle_ge_constraint(expr, rhs)
            else:
                raise ValueError("Constraint without bounds encountered.")
            
            # Add the new constraint to the model
            con_name = con.name + '_std'
            setattr(model, con_name, new_con)
            new_constraints.append(new_con)
        
        # Update the list of constraints and variables
        self.constraints = new_constraints
        self.variables = list(ModelProcessor.get_variables(self.standard_model))
        
        # Adjust the objective function for Phase I if there are artificial variables
        if self.artificial_vars:
            self.adjust_objective_for_phase1()
        else:
            self.objective = self.original_objective

    def adjust_for_negative_rhs(self, expr, lb, ub):
        # Determine if adjustment is needed
        # Prefer lb or ub based on which is not None
        rhs = lb if lb is not None else ub
        if rhs is not None and rhs < 0:
            # Multiply expr and rhs by -1
            expr = -expr
            rhs = -rhs
            # Reverse inequality direction
            if lb is not None and ub is not None and abs(lb - ub) <= 1e-8:
                # Equality constraint, no direction to reverse
                lb, ub = rhs, rhs
            elif ub is not None:
                # Less-than-or-equal-to (expr <= rhs) becomes greater-than-or-equal-to (-expr >= -rhs)
                lb, ub = rhs, None
            elif lb is not None:
                # Greater-than-or-equal-to (expr >= rhs) becomes less-than-or-equal-to (-expr <= -rhs)
                lb, ub = None, rhs
        return expr, lb, ub

    def handle_equality_constraint(self, expr, rhs):
        model = self.standard_model
        # Add an artificial variable
        artificial_var = Var(domain=NonNegativeReals)
        setattr(model, 'artificial_' + str(len(self.artificial_vars)), artificial_var)
        self.artificial_vars.append(artificial_var)
        
        # Create the new equality constraint
        new_expr = expr + artificial_var == rhs

        print(f"handled ge constraint and added {artificial_var}")

        return Constraint(expr=new_expr), artificial_var

    def handle_le_constraint(self, expr, rhs):
        model = self.standard_model
        # Add a slack variable
        slack_var = Var(domain=NonNegativeReals)
        setattr(model, 'slack_' + str(len(self.variables)), slack_var)
        self.variables.append(slack_var)
        
        # Create the new equality constraint
        new_expr = expr + slack_var == rhs

        print(f"handled le constraint and added {slack_var}")

        return Constraint(expr=new_expr)

    def handle_ge_constraint(self, expr, rhs):
        model = self.standard_model
        # Add a surplus variable
        surplus_var = Var(domain=NonNegativeReals)
        setattr(model, 'surplus_' + str(len(self.variables)), surplus_var)
        self.variables.append(surplus_var)
        
        # Add an artificial variable
        artificial_var = Var(domain=NonNegativeReals)
        setattr(model, 'artificial_' + str(len(self.artificial_vars)), artificial_var)
        self.artificial_vars.append(artificial_var)
        
        # Create the new equality constraint
        new_expr = expr - surplus_var + artificial_var == rhs

        print(f"handled ge constraint and added {surplus_var} and {artificial_var}")

        return Constraint(expr=new_expr), artificial_var

    def adjust_objective_for_phase1(self):
        model = self.standard_model
        # Create a new objective that minimizes the sum of artificial variables
        artificial_vars_sum = sum(self.artificial_vars)
        model.obj_phase1 = Objective(expr=artificial_vars_sum, sense=minimize)
        self.objective = model.obj_phase1
        # Deactivate the original objective
        self.original_objective.deactivate()

    def var_name_to_index(self):
        var_name_to_index = {var.name: idx for idx, var in enumerate(self.variables)}
        return var_name_to_index

    def index_to_var_name(self):
        index_to_var_name = {idx: var.name for idx, var in enumerate(self.variables)}
        return index_to_var_name
    
    def original_var_indices(self):
        original_var_indices = [
            idx for idx, var in enumerate(self.variables)
            if 'slack' not in var.name and 'surplus' not in var.name and 'artificial' not in var.name
        ]
        return original_var_indices
