from pyomo.environ import Constraint, Integers, Binary, maximize, minimize
from .super_simplex_solver import SuperSimplexSolver, InfeasibleProblemError, UnboundedProblemError
from ..utils.model_processor import ModelProcessor
from ..utils.logger import totsu_logger

class BranchAndBoundSolver:
    def __init__(self):
        self.simplex_solver = SuperSimplexSolver()
        self.best_solution = None
        self.best_objective = None  # Will be set based on problem type
        self.nodes_explored = 0
        self.is_minimization = True  # Default to minimization

    def solve(self, model):
        totsu_logger.info("Starting Branch and Bound Solver")
        # Determine if the objective is minimization or maximization
        self.is_minimization = model.obj.sense == minimize
        self.best_objective = float('inf') if self.is_minimization else float('-inf')

        self.branch_and_bound(model)

        if self.best_solution is not None:
            # update model with the current best_solution
            ModelProcessor.set_variable_values(model, self.best_solution)

        return self.best_solution, self.best_objective

    def branch_and_bound(self, model):
        try:
            solution = self.simplex_solver.solve(model)
            objective_value = self.simplex_solver.get_current_objective_value()
        except InfeasibleProblemError:
            return  # Infeasible node is discarded.
        except UnboundedProblemError:
            totsu_logger.debug("BranchAndBoundSolver detected an unbounded problem.")
            raise  # Ensure pytest can catch this!

        if self.is_integer_feasible(solution, model):
            if (self.is_minimization and objective_value < self.best_objective) or \
               (not self.is_minimization and objective_value > self.best_objective):
                self.best_solution = solution.copy()
                self.best_objective = objective_value
                totsu_logger.info(f"New best integer solution found: {self.best_solution} with objective {self.best_objective}")
            return

        self.nodes_explored += 1
        fractional_var = self.select_branching_variable(solution, model)
        if fractional_var is None:
            return  # No fractional variables left

        var = model.find_component(fractional_var)
        lower_bound = int(solution[fractional_var])
        upper_bound = lower_bound + 1

        self.branch(model, var, lower_bound, upper_bound)

    def is_integer_feasible(self, solution, model):
        for var in ModelProcessor.get_variables(model):
            if var.domain is Binary and solution[var.name] not in {0, 1}:  # Ensure Binary stays in {0,1}
                return False
            if var.domain is Integers and not float(solution[var.name]).is_integer():
                return False
        return True

    def select_branching_variable(self, solution, model):
        for var in ModelProcessor.get_variables(model):
            val = float(solution[var.name])
            totsu_logger.debug(f"Checking variable {var.name}: value = {val}")
            if var.domain is Binary and solution[var.name] not in {0, 1}:  # Ensure Binary stays in {0,1}
                totsu_logger.debug(f"Branching on variable {var.name} with fractional value {val}")
                return var.name
            if var.domain is Integers and not val.is_integer():
                totsu_logger.debug(f"Branching on variable {var.name} with fractional value {val}")
                return var.name  # Return first fractional variable found
        return None

    def branch(self, model, var, lower_bound, upper_bound):
        branch_models = []

        # Ensure correct branching for Binary variables
        if var.domain is Binary:
            # Left branch: Fix var = 0
            left_model = model.clone()
            left_var = left_model.find_component(var.name)
            left_var.fix(0)  # Directly fix to 0
            branch_models.append(left_model)

            # Right branch: Fix var = 1
            right_model = model.clone()
            right_var = right_model.find_component(var.name)
            right_var.fix(1)  # Directly fix to 1
            branch_models.append(right_model)
        else:  # Integer variables
            var_lb, var_ub = ModelProcessor.get_bounds(var)  # Retrieve original bounds

            # Adjust bounds to ensure they stay within the variable's valid range
            lower_bound = max(lower_bound, var_lb) if var_lb is not None else lower_bound
            upper_bound = min(upper_bound, var_ub) if var_ub is not None else upper_bound

            # Left branch: var ≤ lower_bound
            left_model = model.clone()
            left_constraint = Constraint(expr=var <= lower_bound)
            setattr(left_model, f"branch_{var.name}_leq_{lower_bound}", left_constraint)
            branch_models.append(left_model)

            # Right branch: var ≥ upper_bound
            right_model = model.clone()
            right_constraint = Constraint(expr=var >= upper_bound)
            setattr(right_model, f"branch_{var.name}_geq_{upper_bound}", right_constraint)
            branch_models.append(right_model)

        for new_model in branch_models:
            self.branch_and_bound(new_model)
