from .tableau import Tableau
from .modelstandardizer import ModelStandardizer
from ..utils.model_processor import ModelProcessor
from ..utils.logger import totsu_logger

class OptimizationError(Exception):
    """Base class for exceptions in this optimization module."""
    pass

class InfeasibleProblemError(OptimizationError):
    """Exception raised when the problem is infeasible."""
    pass

class UnboundedProblemError(OptimizationError):
    """Exception raised when the problem is unbounded."""
    pass

class SuperSimplexSolver:

    def __init__(self, max_itr=100):
        self.max_itr = max_itr
        self.model = None
        self._tableau = None

    def solve(self, model):
        totsu_logger.debug("Solving using Simplex method...")
        self.model = model

        # Standardize the model
        try:
            standardizer = ModelStandardizer(model)
            standardizer.standardize_model()
        except ValueError as e:
            # If standardization fails due to infeasibility
            raise InfeasibleProblemError(f"Model is infeasible during standardization: {e}")
        
        # Check if all variables are fixed
        if standardizer.all_variables_fixed():
            if not standardizer.check_constraints():
                raise InfeasibleProblemError("All variables are fixed but does not meet constraints.")
            solution = {var.name: var.value for var in standardizer.variables}
            return solution

        # Initialize the tableau
        self._tableau = Tableau(standardizer)
        self._tableau.initialize_tableau()

        # Perform Phase I iterations
        success = self.simplex_iterations(phase=1)

        # Check if feasible
        if not success or not self._tableau.is_feasible():
            totsu_logger.debug("Problem is infeasible after Phase I.")
            raise InfeasibleProblemError("Problem is infeasible after Phase I.")

        # Adjust tableau for Phase II
        self._tableau.set_phase2_objective()

        # Perform Phase II iterations
        success = self.simplex_iterations(phase=2)

        if not success:
            totsu_logger.debug("Problem may be unbounded or infeasible in Phase II.")
            if not self._tableau.is_feasible():
                raise InfeasibleProblemError("Problem is infeasible after Phase II.")
            else:
                raise UnboundedProblemError("Problem is unbounded after Phase II.")

        # Extract and store the solution
        solution = self.extract_solution()

        return solution
    
    def simplex_iterations(self, phase):
        totsu_logger.debug(f"Executing simplex iterations at phase{phase}")
        iteration = 0
        while not self._tableau.is_optimal():
            if iteration >= self.max_itr:
                raise OptimizationError("Maximum iterations exceeded.")
            pivot_col = self._tableau.select_pivot_column(phase)
            if pivot_col is None:
                if phase == 2:
                    # No pivot column means optimality is achieved or unboundedness
                    if self._tableau.is_optimal():
                        return True  # Optimality achieved
                    else:
                        # Unboundedness detected
                        raise UnboundedProblemError("Problem is unbounded.")
                else:
                    # In Phase I, inability to find pivot column may indicate infeasibility
                    raise InfeasibleProblemError("Problem is infeasible during Phase I.")
            pivot_row = self._tableau.select_pivot_row(pivot_col)
            if pivot_row is None:
                # No valid pivot row indicates unboundedness
                raise UnboundedProblemError("Problem is unbounded.")
            self._tableau.pivot_operation(pivot_row, pivot_col, phase)
            iteration += 1
        return True


    def get_history(self):
        return self._tableau.history

    def get_current_objective_value(self):
        return self._tableau.get_current_objective_value()

    def is_optimal(self):
        return self._tableau.is_optimal()
    
    def get_dual_variables(self):
        return self._tableau.compute_dual_variables()

    def get_reduced_costs(self, y):
        return self._tableau.compute_reduced_costs(y)

    def extract_solution(self):
        # tableau solution does not contain fixed variables
        tableau_solution =  self._tableau.extract_solution()
        # add fixed variables
        for var in self._tableau.original_variables:
            if var.fixed:
                tableau_solution[var.name] = var.value
        totsu_logger.debug(f"Extracted solution: {tableau_solution}")
        return tableau_solution
