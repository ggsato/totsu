import numpy as np
from pyomo.environ import (
    value, minimize
)
from pyomo.repn import generate_standard_repn

class Tableau:
    def __init__(self, standardizer):
        self.standardizer = standardizer
        self.updated_objective = None
        self._tableau = None  # The simplex tableau
        self.updated_tableau = None
        self.updated_variables = None
        self.basis_vars = []
        self.non_basis_vars = []
        self.history = []  # To store tableau history for visualization
        self.is_dirty = False  # Flag to mark if the basis is "dirty" after a pivot

    @property
    def standard_model(self):
        return self.standardizer.standard_model
    
    @property
    def constraints(self):
        return self.standardizer.constraints
    
    @property
    def original_constraints(self):
        return self.standardizer.original_constraints
    
    @property
    def artificial_vars(self):
        return self.standardizer.artificial_vars

    @property
    def phase1_variables(self):
        return self.standardizer.variables

    @property
    def variables(self):
        if self.updated_variables:
            return self.updated_variables
        return self.standardizer.variables
    
    @variables.setter
    def variables(self, new_variables):
        # This is called after artificial variables were removed
        self.updated_variables = new_variables

    @property
    def phase1_tableau(self):
        return self._tableau

    @property
    def tableau(self):
        if self.updated_tableau is not None:
            return self.updated_tableau
        return self._tableau
    
    @tableau.setter
    def tableau(self, new_tableau):
        if self._tableau is None:
            self._tableau = new_tableau
        else:
            # This is called after artificial variables were removed
            self.updated_tableau = new_tableau

    @property
    def objective(self):
        if self.updated_objective is None:
            return self.standardizer.objective
        return self.updated_objective
    
    @objective.setter
    def objective(self, new_objective):
        self.updated_objective = new_objective

    def initialize_tableau(self):
        # Initialize basis and non-basis variables
        self.identify_basis_variables()

        # Construct the initial tableau matrix
        self.construct_tableau()

    def identify_basis_variables(self):
        # Basis variables are initially the slack and artificial variables
        for idx, var in enumerate(self.variables):
            if 'slack' in var.name or 'artificial' in var.name:
                self.basis_vars.append(idx)
            else:
                self.non_basis_vars.append(idx)
        print(f"initial basis_bars = {self.basis_vars}")

    def construct_tableau(self):
        num_constraints = len(self.constraints)
        num_variables = len(self.variables)

        var_name_to_index = self.standardizer.var_name_to_index()

        # Initialize the tableau matrix
        # Rows: Number of constraints + 1 (for the objective function)
        # Columns: Number of variables + 1 (for the RHS)
        self.tableau = np.zeros((num_constraints + 1, num_variables + 1))

        # Fill in the constraint coefficients
        for i, con in enumerate(self.constraints):
            assert value(con.lower) == value(con.upper), f"Constraint {con.name} is not an equality."

            repn = generate_standard_repn(con.body)
            # Map variable names to coefficients
            coef_map = {var.name: coef for var, coef in zip(repn.linear_vars, repn.linear_coefs)}
            for var_name, coef in coef_map.items():
                var_idx = var_name_to_index[var_name]
                self.tableau[i, var_idx] = coef
            # Set the RHS value
            rhs = value(con.lower)  # Since standardized constraints are equalities
            self.tableau[i, -1] = rhs

        # Initialize the objective function row for Phase I
        # Set the coefficients of artificial variables to +1
        self.tableau[-1, :] = 0  # Start with zeros
        for var in self.artificial_vars:
            var_idx = var_name_to_index[var.name]
            self.tableau[-1, var_idx] = 1  # Coefficient of artificial variables is +1

        # Adjust the objective row for artificial variables in the basis
        for i, var_idx in enumerate(self.basis_vars):
            var_name = self.standardizer.index_to_var_name()[var_idx]
            if 'artificial' in var_name:
                # Subtract the constraint row from the objective row
                self.tableau[-1, :] -= self.tableau[i, :]

        self.print_tableau("constructed")

        # Record the initial tableau in history
        self.history.append(self.take_snapshot())

    def print_tableau(self, message):
        print(f"tableau[{message}]:")
        print(f"{self.tableau}")

    def select_pivot_column(self, phase):
        if phase == 1:
            objective_row = self.tableau[-1, :-1]
            # Find indices where coefficients are negative (< 0)
            pivot_cols = np.where(objective_row < -1e-8)[0]
            # Exclude columns already in basis_vars
            eligible_cols = [col for col in pivot_cols if col not in self.basis_vars]
            print(f"eligible cols = {eligible_cols}")
            if not eligible_cols:
                print(f"No eligible pivot columns found in Phase {phase}.")
                return None  # No eligible columns
            # Bland's Rule: Choose the smallest index among eligible pivot columns
            col = int(min(eligible_cols))
            #print(f"{col} was selected by select_pivot_column (phase {phase})")
            return col
        elif phase == 2:
            col = self.select_pivot_column_phase2()
            #print(f"{col} was selected by pivot_colmn (phase 2)")
            return col
        
    def select_pivot_column_phase2(self):
        min_value = np.min(self.tableau[-1, :-1])
        if min_value >= -1e-8:
            return None
        pivot_cols = np.where(self.tableau[-1, :-1] < -1e-8)[0]

        # Exclude columns already in basis_vars
        eligible_cols = [col for col in pivot_cols if col not in self.basis_vars]
        
        if not eligible_cols:
            return None  # No eligible columns
        
        # No upper bounds check required.
        # The standard simplex method assumes variables are non-negative and unbounded above (i.e., have infinite upper bounds).

        # Bland's Rule: Choose the smallest index among eligible pivot columns
        return int(min(eligible_cols))

    def select_pivot_row(self, pivot_col):
        # Apply the minimum ratio test
        num_constraints = len(self.constraints)
        ratios = []
        for i in range(num_constraints):
            coeff = self.tableau[i, pivot_col]
            rhs = self.tableau[i, -1]
            if coeff > 1e-8:
                ratio = rhs / coeff
                ratios.append((ratio, i))
            elif abs(coeff) < 1e-8 and abs(rhs) < 1e-8:
                # Handle degenerate cases
                ratios.append((float('inf'), i))
        if ratios:
            # Extract only the ratio values for finding the minimum ratio
            min_ratio = min(r[0] for r in ratios)
            # Now find the pivot rows that match this minimum ratio within the tolerance
            pivot_rows = [i for r, i in ratios if abs(r - min_ratio) <= 1e-8]
        else:
            print("no row was selected by pivot_row")
            return None  # Unbounded
        
        #print(f"{pivot_rows[0]} was selected by pivot_row")

        return pivot_rows[0]  # Choose the first one (Bland's Rule)

    def pivot_operation(self, pivot_row, pivot_col):
        pivot_element = self.tableau[pivot_row, pivot_col]
        if abs(pivot_element) < 1e-8:
            raise ZeroDivisionError("Pivot element is too close to zero.")
        
        if pivot_row >= len(self.basis_vars):
            raise RuntimeError(f"pivot row: {pivot_row} is out of basis vars: {self.basis_vars}")

        # Normalize the pivot row
        self.tableau[pivot_row, :] /= pivot_element

        # Eliminate the pivot column entries in other rows
        num_rows, num_cols = self.tableau.shape
        for i in range(num_rows):
            if i != pivot_row:
                factor = self.tableau[i, pivot_col]
                self.tableau[i, :] -= factor * self.tableau[pivot_row, :]

        # Update basis variables
        leaving_var_idx = self.basis_vars[pivot_row]
        entering_var_idx = pivot_col

        self.basis_vars[pivot_row] = entering_var_idx
        if entering_var_idx in self.non_basis_vars:
            self.non_basis_vars.remove(entering_var_idx)
        self.non_basis_vars.append(leaving_var_idx)

        # Adjust the objective row
        factor = self.tableau[-1, pivot_col]
        self.tableau[-1, :] -= factor * self.tableau[pivot_row, :]

        self.is_dirty = True

        # Record the tableau after pivot
        self.history.append(self.take_snapshot(pivot_col, pivot_row, entering_var_idx, leaving_var_idx))

        # Debugging output
        index_to_var_name = self.standardizer.index_to_var_name()
        print(f"Pivoting: Row {pivot_row}, Column {pivot_col}")
        print(f"Leaving variable: {index_to_var_name[leaving_var_idx]}")
        print(f"Entering variable: {index_to_var_name[entering_var_idx]}")
        print(f"After pivot, basis_vars: {[index_to_var_name[idx] for idx in self.basis_vars]} by name, {self.basis_vars} by idx")
        print(f"Tableau after pivot operation:\n{self.tableau}")

    def is_optimal(self):
        if self.updated_tableau is None:
            # We are in Phase I
            """
            1. **Feasibility of the Solution**:
            - The Phase I objective value is zero (within numerical tolerance).
            - All artificial variables have zero values (within numerical tolerance). This confirms that a feasible solution to the original problem has been found.

            2. **Removal of Artificial Variables from the Basis**:
   -        No artificial variables remain in the basis. This is essential because artificial variables are not part of the original problem, and their presence in the basis can cause issues in Phase II.
            """
            phase1_tableau = self.phase1_tableau
            objective_value = phase1_tableau[-1, -1]
            if abs(objective_value) > 1e-8:
                return False  # Not optimal yet
            
            # Optionally check artificial variables' values
            var_name_to_index = self.standardizer.var_name_to_index()
            artificial_indices = [var_name_to_index[var.name] for var in self.artificial_vars]
            
            # Collect values of artificial variables in the basis
            artificial_values = []
            artificial_in_basis = []
            for i, idx in enumerate(self.basis_vars):
                if idx in artificial_indices:
                    value = phase1_tableau[i, -1]
                    artificial_values.append(value)
                    artificial_in_basis.append(idx)

            if any(abs(value) > 1e-8 for value in artificial_values):
                return False  # Artificial variables have positive values
            
            # Check if any artificial variables remain in the basis
            if artificial_in_basis:
                return False  # Artificial variables remain in the basis

            print(f"Is optimal. objective_value = {objective_value}, artificial_values = {artificial_values}")
            return True   # Optimality achieved in Phase I
        else:
            # Standard optimality condition for Phase II
            """
            Check if all coefficients in the objective row are non-negative.
            """
            objective_row = self.tableau[-1, :-1]
            return all(coef >= -1e-8 for coef in objective_row)

    def is_feasible(self):
        if self.updated_tableau is None:
            # phase1
            var_name_to_index = self.standardizer.var_name_to_index()
            # After Phase I, check if the artificial variables are zero in the solution
            artificial_indices = [var_name_to_index[var.name] for var in self.artificial_vars]
            for idx in artificial_indices:
                if idx in self.basis_vars:
                    row_idx = self.basis_vars.index(idx)
                    value = self.tableau[row_idx, -1]
                    if abs(value) > 1e-8:
                        print(f"Infeasible: Artificial variable is positive in basis({self.basis_vars}) at row({row_idx})")
                        return False  # Artificial variable is positive in basis
                else:
                    # Check value of non-basic artificial variable
                    # It should be zero or confirm via tableau calculations
                    pass  # Non-basic variables have zero values in the solution
            # Additionally, check that the objective value (sum of artificial variables) is zero
            objective_value = self.tableau[-1, -1]
            if abs(objective_value) > 1e-8:
                print("Infeasible: Objective value is zero")
                return False
        else:
            # phase2
            # After Phase 2, check if all RHS values are positive
            if np.any(self.tableau[:-1, -1] < 0):
                print('Infeasible: Not all RHS values are positive')
                return False
            
            if not self.check_constraints_satisfied():
                print("Infeasible: All constraints are not satisfied")
                return False

        return True
    
    def check_constraints_satisfied(self):
        solution = self.extract_solution()  # Extract solution from the tableau
        constraints = self.original_constraints       # Get original constraints

        # Evaluate each constraint at the given solution
        for con in constraints:
            repn = generate_standard_repn(con.body)
            lhs_value = sum(value(solution[var.name]) * coef for var, coef in zip(repn.linear_vars, repn.linear_coefs))
            rhs_value = value(con.upper) if con.upper is not None else value(con.lower)
            if con.equality:
                satisfied = np.isclose(lhs_value, rhs_value)
            elif con.upper is not None:
                satisfied = lhs_value <= rhs_value
            else:
                satisfied = lhs_value >= rhs_value

            if not satisfied:
                print(f"Constraint [{con}] is not satisfied by the solution.")
                return False

        print("All constraints are satisfied.")
        return True

    def set_phase2_objective(self):
        """
        Set the objective function for Phase II of the Simplex method.
        This method will also remove artificial variables from the tableau.
        """
        # Remove artificial variables from the tableau
        self.remove_artificial_variables()
        
        # Set the new objective
        self.objective.deactivate()
        self.objective = self.standardizer.original_objective
        self.objective.activate()
        repn = generate_standard_repn(self.objective.expr)
        var_name_to_index = self.standardizer.var_name_to_index()

        # Reset the objective row in the tableau
        self.tableau[-1, :] = 0  # Start with zeros

        # Set the coefficients for the new objective
        for var, coef in zip(repn.linear_vars, repn.linear_coefs):
            var_idx = var_name_to_index[var.name]
            self.tableau[-1, var_idx] = coef if self.objective.sense == minimize else -coef

        # Adjust the objective function row to account for current basis variables
        original_var_indices = self.standardizer.original_var_indices()
        for i, var_index in enumerate(self.basis_vars):
            if var_index in original_var_indices:  # Only consider original variables
                coef = self.tableau[-1, var_index]
                if abs(coef) > 1e-8:
                    self.tableau[-1, :] -= coef * self.tableau[i, :]

        # Record the tableau after setting the Phase II objective
        self.history.append(self.take_snapshot())

        self.print_tableau("phase2 objective set")

    def remove_artificial_variables(self):
        """
        Remove artificial variables from the tableau and basis.
        """
        # Identify indices of artificial variables
        artificial_indices = [var_idx for var_idx, var in enumerate(self.variables) if 'artificial' in var.name]
        
        # Remove columns corresponding to artificial variables from the tableau
        self.tableau = np.delete(self.tableau, artificial_indices, axis=1)

        # Update basis and non-basis variables lists
        self.basis_vars = [var for var in self.basis_vars if var not in artificial_indices]
        self.non_basis_vars = [var for var in self.non_basis_vars if var not in artificial_indices]

        # Remove artificial variables from the variable list
        self.variables = [var for var in self.variables if 'artificial' not in var.name]

    def extract_solution(self):
        solution = {}
        num_constraints = len(self.constraints)
        num_variables = len(self.variables)
        index_to_var_name = self.standardizer.index_to_var_name()
        for i, basic_var_idx in enumerate(self.basis_vars):
            var_name = index_to_var_name[basic_var_idx]
            value = self.tableau[i, -1]
            solution[var_name] = value
        # Non-basic variables are zero
        for non_basic_var_idx in self.non_basis_vars:
            var_name = index_to_var_name[non_basic_var_idx]
            solution[var_name] = 0.0

        # Filter out slack, surplus, and artificial variables if desired
        final_solution = {var_name: value for var_name, value in solution.items()
                        if not ('slack' in var_name or 'surplus' in var_name or 'artificial' in var_name)}
        return final_solution

    def take_snapshot(self, pivot_col=None, pivot_row=None, entering_var_idx=None, leaving_var_idx=None):
        snapshot = {
            "tableau": self.tableau.copy(),
            "entering_var_idx": entering_var_idx,
            "leaving_var_idx": leaving_var_idx,
            "basis_vars": self.basis_vars.copy(),
            "objective_value": self.get_current_objective_value(),
            "pivot_col": pivot_col,
            "pivot_row": pivot_row,
            "optimality_status": self.is_optimal(),
            "feasibility_status": self.is_feasible()
        }
        return snapshot
    
    def get_current_objective_value(self):
        # the last row of the tableau represents the objective function
        # and the last column of that row represents the negative objective value.
        objective_row = self.tableau[-1]  # Get the last row (objective row)
        if self.objective.sense == minimize:
            return -objective_row[-1]  # The last element of the objective row represents -objective_value
        return objective_row[-1]
