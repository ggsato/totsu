from pyomo.opt import SolverResults, TerminationCondition, SolverStatus
from pyomo.opt import SolverFactory
from pyomo.core.kernel.component_map import ComponentMap
from .super_simplex_solver import SuperSimplexSolver, InfeasibleProblemError, UnboundedProblemError, OptimizationError
from ..utils.model_processor import ModelProcessor

@SolverFactory.register("totsu", "Pyomo compatible Simplex Solver")
class TotsuSimplexSolver():
        """
        A custom Pyomo compatible solver that wraps SuperSimplexSolver
        """
        def __init__(self):
            self.solver = SuperSimplexSolver()

        def available(self, exception_flag=False):
            """
            Check if the solver is available.
            """
            # For simplicity, we'll assume the solver is always available.
            return True

        def solve(self, model, tee=False, symbolic_solver_labels=False, **options):
            """
            Solve the given Pyomo model.
            """
            # Create a SolverResults object to store the results
            results = SolverResults()
            
            # solve by SuperSimplexSolver
            try:
                solution = self.solver.solve(model)
            except InfeasibleProblemError as ve:
                results.solver.status = SolverStatus.aborted
                results.solver.termination_condition = TerminationCondition.infeasible
            except UnboundedProblemError as ue:
                results.solver.status = SolverStatus.aborted
                results.solver.termination_condition = TerminationCondition.unbounded
            except OptimizationError as oe:
                results.solver.status = SolverStatus.aborted
                results.solver.termination_condition = TerminationCondition.userLimit
            else:
                results.solver.status = SolverStatus.ok
                results.solver.termination_condition = TerminationCondition.optimal
                # Populate results
                results.solver.objective = self.solver.get_current_objective_value()
                results.solver.variables = solution
                # Store
                self.store_solution_in_model(model, solution)

            return results
        
        def store_solution_in_model(self, model, solution):
            # Store variable values
            for var_name, value in solution.items():
                var = model.find_component(var_name)
                if var is not None:
                    var.set_value(value)
                    print(f"{var_name} = {value}")
                else:
                    print(f"Variable {var_name} not found in model.")

            # Calculate and store dual variables
            if hasattr(model, 'dual'):
                y = self.solver.get_dual_variables()
                # Get the same Constraint instances
                constraints = list(ModelProcessor.get_constraints(model))
                index_to_con = {i: con for i, con in enumerate(constraints)}
                for i, dual_value in enumerate(y):
                    con = index_to_con[i]
                    model.dual[con] = dual_value
                    print(f"Dual variable for constraint {con.name} = {dual_value}")

            # Calculate and store reduced costs
            if hasattr(model, 'rc'):
                reduced_costs = self.solver.get_reduced_costs(y)
                for var_name, rc_value in reduced_costs.items():
                    var = model.find_component(var_name)
                    if var is not None:
                        model.rc[var] = rc_value
                        print(f"Reduced cost for variable {var.name} = {rc_value}")
                    else:
                        print(f"Variable {var_name} not found in model.")
