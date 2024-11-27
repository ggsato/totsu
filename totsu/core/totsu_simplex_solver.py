from pyomo.opt import SolverResults, TerminationCondition, SolverStatus
# Register the custom solver with Pyomo
from pyomo.opt import SolverFactory
from .super_simplex_solver import SuperSimplexSolver, InfeasibleProblemError, UnboundedProblemError, OptimizationError

@SolverFactory.register("totsu_simplex_solver", "Pyomo compatible Simplex Solver")
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
            
            # Ensure the model is ready
            if not hasattr(model, 'objective'):
                raise ValueError("The model must have an objective to solve.")
            
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
            pass
        