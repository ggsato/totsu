from pyomo.opt import SolverResults, TerminationCondition, SolverStatus
from pyomo.environ import SolverFactory
from .super_simplex_solver import InfeasibleProblemError, UnboundedProblemError, OptimizationError
from .branch_and_bound_solver import BranchAndBoundSolver
from ..utils.model_processor import ModelProcessor
from ..utils.logger import totsu_logger

@SolverFactory.register("totsubb", "Pyomo compatible Branch And Bound Solver")
class TotsuBranchAndBoundSolver():
        """
        A custom Pyomo compatible solver that wraps BranchAndBoundSolver
        """
        def __init__(self):
            self.solver = BranchAndBoundSolver()

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
                solution, objective = self.solver.solve(model)
            except InfeasibleProblemError as ve:
                results.solver.status = SolverStatus.aborted
                results.solver.termination_condition = TerminationCondition.infeasible
            except UnboundedProblemError as ue:
                results.solver.status = SolverStatus.aborted
                results.solver.termination_condition = TerminationCondition.unbounded
            except OptimizationError as oe:
                results.solver.status = SolverStatus.aborted
                results.solver.termination_condition = TerminationCondition.maxIterations
            else:
                results.solver.status = SolverStatus.ok
                results.solver.termination_condition = TerminationCondition.optimal
                # Populate results
                results.solver.objective = objective
                results.solver.variables = solution

            return results
