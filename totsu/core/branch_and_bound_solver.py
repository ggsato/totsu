from pyomo.environ import Constraint, Integers, Binary, maximize, minimize
from .super_simplex_solver import SuperSimplexSolver, InfeasibleProblemError, UnboundedProblemError
from ..utils.model_processor import ModelProcessor
from ..utils.logger import totsu_logger
import heapq

class BBNode:
    def __init__(self, is_minimization, model, bound, solution):
        self.is_minimization = is_minimization
        self.model = model
        self.bound = bound  # float
        self.solution = solution
    
    # For a min problem, we compare nodes by bound ascending
    # For a max problem, we compare negative of the bound.

    def __lt__(self, other):
        # This is used by heapq to compare nodes.
        if self.is_minimization:
            return self.bound < other.bound
        else:
            return -self.bound < -other.bound

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
        self.is_minimization = (model.obj.sense == minimize)
        if self.is_minimization:
            self.best_objective = float('inf')
        else:
            self.best_objective = float('-inf')

        # Reset counters/trackers
        self.nodes_explored = 0
        self.best_solution = None

        # Start best-first search
        self.best_first_branch_and_bound(model)
        #self.branch_and_bound(model) Depth-First

        if self.best_solution is not None:
            # update model with the current best_solution
            ModelProcessor.set_variable_values(model, self.best_solution)
            model.display()

        return self.best_solution, self.best_objective

    def best_first_branch_and_bound(self, root_model):
        """Perform a best-first branch-and-bound search."""
        # Priority queue of (bound, BBNode)
        pq = []

        # 1. Solve the LP relaxation of the root node
        root_solution = self.simplex_solver.solve(root_model)
        root_bound = self.simplex_solver.get_objective_value()

        # Create the root node
        root_node = BBNode(self.is_minimization, root_model, root_bound, root_solution)
        heapq.heappush(pq, root_node)

        while pq:
            # 2. Pop the node with the best bound
            node = heapq.heappop(pq)
            current_bound = node.bound

            # 3. Bounding / Pruning check
            if self.is_minimization:
                # If the node's bound is >= best known integer solution, prune
                if current_bound >= self.best_objective:
                    continue
            else:
                # For maximization
                if current_bound <= self.best_objective:
                    continue

            # 4. Check solution
            self.nodes_explored += 1
            solution = node.solution
            objective_value = node.bound
            totsu_logger.info(f"{self.nodes_explored} nodes explored with the current objective value = {objective_value} with best = {self.best_objective}")

            # Check bounding again with updated solution
            if self.is_minimization and objective_value >= self.best_objective:
                continue
            if not self.is_minimization and objective_value <= self.best_objective:
                continue

            # 5. Check integer feasibility
            if self.is_integer_feasible(solution, node.model):
                # If feasible, update best known solution if better
                if self.is_better_solution(objective_value):
                    self.best_solution = solution.copy()
                    self.best_objective = objective_value
                    totsu_logger.info(f"new best objective {self.best_objective} found on {self.best_solution}")
                # We don't branch further because it’s already integer feasible
                continue
            else:
                # 6. Branch
                fractional_var = self.select_branching_variable(solution, node.model)
                if fractional_var is not None:
                    # Create child nodes
                    branch_models = self.make_branches(node.model, fractional_var, solution)
                    
                    # For each child, solve them right away to get their bound
                    for child_model in branch_models:
                        try:
                            child_solution = self.simplex_solver.solve(child_model)
                            child_bound = self.simplex_solver.get_objective_value()

                            # Bound check — if it’s not obviously pruned, push in PQ
                            if not self.prune(child_bound):
                                child_node = BBNode(self.is_minimization, child_model, child_bound, child_solution)
                                heapq.heappush(pq, child_node)
                            
                        except InfeasibleProblemError:
                            pass  # discard child
                        except UnboundedProblemError:
                            # Mark the entire problem unbounded 
                            raise

    def prune(self, bound):
        """Check if bound is worse than our current best."""
        if self.is_minimization:
            return bound >= self.best_objective
        else:
            return bound <= self.best_objective

    def is_better_solution(self, obj_val):
        """Check if this objective is better than the best known so far."""
        if self.is_minimization:
            return obj_val < self.best_objective
        else:
            return obj_val > self.best_objective

    def make_branches(self, model, fractional_var, solution):
        """Return a list of two branched child models."""
        var = model.find_component(fractional_var)
        lower_bound = int(solution[fractional_var])
        upper_bound = lower_bound + 1

        branch_models = []
        # For binary
        if var.domain is Binary:
            left_model = model.clone()
            left_var = left_model.find_component(var.name)
            left_var.fix(0)
            branch_models.append(left_model)

            right_model = model.clone()
            right_var = right_model.find_component(var.name)
            right_var.fix(1)
            branch_models.append(right_model)

        else:  # integer variable
            var_lb, var_ub = ModelProcessor.get_bounds(var)
            # Possibly clamp the branch bounds if out-of-range
            if var_lb is not None:
                lower_bound = max(lower_bound, var_lb)
            if var_ub is not None:
                upper_bound = min(upper_bound, var_ub)

            left_model = model.clone()
            left_con = Constraint(expr=(var <= lower_bound))
            setattr(left_model, f"branch_{var.name}_le_{lower_bound}", left_con)
            branch_models.append(left_model)

            right_model = model.clone()
            right_con = Constraint(expr=(var >= upper_bound))
            setattr(right_model, f"branch_{var.name}_ge_{upper_bound}", right_con)
            branch_models.append(right_model)

        return branch_models

    def branch_and_bound(self, model):
        try:
            solution = self.simplex_solver.solve(model)
            objective_value = self.simplex_solver.get_objective_value()

            # Bounding
            if self.is_minimization and objective_value >= self.best_objective:
                return  # No need to explore further
            if (not self.is_minimization) and objective_value <= self.best_objective:
                return

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

        # Branching
        self.branch(model, var, lower_bound, upper_bound)

    def is_integer_feasible(self, solution, model):
        for var in ModelProcessor.get_variables(model):
            if var.domain is Binary and solution[var.name] not in {0, 1}:  # Ensure Binary stays in {0,1}
                return False
            if var.domain is Integers and not abs(round(solution[var.name]) - solution[var.name]) < 1e-6:
                return False
        return True

    def select_branching_variable(self, solution, model):
        fractional_vars = []
        for var in ModelProcessor.get_variables(model):
            val = float(solution[var.name])
            if var.domain is Binary and val not in {0, 1}:  
                fractional_vars.append((var.name, abs(val - 0.5)))  # Prioritize closest to 0.5
            elif var.domain is Integers and not val.is_integer():
                fractional_vars.append((var.name, abs(val - round(val))))  # Prioritize closest to an integer

        if not fractional_vars:
            return None

        # **Choose the most fractional variable for best branching**
        fractional_vars.sort(key=lambda x: x[1], reverse=True)
        return fractional_vars[0][0]

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
