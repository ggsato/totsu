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
    def __init__(self,
                 pricing: str = "bland",          # "bland" | "dantzig" | "devex"
                 lex_ratio: bool = False,         # lexicographic minimum-ratio test
                 phase1_cost_perturb: float = 0.0,
                 phase1_rhs_perturb: float = 0.0,
                 max_itr: int = 10_000,
                 tol: float = 1e-9,
                 use_tableau_history: bool = False):
        self.pricing = pricing
        self.lex_ratio = lex_ratio
        self.phase1_cost_perturb = phase1_cost_perturb
        self.phase1_rhs_perturb = phase1_rhs_perturb
        self.max_itr = max_itr
        self.tol = tol
        self.use_tableau_history = use_tableau_history

        self._deg_pivots = 0
        self._saved_pricing = None
        self._bland_left = 0

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
        self._tableau = Tableau(standardizer, use_history=self.use_tableau_history)
        self._tableau.initialize_tableau()

        # Initialize degeneration tracking
        self._deg_pivots = 0
        self._saved_pricing = None
        self._bland_left = 0

        # Perform Phase I iterations
        self._tableau.phase1_cleanup_artificials(tol=1e-12)
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
        # ---- watchdog state ----
        prev_obj = None
        stalled = 0
        IMPR_TOL = 1e-8 if phase == 1 else 1e-12    # how much Phase objective must improve to reset stall
        STALL_LIMIT = 50 if phase == 1 else 200     # pivots with no meaningful progress before stricter mode

        # flip to stricter Bland-style tie-breaks only when truly stalled
        strict_bland = False
        # store on tableau so select_* can consult it without changing their signatures
        if hasattr(self._tableau, "enable_strict_bland"):
            self._tableau.enable_strict_bland(False)
        else:
            # fallback: attach attribute if setter doesn't exist
            setattr(self._tableau, "strict_bland", False)

        while not self._tableau.is_optimal():
            if iteration >= self.max_itr:
                raise OptimizationError(f"Maximum iterations({self.max_itr}) exceeded.")

            # --- EARLY PHASE-I GATE ---
            if phase == 1 and self._tableau.is_phase1_feasible():
                totsu_logger.info("[Phase-I] Feasible before pivot selection → switch to Phase II.")
                # keep pricing/lex options in sync
                self._tableau.set_pivot_options(pricing=self.pricing, lex_ratio=self.lex_ratio)
                # reset anti-cyclers if you maintain them
                self._deg_pivots = 0
                self._bland_left = 0
                return True  # exit simplex loop to Phase II

            # decay tableau tabu TTL once per iteration
            if getattr(self._tableau, "_tabu_ttl", 0) > 0:
                self._tableau._tabu_ttl -= 1
                if self._tableau._tabu_ttl == 0:
                    self._tableau._tabu_leave_col = None

            # ensure tableau pricing matches solver field
            self._tableau.set_pivot_options(pricing=self.pricing, lex_ratio=self.lex_ratio)

            # Phase I: kick out zero-RHS artificials ASAP
            if phase == 1:
                if self._tableau.phase1_cleanup_artificials(tol=1e-12):
                    iteration += 1
                    if iteration % 1000 == 0:
                        totsu_logger.warning(f"[Phase-I] Removed zero-RHS artificials, continuing...")
                    continue

            # ---- progress watchdog (Phase I benefits most, but harmless in Phase II) ----
            # Try to read the Phase objective RHS; fall back to tableau[-1,-1]
            try:
                obj = float(self._tableau.get_objective_value())
            except Exception:
                obj = float(self._tableau.tableau[-1, -1])

            if prev_obj is not None and obj >= prev_obj - IMPR_TOL:
                stalled += 1
            else:
                stalled = 0
            prev_obj = obj

            if not strict_bland and stalled >= STALL_LIMIT:
                strict_bland = True
                if hasattr(self._tableau, "enable_strict_bland"):
                    self._tableau.enable_strict_bland(True)
                else:
                    setattr(self._tableau, "strict_bland", True)
                totsu_logger.debug(
                    f"Watchdog: enabled strict Bland tie-breaks at iter={iteration}, phase={phase}"
                )
                stalled = 0  # reset after mode switch

            # ---- pivot selection ----
            pivot_col = self._tableau.select_pivot_column(phase, strict=(phase == 1 or strict_bland))
            if pivot_col is None:
                if phase == 2:
                    # No pivot column -> either optimal or unbounded
                    if self._tableau.is_optimal():
                        return True
                    raise UnboundedProblemError("Problem is unbounded.")
                else:
                    # Phase I: no entering column usually means infeasible,
                    # unless your is_optimal() includes 'Phase I objective ~ 0' check.
                    totsu_logger.error("No valid pivot column found in Phase I.")
                    raise InfeasibleProblemError("Problem is infeasible during Phase I.")

            pivot_row = self._tableau.select_pivot_row_harris(pivot_col, phase=phase)
            if pivot_row is None:
                if phase == 1:
                    # Fake-unbounded/blocked direction in Phase I: try a different column
                    # (your (C) guard—just keep looping; the watchdog may flip strict mode)
                    iteration += 1  # count this attempt to avoid infinite loop
                    if iteration % 1 == 0:
                        totsu_logger.warning(f"[Phase-I] Pivot blocked/unbounded on col {pivot_col}") 
                    # --- feasibility gate when Phase I direction is blocked ---
                    if self._tableau.is_phase1_feasible():
                        totsu_logger.info("[Phase-I] Feasible (RHS≥-tol & artificials≈0) after blocked direction; switching to Phase II.")
                        self._deg_pivots = 0
                        self._bland_left = 0
                        self._tableau.set_pivot_options(pricing=self.pricing, lex_ratio=self.lex_ratio)
                        return True  # exit simplex loop to Phase II

                    # --- otherwise: escape the block ---
                    # 1) one-step tabu on this entering column
                    self._tableau._tabu_leave_col = int(pivot_col)
                    self._tableau._tabu_ttl = max(getattr(self._tableau, "_tabu_ttl", 0), 1)

                    # 2) optionally force a stricter pivot-column choice for the next try
                    alt_col = self._tableau.select_pivot_column(phase, strict=True)
                    if alt_col is not None and alt_col != pivot_col:
                        alt_row = self._tableau.select_pivot_row_harris(alt_col, phase=phase)
                        if alt_row is not None:
                            # perform the pivot immediately and loop
                            obj_before = float(self._tableau.tableau[-1, -1])
                            self._tableau.pivot_operation(alt_row, alt_col, phase)
                            iteration += 1

                            # degeneracy accounting remains the same
                            degenerate = bool(getattr(self._tableau, "_last_deg", False))
                            self._deg_pivots = self._deg_pivots + 1 if degenerate else 0
                            # (leave your Bland cooldown code as-is)
                            continue

                    continue

                # Phase II: truly unbounded for this direction
                totsu_logger.error("No valid pivot row found in Phase II; unbounded.")
                raise UnboundedProblemError("Problem is unbounded.")

            # --- capture objective before pivot (optional but handy) ---
            obj_before = float(self._tableau.tableau[-1, -1])

            # ---- pivot ----
            totsu_logger.debug(f"Iter {iteration}: Pivoting on row {pivot_row}, col {pivot_col} (phase {phase})")
            self._tableau.pivot_operation(pivot_row, pivot_col, phase)
            iteration += 1

            # --- read degeneracy flag from tableau ---
            degenerate = bool(getattr(self._tableau, "_last_deg", False))
            if degenerate:
                self._deg_pivots += 1
            else:
                self._deg_pivots = 0

            # --- Bland fallback trigger ---
            if self.pricing == "devex" and self._deg_pivots >= 5 and self._bland_left == 0:
                self._saved_pricing = self.pricing
                self.pricing = "bland"
                self._tableau.set_pivot_options(pricing="bland", lex_ratio=self.lex_ratio)
                self._bland_left = 10
                # optional: clear tabu when switching policy
                self._tableau._tabu_leave_col = None
                self._tableau._tabu_ttl = 0
                totsu_logger.info("[Anticycle] Switch to Bland for 10 pivots")

            # --- Bland cooldown ---
            if self._bland_left > 0:
                self._bland_left -= 1
                if self._bland_left == 0 and self._saved_pricing:
                    self.pricing = self._saved_pricing
                    self._tableau.set_pivot_options(pricing=self.pricing, lex_ratio=self.lex_ratio)
                    totsu_logger.info("[Anticycle] Restore Devex")

        return True

    def get_history(self):
        return self._tableau.history

    def get_current_objective_value(self):
        return self._tableau.get_current_objective_value()

    def get_objective_value(self):
        return ModelProcessor.get_active_objective_value(self.model)

    def is_optimal(self):
        return self._tableau.is_optimal()
    
    def get_dual_variables(self):
        return self._tableau.compute_dual_variables()

    def get_reduced_costs(self, y):
        return self._tableau.compute_reduced_costs(y)

    def extract_solution(self):
        solution =  self._tableau.extract_solution()
        # Finally, let the standardizer post-process the solution
        self._tableau.standardizer.post_process_solution(solution)
        # Store the solution in the model
        ModelProcessor.set_variable_values(self.model, solution)
        return solution
