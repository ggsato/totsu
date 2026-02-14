import numpy as np
from pyomo.environ import (
    value, minimize
)
from pyomo.repn import generate_standard_repn
from ..utils.logger import totsu_logger

ARTIFICIAL_MARKER = -1

# --- in tableau.py ---

class Tableau:
    def __init__(self, standardizer, use_history=False):
        self.standardizer = standardizer # Be careful in phase 2 because it is not synchronized
        self.updated_objective = None
        self._tableau = None  # The simplex tableau
        self.updated_tableau = None
        self.updated_variables = None
        self.basis_vars = []
        self.non_basis_vars = []
        self.history = [] if use_history else None  # To store tableau history for visualization

        self._pricing = "bland"
        self._lex_ratio = False
        self._phase1_cost_eps = 0.0
        self._phase1_rhs_eps  = 0.0

        # Devex bookkeeping
        self._devex_weights = None   # np.ndarray of shape (num_cols,), set later
        self._devex_basis   = None   # set of basic column indices (for refresh)
        # Degeneration tracking and prevention
        self._tabu_leave_col = None
        self._tabu_ttl = 0
        self._last_deg = False
        self._last_delta_obj = None

    @property
    def standard_model(self):
        return self.standardizer.standard_model
    
    @property
    def constraints(self):
        return self.standardizer.constraints
    
    @property
    def constraint_names(self):
        return [con.name for con in self.constraints]
    
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
    def original_variables(self):
        return self.standardizer.original_variables 

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

    def enable_strict_bland(self, flag: bool):
        self.strict_bland = bool(flag)

    def set_pivot_options(self, pricing: str = "bland", lex_ratio: bool = False):
        self._pricing = pricing
        self._lex_ratio = lex_ratio

    def set_phase1_perturbations(self, cost_eps: float = 0.0, rhs_eps: float = 0.0):
        self._phase1_cost_eps = float(cost_eps or 0.0)
        self._phase1_rhs_eps  = float(rhs_eps or 0.0)

    def initialize_tableau(self):
        # Initialize basis and non-basis variables
        self.identify_basis_variables()

        # Construct the initial tableau matrix
        self.construct_tableau()

        # Initialize Devex weights when we know number of columns (exclude RHS)
        num_cols = self.tableau.shape[1] - 1
        if self._devex_weights is None or len(self._devex_weights) != num_cols:
            # Start with all ones (standard Devex)
            import numpy as np
            self._devex_weights = np.ones(num_cols, dtype=float)

        # If Phase I is active (has artificial vars), apply tiny perturbations to help tie-break degeneracy
        if self._phase1_cost_eps > 0.0:
            # Add distinct epsilons to objective row for artificial columns only
            # heuristic: scan var names and nudge their objective coefficients
            var_idx_map = self.var_name_to_index()
            k = 1
            for name, j in var_idx_map.items():
                if name.startswith("artificial_"):
                    self.tableau[-1, j] += self._phase1_cost_eps * k
                    k += 1

        if self._phase1_rhs_eps > 0.0:
            # Nudge RHS of rows whose basic variable is artificial (to break ties in feasibility)
            for i, var_j in enumerate(self.basis_vars):
                var_name = self.index_to_var_name()[var_j]
                if isinstance(var_name, str) and var_name.startswith("artificial_"):
                    self.tableau[i, -1] += self._phase1_rhs_eps

    def identify_basis_variables(self):
        # Basis variables are initially the slack and artificial variables
        for idx, var in enumerate(self.variables):
            if 'slack' in var.name or 'artificial' in var.name:
                self.basis_vars.append(idx)
            else:
                self.non_basis_vars.append(idx)
        totsu_logger.debug(f"initial basis_vars = {self.basis_vars}")

    def construct_tableau(self):
        num_constraints = len(self.constraints)
        num_variables = len(self.variables)
        totsu_logger.debug(f"constructing tableau with constraints = {[con.name for con in self.constraints]} and variables = {[var.name for var in self.variables]}")

        var_name_to_index = self.var_name_to_index()

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
                if var_name not in var_name_to_index:
                    totsu_logger.debug(f"Skipping fixed variable '{var_name}' in constraint '{con.name}'")
                    continue  # Skip fixed variables
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
            var_name = self.index_to_var_name()[var_idx]
            if 'artificial' in var_name:
                # Subtract the constraint row from the objective row
                self.tableau[-1, :] -= self.tableau[i, :]

        self.print_tableau("constructed")

        # Record the initial tableau in history
        if self.history is not None:
            self.history.append(self.take_snapshot(1))

    def print_tableau(self, message):
        totsu_logger.debug(f"tableau[{message}]:")
        totsu_logger.debug(f"{self.tableau}")

    def phase1_cleanup_artificials(self, tol=1e-12):
        """
        For each row whose basic var is 'artificial_*' and RHS ~ 0,
        pivot it out by bringing in the smallest-index non-artificial column
        with a nonzero coefficient in that row.
        """
        idx2name = self.index_to_var_name()
        m, n = self.tableau.shape
        last_col = n - 1

        changed = False
        for i, b_idx in enumerate(self.basis_vars):
            if "artificial" not in idx2name[b_idx]:
                continue
            rhs = float(self.tableau[i, last_col])
            if abs(rhs) > tol:
                continue  # positive artificial must be handled by normal Phase I

            # Find a structural/slack column to enter (avoid artificials)
            row = self.tableau[i, :last_col]
            candidates = [j for j, aij in enumerate(row)
                        if abs(aij) > tol and "artificial" not in idx2name[j]]
            if not candidates:
                continue  # nothing to pivot with; leave for normal iterations

            enter = min(candidates)  # Bland style
            self.pivot_operation(i, enter, phase=1)
            changed = True
        return changed

    def _phase1_choose_artificial_row(self) -> int | None:
        tol = getattr(self, "tol", 1e-9)
        best_i = None
        best_rhs = 0.0
        var_names = self.index_to_var_name()  # listlike: col_idx -> name
        for i, col_idx in enumerate(self.basis_vars):
            name = var_names[col_idx]
            rhs = float(self.tableau[i, -1])
            if rhs > tol and name.startswith("artificial_"):
                if rhs > best_rhs:
                    best_rhs = rhs
                    best_i = i
        return best_i

    def select_pivot_column_phase1_feasibility_directed(self, strict: bool = False):
        # 1) Pick an artificial basic row that still has positive RHS
        i_art = self._phase1_choose_artificial_row()
        if i_art is None:
            totsu_logger.warning("Phase-I feasibility-directed: no artificial row with positive RHS found")
            return None

        tol = getattr(self, "tol", 1e-9)
        row = self.tableau[i_art, :-1]
        rhs = self.tableau[:-1, -1]

        # 2) Candidate columns must (a) reduce that artificial row (a_ij>0 on i_art),
        #    (b) be nonbasic, and (c) NOT be any artificial variable.
        names = self.index_to_var_name()
        cand = [
            j for j, aij in enumerate(row)
            if aij > tol
            and j not in self.basis_vars
            and not names[j].startswith("artificial_")
        ]
        if not cand:
            totsu_logger.warning("Phase-I feasibility-directed: no candidate columns found to reduce artificial row")
            return None

        # ---- B) Strict zero-RHS blocker filtering ----
        # Disallow entering columns that have a positive coefficient on ANY zero-RHS row.
        tol_zero = 1e-12
        zero_rows = [i for i, b in enumerate(rhs) if abs(b) <= tol_zero]

        def is_safe_column(j: int) -> bool:
            col = self.tableau[:-1, j]
            # safe if no zero-RHS row has a positive coefficient in this column
            return all(col[i] <= tol for i in zero_rows)

        safe = [j for j in cand if is_safe_column(j)]
        if safe:
            cand = safe
        # else: keep original 'cand' (we will do a safe degenerate fallback below)

        # 3) Viability check with degeneracy guard:
        #    (a) column must have at least one positive entry somewhere, AND
        #    (b) avoid columns that have any positive entry in a row with RHS ~= 0
        #        (those create rmin=0 blockers and force degenerate pivots).
        def classify_column(j: int):
            col = self.tableau[:-1, j]
            totsu_logger.debug(
                f"Phase-I feasibility-directed: checking viability of col {j}, entries = {col}"
            )
            any_pos = (col > tol).any()
            tol_zero = 1e-12
            has_zero_blocker = any((col[i] > tol) and (abs(rhs[i]) <= tol_zero) for i in range(len(rhs)))
            return any_pos, has_zero_blocker

        good, risky = [], []
        for j in cand:
            any_pos, has_zero_blocker = classify_column(j)
            if not any_pos:
                continue
            (risky if has_zero_blocker else good).append(j)

        # Prefer columns without zero-RHS blockers; fall back to risky ones only if nothing good exists.
        cand = good if good else risky
        if not cand:
            totsu_logger.warning("Phase-I feasibility-directed: no viable candidate columns found")
            return None  # all blocked → let caller handle as blocked/unbounded

        # 4) Prefer columns where the *artificial row* wins the min-ratio test (so it leaves).
        def best_leaving_row(j: int):
            col = self.tableau[:-1, j]
            pos = [(i, rhs[i] / col[i]) for i in range(len(rhs)) if col[i] > tol]
            # lex: tie-break by row index for stability
            totsu_logger.debug(f"Phase-I feasibility-directed: ratios for col {j} are {pos}")
            return min(pos, key=lambda t: (t[1], t[0]))  # (i*, theta*)

        # First pass: pick a candidate where i_art is the leaving row
        for j in sorted(cand):
            i_star, _theta = best_leaving_row(j)
            if i_star == i_art:
                totsu_logger.debug(f"Phase-I feasibility-directed: picked col {j} where artificial row {i_art} leaves")
                # hint to solver: remember this forced pivot
                setattr(self, "_phase1_forced_pivot", (int(i_art), int(j)))
                return int(j)

        # 5) Fallback: still progress the targeted artificial row.
        # ---- Degenerate-safe fallback (do NOT force; allow θ=0 pivot) ----
        totsu_logger.debug(
            f"Phase-I feasibility-directed: no candidate lets artificial row {i_art} leave; choosing min-harm reducer"
        )

        def harm(j: int) -> float:
            col = self.tableau[:-1, j]
            # sum of positive coefficients on zero-RHS rows (the “blockers”)
            return sum(col[i] for i in zero_rows if col[i] > tol)

        # Prefer columns with the least harm to zero-RHS rows; then largest reducer on the artificial row; then smallest j
        j_pick = min(cand, key=lambda j: (harm(j), -row[j], j))

        # IMPORTANT: do NOT set _phase1_forced_pivot here. Let Harris pick.
        # This will typically select a zero-RHS row (θ=0) => feasibility preserved, basis changes.
        return int(j_pick)

    def _has_viable_leaving_row(self, j: int, tol: float) -> bool:
        # Any positive column entry in row-space? (primal feasibility, RHS assumed >= -tol)
        col = self.tableau[:-1, j]
        return (col > tol).any()

    def select_pivot_column(self, phase, strict=False):

        # If not using Devex, keep your original behavior
        if getattr(self, "_pricing", "bland") != "devex":
            if phase == 1:
                return self.select_pivot_column_phase1_feasibility_directed()
            elif phase == 2:
                return self.select_pivot_column_phase2()

        A  = self.tableau
        rc = A[-1, :-1].copy()  # reduced costs (objective row without RHS)

        # --- legacy pick (used to infer sign + for override) ---
        if phase == 1:
            j0 = self.select_pivot_column_phase1_feasibility_directed()
            if j0 is not None:
                names = self.index_to_var_name()
                totsu_logger.debug("Phase-I: using feasibility-directed entering col %d (%s)", j0, names[j0])
                return j0
        else:
            j0 = self.select_pivot_column_phase2()
        # j0 may be None if optimal/degenerate; we’ll guard later.

        # Your helper for sign convention
        improving = self._improving_mask(rc, phase)
        cand_idx = np.where(improving)[0]
        if cand_idx.size == 0:
            totsu_logger.debug("no column was selected by devex pivot_column (no improving candidates)")
            return j0  # fall back (usually None → signals optimal/phase-done)

        # --- tabu filter (only if alternatives exist) ---
        if not hasattr(self, "_tabu_ttl"):       self._tabu_ttl = 0
        if not hasattr(self, "_tabu_leave_col"): self._tabu_leave_col = None
        if self._tabu_ttl > 0 and self._tabu_leave_col is not None:
            cand_wo_tabu = [j for j in cand_idx if j != self._tabu_leave_col]
            if cand_wo_tabu:
                cand_idx = np.array(cand_wo_tabu, dtype=int)

        # (If you moved TTL countdown to the solver loop, you can drop this)
        if self._tabu_ttl > 0:
            self._tabu_ttl -= 1
            if self._tabu_ttl == 0:
                self._tabu_leave_col = None

        # --- avoid artificial re-entry when alternatives exist ---
        names = self.index_to_var_name()
        basis_set = set(self.basis_vars)
        filtered = [j for j in cand_idx
                    if not (isinstance(names[j], str)
                            and names[j].startswith("artificial_")
                            and (j not in basis_set))]
        if filtered:
            cand_idx = np.array(filtered, dtype=int)

        # --- Devex weights + scoring ---
        if getattr(self, "_devex_weights", None) is None or len(self._devex_weights) != rc.size:
            self._devex_weights = np.ones(rc.size, dtype=float)

        w = np.maximum(self._devex_weights[cand_idx], 1e-16)
        scores = np.abs(rc[cand_idx]) / np.sqrt(w)
        if scores.size == 0:
            totsu_logger.debug("no column was selected by devex pivot_column (no scores computed)")
            return j0

        j_enter = cand_idx[int(np.argmax(scores))]

        # --- legacy override: avoid artificial if legacy found non-artificial ---
        if j0 is not None:
            name_devex  = names[j_enter]
            name_legacy = names[j0]
            if (isinstance(name_devex, str) and name_devex.startswith("artificial_")
                and not (isinstance(name_legacy, str) and name_legacy.startswith("artificial_"))
                and improving[j0]):
                totsu_logger.debug(f"Devex pivot column overridden: picking legacy {j0} ({name_legacy}) over devex {j_enter} ({name_devex})")
                return int(j0)

        totsu_logger.debug(f"Devex pivot column selected: {j_enter} with score {scores[np.argmax(scores)]}")
        return int(j_enter)

        
    def select_pivot_column_phase2(self):
        objective_row = self.tableau[-1, :-1]
        min_value = np.min(objective_row)
        if min_value >= -1e-8:
            return None
        pivot_cols = np.where(self.tableau[-1, :-1] < -1e-8)[0]

        # Exclude columns already in basis_vars
        eligible_cols = [col for col in pivot_cols if col not in self.basis_vars]
        
        if not eligible_cols:
            return None  # No eligible columns
        
        # No upper bounds check required.
        # The standard simplex method assumes variables are non-negative and unbounded above (i.e., have infinite upper bounds).

        # Choose the column with the most negative coefficient
        obj_values = objective_row[eligible_cols]
        col = eligible_cols[np.argmin(obj_values)]
        totsu_logger.debug(f"The column of the most negative coefficient out of [{obj_values}/{eligible_cols}] is {col}")
        return int(col)

    def select_pivot_row(self, pivot_col, phase: int = 2, strict=False):
        ratios = self.compute_ratios(pivot_col, len(self.constraints))
        if not ratios:
            # No valid leaving row for this column
            totsu_logger.debug("no row was selected by pivot_row")
            return None  # Unbounded

        # --- tie set ---
        rmin = min(r for r, _ in ratios)
        tol = 1e-10
        ties = [i for r, i in ratios if abs(r - rmin) <= tol]

        if strict or phase == 1:
            # prefer removing artificials when tied
            idx2name = self.index_to_var_name()
            arti = [i for i in ties if "artificial" in idx2name[self.basis_vars[i]]]
            if arti:
                return min(arti)
            return min(ties)  # Bland fallback
        else:
            # original behavior (first of ties)
            return ties[0]

    def select_pivot_row_harris(self, col: int, phase: int) -> int | None:
        eps_piv = 1e-12
        tol = 1e-12 if phase == 2 else 1e-10
        
        # ---------- Phase-I feasibility-directed: honor forced leaving row ----------
        forced = getattr(self, "_phase1_forced_pivot", None)
        if phase == 1 and forced is not None:
            i_forced, j_forced = forced
            if j_forced == col:
                A = self.tableau
                tol = getattr(self, "tol", 1e-9)
                eps_piv = 1e-12
                harris_tol = 1e-12

                Acol = A[:-1, col]
                rhs  = A[:-1, -1]

                a_forced = A[i_forced, col]
                # Compute full Harris ratios first
                ratios = [(rhs[i] / Acol[i], i) for i in range(len(rhs)) if Acol[i] > eps_piv]
                if not ratios:
                    setattr(self, "_phase1_forced_pivot", None)
                    return None

                rmin, i_min = min(ratios, key=lambda t: t[0])
                gamma = 1e-7 * (1.0 + abs(rmin))

                if a_forced > eps_piv:
                    r_forced = rhs[i_forced] / a_forced
                    # Only force if the forced row is within the Harris band of the min ratio
                    if r_forced <= rmin + gamma and r_forced >= -harris_tol:
                        totsu_logger.debug(
                            f"[Phase-I] Forcing leaving row {i_forced} on col {col} (a_ij={a_forced:.3g}, step={r_forced:.3g})"
                        )
                        setattr(self, "_phase1_forced_pivot", None)
                        return i_forced
                    else:
                        totsu_logger.debug(
                            f"[Phase-I] Reject force: forced row {i_forced} ratio={r_forced:.3g} "
                            f"not within Harris band of min ratio={rmin:.3g} (row {i_min})"
                        )
                else:
                    totsu_logger.debug(
                        f"[Phase-I] Forced row {i_forced} invalid on col {col} (a_ij={a_forced:.3g} ≤ tol); falling back."
                    )

                # consume hint after evaluating
                setattr(self, "_phase1_forced_pivot", None)

        A = self.tableau
        Acol = A[:-1, col]           # exclude objective row
        rhs  = A[:-1, -1]

        # First pass: eligible rows and ratios
        ratios = []
        for i, aij in enumerate(Acol):
            if aij > eps_piv:
                ratios.append((rhs[i] / aij, i, aij))
        if not ratios:
            totsu_logger.debug("no row was selected by Harris pivot_row")
            return None

        rmin = min(r for r, _, _ in ratios)

        # Harris band
        gamma = 1e-7 * (1.0 + abs(rmin))
        band = [(r, i, aij) for (r, i, aij) in ratios if r <= rmin + gamma]
        if not band:
            totsu_logger.debug("no row was selected by Harris pivot_row after banding")
            return None

        # Your existing preference: (remove artificials first, then strongest pivot, then row index)
        def pref_key(t):
            r, i, aij = t
            is_art = 0 if str(self.basis_vars[i]).startswith("artificial_") else 1
            return (is_art, -abs(aij), i)

        band.sort(key=pref_key)

        # If lex disabled or no tie after your key, keep your choice
        if not getattr(self, "_lex_ratio", False):
            return band[0][1]

        # Detect ties under your preference key’s first two components
        # (artificial priority and |aij|). We keep only those sharing the best pair.
        best_is_art, best_neg_abs = pref_key(band[0])[:2]
        tied = [(r, i, aij) for (r, i, aij) in band
                if pref_key((r, i, aij))[:2] == (best_is_art, best_neg_abs)]

        if len(tied) == 1:
            return tied[0][1]

        # Lexicographic tie-break among 'tied'
        cols_wo_rhs = A.shape[1] - 1
        vecs = []
        for r, i, aij in tied:
            piv = A[i, col]
            if abs(piv) <= tol:
                continue
            v = (A[i, :cols_wo_rhs] / piv).copy()
            v[col] = 0.0                     # ignore pivot column itself
            vecs.append((i, tuple(np.round(v, 15))))  # stable compare

        if not vecs:
            # fallback to your original pick if something filtered everything out
            return band[0][1]

        # Pick lexicographically smallest normalized row
        i_star = min(vecs, key=lambda t: t[1])[0]
        totsu_logger.debug(f"[Lex] candidates={ [i for _,i,_ in tied] } chosen={i_star}")
        return int(i_star)

    def pivot_operation(self, pivot_row, pivot_col, phase=1):
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

        # Record the tableau after pivot
        if self.history is not None:
            self.history.append(self.take_snapshot(phase, pivot_col, pivot_row, entering_var_idx, leaving_var_idx))

        # reward columns that actually pivoted
        if getattr(self, "_devex_weights", None) is not None:
            # compute step length for this pivot
            step = float(self.tableau[pivot_row, -1])  # RHS of the pivot row *after* normalization is 1
            # but better compute *before* row normalization:
            # step = rhs_before / aij_before   # store these right before you pivot

            obj_before = getattr(self, "_obj_before", None)
            obj_after  = float(self.tableau[-1, -1])  # or your current objective getter

            degenerate = (step <= 1e-14) or (obj_before is not None and abs(obj_after - obj_before) <= 1e-14)

            # --- UPDATE WEIGHTS ONLY IF NOT DEGENERATE ---
            if degenerate:
                # expose to solver loop:
                self._last_deg = bool(degenerate)
                self._last_delta_obj = None if obj_before is None else (obj_after - obj_before)
                totsu_logger.warning(f"[Devex] degenerate pivot detected (step={step}, obj_before={obj_before}, obj_after={obj_after}), weights unchanged")
            else:
                j = int(pivot_col)
                self._devex_weights[j] = max(1.0, self._devex_weights[j] + 1.0)
                totsu_logger.debug(f"[Devex] weight[{pivot_col}] -> {self._devex_weights[pivot_col]}")

                # after pivot completes & you know the leaving column index j_leave
                self._tabu_leave_col = int(leaving_var_idx)
                self._tabu_ttl = 5  # forbid re-entry for next 5 pivots

            # remember objective for next iteration
            self._obj_before = obj_after

        # Debugging output
        index_to_var_name = self.index_to_var_name()
        totsu_logger.debug(f"Pivoting: Row {pivot_row}, Column {pivot_col}")
        totsu_logger.debug(f"Leaving variable: {index_to_var_name[leaving_var_idx]}")
        totsu_logger.debug(f"Entering variable: {index_to_var_name[entering_var_idx]}")
        totsu_logger.debug(f"After pivot, basis_vars: {[index_to_var_name[idx] for idx in self.basis_vars if idx != ARTIFICIAL_MARKER]} by name, {self.basis_vars} by idx")
        totsu_logger.debug(f"Tableau after pivot operation:\n{self.tableau}")

    def is_optimal(self):
        if self.updated_tableau is None:
            # We are in Phase I
            # For an equality constraint with rhs = 0, the initial tableau can be optimal
            # This check prevents premature optimality in Phase I
            if self.select_pivot_column(1) is not None:
                return False
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
            
            # Check artificial variables' values
            var_name_to_index = self.var_name_to_index()
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

            totsu_logger.debug(f"Is optimal. objective_value = {objective_value}, artificial_values = {artificial_values}, artificial_indices = {artificial_indices}")
            return True   # Optimality achieved in Phase I
        else:
            # Standard optimality condition for Phase II
            """
            Check if all coefficients in the objective row are non-negative.
            """
            objective_row = self.tableau[-1, :-1]
            return all(coef >= -1e-8 for coef in objective_row)

    def is_phase1_feasible(self, tol: float = 1e-9) -> bool:
        """
        Phase-I feasibility gate:
        - All RHS must be >= -tol (primal feasible),
        - Any basic artificial must have RHS <= tol (i.e., driven to ~0).
        """
        import numpy as np

        rhs = self.tableau[:-1, -1]
        if np.any(rhs < -tol):
            totsu_logger.warning(f"Phase-I infeasible: some RHS < -tol ({tol}): {rhs}")
            return False

        names = self.index_to_var_name()
        for i, j in enumerate(self.basis_vars):
            nm = names[j]
            if isinstance(nm, str) and nm.startswith("artificial_"):
                # Artificial still carrying positive mass → not feasible yet
                if self.tableau[i, -1] > tol:
                    totsu_logger.warning(f"Phase-I infeasible: artificial basic var {nm} has RHS = {self.tableau[i, -1]} > tol ({tol})")    
                    return False

        # also check the Phase-I objective value
        obj_val = float(self.tableau[-1, -1])
        if abs(obj_val) > 1e-7:   # slightly looser than tol
            totsu_logger.warning(f"Phase-I infeasible: objective value = {obj_val} > tol ({tol})")
            return False

        return True

    def is_feasible(self):
        if self.updated_tableau is None:
            # phase1
            var_name_to_index = self.var_name_to_index()
            # After Phase I, check if the artificial variables are zero in the solution
            artificial_indices = [var_name_to_index[var.name] for var in self.artificial_vars]
            for idx in artificial_indices:
                if idx in self.basis_vars:
                    row_idx = self.basis_vars.index(idx)
                    value = self.tableau[row_idx, -1]
                    if abs(value) > 1e-8:
                        totsu_logger.debug(f"Infeasible: Artificial variable is positive in basis({self.basis_vars}) at row({row_idx})")
                        return False  # Artificial variable is positive in basis
                else:
                    # Check value of non-basic artificial variable
                    # It should be zero or confirm via tableau calculations
                    pass  # Non-basic variables have zero values in the solution
            # Additionally, check that the objective value (sum of artificial variables) is zero
            objective_value = self.tableau[-1, -1]
            if abs(objective_value) > 1e-8:
                totsu_logger.debug("Infeasible: Objective value is zero")
                return False
        else:
            # phase2
            # After Phase 2, check if all RHS values are positive
            if np.any(self.tableau[:-1, -1] < 0):
                totsu_logger.debug('Infeasible: Not all RHS values are positive')
                return False
            
            if not self.check_constraints_satisfied():
                totsu_logger.debug("Infeasible: All constraints are not satisfied")
                return False

        return True
    
    def check_constraints_satisfied(self, phase=2):
        solution = self.extract_solution()  # Extract solution from the tableau
        if phase == 1:
            constraints = self.constraints       # Get standardized constraints
        elif phase == 2:
            constraints = self.original_constraints

        # Evaluate each constraint at the given solution
        for con in constraints:
            repn = generate_standard_repn(con.body)
            lhs_value = sum(value(solution[var.name]) * coef for var, coef in zip(repn.linear_vars, repn.linear_coefs)) + repn.constant
            rhs_value = value(con.upper) if con.upper is not None else value(con.lower)
            if con.equality:
                satisfied = np.isclose(lhs_value, rhs_value)
            elif con.upper is not None:
                satisfied = lhs_value <= rhs_value
            else:
                satisfied = lhs_value >= rhs_value

            if not satisfied:
                totsu_logger.debug(f"Constraint [{con}: {repn}, {lhs_value} = {rhs_value}] is not satisfied by the solution.")
                return False

        totsu_logger.debug("All constraints are satisfied.")
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
        var_name_to_index = self.var_name_to_index()

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
        if self.history is not None:
            self.history.append(self.take_snapshot(2))

        self.print_tableau("phase2 objective set")

    def remove_artificial_variables(self):
        """
        Remove artificial variables from the tableau and basis.
        """
        # Identify indices of artificial variables
        artificial_indices = [var_idx for var_idx, var in enumerate(self.variables) if 'artificial' in var.name]
        
        # Remove columns corresponding to artificial variables from the tableau
        self.tableau = np.delete(self.tableau, artificial_indices, axis=1)

        # Remove artificial variables from the variable list
        new_variables = [var for var in self.variables if 'artificial' not in var.name]
        
        # Build mapping from old index to new index
        old_index_to_new_index = {}
        new_idx = 0
        for old_idx, var in enumerate(self.variables):
            if 'artificial' not in var.name:
                old_index_to_new_index[old_idx] = new_idx
                new_idx += 1
            else:
                old_index_to_new_index[old_idx] = ARTIFICIAL_MARKER

        # Update basis and non-basis variables lists
        # Replace indices corresponding to artificial variables with the marker
        self.basis_vars = [old_index_to_new_index.get(var_idx, ARTIFICIAL_MARKER) for var_idx in self.basis_vars]
        self.non_basis_vars = [old_index_to_new_index.get(var_idx, ARTIFICIAL_MARKER) for var_idx in self.non_basis_vars]

        # Remove the marker from non_basis_vars
        self.non_basis_vars = [var_idx for var_idx in self.non_basis_vars if var_idx != ARTIFICIAL_MARKER]

        # Update variables list
        self.variables = new_variables

    def extract_solution(self):
        solution = {}
        index_to_var_name = self.index_to_var_name()
        for i, basic_var_idx in enumerate(self.basis_vars):
            if basic_var_idx == ARTIFICIAL_MARKER:
                continue
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

    def take_snapshot(self, phase, pivot_col=None, pivot_row=None, entering_var_idx=None, leaving_var_idx=None):
        snapshot = {
            "tableau": self.tableau.copy(),
            "entering_var_idx": entering_var_idx,
            "leaving_var_idx": leaving_var_idx,
            "basis_vars": self.basis_vars.copy(),
            "objective_value": self.get_current_objective_value(),
            "pivot_col": pivot_col,
            "pivot_row": pivot_row,
            "optimality_status": self.is_optimal(),
            "feasibility_status": self.is_feasible(),
            "objective_row": self.tableau[-1, :-1].copy(),  # Coefficients of the objective row
            "ratios": self.compute_ratios(pivot_col, len(self.constraints)) if pivot_col is not None else [],  # Ratios for the selected pivot column
            "variable_names": [var.name for var in self.variables],  # Include variable names
            "phase": phase
        }
        return snapshot
    
    def compute_ratios(self, pivot_col, num_constraints):
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
        totsu_logger.debug(f"computing ratio for {num_constraints} constraints = {ratios}")
        return ratios
    
    def get_current_objective_value(self):
        # the last row of the tableau represents the objective function
        # and the last column of that row represents the negative objective value.
        objective_row = self.tableau[-1]  # Get the last row (objective row)
        if self.objective.sense == minimize:
            return -objective_row[-1]  # The last element of the objective row represents -objective_value
        return objective_row[-1]

    def compute_dual_variables(self):
        """
        Compute the dual variables y^T = c_B^T * B^{-1}
        Handles cases where basis_vars contain ARTIFICIAL_MARKER (-1).
        Returns a dual variables array aligned with all constraints, setting y_i=0
        for constraints with ARTIFICIAL_MARKER.
        """
        index_to_var_name = self.index_to_var_name()
        num_constraints = len(self.constraints)
        
        # Initialize dual variables with zeros for all constraints
        y = np.zeros(num_constraints)
        
        # Identify constraints with valid basis variables
        valid_constraints = [i for i, idx in enumerate(self.basis_vars) if idx != ARTIFICIAL_MARKER]
        
        if not valid_constraints:
            totsu_logger.debug("No valid basis variables found for dual variables computation.")
            return y  # All y_i are zero
        
        # Extract c_B (coefficients of valid basic variables in the objective function)
        c_B = []
        for i in valid_constraints:
            idx = self.basis_vars[i]
            var_name = index_to_var_name[idx]
            var = self.get_variable_by_name(var_name)
            coef = self.get_objective_coefficient(var)
            c_B.append(coef)
        c_B = np.array(c_B)
        
        # Extract A_S: columns of A for basis_vars in valid_constraints
        A_S = []
        for i in valid_constraints:
            idx = self.basis_vars[i]
            column = self.get_variable_column_in_constraints_by_index(idx)
            A_S.append(column)
        A_S = np.column_stack(A_S)  # Shape: (num_constraints, num_valid_basis_vars)
        
        # Check if A_S is square
        num_valid_basis_vars = len(valid_constraints)
        if A_S.shape[1] != num_valid_basis_vars:
            totsu_logger.error("Basis matrix A_S is not square. Cannot compute dual variables.")
            raise ValueError("Basis matrix A_S must be square to compute dual variables.")
        
        # Solve for y_S in the system A_S^T y_S = c_B
        # This requires A_S^T to be invertible
        try:
            # Check if A_S is square
            if A_S.shape[0] != A_S.shape[1]:
                # Use least squares if A_S is not square
                y_S, residuals, rank, s = np.linalg.lstsq(A_S.T, c_B, rcond=None)
                totsu_logger.debug(f"Dual variables (least squares): {y_S}")
            else:
                # Direct inversion if A_S is square
                B_inv = np.linalg.inv(A_S)
                y_S = c_B @ B_inv
                totsu_logger.debug(f"Dual variables (inverted basis matrix): {y_S}")
        except np.linalg.LinAlgError as e:
            totsu_logger.error(f"Cannot compute dual variables: {e}")
            raise
        
        # Assign y_S to the corresponding positions in y
        for i, constraint_idx in enumerate(valid_constraints):
            y[constraint_idx] = y_S[i]
            totsu_logger.debug(f"Dual variable y[{constraint_idx}] set to {y_S[i]}")
        
        # y remains 0 for constraints with ARTIFICIAL_MARKER
        for i in range(num_constraints):
            if i not in valid_constraints:
                totsu_logger.debug(f"Dual variable y[{i}] remains 0 (ARTIFICIAL_MARKER).")
        
        return y

    def compute_reduced_costs(self, y):
        """
        Compute reduced costs for all variables.
        Handles cases where dual variables y_i are zero for constraints with ARTIFICIAL_MARKER.
        """
        index_to_var_name = self.index_to_var_name()
        reduced_costs = {}
        for idx in self.non_basis_vars:
            var_name = index_to_var_name[idx]
            var = self.get_variable_by_name(var_name)
            c_j = self.get_objective_coefficient(var)
            
            # Extract A_j (column of variable in constraints)
            A_j = self.get_variable_column_in_constraints(var_name)
            
            # Compute reduced cost: c_j - y^T A_j
            reduced_cost_j = c_j - y @ A_j
            reduced_costs[var_name] = reduced_cost_j
            totsu_logger.debug(f"Reduced cost for {var_name}: {reduced_cost_j}")
        
        # Reduced costs of basic variables are zero
        for idx in self.basis_vars:
            if idx == ARTIFICIAL_MARKER:
                continue  # Skip artificial variables
            var_name = index_to_var_name[idx]
            reduced_costs[var_name] = 0.0
            totsu_logger.debug(f"Reduced cost for basic variable {var_name}: 0.0")
        
        return reduced_costs

    def get_basis_matrix(self, valid_basis_indices=None):
        """
        Extract the basis matrix B from the tableau.
        If valid_basis_indices is provided, only those indices are considered.
        """
        num_constraints = len(self.constraints)
        if valid_basis_indices is None:
            valid_basis_indices = self.basis_vars
        
        B = []
        for i, idx in enumerate(valid_basis_indices):
            if idx == ARTIFICIAL_MARKER:
                continue  # Skip artificial variables
            column = self.get_variable_column_in_constraints_by_index(idx)
            B.append(column)
        
        if not B:
            return np.array([])  # Return empty array if no valid basis variables
        B = np.column_stack(B)
        return B

    def get_variable_column_in_constraints_by_index(self, var_idx):
        """
        Extract the column of a variable in the constraints by variable index
        """
        var_name = self.index_to_var_name()[var_idx]
        return self.get_variable_column_in_constraints(var_name)

    def get_variable_column_in_constraints(self, var_name):
        """
        Extract the column of a variable in the constraints
        """
        column = []
        for con in self.constraints:
            repn = generate_standard_repn(con.body)
            coef_map = {var.name: coef for var, coef in zip(repn.linear_vars, repn.linear_coefs)}
            column.append(coef_map.get(var_name, 0.0))
        return np.array(column)

    def get_objective_coefficient(self, var):
        """
        Get the coefficient of the variable in the objective function.
        """
        repn = generate_standard_repn(self.objective.expr)
        # Build a mapping between variable names and coefficients
        coef_map = {v.name: coef for v, coef in zip(repn.linear_vars, repn.linear_coefs)}
        return coef_map.get(var.name, 0.0)

    def get_variable_by_name(self, var_name):
        if not hasattr(self, 'var_name_to_var'):
            self.var_name_to_var = {var.name: var for var in self.variables}
        return self.var_name_to_var[var_name]

    def index_to_var_name(self):
        index_to_var_name = {idx: var.name for idx, var in enumerate(self.variables)}
        return index_to_var_name

    def var_name_to_index(self):
        var_name_to_index = {var.name: idx for idx, var in enumerate(self.variables)}
        return var_name_to_index

    def _improving_mask(self, rc: "np.ndarray", phase: int) -> "np.ndarray[bool]":
        # If your current code uses different signs per phase, encode it here.
        # Example (common):
        #   Phase I (min sum artificials): positive reduced cost improves
        #   Phase II (max original objective with negated row): negative improves
        tol = getattr(self, "tol", 1e-9)
        if phase == 1:
            return rc > tol
        else:
            return rc < -tol