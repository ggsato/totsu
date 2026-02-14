from dataclasses import dataclass, field
import math, time, heapq
from typing import Dict, List, Optional, Tuple, Callable

from pyomo.environ import Var, Constraint, Binary, Integers, minimize, maximize, value
from .super_simplex_solver import SuperSimplexSolver, InfeasibleProblemError, UnboundedProblemError, OptimizationError
from ..utils.model_processor import ModelProcessor
from ..utils.logger import totsu_logger

@dataclass(order=True)
class _Node:
    # Priority in heap: best bound first for minimization; reverse for maximization via storing bound appropriately
    priority: float
    id: int
    model: object = field(compare=False)
    depth: int = field(compare=False, default=0)
    # bookkeeping
    parent_id: Optional[int] = field(compare=False, default=None)
    branched_on: Optional[str] = field(compare=False, default=None)
    branch_rule: Optional[str] = field(compare=False, default=None)  # "≤" or "≥"
    branch_value: Optional[float] = field(compare=False, default=None)

@dataclass
class Families:
    """Optional structure to tell the solver which binaries form assignment 'families'.
       Example: family_keys might be tuples like (worker, day); each maps to a list of var names that are 'choose ≤ 1' or '= 1'."""
    groups: Dict[Tuple, List[str]]
    equals_one: bool = True  # set False if families are '≤ 1'

class AdvancedBranchAndBound:
    """ a 4-way 0/1 assignment (workers × customers × days × skills) explodes combinatorially. 
    The only way to “beat” the exponent is to (1) make the LP relaxations tighter, 
    (2) get good incumbents early, 
    and (3) explore as few nodes as possible via smart branching and pruning.
    """
    def __init__(
        self,
        strong_branch_k: int = 8,         # evaluate up to K most fractional candidates
        strong_branch_depth: int = 2,     # only do strong branching near the top
        node_selection: str = "best_bound",  # or "depth"
        mip_gap: Optional[float] = None,  # absolute mipgap; None = no early stop by gap
        time_limit_s: Optional[float] = None,
        node_limit: Optional[int] = None,
        families: Optional[Families] = None,        # optional: helps heuristic rounding/repair a LOT
        branch_priority: Optional[Dict[str,int]] = None,  # larger = branch earlier
        log_every: int = 200,
        simplex_max_itr=2000,          # NEW: default much higher than 100
        simplex_backoff=[2000, 5000, 12000],  # NEW: retry ladder,
    ):
        self.sb_k = strong_branch_k
        self.sb_depth = strong_branch_depth
        self.node_selection = node_selection
        self.mip_gap = mip_gap
        self.time_limit_s = time_limit_s
        self.node_limit = node_limit
        self.families = families
        self.branch_priority = branch_priority or {}
        self.log_every = log_every

        self.simplex = SuperSimplexSolver(
            pricing="devex",
            lex_ratio=True,
            phase1_cost_perturb=1e-9,
            phase1_rhs_perturb=1e-10,
        )
        # try to set, whether exposed in __init__ or as attribute
        if hasattr(self.simplex, "max_itr"):
            self.simplex.max_itr = simplex_max_itr
        self._simplex_backoff = simplex_backoff
        self.is_min = True
        self.best_sol = None
        self.best_obj = math.inf
        self.nodes = 0
        self.open = []   # heap for best-bound, or stack emulated if depth-first
        self.start_time = None
        self.next_id = 0

        self.history = []  # list of (t_sec, best_bound, incumbent)

        self.rel_gap = None  # e.g., 0.01 for 1%

        self.pseudocost = {}  # var -> (score0, score1, n0, n1)

    def solve(self, model):
        self.start_time = time.time()
        self.is_min = (ModelProcessor.get_active_objective(model).sense == minimize)
        self.best_obj = math.inf if self.is_min else -math.inf
        self.best_sol = None
        self.nodes = 0
        self.open = []
        self.next_id = 0

        # Root node
        self._push_node(model.clone(), depth=0, bound=self._infty_bound())
        self._record_progress()

        while self.open:
            if self._out_of_budget():
                break

            if self.nodes % 1 == 0:
                totsu_logger.info(f" B&B nodes explored: {self.nodes}, open nodes: {len(self.open)}, best obj: {self.best_obj:.6f}")
                self._record_progress()

            node = self._pop_node()

            try:
                sol = self._lp_solve_with_backoff(node.model)
                self._rins_lite(node.model, sol)
                try:
                    lp = float(self.simplex.get_current_objective_value())
                except Exception:
                    lp = self._eval_objective_on(node.model, sol)
            except InfeasibleProblemError:
                continue
            except UnboundedProblemError:
                # In pure 0-1 assignment models with proper bounds this is rare, but we safely discard.
                continue

            # Prune by bound
            if self._worse_than_incumbent(lp):
                continue

            # Try family-aware rounding → fix-and-optimize to get a good incumbent early
            self._incumbent_heuristic(node, sol)
            self._record_progress()

            # Check integrality
            # integral?
            frac_name = self._first_fractional(sol, node.model)
            if frac_name is None:
                obj_val = self._eval_objective_on(node.model, sol)
                self._update_incumbent_if_better(sol, obj_val)
                continue


            # Choose branching variable (priority → strong-branching-lite → most fractional)
            var_name = self._choose_branch_var(sol, node.model, node.depth)
            var = node.model.find_component(var_name)
            floor_v = math.floor(float(sol[var_name]))
            ceil_v = floor_v + 1

            # Left: var ≤ floor_v ; Right: var ≥ ceil_v   (Binary reduces to {0,1})
            left = node.model.clone()
            setattr(left, f"bb_le_{var_name}_{floor_v}", Constraint(expr=left.find_component(var_name) <= floor_v))

            right = node.model.clone()
            setattr(right, f"bb_ge_{var_name}_{ceil_v}", Constraint(expr=right.find_component(var_name) >= ceil_v))

            # Bound estimates for queue ordering (optional quick look-ahead)
            left_bound = self._estimate_child_bound(left) if self.node_selection == "best_bound" else self._infty_bound()
            right_bound = self._estimate_child_bound(right) if self.node_selection == "best_bound" else self._infty_bound()

            # Push children (order so better bound is popped sooner)
            self._push_node(left, node.depth+1, bound=left_bound, parent=node.id, branched_on=var_name, rule="≤", value=floor_v)
            self._push_node(right, node.depth+1, bound=right_bound, parent=node.id, branched_on=var_name, rule="≥", value=ceil_v)

        # Write back best incumbent (if any)
        ModelProcessor.set_variable_values(model, self.best_sol)
        final_obj = self._eval_objective_on(model, self.best_sol) if self.best_sol is not None else (math.inf if self.is_min else -math.inf)
        return self.best_sol, final_obj

    # -------------- helpers --------------

    def _get_obj_value(self) -> float:
        return self.simplex.get_current_objective_value()

    def _update_incumbent_if_better(self, sol: Dict[str,float], obj: float):
        if (self.is_min and obj < self.best_obj) or ((not self.is_min) and obj > self.best_obj):
            self.best_obj = obj
            self.best_sol = sol.copy()

    def _worse_than_incumbent(self, bound: float) -> bool:
        if self.best_sol is None:
            return False
        return (self.is_min and bound >= self.best_obj - 1e-9) or ((not self.is_min) and bound <= self.best_obj + 1e-9)

    def _infty_bound(self):
        return math.inf if self.is_min else -math.inf

    def _out_of_budget(self):
        if self.time_limit_s and (time.time() - self.start_time) >= self.time_limit_s:
            return True
        if self.node_limit and self.nodes >= self.node_limit:
            return True
        if self.best_sol is not None:
            best_bound = self._best_open_bound()
            if best_bound is not None:
                if self.mip_gap is not None and abs(best_bound - self.best_obj) <= self.mip_gap:
                    return True
                if self.rel_gap is not None and self.best_obj != 0:
                    rg = abs(best_bound - self.best_obj) / (1e-9 + abs(self.best_obj))
                    if rg <= self.rel_gap:
                        return True
        return False

    def _best_open_bound(self) -> Optional[float]:
        if not self.open:
            return None
        if self.node_selection == "best_bound":
            # heap stores priority already as a bound-like number
            return self.open[0][0]
        return None

    def _push_node(self, model, depth: int, bound: float, parent=None, branched_on=None, rule=None, value=None):
        self.nodes += 1
        nid = self.next_id; self.next_id += 1
        # heap priority: minimization uses bound directly; maximization invert sign so "largest" becomes "smallest priority"
        if self.node_selection == "best_bound":
            priority = bound if self.is_min else -bound
            heapq.heappush(self.open, (priority, nid, _Node(priority, nid, model, depth, parent, branched_on, rule, value)))
        else:
            # depth-first: emulate stack with negative depth/ID
            priority = -(depth*10_000 + nid)
            heapq.heappush(self.open, (priority, nid, _Node(priority, nid, model, depth, parent, branched_on, rule, value)))

    def _pop_node(self) -> _Node:
        _, _, node = heapq.heappop(self.open)
        return node

    def _first_fractional(self, sol, model) -> Optional[str]:
        # honor priorities when available
        candidates = []
        for v in ModelProcessor.get_variables(model):
            xv = float(sol[v.name])
            if v.domain is Binary and xv not in (0.0, 1.0):
                candidates.append((v.name, abs(xv - 0.5)))  # fractionality
            elif v.domain is Integers and abs(xv - round(xv)) > 1e-9:
                candidates.append((v.name, abs(xv - round(xv))))
        if not candidates:
            return None
        # sort by branch_priority (desc), then by fractionality (desc)
        def keyfn(item):
            name, frac = item
            return (self.branch_priority.get(name, 0), frac)
        candidates.sort(key=keyfn, reverse=True)
        return candidates[0][0]

    def _choose_branch_var(self, sol, model, depth: int) -> str:
        # strong-branching-lite near the top: evaluate up to K candidates with quick LP solves
        if depth <= self.sb_depth:
            cand = self._top_fractional_candidates(sol, model, self.sb_k)
            best_name = None
            best_score = -math.inf
            for name in cand:
                score = self._probe_score(model, name, sol)
                self._update_pseudocost(name, 0, float(self.simplex.get_current_objective_value()), score)
                if score > best_score:
                    best_name, best_score = name, score
            if best_name is not None:
                return best_name
        return self._first_fractional(sol, model)

    def _top_fractional_candidates(self, sol, model, k: int) -> List[str]:
        pool = []
        for v in ModelProcessor.get_variables(model):
            xv = float(sol[v.name])
            if v.domain is Binary and xv not in (0.0, 1.0):
                pool.append((v.name, abs(xv - 0.5)))
        pool.sort(key=lambda t: t[1], reverse=True)
        return [n for n,_ in pool[:k]]

    def _probe_score(self, model, var_name: str, sol) -> float:
        # Fix to 0 and 1, solve both quickly and use the better bound improvement as score.
        v = model.find_component(var_name)
        scores = []
        for fix_val in (0, 1):
            m = model.clone()
            mv = m.find_component(var_name)
            mv.fix(fix_val)
            try:
                self.simplex.solve(m)
                child_bound = self._get_obj_value()
                scores.append(child_bound)
            except (InfeasibleProblemError, UnboundedProblemError):
                # strong pruning if infeasible
                scores.append(-math.inf if self.is_min else math.inf)
        if self.is_min:
            return -min(scores)  # larger is better improvement
        else:
            return max(scores)

    def _estimate_child_bound(self, m) -> float:
        try:
            self.simplex.solve(m)
            return self._get_obj_value()
        except (InfeasibleProblemError, UnboundedProblemError):
            return self._infty_bound()

    # ---------- incumbent heuristic (family-aware rounding + fix-and-optimize) ----------
    def _incumbent_heuristic(self, node: _Node, sol: Dict[str,float]):
        if self.families is None:
            return
        m = node.model.clone()

        # 1) Round within each family to honor =1 / ≤1
        for key, var_names in self.families.groups.items():
            # pick the argmax fractional value
            best = max(var_names, key=lambda n: sol.get(n, 0.0))
            if self.families.equals_one:
                # fix best to 1, rest to 0
                for n in var_names:
                    v = m.find_component(n)
                    if n == best:
                        v.fix(1)
                    else:
                        v.fix(0)
            else:
                # ≤1 family: fix the winner to 1 if it's > 0.5, otherwise fix all 0
                if sol.get(best, 0.0) >= 0.5:
                    for n in var_names:
                        v = m.find_component(n)
                        v.fix(1 if n == best else 0)
                else:
                    for n in var_names:
                        m.find_component(n).fix(0)

        # 2) Re-optimize (LP) with these fixes. If feasible and better → update incumbent.
        try:
            self.simplex.solve(m)
            # Build a solution dict from m
            sol_m = {var.name: float(var.value) for var in ModelProcessor.get_variables(m)}
            obj = self._eval_objective_on(m, sol_m)              # ✅ real value
            if not self._worse_than_incumbent(obj):
                self._update_incumbent_if_better(sol_m, obj)
        except (InfeasibleProblemError, UnboundedProblemError):
            pass

    def _active_objective(self, model):
        return ModelProcessor.get_active_objective(model)

    def _eval_objective_on(self, model, sol_dict):
        # Temporarily set variable values, evaluate objective, then (optionally) restore
        # In practice, we can set and leave them; the solver will overwrite on next LP solve.
        for v in model.component_data_objects(Var, descend_into=True):
            if v.name in sol_dict:
                v.set_value(float(sol_dict[v.name]), skip_validation=True)
        obj = self._active_objective(model)
        return float(value(obj.expr))

    def _lp_solve_with_backoff(self, model):
        """
        Try solving the LP relaxation; if iteration cap hits, increase max_itr and retry.
        """
        totsu_logger.info("Solving LP relaxation with backoff strategy. backoff levels: %s", self._simplex_backoff)
        # first try with current setting
        try:
            sol = self.simplex.solve(model)
            return sol
        except OptimizationError as e:
            totsu_logger.warning(f"LP solve hit iteration cap or failed: {e}")
            pass

        # backoff retries
        original = getattr(self.simplex, "max_itr", None)
        for cap in self._simplex_backoff:
            if hasattr(self.simplex, "max_itr"):
                self.simplex.max_itr = cap
            try:
                totsu_logger.info(f"LP solve hit iteration cap; retrying with max_itr={cap}")
                sol = self.simplex.solve(model)
                return sol
            except OptimizationError as e:
                totsu_logger.warning(f"LP solve hit iteration cap or failed: {e}")
                continue

        # restore original cap if we changed it
        if hasattr(self.simplex, "max_itr") and original is not None:
            self.simplex.max_itr = original

        # give up on this node (treat as failed LP)
        raise OptimizationError("LP relaxation hit iteration cap at all backoff levels.")

    def _record_progress(self):
        t = time.time() - self.start_time
        best_bound = self._best_open_bound()
        incumbent = self.best_obj if self.best_sol is not None else (math.inf if self.is_min else -math.inf)
        self.history.append((t, best_bound, incumbent))

    def _update_pseudocost(self, var_name, fix_val, parent_bound, child_bound):
        improve = (parent_bound - child_bound) if self.is_min else (child_bound - parent_bound)
        s0, s1, n0, n1 = self.pseudocost.get(var_name, (0.0, 0.0, 0, 0))
        if fix_val == 0:
            self.pseudocost[var_name] = (s0 + improve, s1, n0 + 1, n1)
        else:
            self.pseudocost[var_name] = (s0, s1 + improve, n0, n1)

    def _pseudocost_score(self, var_name):
        s0, s1, n0, n1 = self.pseudocost.get(var_name, (0.0, 0.0, 0, 0))
        a0 = s0 / max(1, n0); a1 = s1 / max(1, n1)
        return max(a0, a1)

    def _rins_lite(self, model, sol, tol=1e-3, max_fix=200):
        m = model.clone(); fixed = 0
        for v in ModelProcessor.get_variables(m):
            xv = float(sol[v.name])
            if v.domain is Binary and (xv <= tol or xv >= 1 - tol):
                m.find_component(v.name).fix(1 if xv >= 0.5 else 0)
                fixed += 1
                if fixed >= max_fix: break
        try:
            self.simplex.solve(m)
            sol_m = {var.name: float(var.value) for var in ModelProcessor.get_variables(m)}
            obj = self._eval_objective_on(m, sol_m)
            if not self._worse_than_incumbent(obj):
                self._update_incumbent_if_better(sol_m, obj)
        except (InfeasibleProblemError, UnboundedProblemError):
            pass
