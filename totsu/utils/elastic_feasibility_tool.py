# totsu/core/elastic_feasibility_tool.py

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pyomo.environ import Constraint, Objective, Var, NonNegativeReals, minimize, value
from pyomo.core.base.constraint import _ConstraintData

from ..utils.model_processor import ModelProcessor
from ..utils.logger import totsu_logger


@dataclass
class ElasticDeviation:
    """
    Records one deviation variable created by the elastic tool.

    component_name : name of the parent Constraint component
                     e.g. "client_demand", "worker_daily", "window"
    index          : index tuple of that constraint, e.g. ("c1",) or ("w2", 3)
    original_name  : full Pyomo name, e.g. "client_demand[ c1,3 ]"
    kind           : "slack", "excess", "slack_le", "excess_ge", etc.
    """
    var: Var
    penalty: float
    kind: str
    component_name: str
    index: Tuple[Any, ...]
    original_name: str


@dataclass
class ElasticResult:
    model: Any
    deviations: List[ElasticDeviation] = field(default_factory=list)


class ElasticFeasibilityTool:
    """
    Generic elastic-feasibility transformer for Pyomo models.

    * Works purely on constraints, bounds, and variables (no domain semantics).
    * For each selected constraint:
        - Detect <=, >=, ==, or ranged (lb <= body <= ub).
        - Add non-negative deviation variables (slack/excess).
        - Replace with an equality including deviations:
              body <= ub  →  body + s = ub
              body >= lb  →  body - e = lb
              body == rhs →  body + s - e = rhs
              lb <= body <= ub → split into GE & LE pieces.
    * Optionally replaces the objective by "minimize total violation".
    """

    def __init__(self, default_penalty: float = 1.0, tol: float = 1e-8):
        self.default_penalty = default_penalty
        self.tol = tol
        self._counter = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def apply(
        self,
        model,
        constraints: Optional[Iterable[Any]] = None,
        penalty_map: Optional[Dict[Any, float]] = None,
        deactivate_original_objective: bool = True,
        clone: bool = True,
    ) -> ElasticResult:
        """
        Transform `model` into an elastic-feasibility model.

        Parameters
        ----------
        model:
            Pyomo ConcreteModel or model instance.
        constraints:
            Which constraints to elasticize. Options:
              * None: all active ConstraintData in the model.
              * Iterable of:
                  - ConstraintData
                  - Constraint components (including ConstraintList)
                  - Strings (names of Constraint components on the model)
        penalty_map:
            Optional mapping to override the default penalty.
            Keys may be:
              * ConstraintData objects
              * Parent Constraint components
              * Component names (strings)
        deactivate_original_objective:
            If True, deactivate the original active objective(s) and
            create a new objective that minimizes total violation.
        clone:
            If True, operate on a cloned model (leaving `model` untouched).

        Returns
        -------
        ElasticResult
            Contains the transformed model and a list of deviations.
        """
        working_model = model.clone() if clone else model
        penalty_map = penalty_map or {}

        # 1. Collect constraints to elasticize
        if constraints is None:
            constraint_datas = self._collect_all_constraints(working_model)
        else:
            constraint_datas = self._normalize_constraint_selection(working_model, constraints)

        deviations: List[ElasticDeviation] = []

        # 2. Elasticize each selected constraint
        for con in constraint_datas:
            new_devs = self._elasticize_constraint(working_model, con, penalty_map)
            deviations.extend(new_devs)

        # 3. Replace objective with "minimize total violation", if requested
        if deactivate_original_objective:
            self._replace_objective_with_violation_min(working_model, deviations)

        return ElasticResult(model=working_model, deviations=deviations)

    # ------------------------------------------------------------------
    # Core transformations
    # ------------------------------------------------------------------
    def _collect_all_constraints(self, model) -> List[_ConstraintData]:
        """Return all active ConstraintData objects in the model."""
        return list(model.component_data_objects(Constraint, active=True, descend_into=True))

    def _normalize_constraint_selection(
        self,
        model,
        selection: Iterable[Any],
    ) -> List[_ConstraintData]:
        """Normalize various user selectors to a flat list of ConstraintData."""
        out: List[_ConstraintData] = []

        def add_component(comp):
            if isinstance(comp, _ConstraintData):
                out.append(comp)
            elif isinstance(comp, Constraint):
                out.extend(list(comp.values()))
            else:
                raise TypeError(f"Unsupported constraint selector type: {type(comp)}")

        for item in selection:
            if isinstance(item, str):
                comp = getattr(model, item)
                add_component(comp)
            else:
                add_component(item)

        return out

    # Small helper: get component name & index tuple for a ConstraintData
    def _component_and_index(self, con: _ConstraintData) -> Tuple[str, Tuple[Any, ...]]:
        parent = con.parent_component()
        component_name = parent.name

        # Normalize index into a tuple, even for scalar constraints
        try:
            idx = con.index()
        except Exception:
            idx = None

        if idx is None:
            index = ()
        elif isinstance(idx, tuple):
            index = idx
        else:
            index = (idx,)

        return component_name, index

    def _get_penalty(self, con: _ConstraintData, penalty_map: Dict[Any, float]) -> float:
        """Find penalty from penalty_map (by ConstraintData, parent, or name), or use default."""
        if con in penalty_map:
            return penalty_map[con]

        parent = con.parent_component()
        if parent in penalty_map:
            return penalty_map[parent]

        if isinstance(parent.name, str) and parent.name in penalty_map:
            return penalty_map[parent.name]

        return self.default_penalty

    def _elasticize_constraint(
        self,
        model,
        con: _ConstraintData,
        penalty_map: Dict[Any, float],
    ) -> List[ElasticDeviation]:
        """
        Convert a single ConstraintData `con` into an elastic form:

          * Deactivate the original constraint.
          * Add deviation variables (slack / excess).
          * Add one or two new equality constraints that include the deviations.

        Handles:
          - body <= ub
          - body >= lb
          - body == rhs
          - lb <= body <= ub (split into GE + LE parts)
        """
        lb = value(con.lower) if con.has_lb() else None
        ub = value(con.upper) if con.has_ub() else None
        body = con.body

        name_base = con.name.replace(":", "_").replace("[", "_").replace("]", "_")

        component_name, index = self._component_and_index(con)
        original_name = con.name
        penalty = self._get_penalty(con, penalty_map)

        deviations: List[ElasticDeviation] = []

        # Equality case: lb == ub
        if lb is not None and ub is not None and abs(lb - ub) <= self.tol:
            rhs = lb
            con.deactivate()

            s = self._new_deviation_var(model, f"{name_base}_posdev")
            e = self._new_deviation_var(model, f"{name_base}_negdev")

            deviations.append(
                ElasticDeviation(
                    var=s,
                    penalty=penalty,
                    kind="slack",
                    component_name=component_name,
                    index=index,
                    original_name=original_name,
                )
            )
            deviations.append(
                ElasticDeviation(
                    var=e,
                    penalty=penalty,
                    kind="excess",
                    component_name=component_name,
                    index=index,
                    original_name=original_name,
                )
            )

            new_con = Constraint(expr=body + s - e == rhs)
            setattr(model, f"{name_base}_elastic_eq", new_con)

            totsu_logger.debug(
                f"Elasticized equality '{original_name}' with slack {s.name} and excess {e.name}."
            )
            return deviations

        # Ranged case: lb < body < ub
        if lb is not None and ub is not None and lb < ub - self.tol:
            con.deactivate()
            totsu_logger.debug(f"Splitting ranged constraint '{original_name}' into GE/LE.")

            # GE part: body >= lb → body - e_ge = lb
            e_ge = self._new_deviation_var(model, f"{name_base}_ge_negdev")
            deviations.append(
                ElasticDeviation(
                    var=e_ge,
                    penalty=penalty,
                    kind="excess_ge",
                    component_name=component_name,
                    index=index,
                    original_name=original_name,
                )
            )
            con_ge = Constraint(expr=body - e_ge == lb)
            setattr(model, f"{name_base}_elastic_ge", con_ge)

            # LE part: body <= ub → body + s_le = ub
            s_le = self._new_deviation_var(model, f"{name_base}_le_posdev")
            deviations.append(
                ElasticDeviation(
                    var=s_le,
                    penalty=penalty,
                    kind="slack_le",
                    component_name=component_name,
                    index=index,
                    original_name=original_name,
                )
            )
            con_le = Constraint(expr=body + s_le == ub)
            setattr(model, f"{name_base}_elastic_le", con_le)

            return deviations

        # Pure GE: body >= lb
        if lb is not None and ub is None:
            con.deactivate()
            e = self._new_deviation_var(model, f"{name_base}_negdev")
            deviations.append(
                ElasticDeviation(
                    var=e,
                    penalty=penalty,
                    kind="excess",
                    component_name=component_name,
                    index=index,
                    original_name=original_name,
                )
            )

            new_con = Constraint(expr=body - e == lb)
            setattr(model, f"{name_base}_elastic_ge", new_con)

            totsu_logger.debug(
                f"Elasticized GE constraint '{original_name}' with excess {e.name}."
            )
            return deviations

        # Pure LE: body <= ub
        if ub is not None and lb is None:
            con.deactivate()
            s = self._new_deviation_var(model, f"{name_base}_posdev")
            deviations.append(
                ElasticDeviation(
                    var=s,
                    penalty=penalty,
                    kind="slack",
                    component_name=component_name,
                    index=index,
                    original_name=original_name,
                )
            )

            new_con = Constraint(expr=body + s == ub)
            setattr(model, f"{name_base}_elastic_le", new_con)

            totsu_logger.debug(
                f"Elasticized LE constraint '{original_name}' with slack {s.name}."
            )
            return deviations

        # No bounds: nothing to do (e.g., logical or malformed constraints)
        totsu_logger.debug(f"Skipped constraint '{original_name}' – no bounds detected.")
        return deviations

    def _new_deviation_var(self, model, base_name: str) -> Var:
        """
        Create a new non-negative deviation variable with a unique name.

        IMPORTANT: names are prefixed with 'elastic_dev_' and must NOT contain
        substrings like 'slack' or 'artificial', otherwise Tableau.identify_basis_variables
        will mistakenly treat them as initial basic variables.
        """
        name = f"elastic_dev_{base_name}_{self._counter}"
        self._counter += 1
        v = Var(within=NonNegativeReals)
        setattr(model, name, v)
        return v

    def _replace_objective_with_violation_min(
        self,
        model,
        deviations: List[ElasticDeviation],
    ) -> None:
        """Deactivate existing objective(s) and add 'minimize total deviation' objective."""
        try:
            active_obj = ModelProcessor.get_active_objective(model)
            active_obj.deactivate()
        except ValueError:
            active_obj = None  # no active objective; that's fine

        if not deviations:
            totsu_logger.warning(
                "ElasticFeasibilityTool: no deviations created; "
                "keeping original objective."
            )
            if active_obj is not None:
                active_obj.activate()
            return

        total_violation = sum(dev.penalty * dev.var for dev in deviations)

        if hasattr(model, "elastic_obj"):
            delattr(model, "elastic_obj")

        model.elastic_obj = Objective(expr=total_violation, sense=minimize)
        totsu_logger.debug("Replaced objective with 'minimize total elastic violation'.")
