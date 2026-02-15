# totsu/core/elastic_feasibility_tool.py

from dataclasses import dataclass, field
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pyomo.environ import Block, Constraint, Objective, Var, NonNegativeReals, maximize, minimize, value
try:
    from pyomo.core.base.constraint import ConstraintData
except ImportError:  # pragma: no cover - compatibility with older Pyomo
    from pyomo.core.base.constraint import _ConstraintData as ConstraintData
from pyomo.core.expr.visitor import identify_variables
from pyomo.repn import generate_standard_repn

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
    sense: str
    bound: float
    generated_constraint_name: str
    original_constraint: Optional[ConstraintData] = None


@dataclass
class ElasticResult:
    model: Any
    deviations: List[ElasticDeviation] = field(default_factory=list)
    total_violation_cost: float = 0.0
    violation_breakdown: List[dict] = field(default_factory=list)
    objective_mode: Optional[str] = None
    active_objective_value: Optional[float] = None
    original_objective_value: Optional[float] = None
    violation_objective_value: Optional[float] = None
    combined_objective_value: Optional[float] = None
    _original_objective_expr: Any = field(default=None, repr=False, compare=False)
    _violation_objective_expr: Any = field(default=None, repr=False, compare=False)
    _combined_objective_expr: Any = field(default=None, repr=False, compare=False)


class ElasticFeasibilityTool:
    """
    Generic elastic-feasibility transformer for Pyomo models.

    * Works purely on constraints, bounds, and variables (no domain semantics).
    * For each selected constraint:
        - Detect <=, >=, ==, or ranged (lb <= body <= ub).
        - Add non-negative deviation variables (slack/excess).
        - Replace with an equality including deviations:
              body <= ub  →  body - e = ub
              body >= lb  →  body + s = lb
              body == rhs →  body + s - e = rhs
              lb <= body <= ub → split into GE & LE pieces.
    * Applies objective mode selected in `apply()`.
    """

    def __init__(self, default_penalty: float = 1.0, tol: float = 1e-8):
        self.default_penalty = default_penalty
        self.tol = tol
        self._var_counter = 0
        self._constraint_counter = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def apply(
        self,
        model,
        constraints: Optional[Iterable[Any]] = None,
        penalty_map: Optional[Dict[Any, float]] = None,
        deactivate_original_objective: bool = True,
        objective_mode: str = "violation_only",
        original_objective_weight: float = 1.0,
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
            Backward-compatibility flag.
            - True: maps to objective_mode="violation_only"
            - False: maps to objective_mode="keep_original" when
              objective_mode is left at default.
        objective_mode:
            One of:
              * "violation_only"
              * "original_plus_violation"
              * "keep_original"
        original_objective_weight:
            Weight for the original objective expression when
            objective_mode="original_plus_violation".
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

        # 3. Apply objective mode (with backward compatibility mapping)
        resolved_objective_mode = self._resolve_objective_mode(
            objective_mode=objective_mode,
            deactivate_original_objective=deactivate_original_objective,
        )
        objective_terms = self._apply_objective_mode(
            working_model,
            deviations=deviations,
            objective_mode=resolved_objective_mode,
            original_objective_weight=original_objective_weight,
        )

        result = ElasticResult(
            model=working_model,
            deviations=deviations,
            objective_mode=resolved_objective_mode,
            _original_objective_expr=objective_terms.get("original_expr"),
            _violation_objective_expr=objective_terms.get("violation_expr"),
            _combined_objective_expr=objective_terms.get("combined_expr"),
        )
        # Initial values before solve (all deviation vars are typically unset).
        self.populate_violation_summary(result, tol=self.tol)
        return result

    @staticmethod
    def populate_violation_summary(
        result: ElasticResult,
        tol: float = 0.0,
        include_variable_contributions: bool = False,
        max_contrib_vars: int = 10,
    ) -> ElasticResult:
        """
        Populate `result.total_violation_cost` and `result.violation_breakdown`
        from current deviation variable values.

        Call this after solve to get the final structural diagnosis numbers.
        """
        breakdown: List[dict] = []

        for dev in result.deviations:
            dev_val = dev.var.value
            if dev_val is None or dev_val <= tol:
                continue

            cost = dev.penalty * dev_val
            row = {
                "component_name": dev.component_name,
                "deviation": dev_val,
                "penalty": dev.penalty,
                "cost": cost,
                "constraint_name": dev.original_name,
                "index": tuple(dev.index) if dev.index is not None else (),
                "sense": dev.sense,
                "bound": float(dev.bound),
                "kind": dev.kind,
                "deviation_var": dev.var.name,
                "generated_constraint_name": dev.generated_constraint_name,
            }
            if include_variable_contributions:
                row["violation_amount"] = dev_val
                row["body_value"] = ElasticFeasibilityTool._safe_value_of_constraint_body(
                    dev.original_constraint
                )
                row["variable_contributions"] = ElasticFeasibilityTool._collect_variable_contributions(
                    dev.original_constraint,
                    max_contrib_vars=max_contrib_vars,
                )

            breakdown.append(row)

        # Stable and deterministic ordering for reporting:
        # highest cost first, then name, deviation, penalty.
        breakdown.sort(
            key=lambda row: (
                -row["cost"],
                str(row["component_name"]),
                -row["deviation"],
                -row["penalty"],
                str(row["constraint_name"]),
            )
        )

        result.violation_breakdown = breakdown
        result.total_violation_cost = sum(row["cost"] for row in breakdown)
        ElasticFeasibilityTool.populate_objective_summary(result)
        return result

    @staticmethod
    def _safe_eval_float(expr) -> Optional[float]:
        if expr is None:
            return None
        try:
            for var in identify_variables(expr, include_fixed=True):
                if var.value is None:
                    return None
        except Exception:
            # Fall back to safe evaluation path below.
            pass
        try:
            evaluated = value(expr, exception=False)
            if evaluated is None:
                return None
            return float(evaluated)
        except Exception:
            return None

    @staticmethod
    def populate_objective_summary(result: ElasticResult) -> ElasticResult:
        """
        Populate objective value fields from the current solved model state.

        This is safe to call before solve; fields remain None when values
        cannot be evaluated yet.
        """
        active_obj = next(
            result.model.component_data_objects(Objective, active=True, descend_into=True),
            None,
        )
        result.active_objective_value = (
            ElasticFeasibilityTool._safe_eval_float(active_obj.expr) if active_obj is not None else None
        )
        result.original_objective_value = ElasticFeasibilityTool._safe_eval_float(
            result._original_objective_expr
        )
        result.violation_objective_value = ElasticFeasibilityTool._safe_eval_float(
            result._violation_objective_expr
        )
        result.combined_objective_value = ElasticFeasibilityTool._safe_eval_float(
            result._combined_objective_expr
        )
        return result

    @staticmethod
    def _safe_value_of_constraint_body(con: Optional[ConstraintData]) -> Optional[float]:
        if con is None:
            return None
        try:
            return float(value(con.body))
        except Exception:
            return None

    @staticmethod
    def _collect_variable_contributions(
        con: Optional[ConstraintData],
        max_contrib_vars: int,
    ) -> List[Dict[str, Any]]:
        if con is None:
            return []

        try:
            repn = generate_standard_repn(con.body, compute_values=False)
        except Exception:
            repn = None

        contributions: List[Dict[str, Any]] = []
        if repn is not None and repn.is_linear():
            for var, coef in zip(repn.linear_vars, repn.linear_coefs):
                var_val = var.value if var.value is not None else 0.0
                coef_val = float(coef)
                contribution = coef_val * float(var_val)
                contributions.append(
                    {
                        "var": var.name,
                        "value": float(var_val),
                        "coef": coef_val,
                        "contribution": contribution,
                        "abs_contribution": abs(contribution),
                    }
                )
        else:
            seen = set()
            for var in identify_variables(con.body, include_fixed=True):
                if id(var) in seen:
                    continue
                seen.add(id(var))
                var_val = var.value if var.value is not None else 0.0
                contributions.append(
                    {
                        "var": var.name,
                        "value": float(var_val),
                        "coef": None,
                        "contribution": None,
                        "abs_contribution": None,
                    }
                )
            contributions.sort(key=lambda row: abs(row["value"]), reverse=True)

        if repn is not None and repn.is_linear():
            contributions.sort(key=lambda row: row["abs_contribution"], reverse=True)
        return contributions[:max(0, int(max_contrib_vars))]

    # ------------------------------------------------------------------
    # Core transformations
    # ------------------------------------------------------------------
    def _collect_all_constraints(self, model) -> List[ConstraintData]:
        """Return all active ConstraintData objects in the model."""
        return list(model.component_data_objects(Constraint, active=True, descend_into=True))

    def _normalize_constraint_selection(
        self,
        model,
        selection: Iterable[Any],
    ) -> List[ConstraintData]:
        """Normalize various user selectors to a flat list of ConstraintData."""
        out: List[ConstraintData] = []

        def add_component(comp):
            if isinstance(comp, ConstraintData):
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
    def _component_and_index(self, con: ConstraintData) -> Tuple[str, Tuple[Any, ...]]:
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

    def _get_penalty(self, con: ConstraintData, penalty_map: Dict[Any, float]) -> float:
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
        con: ConstraintData,
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

        name_base = self._sanitize_component_name(con.name)

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
                    sense="EQ",
                    bound=rhs,
                    generated_constraint_name="",
                    original_constraint=con,
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
                    sense="EQ",
                    bound=rhs,
                    generated_constraint_name="",
                    original_constraint=con,
                )
            )

            generated_name = self._new_elastic_constraint(
                model,
                name_base,
                "eq",
                body + s - e == rhs,
            )
            deviations[-1].generated_constraint_name = generated_name
            deviations[-2].generated_constraint_name = generated_name

            totsu_logger.debug(
                f"Elasticized equality '{original_name}' with slack {s.name} and excess {e.name}."
            )
            return deviations

        # Ranged case: lb < body < ub
        if lb is not None and ub is not None and lb < ub - self.tol:
            con.deactivate()
            totsu_logger.debug(f"Splitting ranged constraint '{original_name}' into GE/LE.")

            # GE part: body >= lb → body + s_ge = lb
            s_ge = self._new_deviation_var(model, f"{name_base}_ge_posdev")
            deviations.append(
                ElasticDeviation(
                    var=s_ge,
                    penalty=penalty,
                    kind="slack_ge",
                    component_name=component_name,
                    index=index,
                    original_name=original_name,
                    sense="RANGE_GE",
                    bound=lb,
                    generated_constraint_name="",
                    original_constraint=con,
                )
            )
            generated_ge = self._new_elastic_constraint(
                model,
                name_base,
                "range_ge",
                body + s_ge == lb,
            )
            deviations[-1].generated_constraint_name = generated_ge

            # LE part: body <= ub → body - e_le = ub
            e_le = self._new_deviation_var(model, f"{name_base}_le_negdev")
            deviations.append(
                ElasticDeviation(
                    var=e_le,
                    penalty=penalty,
                    kind="excess_le",
                    component_name=component_name,
                    index=index,
                    original_name=original_name,
                    sense="RANGE_LE",
                    bound=ub,
                    generated_constraint_name="",
                    original_constraint=con,
                )
            )
            generated_le = self._new_elastic_constraint(
                model,
                name_base,
                "range_le",
                body - e_le == ub,
            )
            deviations[-1].generated_constraint_name = generated_le

            return deviations

        # Pure GE: body >= lb
        if lb is not None and ub is None:
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
                    sense="GE",
                    bound=lb,
                    generated_constraint_name="",
                    original_constraint=con,
                )
            )

            generated_name = self._new_elastic_constraint(
                model,
                name_base,
                "ge",
                body + s == lb,
            )
            deviations[-1].generated_constraint_name = generated_name

            totsu_logger.debug(
                f"Elasticized GE constraint '{original_name}' with slack {s.name}."
            )
            return deviations

        # Pure LE: body <= ub
        if ub is not None and lb is None:
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
                    sense="LE",
                    bound=ub,
                    generated_constraint_name="",
                    original_constraint=con,
                )
            )

            generated_name = self._new_elastic_constraint(
                model,
                name_base,
                "le",
                body - e == ub,
            )
            deviations[-1].generated_constraint_name = generated_name

            totsu_logger.debug(
                f"Elasticized LE constraint '{original_name}' with excess {e.name}."
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
        safe_base_name = re.sub(r"slack", "dev", base_name, flags=re.IGNORECASE)
        safe_base_name = re.sub(r"artificial", "dev", safe_base_name, flags=re.IGNORECASE)
        name = f"elastic_dev_{safe_base_name}_{self._var_counter}"
        self._var_counter += 1
        elastic_block = self._get_or_create_elastic_block(model)
        v = Var(within=NonNegativeReals)
        setattr(elastic_block, name, v)
        return v

    def _new_elastic_constraint(self, model, base_name: str, suffix: str, expr) -> str:
        """Create one elastic constraint under model.elastic and return its full name."""
        elastic_block = self._get_or_create_elastic_block(model)
        name = f"elastic_con_{base_name}_{suffix}_{self._constraint_counter}"
        self._constraint_counter += 1
        setattr(elastic_block, name, Constraint(expr=expr))
        return getattr(elastic_block, name).name

    def _get_or_create_elastic_block(self, model):
        """Get model.elastic block, creating it when needed."""
        if hasattr(model, "elastic"):
            elastic_block = getattr(model, "elastic")
            if not isinstance(elastic_block, Block):
                raise TypeError("model.elastic exists but is not a Pyomo Block.")
            return elastic_block

        model.elastic = Block()
        return model.elastic

    def _sanitize_component_name(self, raw_name: str) -> str:
        """Create a component-safe and readable base name from a Pyomo component name."""
        sanitized = re.sub(r"[^0-9A-Za-z_]+", "_", raw_name)
        sanitized = re.sub(r"_+", "_", sanitized).strip("_")
        return sanitized or "constraint"

    def _resolve_objective_mode(
        self,
        objective_mode: str,
        deactivate_original_objective: bool,
    ) -> str:
        allowed = {"violation_only", "original_plus_violation", "keep_original"}
        if objective_mode not in allowed:
            raise ValueError(
                f"Unsupported objective_mode '{objective_mode}'. "
                f"Expected one of {sorted(allowed)}."
            )

        if objective_mode == "violation_only" and not deactivate_original_objective:
            totsu_logger.warning(
                "deactivate_original_objective=False is deprecated. "
                "Mapping to objective_mode='keep_original'."
            )
            return "keep_original"

        return objective_mode

    def _apply_objective_mode(
        self,
        model,
        deviations: List[ElasticDeviation],
        objective_mode: str,
        original_objective_weight: float,
    ) -> Dict[str, Any]:
        """Apply objective mode after constraints have been elasticized."""
        try:
            active_obj = ModelProcessor.get_active_objective(model)
        except ValueError:
            active_obj = None  # no active objective; that's fine

        original_expr = active_obj.expr if active_obj is not None else None

        if objective_mode == "keep_original":
            return {
                "original_expr": original_expr,
                "violation_expr": None,
                "combined_expr": original_expr,
            }

        if objective_mode not in ("violation_only", "original_plus_violation"):
            raise ValueError(f"Unknown objective mode: {objective_mode}")

        elastic_block = self._get_or_create_elastic_block(model)

        if objective_mode == "violation_only" and not deviations:
            totsu_logger.warning(
                "ElasticFeasibilityTool: no deviations created; "
                "keeping original objective."
            )
            if active_obj is not None:
                active_obj.activate()
            return {
                "original_expr": original_expr,
                "violation_expr": None,
                "combined_expr": original_expr,
            }

        violation_expr = sum(dev.penalty * dev.var for dev in deviations)

        if active_obj is not None:
            active_obj.deactivate()

        if objective_mode == "violation_only":
            objective_expr = violation_expr
        else:
            if active_obj is None:
                objective_expr = violation_expr
            else:
                self._validate_objective_is_finite(
                    obj_expr=active_obj.expr,
                    objective_name=getattr(active_obj, "name", "<unknown>"),
                    objective_mode=objective_mode,
                )
            if active_obj is None:
                objective_expr = violation_expr
            elif active_obj.sense == maximize:
                objective_expr = -original_objective_weight * active_obj.expr + violation_expr
            else:
                objective_expr = original_objective_weight * active_obj.expr + violation_expr

        combined_expr = (
            (original_expr + violation_expr)
            if (objective_mode == "original_plus_violation" and original_expr is not None)
            else violation_expr
        )

        if hasattr(elastic_block, "elastic_obj"):
            elastic_block.del_component(elastic_block.elastic_obj)

        elastic_block.elastic_obj = Objective(expr=objective_expr, sense=minimize)
        totsu_logger.debug(f"Applied elastic objective mode '{objective_mode}'.")
        return {
            "original_expr": original_expr,
            "violation_expr": violation_expr,
            "combined_expr": combined_expr,
        }

    @staticmethod
    def _validate_objective_is_finite(
        obj_expr,
        objective_name: Optional[str] = None,
        objective_mode: Optional[str] = None,
    ) -> None:
        """
        Validate that objective constant/coefficients are finite.
        Raises ValueError when inf/-inf/nan is detected.
        """
        repn = None
        repn_error = None
        for compute_values in (True, False):
            try:
                repn = generate_standard_repn(obj_expr, compute_values=compute_values)
                repn_error = None
                break
            except Exception as err:
                repn_error = err
                repn = None

        if repn is None:
            raise ValueError(
                "ElasticFeasibilityTool cannot analyze the original objective for "
                f"objective_mode='{objective_mode or 'original_plus_violation'}'. "
                f"Objective '{objective_name or '<unknown>'}' could not be represented: {repn_error}"
            )

        problems: List[str] = []

        def _non_finite_label(val) -> Optional[str]:
            try:
                num = float(val)
            except Exception:
                try:
                    num = float(value(val))
                except Exception:
                    return None
            if math.isnan(num):
                return "nan"
            if math.isinf(num):
                return "inf" if num > 0 else "-inf"
            return None

        const_label = _non_finite_label(getattr(repn, "constant", None))
        if const_label is not None:
            problems.append(f"constant={const_label}")

        for var, coef in zip(getattr(repn, "linear_vars", ()), getattr(repn, "linear_coefs", ())):
            label = _non_finite_label(coef)
            if label is not None:
                problems.append(f"linear coef for '{var.name}'={label}")

        q_vars = getattr(repn, "quadratic_vars", ()) or ()
        q_coefs = getattr(repn, "quadratic_coefs", ()) or ()
        for pair, coef in zip(q_vars, q_coefs):
            label = _non_finite_label(coef)
            if label is not None:
                try:
                    v0, v1 = pair
                    term = f"('{v0.name}', '{v1.name}')"
                except Exception:
                    term = str(pair)
                problems.append(f"quadratic coef for {term}={label}")

        if problems:
            detail = "; ".join(problems[:5])
            raise ValueError(
                "ElasticFeasibilityTool detected non-finite values in the original objective "
                f"while objective_mode='{objective_mode or 'original_plus_violation'}' "
                f"(objective '{objective_name or '<unknown>'}'): {detail}. "
                "Do not use inf/-inf/nan in objectives. For forbidden arcs, remove the variable/domain, "
                "fix x=0, or use a finite Big-M. In the transportation example, None costs converted to inf "
                "are a common cause."
            )
