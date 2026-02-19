from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from pyomo.environ import SolverFactory, TerminationCondition

from totsu.utils.elastic_feasibility_tool import ElasticFeasibilityTool
from totsu.utils.solver_utils import resolve_solver_name


def _is_feasible_termination(termination_condition) -> bool:
    return termination_condition in {
        TerminationCondition.optimal,
        TerminationCondition.feasible,
        TerminationCondition.locallyOptimal,
        TerminationCondition.globallyOptimal,
    }


def _format_direction(row: Dict) -> str:
    violation = float(row.get("violation", 0.0))
    sense = str(row.get("sense", "")).upper()
    if sense in {"EQ"}:
        return f"relax by ±{violation:.6g}"
    if sense in {"GE", "RANGE_GE"}:
        return f"relax lower bound by -{violation:.6g}"
    return f"relax upper bound by +{violation:.6g}"


def _decorate_top_relaxations(
    elastic_result,
    max_items: int,
    pretty_name: Optional[Callable] = None,
) -> List[Dict]:
    deviations_by_var = {dev.var.name: dev for dev in elastic_result.deviations}
    rows: List[Dict] = []
    for raw_row in elastic_result.violation_breakdown[:max_items]:
        row = dict(raw_row)
        row["direction"] = _format_direction(row)

        if pretty_name is not None:
            dev = deviations_by_var.get(row.get("violation_var", ""))
            con = getattr(dev, "original_constraint", None) if dev is not None else None
            if con is not None:
                try:
                    pretty = pretty_name(con)
                except Exception:
                    pretty = None
                if pretty:
                    row["pretty_name"] = str(pretty)
        rows.append(row)
    return rows


@dataclass
class AnalysisResult:
    solver_name: str
    is_feasible_original: bool
    is_feasible_elastic: Optional[bool]
    original_results: object
    elastic_solve_results: Optional[object]
    elastic_model: Optional[object]
    top_relaxations: List[Dict]
    margin_summary: List[Dict]

    def print_summary(self) -> None:
        print("=== Totsu Infeasibility Analysis ===")
        print(f"Solver: {self.solver_name}")
        print(f"Original model feasible: {self.is_feasible_original}")
        if self.is_feasible_elastic is not None:
            print(f"Elastic model feasible: {self.is_feasible_elastic}")
        if not self.top_relaxations:
            print("Top relaxations: none")
            return
        print("Top relaxations:")
        for row in self.top_relaxations:
            raw_name = row.get("constraint_name", "<unknown>")
            pretty = row.get("pretty_name")
            name_out = f"{raw_name} ({pretty})" if pretty else raw_name
            print(
                f"  - {name_out} [index={row.get('index', ())}]: "
                f"violation={row['violation']:.6g}, cost={row['cost']:.6g}, "
                f"{row.get('direction', _format_direction(row))}"
            )
        if self.margin_summary:
            print("Top tight constraints (by margin):")
            for row in self.margin_summary:
                margin = row.get("margin")
                margin_txt = "None" if margin is None else f"{margin:.6g}"
                print(
                    f"  - {row.get('constraint_name', '<unknown>')} "
                    f"[index={row.get('index', ())}, sense={row.get('sense', '?')}]: "
                    f"margin={margin_txt}, is_tight={row.get('is_tight', False)}"
                )

    def to_dict(self) -> Dict:
        return {
            "solver_name": self.solver_name,
            "is_feasible_original": self.is_feasible_original,
            "is_feasible_elastic": self.is_feasible_elastic,
            "original_results": self.original_results,
            "elastic_solve_results": self.elastic_solve_results,
            "elastic_model": self.elastic_model,
            "top_relaxations": list(self.top_relaxations),
            "margin_summary": list(self.margin_summary),
        }


def analyze_infeasibility(
    model,
    solver: str = "auto",
    *,
    tee: bool = False,
    violation_only: bool = True,
    default_penalty: float = 1.0,
    max_items: int = 10,
    pretty_name: Optional[Callable] = None,
    include_margin: bool = False,
    margin_scope: str = "non_elastic",
    margin_max_items: int = 10,
) -> AnalysisResult:
    solver_name = resolve_solver_name(solver, solver_factory=SolverFactory)
    solver_instance = SolverFactory(solver_name)
    if solver_instance is None or not solver_instance.available(exception_flag=False):
        raise RuntimeError(f"Solver is not available via Pyomo: {solver_name!r}")

    original_results = solver_instance.solve(model, tee=tee)
    original_term = getattr(original_results.solver, "termination_condition", None)
    is_feasible_original = _is_feasible_termination(original_term)

    if is_feasible_original:
        return AnalysisResult(
            solver_name=solver_name,
            is_feasible_original=True,
            is_feasible_elastic=None,
            original_results=original_results,
            elastic_solve_results=None,
            elastic_model=None,
            top_relaxations=[],
            margin_summary=[],
        )

    objective_mode = "violation_only" if violation_only else "original_plus_violation"
    tool = ElasticFeasibilityTool(default_penalty=float(default_penalty))
    elastic_result = tool.apply(
        model,
        objective_mode=objective_mode,
        clone=True,
    )
    elastic_solve_results = solver_instance.solve(elastic_result.model, tee=tee)
    elastic_term = getattr(elastic_solve_results.solver, "termination_condition", None)
    is_feasible_elastic = _is_feasible_termination(elastic_term)

    ElasticFeasibilityTool.populate_violation_summary(elastic_result, tol=1e-9)
    margin_summary: List[Dict] = []
    if include_margin:
        ElasticFeasibilityTool.populate_margin_summary(
            elastic_result,
            model=elastic_result.model,
            tol=1e-9,
            scope=margin_scope,
            max_items=margin_max_items,
        )
        margin_summary = list(elastic_result.margin_summary)
    return AnalysisResult(
        solver_name=solver_name,
        is_feasible_original=False,
        is_feasible_elastic=is_feasible_elastic,
        original_results=original_results,
        elastic_solve_results=elastic_solve_results,
        elastic_model=elastic_result.model,
        top_relaxations=_decorate_top_relaxations(
            elastic_result,
            max_items=max_items,
            pretty_name=pretty_name,
        ),
        margin_summary=margin_summary,
    )


__all__ = ["AnalysisResult", "analyze_infeasibility"]
