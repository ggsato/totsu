from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from pyomo.opt import SolverFactory, TerminationCondition

from totsu.utils.elastic_feasibility_tool import ElasticFeasibilityTool
from totsu.utils.solver_utils import resolve_solver_name


def _is_feasible_termination(termination_condition) -> bool:
    return termination_condition in {
        TerminationCondition.optimal,
        TerminationCondition.feasible,
        TerminationCondition.locallyOptimal,
        TerminationCondition.globallyOptimal,
    }


@dataclass
class AnalysisResult:
    solver_name: str
    is_feasible_original: bool
    is_feasible_elastic: Optional[bool]
    top_relaxations: List[Dict]

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
            print(
                f"  - {row['constraint_name']}: deviation={row['deviation']:.6g}, "
                f"penalty={row['penalty']:.6g}, cost={row['cost']:.6g}"
            )

    def to_dict(self) -> Dict:
        return {
            "solver_name": self.solver_name,
            "is_feasible_original": self.is_feasible_original,
            "is_feasible_elastic": self.is_feasible_elastic,
            "top_relaxations": list(self.top_relaxations),
        }


def analyze_infeasibility(
    model,
    solver: str = "auto",
    *,
    tee: bool = False,
    violation_only: bool = True,
    max_items: int = 10,
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
            top_relaxations=[],
        )

    objective_mode = "violation_only" if violation_only else "original_plus_violation"
    tool = ElasticFeasibilityTool(default_penalty=1.0)
    elastic_result = tool.apply(
        model,
        objective_mode=objective_mode,
        clone=True,
    )
    elastic_solve_results = solver_instance.solve(elastic_result.model, tee=tee)
    elastic_term = getattr(elastic_solve_results.solver, "termination_condition", None)
    is_feasible_elastic = _is_feasible_termination(elastic_term)

    ElasticFeasibilityTool.populate_violation_summary(elastic_result, tol=1e-9)
    return AnalysisResult(
        solver_name=solver_name,
        is_feasible_original=False,
        is_feasible_elastic=is_feasible_elastic,
        top_relaxations=elastic_result.violation_breakdown[:max_items],
    )


__all__ = ["AnalysisResult", "analyze_infeasibility"]
