from __future__ import annotations

from typing import Callable, Iterable, Optional

from pyomo.opt import SolverFactory

AUTO_SOLVER_CANDIDATES = ("highs", "cbc", "glpk")
NO_SOLVER_FOUND_MESSAGE = (
    "No LP/MILP solver found by Pyomo. Install one of: HiGHS / CBC / GLPK."
)


def _is_solver_available(solver) -> bool:
    if solver is None:
        return False
    try:
        return bool(solver.available(exception_flag=False))
    except TypeError:
        return bool(solver.available(False))
    except Exception:
        return False


def resolve_solver_name(
    solver_name: str = "auto",
    *,
    candidates: Iterable[str] = AUTO_SOLVER_CANDIDATES,
    solver_factory: Callable[[str], Optional[object]] = SolverFactory,
) -> str:
    """Resolve a solver name, supporting automatic fallback discovery."""
    if solver_name != "auto":
        return solver_name

    for candidate in candidates:
        solver = solver_factory(candidate)
        if _is_solver_available(solver):
            return candidate

    raise RuntimeError(NO_SOLVER_FOUND_MESSAGE)
