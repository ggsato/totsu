from __future__ import annotations

import io
import logging
from contextlib import contextmanager, redirect_stderr, redirect_stdout
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


@contextmanager
def _silent_solver_probe():
    """
    Suppress noisy logging / stderr output during solver discovery.

    Pyomo may emit warnings when a plugin is missing; auto-discovery should
    quietly skip unavailable candidates.
    """
    previous_disable_level = logging.root.manager.disable
    sink = io.StringIO()
    try:
        logging.disable(logging.CRITICAL)
        with redirect_stdout(sink), redirect_stderr(sink):
            yield
    finally:
        logging.disable(previous_disable_level)


def select_solver_auto(
    candidates: Iterable[str] = AUTO_SOLVER_CANDIDATES,
    *,
    solver_factory: Callable[[str], Optional[object]] = SolverFactory,
) -> Optional[str]:
    """
    Return first available solver from `candidates`, or None when none are usable.

    Candidate probing is silent and resilient:
    - Any exception from `solver_factory(...)` is ignored.
    - Any exception from `available(...)` is ignored.
    """
    for candidate in candidates:
        with _silent_solver_probe():
            try:
                solver = solver_factory(candidate)
            except Exception:
                continue
            if _is_solver_available(solver):
                return candidate
    return None


def resolve_solver_name(
    solver_name: str = "auto",
    *,
    candidates: Iterable[str] = AUTO_SOLVER_CANDIDATES,
    solver_factory: Callable[[str], Optional[object]] = SolverFactory,
) -> str:
    """Resolve a solver name, supporting automatic fallback discovery."""
    if solver_name != "auto":
        return solver_name

    selected = select_solver_auto(candidates=candidates, solver_factory=solver_factory)
    if selected is not None:
        return selected

    raise RuntimeError(NO_SOLVER_FOUND_MESSAGE)
