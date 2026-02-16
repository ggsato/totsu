import pytest

from totsu.utils.solver_utils import NO_SOLVER_FOUND_MESSAGE, resolve_solver_name


class _FakeSolver:
    def __init__(self, available):
        self._available = available

    def available(self, exception_flag=False):
        return self._available


def test_resolve_solver_name_auto_prefers_highs_then_cbc_then_glpk():
    availability = {
        "highs": False,
        "cbc": True,
        "glpk": True,
    }

    def fake_factory(name):
        return _FakeSolver(availability.get(name, False))

    selected = resolve_solver_name("auto", solver_factory=fake_factory)
    assert selected == "cbc"


def test_resolve_solver_name_auto_raises_when_none_available():
    def fake_factory(name):
        return _FakeSolver(False)

    with pytest.raises(RuntimeError, match=NO_SOLVER_FOUND_MESSAGE):
        resolve_solver_name("auto", solver_factory=fake_factory)


def test_resolve_solver_name_non_auto_passthrough():
    assert resolve_solver_name("glpk") == "glpk"
