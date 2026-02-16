import pytest

from totsu.utils.solver_utils import (
    NO_SOLVER_FOUND_MESSAGE,
    resolve_solver_name,
    select_solver_auto,
)


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


def test_select_solver_auto_skips_factory_exceptions():
    calls = []

    def fake_factory(name):
        calls.append(name)
        if name == "highs":
            raise RuntimeError("plugin missing")
        if name == "cbc":
            return _FakeSolver(True)
        return _FakeSolver(False)

    selected = select_solver_auto(candidates=("highs", "cbc", "glpk"), solver_factory=fake_factory)
    assert selected == "cbc"
    assert calls == ["highs", "cbc"]


def test_select_solver_auto_suppresses_probe_output(capsys):
    def fake_factory(name):
        print(f"noise from {name}")
        if name == "highs":
            raise RuntimeError("missing")
        return _FakeSolver(name == "cbc")

    selected = select_solver_auto(candidates=("highs", "cbc"), solver_factory=fake_factory)
    assert selected == "cbc"
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
