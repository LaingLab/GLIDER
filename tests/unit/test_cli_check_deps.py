"""`--check-deps` reports whether each optional stack can actually be imported.

The flag exists because a packaged build broke in a way no availability check
could see: every torch file was present on disk and ``importlib.util.find_spec``
called it installed, while the real import died loading ``c10.dll``. A frozen
bundle is exactly where you cannot check by hand, so the diagnostic has to
perform the real import -- and these tests pin that, because "does it exist"
is the easier thing to write and would silently pass while helping nobody.
"""

import sys

import pytest

from glider.__main__ import _DEP_STACKS, _print_dep_check, parse_args


def test_the_flag_parses(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["glider", "--check-deps"])
    assert parse_args().check_deps is True


def test_it_is_off_by_default(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["glider"])
    assert parse_args().check_deps is False


# ---------------------------------------------------------------------------
# What it reports
# ---------------------------------------------------------------------------


def test_everything_importable_exits_zero(capsys, monkeypatch):
    monkeypatch.setattr("importlib.import_module", lambda name: object())
    assert _print_dep_check() == 0
    out = capsys.readouterr().out
    assert "All optional stacks available." in out
    assert "[FAIL]" not in out


def test_a_broken_stack_exits_nonzero(capsys, monkeypatch):
    def fake_import(name):
        if name == "hdbscan":
            raise ImportError("No module named 'hdbscan'")
        return object()

    monkeypatch.setattr("importlib.import_module", fake_import)
    assert _print_dep_check() == 1
    out = capsys.readouterr().out
    assert "[FAIL] Behavior analysis" in out
    assert "hdbscan" in out


def test_a_dll_failure_is_reported_not_raised(capsys, monkeypatch):
    """The failure that motivated the flag: present on disk, dead on import.

    find_spec calls this torch installed. Only the real import disagrees, and
    an OSError escaping here would take the whole diagnostic down.
    """

    def fake_import(name):
        if name == "torch":
            raise OSError("[WinError 1114] ... Error loading c10.dll")
        return object()

    monkeypatch.setattr("importlib.import_module", fake_import)
    assert _print_dep_check() == 1
    out = capsys.readouterr().out
    assert "[FAIL] Pose tracking (YOLO)" in out
    assert "OSError" in out
    assert "c10.dll" in out


def test_a_healthy_stack_is_not_reported_as_broken(capsys, monkeypatch):
    """One failing stack must not condemn the others."""

    def fake_import(name):
        if name == "hdbscan":
            raise ImportError("nope")
        return object()

    monkeypatch.setattr("importlib.import_module", fake_import)
    _print_dep_check()
    assert "[ ok ] Audio" in capsys.readouterr().out


def test_it_names_the_extra_that_would_fix_it(capsys, monkeypatch):
    def fake_import(name):
        if name == "sounddevice":
            raise ImportError("nope")
        return object()

    monkeypatch.setattr("importlib.import_module", fake_import)
    _print_dep_check()
    assert "GLIDER[audio]" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The stack table
# ---------------------------------------------------------------------------


def test_the_output_is_ascii_only(capsys, monkeypatch):
    """Windows hands a frozen app a cp1252 stdout when its output is piped.

    _print_gpu_check's tick marks crashed it there mid-report -- precisely when
    someone was capturing the output to send on. This one must survive that.
    """
    monkeypatch.setattr("importlib.import_module", lambda name: object())
    _print_dep_check()
    capsys.readouterr().out.encode("cp1252")  # raises if anything is unmappable


def test_it_covers_the_behavior_gate(monkeypatch):
    """These four are what behavior_available() requires -- keep them in step.

    hdbscan is the one that went missing from the bundle and disabled the whole
    Behavior menu, so a check that omitted it would have reported all clear.
    """
    from glider.gui.behavior.availability import _REQUIRED

    checked = {mod for _, _, mods in _DEP_STACKS for mod in mods}
    missing = [mod for mod, _pip in _REQUIRED if mod not in checked]
    assert not missing, f"the behavior gate gets checked for {missing}, but --check-deps does not"


@pytest.mark.parametrize("label,extra,modules", _DEP_STACKS)
def test_every_stack_is_well_formed(label, extra, modules):
    assert label.strip()
    assert modules, f"{label} lists no modules"
