"""`glider doctor` reports whether this install can actually run GLIDER.

It grew out of ``--check-deps``, which exists because a packaged build broke in
a way no availability check could see: every torch file was present on disk
and ``importlib.util.find_spec`` called it installed, while the real import
died loading ``c10.dll``. A frozen bundle is exactly where you cannot check by
hand, so the diagnostic has to perform the real import -- and these tests pin
that, because "does it exist" is the easier thing to write and would silently
pass while helping nobody.
"""

import os
import subprocess
import sys

import pytest

from glider import env_doctor
from glider.env_doctor import DEP_STACKS, run_doctor


@pytest.fixture(autouse=True)
def _no_real_torch(monkeypatch):
    """The inference-device line would import the real torch; seconds per test."""
    monkeypatch.setattr(env_doctor, "_inference_device", lambda: "cpu")


def _imports_except(failures):
    """A stand-in for importlib.import_module that fails only the named modules."""

    def fake_import(name):
        if name in failures:
            raise failures[name]
        return object()

    return fake_import


# ---------------------------------------------------------------------------
# What it reports
# ---------------------------------------------------------------------------


def test_everything_importable_exits_zero(capsys, monkeypatch):
    monkeypatch.setattr("importlib.import_module", lambda name: object())
    assert run_doctor() == 0
    out = capsys.readouterr().out
    assert "[FAIL]" not in out
    assert "Inference device: cpu" in out


def test_an_extra_never_installed_is_not_a_failure(capsys, monkeypatch):
    """A lab that never asked for YOLO has nothing wrong with it."""
    missing = ModuleNotFoundError("No module named 'ultralytics'", name="ultralytics")
    monkeypatch.setattr("importlib.import_module", _imports_except({"ultralytics": missing}))
    assert run_doctor() == 0
    out = capsys.readouterr().out
    assert "[ -- ] Pose tracking (YOLO): not installed" in out
    assert "--extra vision" in out
    assert "Inference device" not in out


def test_a_half_installed_stack_is_a_failure(capsys, monkeypatch):
    """hdbscan going missing from a bundle greyed out the whole Behavior menu."""
    missing = ModuleNotFoundError("No module named 'hdbscan'", name="hdbscan")
    monkeypatch.setattr("importlib.import_module", _imports_except({"hdbscan": missing}))
    assert run_doctor() == 1
    out = capsys.readouterr().out
    assert "[FAIL] Behavior analysis" in out
    assert "hdbscan" in out


def test_a_package_missing_its_own_dependency_is_broken_not_absent(capsys, monkeypatch):
    """ultralytics installed without polars raises ModuleNotFoundError too.

    That names polars, not ultralytics -- the extra *was* chosen, and calling
    it "not installed" would send the operator to install what they have.
    """
    missing = ModuleNotFoundError("No module named 'polars'", name="polars")
    monkeypatch.setattr("importlib.import_module", _imports_except({"ultralytics": missing}))
    assert run_doctor() == 1
    assert "[FAIL] Pose tracking (YOLO)" in capsys.readouterr().out


def test_core_missing_is_a_failure_not_an_option(capsys, monkeypatch):
    missing = ModuleNotFoundError("No module named 'PyQt6'", name="PyQt6")
    monkeypatch.setattr("importlib.import_module", _imports_except({"PyQt6.QtWidgets": missing}))
    assert run_doctor() == 1
    out = capsys.readouterr().out
    assert "[FAIL] Core" in out
    assert "PyQt6.QtWidgets is not installed" in out


def test_a_dll_failure_is_reported_not_raised(capsys, monkeypatch):
    """The failure that motivated the check: present on disk, dead on import.

    find_spec calls this torch installed. Only the real import disagrees, and
    an OSError escaping here would take the whole diagnostic down.
    """
    dll = OSError("[WinError 1114] ... Error loading c10.dll")
    monkeypatch.setattr("importlib.import_module", _imports_except({"torch": dll}))
    assert run_doctor() == 1
    out = capsys.readouterr().out
    assert "[FAIL] Pose tracking (YOLO)" in out
    assert "OSError" in out
    assert "c10.dll" in out


def test_a_healthy_stack_is_not_reported_as_broken(capsys, monkeypatch):
    """One failing stack must not condemn the others."""
    monkeypatch.setattr(
        "importlib.import_module", _imports_except({"hdbscan": ImportError("nope")})
    )
    run_doctor()
    assert "[ ok ] Audio" in capsys.readouterr().out


def test_it_names_the_extra_that_would_fix_it(capsys, monkeypatch):
    monkeypatch.setattr(
        "importlib.import_module", _imports_except({"sounddevice": ImportError("nope")})
    )
    run_doctor()
    assert "uv sync --inexact --extra audio" in capsys.readouterr().out


def test_the_output_is_ascii_only(capsys, monkeypatch):
    """Windows hands a frozen app a cp1252 stdout when its output is piped.

    _print_gpu_check's tick marks crashed it there mid-report -- precisely when
    someone was capturing the output to send on. This one must survive that.
    """
    monkeypatch.setattr("importlib.import_module", lambda name: object())
    run_doctor()
    capsys.readouterr().out.encode("cp1252")  # raises if anything is unmappable


def test_a_non_ascii_path_on_a_cp1252_pipe_does_not_crash_it():
    """The report's text is ASCII; the paths in it are whatever the user's are."""
    blocked = ("ultralytics", "umap", "lightgbm", "sounddevice")
    code = (
        "import sys\n"
        f"for name in {blocked!r}: sys.modules[name] = None\n"
        "sys.prefix = 'C:\\\\Users\\\\\u5f20\u4f1f\\\\glider\\\\.venv'\n"
        "from glider.env_doctor import run_doctor\n"
        "run_doctor()\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        env={**os.environ, "PYTHONIOENCODING": "cp1252"},
        timeout=120,
    )
    out = result.stdout.decode("utf-8")
    assert "\u5f20\u4f1f" in out, result.stderr.decode("utf-8", "replace")
    assert "-" * 60 in out.split("Virtual environment", 1)[1]  # it reached the end


# ---------------------------------------------------------------------------
# It runs where GLIDER cannot
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("argv", [["doctor"], ["--check-deps"]])
def test_it_runs_when_qt_will_not_load(argv):
    """A Qt that will not load is the commonest thing it has to report.

    glider.__main__ imports PyQt6 at module level, so anything dispatched after
    that import dies with a bare DLL error instead of a report. The heavy
    optional stacks are blocked too, only to keep this subprocess quick.
    """
    blocked = ("PyQt6", "ultralytics", "umap", "lightgbm", "sounddevice")
    code = (
        "import runpy, sys\n"
        f"for name in {blocked!r}: sys.modules[name] = None\n"
        f"sys.argv = ['glider', *{argv!r}]\n"
        "runpy.run_module('glider', run_name='__main__')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert "[FAIL] Core" in result.stdout, result.stderr
    assert "PyQt6" in result.stdout
    assert result.returncode == 1


# ---------------------------------------------------------------------------
# The stack table
# ---------------------------------------------------------------------------


def test_it_covers_the_behavior_gate():
    """These four are what behavior_available() requires -- keep them in step.

    hdbscan is the one that went missing from the bundle and disabled the whole
    Behavior menu, so a check that omitted it would have reported all clear.
    """
    from glider.gui.behavior.availability import _REQUIRED

    checked = {mod for _, _, mods in DEP_STACKS for mod in mods}
    missing = [mod for mod, _pip in _REQUIRED if mod not in checked]
    assert not missing, f"the behavior gate gets checked for {missing}, but doctor does not"


def test_core_lists_only_what_glider_itself_requires():
    """polars sat in Core while only ultralytics pulls it in, so a plain
    `--extra pc` install was told its core was broken."""
    core = next(mods for label, extra, mods in DEP_STACKS if extra == "-")
    assert "polars" not in core


@pytest.mark.parametrize("label,extra,modules", DEP_STACKS)
def test_every_stack_is_well_formed(label, extra, modules):
    assert label.strip()
    assert modules, f"{label} lists no modules"
