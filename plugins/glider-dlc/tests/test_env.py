"""Building the DeepLabCut environment GLIDER converts in.

Every ``uv`` call is faked. What is being checked is the decision-making around
them -- when to build, when not to, what to do with a half-built environment,
and what a person is told when it fails -- because that is what a researcher
meets, and because actually running these would download over a gigabyte.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from glider_dlc import env as env_module
from glider_dlc.env import (
    ENV_PACKAGES,
    ENV_PYTHON,
    STAMP_NAME,
    ProvisioningError,
    env_dir,
    interpreter,
    is_provisioned,
    provision,
    run_converter,
)


@pytest.fixture
def uv_calls() -> list[list[str]]:
    """Every argv a fake ``uv`` was invoked with, in order."""
    return []


@pytest.fixture
def env(tmp_path, monkeypatch, uv_calls):
    """An environment path, with ``uv`` faked out.

    The fake creates the interpreter on ``uv venv`` and otherwise succeeds
    silently, so provisioning reaches the state a real one would without
    downloading a gigabyte.
    """
    target = tmp_path / "dlc-env"

    def _uv(args, **kwargs):
        uv_calls.append(args)
        if args[1] == "venv":
            interpreter(target).parent.mkdir(parents=True, exist_ok=True)
            interpreter(target).write_text("#!fake")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(env_module.shutil, "which", lambda name: "uv" if name == "uv" else None)
    monkeypatch.setattr(subprocess, "run", _uv)
    return target


# --- where it lives ----------------------------------------------------------


def test_it_lives_under_the_glider_directory():
    assert env_dir().parts[-3:] == (".glider", "envs", "deeplabcut")


def test_an_existing_environment_can_be_pointed_at(monkeypatch, tmp_path):
    """The escape hatch for a lab that already has DeepLabCut working and would
    rather not have a second copy of it."""
    monkeypatch.setenv("GLIDER_DLC_ENV", str(tmp_path / "mine"))

    assert env_dir() == tmp_path / "mine"


def test_the_interpreter_is_platform_correct(tmp_path):
    path = interpreter(tmp_path)

    assert path.name == ("python.exe" if sys.platform == "win32" else "python")


# --- when to build -----------------------------------------------------------


def test_a_fresh_machine_is_not_provisioned(tmp_path):
    assert is_provisioned(tmp_path / "nothing-here") is False


def test_an_interpreter_without_a_stamp_is_not_provisioned(env):
    """A half-finished install leaves an interpreter that imports nothing. The
    stamp is written last, so its absence is what says 'incomplete'."""
    interpreter(env).parent.mkdir(parents=True, exist_ok=True)
    interpreter(env).write_text("#!fake")

    assert is_provisioned(env) is False


def test_a_complete_environment_is_provisioned(env):
    provision(env)

    assert is_provisioned(env) is True


def test_provisioning_twice_does_nothing_the_second_time(env, uv_calls):
    provision(env)
    before = len(uv_calls)

    provision(env)

    assert len(uv_calls) == before


def test_a_changed_spec_rebuilds(env):
    """Otherwise a fix to the pinned DeepLabCut version would apply to new
    users only, and silently -- the shape of bug that gets blamed on the
    model."""
    provision(env)
    (env / STAMP_NAME).write_text(json.dumps({"python": "3.10", "packages": ["deeplabcut"]}))

    assert is_provisioned(env) is False


def test_an_environment_the_lab_manages_is_taken_at_its_word(env, monkeypatch):
    """A GLIDER_DLC_ENV has no stamp and is not ours to rebuild."""
    interpreter(env).parent.mkdir(parents=True, exist_ok=True)
    interpreter(env).write_text("#!fake")
    monkeypatch.setenv("GLIDER_DLC_ENV", str(env))

    assert is_provisioned(env) is True


# --- how it builds -----------------------------------------------------------


def test_it_creates_the_venv_then_installs_into_it(env, uv_calls):
    provision(env)

    assert [c[1] for c in uv_calls] == ["venv", "pip"]
    assert "--python" in uv_calls[0] and ENV_PYTHON in uv_calls[0]
    assert all(pkg in uv_calls[1] for pkg in ENV_PACKAGES)


def test_the_install_targets_the_new_interpreter(env, uv_calls):
    """Without --python, uv picks its own target and DeepLabCut lands somewhere
    GLIDER never runs -- an install that reports success and changes nothing."""
    provision(env)

    install = uv_calls[1]

    assert install[install.index("--python") + 1] == str(interpreter(env))


def test_a_half_built_environment_is_removed_first(env):
    """Installing into a virtualenv whose interpreter is missing does not work,
    and the error it produces is unreadable."""
    env.mkdir(parents=True)
    (env / "leftovers.txt").write_text("from an interrupted download")

    provision(env)

    assert not (env / "leftovers.txt").exists()
    assert is_provisioned(env) is True


def test_progress_is_reported_before_the_long_wait(env):
    """The download is minutes long. A caller with nothing to show during it
    can only show a frozen window."""
    said: list[str] = []

    provision(env, on_progress=said.append)

    assert any("GB" in m for m in said), said


# --- when it cannot ----------------------------------------------------------


def test_no_uv_is_reported_with_a_way_out(env, monkeypatch):
    monkeypatch.setattr(env_module.shutil, "which", lambda name: None)

    with pytest.raises(ProvisioningError, match="uv"):
        provision(env)


def test_a_failed_install_shows_what_uv_said(env, monkeypatch):
    """uv's last lines name the actual problem -- an unresolvable pin, no
    network, no matching Python. A bare 'failed' would send someone to a log
    file they do not know about."""

    def _fail(args, **kwargs):
        if args[1] == "venv":
            interpreter(env).parent.mkdir(parents=True, exist_ok=True)
            interpreter(env).write_text("#!fake")
            return subprocess.CompletedProcess(args, 0, "", "")
        return subprocess.CompletedProcess(args, 1, "", "error: no solution found")

    monkeypatch.setattr(subprocess, "run", _fail)

    with pytest.raises(ProvisioningError, match="no solution found"):
        provision(env)


def test_a_failed_install_leaves_nothing_that_reads_as_ready(env, monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda args, **kw: subprocess.CompletedProcess(args, 1, "", "boom"),
    )

    with pytest.raises(ProvisioningError):
        provision(env)

    assert is_provisioned(env) is False


def test_a_wedged_install_is_reported_in_minutes(env, monkeypatch):
    def _timeout(args, **kwargs):
        raise subprocess.TimeoutExpired(args, 3600)

    monkeypatch.setattr(subprocess, "run", _timeout)

    with pytest.raises(ProvisioningError, match="minutes"):
        provision(env)


def test_an_empty_managed_environment_says_what_to_do(env, monkeypatch):
    """GLIDER_DLC_ENV pointing somewhere empty is a typo, not an invitation to
    build 1.3 GB in a place the lab chose for something else."""
    monkeypatch.setenv("GLIDER_DLC_ENV", str(env))

    with pytest.raises(ProvisioningError, match="GLIDER_DLC_ENV"):
        provision(env)

    assert not env.exists()


# --- running the conversion --------------------------------------------------


def test_the_converter_runs_in_the_provisioned_interpreter(env, monkeypatch, tmp_path):
    seen: dict = {}

    def _run(args, **kwargs):
        seen["args"] = args
        seen["env"] = kwargs.get("env", {})
        return subprocess.CompletedProcess(args, 0, "{}", "")

    monkeypatch.setattr(subprocess, "run", _run)
    script = tmp_path / "convert.py"

    run_converter(script, tmp_path / "model", env=env)

    assert seen["args"][0] == str(interpreter(env))
    assert seen["args"][1] == str(script)


def test_the_child_is_told_to_use_utf8(env, monkeypatch, tmp_path):
    """torch's exporter prints a check mark on success, which raises
    UnicodeEncodeError on a Windows console still defaulting to cp1252 -- a
    conversion that worked, reported as a crash."""
    seen: dict = {}

    def _run(args, **kwargs):
        seen.update(kwargs.get("env", {}))
        return subprocess.CompletedProcess(args, 0, "{}", "")

    monkeypatch.setattr(subprocess, "run", _run)

    run_converter(tmp_path / "convert.py", tmp_path / "model", env=env)

    assert seen.get("PYTHONIOENCODING") == "utf-8"


def test_a_failed_conversion_raises_the_childs_message(env, monkeypatch, tmp_path):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda args, **kw: subprocess.CompletedProcess(args, 1, "", "no snapshot-*.pt found"),
    )

    with pytest.raises(ProvisioningError, match="no snapshot"):
        run_converter(tmp_path / "convert.py", tmp_path / "model", env=env)
