"""Installing a plugin, and refusing to.

pip is driven as a subprocess rather than imported, so every test here fakes the
subprocess. That is the point: the failures worth testing are pip's, and pip's
failures are exit codes and text, not exceptions.

The default runner is the exception: it is the only code here that really spawns
a process, so faking it would leave every one of its failure modes untested, and
each of those failures is silent. Those tests drive a short ``python -c`` program
instead -- a real subprocess, but no pip and no network.
"""

import asyncio
import sys

import pytest

from glider.plugins.installer import (
    InstallResult,
    NoInstallerError,
    _default_runner,
    incompatibility_message,
    install,
    installer_command,
    is_compatible,
)

ENTRY = {
    "name": "glider-harp",
    "pypi": "glider-harp",
    "version": "0.1.0",
    "glider_requires": ">=1.0,<2.0",
}


def _runner(returncode: int, output: str):
    async def run(args, on_output=None):
        if on_output:
            for line in output.splitlines():
                on_output(line)
        return returncode, output

    return run


async def test_a_compatible_plugin_installs():
    result = await install(
        ENTRY, glider_version="1.0.0", runner=_runner(0, "Successfully installed glider-harp")
    )

    assert isinstance(result, InstallResult)
    assert result.ok is True
    assert "Successfully installed" in result.output


async def test_the_default_command_targets_this_interpreter():
    """Installing with the wrong python puts the plugin somewhere GLIDER will
    never import from -- and it would look like it worked.

    This used to assert the literal pip argv, which encoded the assumption that
    pip exists. On GLIDER's documented setup (``uv venv``) it does not, and the
    hard-coded shape was the bug. The invariant is not "pip is used"; it is
    "whatever installer is used, it installs into *this* interpreter" -- so the
    assertion is now against ``installer_command``'s real answer for this
    environment, which the shape tests above pin for both branches.
    """
    seen = {}

    async def run(args, on_output=None):
        seen["args"] = args
        return 0, ""

    await install(ENTRY, glider_version="1.0.0", runner=run)

    assert seen["args"] == installer_command("glider-harp")
    assert sys.executable in seen["args"]
    assert seen["args"][-1] == "glider-harp"


async def test_too_old_a_glider_is_refused_naming_both_versions():
    entry = {**ENTRY, "glider_requires": ">=2.0"}

    result = await install(entry, glider_version="1.0.0", runner=_runner(0, ""))

    assert result.ok is False
    assert "2.0" in result.message
    assert "1.0.0" in result.message


async def test_a_refusal_never_runs_pip():
    entry = {**ENTRY, "glider_requires": ">=2.0"}
    ran = False

    async def run(args, on_output=None):
        nonlocal ran
        ran = True
        return 0, ""

    await install(entry, glider_version="1.0.0", runner=run)

    assert ran is False


async def test_pip_failure_surfaces_its_own_output():
    output = "ERROR: Could not find a version that satisfies the requirement zmq>=26"
    result = await install(ENTRY, glider_version="1.0.0", runner=_runner(1, output))

    assert result.ok is False
    assert "zmq>=26" in result.output
    assert "1" in result.message


async def test_progress_lines_reach_the_caller():
    """The window streams pip's output onto the row, so the callback has to fire
    as lines arrive rather than once at the end."""
    lines = []
    await install(
        ENTRY,
        glider_version="1.0.0",
        runner=_runner(0, "Collecting glider-harp\nInstalling collected packages"),
        on_output=lines.append,
    )

    assert lines == ["Collecting glider-harp", "Installing collected packages"]


@pytest.mark.parametrize(
    "spec,version,ok",
    [
        (">=1.0,<2.0", "1.0.0", True),
        (">=1.0,<2.0", "1.9.9", True),
        (">=1.0,<2.0", "2.0.0", False),
        (">=1.0,<2.0", "0.9.0", False),
        ("", "1.0.0", True),
    ],
)
async def test_the_version_gate(spec, version, ok):
    result = await install(
        {**ENTRY, "glider_requires": spec}, glider_version=version, runner=_runner(0, "")
    )
    assert result.ok is ok


# ---------------------------------------------------------------------------
# A catalogue entry the maintainer got wrong.
#
# The index comes over the network and nothing validates its fields, so a
# malformed `glider_requires` is data arriving from outside -- not a programming
# error. `"1.0"` where `">=1.0"` was meant is the natural authoring mistake, and
# it used to raise `InvalidSpecifier` out of `is_compatible`, up through the
# window's constructor, into a task nobody was watching.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("requires", ["1.0", "not a specifier", ">=<1", "1.0.0"])
def test_an_unparseable_requirement_is_answered_not_raised(requires):
    assert is_compatible({**ENTRY, "glider_requires": requires}, "1.0.0") is False


def test_an_unparseable_requirement_says_the_entry_is_unreadable(requires="1.0"):
    """ "needs GLIDER 1.0, you are running 1.0.0" would be a nonsense sentence --
    it reads as a version mismatch when the entry itself is the problem."""
    message = incompatibility_message({**ENTRY, "glider_requires": requires}, "1.0.0")

    assert "unreadable" in message.lower()
    assert "1.0" in message


async def test_an_unparseable_requirement_never_runs_pip():
    ran = False

    async def run(args, on_output=None):
        nonlocal ran
        ran = True
        return 0, ""

    result = await install({**ENTRY, "glider_requires": "1.0"}, glider_version="1.0.0", runner=run)

    assert result.ok is False
    assert ran is False


async def test_an_entry_with_no_package_name_is_refused_rather_than_crashing():
    """`entry["pypi"]` was the one bracket access to a key everything else
    tolerated the absence of."""
    ran = False

    async def run(args, on_output=None):
        nonlocal ran
        ran = True
        return 0, ""

    result = await install({"version": "1.0.0"}, glider_version="1.0.0", runner=run)

    assert result.ok is False
    assert ran is False


async def test_an_entry_with_only_a_name_installs_that_name():
    seen = {}

    async def run(args, on_output=None):
        seen["args"] = args
        return 0, ""

    await install({"name": "glider-harp"}, glider_version="1.0.0", runner=run)

    assert seen["args"][-1] == "glider-harp"


# ---------------------------------------------------------------------------
# The default runner, against a real subprocess.
#
# Every test above fakes the runner, which leaves the one function that actually
# spawns a process uncovered -- and its failure modes are all silent. A lost
# stderr merge drops the error text that is usually the whole answer; a wrong
# returncode reports a failed install as a success; a decode error on unusual
# output crashes an install that in fact worked. These drive `python -c` rather
# than pip: a real process, no network, no package index, exits immediately.
# ---------------------------------------------------------------------------


async def _run_program(source: str, on_output=None):
    """Run a short Python program through the real runner."""
    try:
        return await _default_runner([sys.executable, "-c", source], on_output)
    except NotImplementedError:  # pragma: no cover - loop without subprocess support
        pytest.skip("this event loop does not support subprocesses")


async def test_the_runner_streams_each_line_as_a_separate_call():
    """Not just 'the joined output is right' -- an implementation that buffered
    everything and flushed once at the end would satisfy that, and it is exactly
    the bug that freezes the progress row until the install is already over."""
    lines = []
    returncode, output = await _run_program(
        "for line in ('Collecting glider-harp', 'Downloading', 'Installing'):\n"
        "    print(line, flush=True)\n",
        lines.append,
    )

    assert lines == ["Collecting glider-harp", "Downloading", "Installing"]
    assert returncode == 0
    assert output == "Collecting glider-harp\nDownloading\nInstalling"


async def test_the_runner_reports_a_nonzero_exit():
    """`returncode or 0` must not launder a failure into a success."""
    returncode, _ = await _run_program("import sys; sys.exit(3)")

    assert returncode == 3


async def test_the_runner_does_not_launder_an_unknown_exit_into_a_success(monkeypatch):
    """`returncode or 0` mapped None -- "the process never reported" -- onto zero,
    which is the wrong default direction for the value gating "did pip work"."""

    class _NeverReports:
        returncode = None

        def __init__(self):
            self.stdout = self

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def wait(self):
            return None

    async def fake_exec(*args, **kwargs):
        return _NeverReports()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    returncode, _ = await _default_runner([sys.executable, "-c", "pass"])

    assert returncode != 0


async def test_the_runner_captures_stderr_too():
    """pip writes its errors to stderr, so dropping the stderr=STDOUT merge would
    leave a failed install with an empty explanation."""
    _, output = await _run_program(
        "import sys; sys.stderr.write('ERROR: no matching distribution\\n')"
    )

    assert "ERROR: no matching distribution" in output


async def test_the_runner_survives_undecodable_output():
    """A stray non-UTF-8 byte in a dependency's output must not raise and abort
    an install that is otherwise working."""
    returncode, output = await _run_program(
        "import sys; sys.stdout.buffer.write(b'Collecting \\xff\\xfe harp\\n')"
    )

    assert returncode == 0
    assert "Collecting" in output
    assert "harp" in output


# --- choosing an installer -------------------------------------------------
#
# GLIDER's documented setup is `uv venv` + `uv sync` (CLAUDE.md), and `uv venv`
# does not install pip. So `sys.executable -m pip` -- the obvious command --
# fails with "No module named pip" on the *primary* supported environment. Every
# other test in this file injects a fake runner, which is exactly why this
# survived: the fake stood in for the one thing that was broken.


def test_pip_is_used_when_it_is_importable():
    cmd = installer_command("glider-harp", pip_available=lambda: True, uv_path=lambda: None)

    assert cmd == [sys.executable, "-m", "pip", "install", "glider-harp"]


def test_uv_is_used_when_pip_is_absent():
    cmd = installer_command(
        "glider-harp", pip_available=lambda: False, uv_path=lambda: "/opt/bin/uv"
    )

    assert cmd == [
        "/opt/bin/uv",
        "pip",
        "install",
        "--python",
        sys.executable,
        "glider-harp",
    ]


def test_uv_installs_into_this_interpreter_not_uv_s_default():
    """Without --python, uv picks its own target and the plugin lands somewhere
    GLIDER will never import from -- which would look like a successful install."""
    cmd = installer_command("x", pip_available=lambda: False, uv_path=lambda: "uv")

    assert "--python" in cmd
    assert cmd[cmd.index("--python") + 1] == sys.executable


def test_pip_wins_when_both_are_available():
    cmd = installer_command("x", pip_available=lambda: True, uv_path=lambda: "uv")

    assert cmd[:3] == [sys.executable, "-m", "pip"]


def test_neither_available_raises_with_a_message_naming_both():
    with pytest.raises(NoInstallerError) as excinfo:
        installer_command("x", pip_available=lambda: False, uv_path=lambda: None)

    message = str(excinfo.value)
    assert "pip" in message and "uv" in message


async def test_install_reports_the_missing_installer_instead_of_crashing():
    """The window must show this on the row, not raise out of the click."""
    result = await install(
        ENTRY,
        glider_version="1.0.0",
        runner=_runner(0, ""),
        command=lambda pkg: (_ for _ in ()).throw(NoInstallerError("no pip, no uv")),
    )

    assert result.ok is False
    assert "no pip, no uv" in result.message


async def test_the_real_detectors_agree_with_this_interpreter():
    """Guard against the detector drifting from reality: whatever it reports for
    pip must match whether `import pip` actually works here."""
    import importlib.util

    from glider.plugins.installer import _pip_is_importable

    assert _pip_is_importable() is (importlib.util.find_spec("pip") is not None)
