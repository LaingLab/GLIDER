"""Installing a plugin, and refusing to.

pip is driven as a subprocess rather than imported, so every test here fakes the
subprocess. That is the point: the failures worth testing are pip's, and pip's
failures are exit codes and text, not exceptions.

The default runner is the exception: it is the only code here that really spawns
a process, so faking it would leave every one of its failure modes untested, and
each of those failures is silent. Those tests drive a short ``python -c`` program
instead -- a real subprocess, but no pip and no network.
"""

import sys

import pytest

from glider.plugins.installer import InstallResult, _default_runner, install

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


async def test_the_pip_command_targets_this_interpreter():
    """Installing with the wrong python puts the plugin somewhere GLIDER will
    never import from -- and it would look like it worked."""
    seen = {}

    async def run(args, on_output=None):
        seen["args"] = args
        return 0, ""

    await install(ENTRY, glider_version="1.0.0", runner=run)

    assert seen["args"][:4] == [sys.executable, "-m", "pip", "install"]
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
