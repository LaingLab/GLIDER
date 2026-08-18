"""Installing a plugin, and refusing to.

pip is driven as a subprocess rather than imported, so every test here fakes the
subprocess. That is the point: the failures worth testing are pip's, and pip's
failures are exit codes and text, not exceptions.
"""

import pytest

from glider.plugins.installer import InstallResult, install

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

    import sys

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
