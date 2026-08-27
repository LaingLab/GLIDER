"""Shared helpers for the glider-sleap-nn tests."""

import subprocess
import sys

import pytest

#: Import cost is what makes `claims()` cheap enough to run on every model
#: selection, so it is worth a test -- but "did importing X pull in torch" is a
#: property of a *fresh* interpreter, not of this one. Asserting
#: ``"torch" not in sys.modules`` inline passes in isolation and fails as soon
#: as any earlier test in the session has imported torch for its own reasons,
#: which says nothing about this plugin. So the question is asked in a
#: subprocess, where the only thing that has been imported is the module under
#: test.
_PROBE = (
    "import importlib, sys;"
    "importlib.import_module({module!r});"
    "heavy = sorted(m for m in ('torch', 'sleap_nn', 'lightning') if m in sys.modules);"
    "print(','.join(heavy))"
)


@pytest.fixture
def heavy_imports_after():
    """Return the heavy modules pulled in by importing a given module name."""

    def _probe(module: str) -> list[str]:
        completed = subprocess.run(
            [sys.executable, "-c", _PROBE.format(module=module)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert completed.returncode == 0, completed.stderr
        return [name for name in completed.stdout.strip().split(",") if name]

    return _probe
