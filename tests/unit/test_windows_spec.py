"""The Windows bundle must contain what the app checks for at startup.

Every optional dependency is lazy-imported so a bare ``[pc]`` install stays
importable. PyInstaller's static analysis therefore never sees them, and a
build that omits one does not fail — it ships a working app with a feature
silently switched off, because ``behavior_available()`` probes for the very
modules the bundle is missing and disables the menu without an error anywhere.

This is a static check over the spec rather than a build, so it runs in
milliseconds on every commit instead of only when someone cuts a release.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# Deliberately not in a tests/unit/packaging/ package: that name shadows the
# real `packaging` library, which glider.updater imports.
SPEC = Path(__file__).resolve().parents[2] / "packaging" / "windows" / "glider.spec"


@pytest.fixture(scope="module")
def spec_text() -> str:
    if not SPEC.is_file():
        pytest.skip(f"{SPEC} not present")
    return SPEC.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def collected(spec_text: str) -> set[str]:
    """Package names the spec collects or names as hidden imports."""
    names = set(re.findall(r'collect_all\(\s*"([^"]+)"', spec_text))
    names |= set(re.findall(r'collect_submodules\(\s*"([^"]+)"', spec_text))
    names |= set(re.findall(r'collect_dynamic_libs\(\s*"([^"]+)"', spec_text))
    # The behavior stack is collected from a tuple in a loop.
    for block in re.findall(r"for _package in \(([^)]*)\)", spec_text):
        names |= set(re.findall(r'"([^"]+)"', block))
    names |= set(re.findall(r'^\s*"([\w.]+)",\s*$', spec_text, flags=re.MULTILINE))
    return names


def test_the_behavior_gate_modules_are_all_collected(collected):
    """Whatever behavior_available() probes for has to be in the bundle.

    Miss one and Tools > Behavior Analysis is greyed out in the installer
    with nothing on screen saying why.
    """
    from glider.gui.behavior.availability import _REQUIRED

    for module, _pip_name in _REQUIRED:
        assert module in collected, (
            f"behavior_available() requires {module!r}, but the Windows spec "
            f"never collects it — the release would disable the Behavior "
            f"Analysis menu with no error"
        )


def test_model_bundles_can_be_read(collected):
    """joblib is how every BehaviorModel is loaded."""
    assert "joblib" in collected


def test_the_classifier_itself_is_collected(collected):
    assert "lightgbm" in collected


def test_agpl_dependencies_stay_out(spec_text):
    """ultralytics is AGPL-3.0 and ships as a lazy first-run download, so it
    must never be swept into the permissively-licensed installer."""
    excludes = re.search(r"excludes = \[(.*?)\n\]", spec_text, flags=re.DOTALL)
    assert excludes is not None
    for package in ("ultralytics", "torch", "torchvision"):
        assert f'"{package}"' in excludes.group(1)


def test_pi_only_drivers_stay_out(spec_text):
    excludes = re.search(r"excludes = \[(.*?)\n\]", spec_text, flags=re.DOTALL)
    for package in ("gpiozero", "lgpio"):
        assert f'"{package}"' in excludes.group(1)


def test_collected_behavior_packages_reach_the_analysis(spec_text):
    """Collecting them is useless if the lists are never passed to Analysis."""
    analysis = re.search(r"a = Analysis\((.*?)\n\)", spec_text, flags=re.DOTALL)
    assert analysis is not None
    body = analysis.group(1)
    assert "behavior_binaries" in body
    assert "behavior_datas" in body
    assert "behavior_hiddenimports" in spec_text
