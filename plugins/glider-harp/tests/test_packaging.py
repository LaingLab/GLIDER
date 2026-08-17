"""What has to be true of ``glider-harp`` as a *distribution*, not as source.

Every assertion here covers a failure that a source checkout cannot show you,
because the checkout has properties the wheel does not inherit for free:

* ``profiles/*.json`` is beside ``derivation.py`` on disk whether or not
  anything declares it as package data, so ``load_profile`` passes in the repo
  and raises ``FileNotFoundError`` only once installed.
* ``harp.protocol`` imports under both 0.4.0 and 0.5.x, so a mis-resolved
  dependency is invisible to ``import``.
* ``import glider_harp`` costing no ``glider`` modules is a property of what
  the package body does *not* do, which nothing else would notice regressing.
* the core ``glider`` wheel not containing this package is a property of the
  root ``pyproject.toml``, which lives in a different tree entirely.
"""

import json
import shutil
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PLUGIN_ROOT.parent.parent


def _build_wheel(project_dir: Path, tmp_path: Path, tag: str) -> Path:
    """Build ``project_dir`` into a wheel, from a pristine copy of the tree.

    The copy is not fastidiousness: setuptools writes ``build/`` and
    ``*.egg-info`` next to the source it is given, and a test that leaves those
    in the working tree is a test that dirties ``git status`` every time it
    runs -- and, worse, leaves a stale ``build/lib`` that later builds happily
    reuse. Building a copy means the assertions are about what the declared
    configuration produces from a clean tree, which is the thing under test.
    """
    staging = tmp_path / f"{tag}-src"
    ignore = shutil.ignore_patterns(
        "__pycache__", "*.pyc", ".git", ".venv", ".worktrees", "build", "dist", "*.egg-info"
    )
    staging.mkdir()
    for entry in ("pyproject.toml", "README.md", "LICENSE", "MANIFEST.in", "src", "plugins"):
        source = project_dir / entry
        if not source.exists():
            continue
        if source.is_dir():
            shutil.copytree(source, staging / entry, ignore=ignore)
        else:
            shutil.copy2(source, staging / entry)

    out = tmp_path / f"{tag}-wheel"
    out.mkdir()
    # Call the build backend rather than shelling out to `build` or `uv`:
    # neither is guaranteed present (the project's venv is uv-created and has
    # no pip), while setuptools is the declared backend for both projects.
    code = textwrap.dedent("""
        import sys
        from setuptools import build_meta
        sys.stdout.write(build_meta.build_wheel(sys.argv[1]))
        """)
    result = subprocess.run(
        [sys.executable, "-c", code, str(out)],
        cwd=staging,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"wheel build for {project_dir} failed:\n{result.stdout}\n{result.stderr}")
    return out / result.stdout.strip().splitlines()[-1]


@pytest.fixture(scope="module")
def plugin_wheel(tmp_path_factory) -> Path:
    return _build_wheel(PLUGIN_ROOT, tmp_path_factory.mktemp("plugin"), "glider-harp")


@pytest.fixture(scope="module")
def plugin_wheel_names(plugin_wheel) -> list[str]:
    with zipfile.ZipFile(plugin_wheel) as zf:
        return zf.namelist()


# --------------------------------------------------------------------------
# 1. The dependency resolves to a *usable* harp-protocol.
# --------------------------------------------------------------------------


def test_the_installed_harp_protocol_is_the_one_this_package_was_written_against():
    """A bare ``import harp.protocol`` is not a version check -- 0.4.0 passes it.

    0.4.0 and 0.5.x share no API but the same import path, and ``harp``'s
    unbounded ``Requires-Dist: harp-protocol`` lets a resolve pick the wrong
    one and report success. Only a *name* separates them, which is why
    ``board.connect()`` imports one as a canary and why this asserts on names
    rather than on the module. Retarget both together if upstream renames.
    """
    from harp.protocol import HarpMessage, HarpParseError, MessageType, PayloadType

    assert HarpMessage is not None
    assert issubclass(HarpParseError, Exception)
    assert {"Read", "Write", "Event"} <= set(MessageType.__members__)
    assert {"U8", "U16", "Float"} <= set(PayloadType.__members__)


def test_only_harp_protocol_is_depended_on_not_the_harp_metapackage(plugin_wheel):
    """The declared dependency has to keep matching what the code imports.

    Every ``harp.`` import in this package is ``harp.protocol``. ``harp`` is a
    meta-package that adds harp-device, harp-serial and harp-data -- and their
    pandas/pyserial trees -- for no importer. Depending on it would also route
    the harp-protocol requirement through ``harp``'s unbounded one.

    Read from the built wheel's METADATA rather than from ``pyproject.toml``,
    so this is what a resolver will actually be handed.
    """
    with zipfile.ZipFile(plugin_wheel) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".dist-info/METADATA"))
        metadata = zf.read(name).decode("utf-8").splitlines()

    declared = [
        line.split(":", 1)[1].strip()
        for line in metadata
        if line.lower().startswith("requires-dist:")
    ]
    harp_requirements = [r for r in declared if r.lower().startswith("harp")]
    assert len(harp_requirements) == 1, harp_requirements
    requirement = harp_requirements[0]
    assert requirement.lower().startswith("harp-protocol"), requirement
    # An explicit lower bound is the whole point: without one the incompatible
    # 0.4.0 satisfies the requirement, and pip reports success.
    assert ">=" in requirement, requirement


# --------------------------------------------------------------------------
# 2. Importing the package still costs no `glider`.
# --------------------------------------------------------------------------


def test_importing_the_package_pulls_in_no_glider_modules():
    """The lazy plugin tables must not become eager imports by accident.

    ``import glider_harp`` costs zero ``glider`` modules; importing
    ``glider_harp.board`` costs dozens, including all of ``glider.vision``.
    A single top-level ``from glider_harp.board import HarpBoard`` in
    ``__init__.py`` erases that difference silently, so it is pinned here.
    Run in a subprocess because this process has almost certainly imported
    ``glider`` already.
    """
    code = textwrap.dedent("""
        import sys
        import glider_harp
        leaked = sorted(m for m in sys.modules if m == "glider" or m.startswith("glider."))
        leaked = [m for m in leaked if not m.startswith("glider_harp")]
        print(repr(leaked))
        """)
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=_child_env()
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", result.stdout


def test_the_plugin_tables_resolve_lazily_and_only_on_access():
    """PEP 562 ``__getattr__`` is what ``PluginManager``'s ``hasattr`` sees."""
    code = textwrap.dedent("""
        import sys
        import glider_harp
        assert "glider_harp.board" not in sys.modules
        assert hasattr(glider_harp, "BOARD_DRIVERS")
        assert "glider_harp.board" in sys.modules
        assert glider_harp.BOARD_DRIVERS["harp"].__name__ == "HarpBoard"
        assert glider_harp.DEVICE_TYPES["Harp"].__name__ == "HarpDevice"
        # Resolved values are cached into the module namespace.
        assert "BOARD_DRIVERS" in vars(glider_harp)
        # And unknown names still raise, which is what keeps `hasattr(module,
        # "setup")` False in PluginManager.load_plugin.
        assert not hasattr(glider_harp, "setup")
        try:
            glider_harp.NOPE
        except AttributeError:
            pass
        else:
            raise SystemExit("__getattr__ swallowed an unknown name")
        print("ok")
        """)
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=_child_env()
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "ok"


async def test_plugin_manager_registers_the_board_and_device_from_the_package():
    """The end the packaging exists for: GLIDER's own loader, run for real.

    Not a stand-in for ``PluginManager`` -- the real class, the real
    ``_register_plugin_components``, resolving the tables through the module
    ``__getattr__``. This is the module-valued entry point shape
    (``glider_harp``, no ``:Class``); the ``module:Class`` shape registers
    nothing, which is why the pyproject declares both.
    """
    hardware_manager = pytest.importorskip("glider.core.hardware_manager")
    from glider.hal.base_device import DEVICE_REGISTRY
    from glider.plugins.plugin_manager import PluginInfo, PluginManager

    manager = PluginManager()
    info = PluginInfo(name="glider_harp", entry_point="glider_harp", plugin_type="driver")
    manager._plugins["glider_harp"] = info

    assert await manager.load_plugin("glider_harp") is True, info.error
    assert "harp" in hardware_manager.HardwareManager.get_available_drivers()
    assert hardware_manager.HardwareManager.get_driver_class("harp").__name__ == "HarpBoard"
    assert DEVICE_REGISTRY["Harp"].__name__ == "HarpDevice"


def _child_env() -> dict:
    """Environment for a subprocess that must import ``glider_harp``.

    Inherits the parent's, then makes sure the package's ``src`` is on
    ``PYTHONPATH`` using ``os.pathsep`` -- ``;`` on Windows, ``:`` elsewhere.
    Hardcoding either one silently imports something else.
    """
    import os

    env = dict(os.environ)
    src = str(PLUGIN_ROOT / "src")
    existing = env.get("PYTHONPATH", "")
    if src not in existing.split(os.pathsep):
        env["PYTHONPATH"] = f"{src}{os.pathsep}{existing}" if existing else src
    return env


# --------------------------------------------------------------------------
# 3. The shipped profile survives into the wheel, and loads from it.
# --------------------------------------------------------------------------


def test_the_shipped_profile_is_in_the_wheel(plugin_wheel_names):
    """Under a src layout setuptools ships no non-Python file unless told to.

    Without the ``[tool.setuptools.package-data]`` declaration this list has
    only ``.py`` files in it, and the source tree gives no hint of that.
    """
    profiles = [n for n in plugin_wheel_names if n.startswith("glider_harp/profiles/")]
    assert "glider_harp/profiles/licketysplit.json" in profiles, plugin_wheel_names
    # Everything in the source profiles/ directory, not just the one we name.
    on_disk = sorted(p.name for p in (PLUGIN_ROOT / "src" / "glider_harp" / "profiles").iterdir())
    assert sorted(Path(n).name for n in profiles) == on_disk


def test_load_profile_works_from_an_installed_package_not_the_checkout(plugin_wheel, tmp_path):
    """``PROFILE_DIR`` is ``Path(__file__).parent / "profiles"``, so this is the
    only way to test it: from the checkout it resolves to the source tree and
    passes no matter what the wheel contains."""
    unpacked = tmp_path / "site-packages"
    with zipfile.ZipFile(plugin_wheel) as zf:
        zf.extractall(unpacked)

    code = textwrap.dedent(f"""
        import json, sys
        sys.path.insert(0, {str(unpacked)!r})
        from glider_harp import derivation
        # Prove we are looking at the unpacked wheel and not at the checkout
        # that PYTHONPATH or an editable install may also have on the path.
        assert derivation.__file__.startswith({str(unpacked)!r}), derivation.__file__
        print(json.dumps(derivation.load_profile("licketysplit")))
        """)
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=tmp_path
    )
    assert result.returncode == 0, result.stdout + result.stderr
    profile = json.loads(result.stdout)
    assert profile["name"] == "LicketySplit"
    assert profile["record"]


# --------------------------------------------------------------------------
# 4. Entry points are declared in the groups PluginManager scans.
# --------------------------------------------------------------------------


def test_entry_points_land_in_the_groups_plugin_manager_discovers(plugin_wheel):
    with zipfile.ZipFile(plugin_wheel) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".dist-info/entry_points.txt"))
        text = zf.read(name).decode("utf-8")

    import configparser

    parser = configparser.ConfigParser()
    parser.read_string(text)

    # These are the exact three groups `_discover_from_entry_points` iterates.
    assert set(parser.sections()) <= {"glider.driver", "glider.device", "glider.node"}
    assert parser["glider.driver"]["harp"] == "glider_harp.board:HarpBoard"
    assert parser["glider.driver"]["glider_harp"] == "glider_harp"
    assert parser["glider.device"]["harp"] == "glider_harp.device:HarpDevice"


# --------------------------------------------------------------------------
# 5. The core wheel does not carry this package.
# --------------------------------------------------------------------------


def test_the_core_glider_wheel_does_not_contain_the_plugin(tmp_path_factory):
    """``glider-harp`` ships separately, so a copy inside the GLIDER wheel would
    shadow the installed one and pin users to whatever was vendored.

    The staging copy deliberately includes ``plugins/``, so this fails if the
    root ``pyproject.toml`` ever grows a package/data entry reaching into it --
    which is the only way the files could get in.
    """
    wheel = _build_wheel(REPO_ROOT, tmp_path_factory.mktemp("core"), "glider")
    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()

    assert any(n.startswith("glider/") for n in names), "core wheel looks empty"
    strays = [n for n in names if "glider_harp" in n or n.startswith("plugins/")]
    assert strays == [], strays
