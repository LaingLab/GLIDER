"""Install a catalogue entry with pip, or with uv when pip is absent.

pip runs as a subprocess of *this* interpreter rather than being imported: pip's
API is explicitly not public, and installing into a different environment than
the one GLIDER is running from would look like success and import like failure.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import logging
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

logger = logging.getLogger(__name__)

#: What goes wrong when a catalogue entry's version fields are not version
#: fields. ``TypeError`` is in here because the index is JSON from a web server:
#: ``glider_requires`` may arrive as a list or a number, not just as bad text.
_UNREADABLE = (InvalidSpecifier, InvalidVersion, TypeError)


class NoInstallerError(RuntimeError):
    """Neither pip nor uv is available to install with."""


def _pip_is_importable() -> bool:
    """Can the interpreter we install into actually run ``-m pip``?

    Checked in-process rather than by shelling out, which is exact here: we
    always install into ``sys.executable``, so pip being importable *now* is the
    same question as ``sys.executable -m pip`` working.
    """
    return importlib.util.find_spec("pip") is not None


def _find_uv() -> str | None:
    return shutil.which("uv")


def installer_command(
    package: str,
    requirements: Sequence[str] = (),
    *,
    pip_available: Callable[[], bool] = _pip_is_importable,
    uv_path: Callable[[], str | None] = _find_uv,
) -> list[str]:
    """Build the command that installs ``package`` into this interpreter.

    pip is not a given. GLIDER's documented setup is ``uv venv`` + ``uv sync``
    (CLAUDE.md), and ``uv venv`` does not install pip -- so on the *primary*
    supported environment ``sys.executable -m pip`` fails outright with "No
    module named pip". uv is the fallback, and on such an environment it is
    almost certainly present, since it is what built the venv.

    ``requirements`` are catalogue-declared direct requirements, appended to
    the command. They exist for one reason: uv honours a pre-release marker
    (``>=0.5.0rc1``) only on a *direct* requirement, so a plugin whose
    dependency pins a pre-release resolves under pip and fails under uv unless
    the catalogue names that pin directly. Harmless under pip, which honours
    the marker wherever it appears (PEP 440).

    pip wins when both exist: it is the interpreter's own installer and needs no
    assumption about which environment uv would otherwise choose.

    Raises:
        NoInstallerError: if neither is available. The caller shows this on the
            plugin's row; it is a condition to report, not to crash on.
    """
    if pip_available():
        return [sys.executable, "-m", "pip", "install", package, *requirements]

    uv = uv_path()
    if uv:
        # --python is not optional. Without it uv picks its own target, and the
        # plugin lands in an environment GLIDER never imports from -- an install
        # that reports success and changes nothing.
        return [uv, "pip", "install", "--python", sys.executable, package, *requirements]

    raise NoInstallerError(
        "Neither pip nor uv is available in this environment, so plugins cannot "
        "be installed. Install pip into GLIDER's environment, or install uv."
    )


@dataclass(frozen=True)
class InstallResult:
    ok: bool
    message: str
    output: str = ""


def _requirement(entry: Mapping[str, Any]) -> str:
    return str(entry.get("glider_requires", "") or "")


def package_name(entry: Mapping[str, Any]) -> str:
    """What to hand pip: the ``pypi`` field, or the catalogue name behind it.

    Every other reader of an entry tolerates a missing ``pypi`` this way. This
    one used to subscript it, which turned a one-field omission in a curated
    index into a ``KeyError`` mid-install.
    """
    return str(entry.get("pypi") or entry.get("name") or "").strip()


def is_compatible(entry: Mapping[str, Any], glider_version: str) -> bool:
    """Whether *entry*'s ``glider_requires`` admits *glider_version*.

    Split out of :func:`install` so the Plugins window can grey a row out under
    exactly the rule the installer would refuse it by. A window offering an
    Install button that pip then declines is worse than no button at all.

    A requirement that does not parse -- ``"1.0"`` where ``">=1.0"`` was meant is
    the natural authoring mistake -- is **answered ``False``, not raised**. The
    index arrives over the network and nothing validates its fields, so a
    malformed one is data, not a programming error; raising here took the whole
    window down before a single row was drawn.
    """
    requires = _requirement(entry)
    if not requires:
        return True
    try:
        return Version(str(glider_version)) in SpecifierSet(requires)
    except _UNREADABLE as exc:
        logger.warning(
            "Catalogue entry %r has an unreadable version requirement %r: %s",
            entry.get("name"),
            requires,
            exc,
        )
        return False


def is_readable(entry: Mapping[str, Any], glider_version: str) -> bool:
    """Whether the two version fields can be compared at all."""
    try:
        SpecifierSet(_requirement(entry))
        Version(str(glider_version))
    except _UNREADABLE:
        return False
    return True


def incompatibility_message(entry: Mapping[str, Any], glider_version: str) -> str:
    """Say *which* two versions disagree.

    "incompatible" on its own sends people to the issue tracker to ask which
    half is wrong, so both halves are always named -- and named identically
    whether the refusal came from the window or from :func:`install`.

    When the fields cannot be read at all the sentence has to change rather than
    fill in the blanks: "needs GLIDER 1.0, you are running 1.0.0" reads as a
    version mismatch when the entry itself is what is broken.
    """
    name = str(entry.get("name") or package_name(entry) or "This plugin")
    requires = _requirement(entry)
    if not is_readable(entry, glider_version):
        return (
            f"{name} has an unreadable catalogue entry: {requires!r} is not a version "
            f"requirement GLIDER can compare against {glider_version}. The catalogue "
            "needs correcting; nothing can be installed from this row."
        )
    return f"{name} needs GLIDER {requires}. You are running {glider_version}."


async def _default_runner(args: list[str], on_output: Callable[[str], None] | None = None):
    """Run a command, streaming stdout line by line as it arrives."""
    process = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    chunks: list[str] = []
    assert process.stdout is not None
    async for raw in process.stdout:
        line = raw.decode("utf-8", errors="replace").rstrip()
        chunks.append(line)
        if on_output:
            on_output(line)
    await process.wait()
    # Not `returncode or 0`: that maps None -- "the process never reported a
    # status" -- onto success, and this is the value that decides whether pip
    # worked. An unknown outcome is a failed install, not a clean one.
    returncode = process.returncode
    return (1 if returncode is None else returncode), "\n".join(chunks)


async def install(
    entry: dict[str, Any],
    glider_version: str,
    runner=None,
    on_output: Callable[[str], None] | None = None,
    command: Callable[[str, tuple[str, ...]], list[str]] | None = None,
) -> InstallResult:
    """Install one catalogue entry, refusing before pip runs if it cannot fit."""
    run = runner or _default_runner

    if not is_compatible(entry, glider_version):
        logger.info(
            "Refusing to install %s: needs GLIDER %s, running %s",
            entry.get("name"),
            entry.get("glider_requires", ""),
            glider_version,
        )
        return InstallResult(ok=False, message=incompatibility_message(entry, glider_version))

    package = package_name(entry)
    if not package:
        logger.error("Catalogue entry names no package to install: %r", entry)
        return InstallResult(
            ok=False,
            message="This catalogue entry names no package, so there is nothing to install.",
        )

    build_command = command or installer_command
    try:
        args = build_command(package, tuple(entry.get("requirements") or ()))
    except NoInstallerError as exc:
        logger.error("No installer available: %s", exc)
        return InstallResult(ok=False, message=str(exc))

    returncode, output = await run(args, on_output)

    if returncode != 0:
        tool = "uv" if args and args[0] != sys.executable else "pip"
        return InstallResult(
            ok=False, message=f"{tool} exited with code {returncode}.", output=output
        )

    # A freshly installed plugin has to be importable without a restart.
    importlib.invalidate_caches()
    return InstallResult(
        ok=True, message=f"Installed {entry.get('name') or package}.", output=output
    )
