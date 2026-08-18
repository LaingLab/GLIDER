"""Install a catalogue entry with pip.

pip runs as a subprocess of *this* interpreter rather than being imported: pip's
API is explicitly not public, and installing into a different environment than
the one GLIDER is running from would look like success and import like failure.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from packaging.specifiers import SpecifierSet
from packaging.version import Version

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InstallResult:
    ok: bool
    message: str
    output: str = ""


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
    return process.returncode or 0, "\n".join(chunks)


async def install(
    entry: dict[str, Any],
    glider_version: str,
    runner=None,
    on_output: Callable[[str], None] | None = None,
) -> InstallResult:
    """Install one catalogue entry, refusing before pip runs if it cannot fit."""
    run = runner or _default_runner
    requires = entry.get("glider_requires", "") or ""

    if requires and Version(glider_version) not in SpecifierSet(requires):
        # Name both versions: "incompatible" alone sends people to the issue
        # tracker to ask which half is wrong.
        logger.info(
            "Refusing to install %s: needs GLIDER %s, running %s",
            entry.get("name"),
            requires,
            glider_version,
        )
        return InstallResult(
            ok=False,
            message=(f"{entry['name']} needs GLIDER {requires}. You are running {glider_version}."),
        )

    args = [sys.executable, "-m", "pip", "install", entry["pypi"]]
    returncode, output = await run(args, on_output)

    if returncode != 0:
        return InstallResult(ok=False, message=f"pip exited with code {returncode}.", output=output)

    # A freshly installed plugin has to be importable without a restart.
    importlib.invalidate_caches()
    return InstallResult(ok=True, message=f"Installed {entry['name']}.", output=output)
