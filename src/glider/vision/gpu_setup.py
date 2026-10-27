"""Swap a CPU-only PyTorch for the CUDA build this machine's NVIDIA driver can run.

``uv sync`` installs torch from PyPI, and on Windows PyPI's torch is CPU-only.
So a desktop with an NVIDIA card runs every YOLO and pose model on the CPU,
several times slower, and nothing says so unless someone opens the device check
(#221). This finds the driver, finds the PyTorch build that matches it, and
installs that build in place of the CPU one.

**Which build is read off the PyTorch index, not a table here.** Each torch
release ships for a different handful of CUDA versions (``cu126``, ``cu128``,
``cu130``...), and a table in this file would be wrong by the next release. The
driver says the newest CUDA it can run; :func:`find_wheel_tag` walks down from
there and takes the first index that has a wheel for *this* torch version,
Python and platform.

**The torch version is kept.** Only the build changes (``2.10.0`` becomes
``2.10.0+cu128``), so everything resolved against the lock file still fits.

Qt-free, so the detection is testable without a GUI; the dialogs live in
:mod:`glider.gui.main_window`.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import logging
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "PYTORCH_INDEX",
    "candidate_tags",
    "driver_cuda_version",
    "find_wheel_tag",
    "gpu_support",
    "install_command",
    "install_cuda_torch",
    "installed_torch",
    "needs_cuda_torch",
    "parse_driver_cuda",
]

#: PyTorch's wheel index. ``{PYTORCH_INDEX}/cu128/torch/`` lists every torch
#: wheel built against CUDA 12.8.
PYTORCH_INDEX = "https://download.pytorch.org/whl"

#: The oldest CUDA PyTorch still publishes for; nothing below it is worth asking for.
_OLDEST_CUDA = (11, 8)

# Both header styles: "CUDA Version: 12.4" (drivers before 580) and
# "CUDA UMD Version: 13.3" (later ones).
_DRIVER_CUDA = re.compile(r"CUDA(?: UMD)? Version:\s*(\d+)\.(\d+)")


def parse_driver_cuda(nvidia_smi_output: str) -> tuple[int, int] | None:
    """The newest CUDA the driver supports, from ``nvidia-smi``'s header."""
    match = _DRIVER_CUDA.search(nvidia_smi_output)
    return (int(match.group(1)), int(match.group(2))) if match else None


def driver_cuda_version(run=subprocess.run) -> tuple[int, int] | None:
    """Ask ``nvidia-smi``; None when there is no NVIDIA driver to ask."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        result = run([exe], capture_output=True, text=True, timeout=10)
    except (subprocess.SubprocessError, OSError):
        return None
    if result.returncode != 0:
        return None
    return parse_driver_cuda(result.stdout)


def installed_torch() -> tuple[str, str | None] | None:
    """``(version, cuda)`` of the installed torch, or None if there is none.

    Read from ``torch/version.py`` rather than by importing torch, which takes
    seconds and loads DLLs that must not be loaded after Qt on Windows (see
    ``glider.__main__``). If torch is already imported, its own answer is used.
    ``cuda`` is None for a CPU-only build.
    """
    try:
        version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return None
    torch = sys.modules.get("torch")
    if torch is not None and hasattr(torch, "version"):
        return version, getattr(torch.version, "cuda", None)
    spec = importlib.util.find_spec("torch")
    if spec is None or not spec.submodule_search_locations:
        return None
    try:
        source = (Path(next(iter(spec.submodule_search_locations))) / "version.py").read_text()
    except OSError:
        return version, None
    match = re.search(r"^cuda\b[^=\n]*=\s*['\"]([^'\"]+)['\"]", source, re.MULTILINE)
    return version, match.group(1) if match else None


def gpu_support(
    *,
    torch_info: Callable[[], tuple[str, str | None] | None] = installed_torch,
    driver: Callable[[], tuple[int, int] | None] = driver_cuda_version,
    platform_name: str = sys.platform,
    frozen: bool = bool(getattr(sys, "frozen", False)),
) -> tuple[tuple[int, int] | None, str]:
    """``(driver CUDA version, "")`` if a CUDA torch can be installed, else ``(None, why not)``.

    Installable when torch is installed but CPU-only and an NVIDIA driver
    answers. Not on macOS (no CUDA there), and not in a frozen build, which
    ships the CUDA build already and has no installer to change it with. The
    cheap checks run first, so the common case never starts ``nvidia-smi``.
    The reason is worded for the person who asked, since the Install GPU
    Support entry has to say something when there is nothing to install.
    """
    if frozen:
        return None, "This build of GLIDER already includes GPU support."
    if platform_name == "darwin":
        return None, (
            "Macs have no CUDA. GLIDER uses Apple's GPU (MPS) on its own when "
            "the Mac has Apple Silicon."
        )
    torch = torch_info()
    if torch is None:
        return None, "PyTorch is not installed. Install the vision extra first."
    if torch[1] is not None:
        return None, f"GPU support is already installed (PyTorch built for CUDA {torch[1]})."
    found = driver()
    if found is None:
        return None, (
            "No NVIDIA GPU driver answered (nvidia-smi was not found or failed). "
            "Install the NVIDIA driver for your card, then try again."
        )
    return found, ""


def needs_cuda_torch(**probes) -> tuple[int, int] | None:
    """The driver's CUDA version if a CUDA torch should be offered, else None."""
    return gpu_support(**probes)[0]


def candidate_tags(driver: tuple[int, int]) -> list[str]:
    """Wheel tags to try, newest first, none newer than the driver can run.

    CUDA minor versions are forward compatible within a major on the driver
    side, so a 12.4 driver runs ``cu124`` and below but not ``cu126``.
    """
    tags = []
    major, minor = driver
    while (major, minor) >= _OLDEST_CUDA:
        tags.append(f"cu{major}{minor}")
        if minor == 0:
            # No CUDA x.10 or later exists, so 9 is enough to start each major.
            major, minor = major - 1, 9
        else:
            minor -= 1
    return tags


def _platform_marker(platform_name: str = sys.platform, machine: str = "") -> str:
    machine = (machine or platform.machine()).lower()
    if platform_name == "win32":
        return "win_amd64"
    return "aarch64" if machine in ("aarch64", "arm64") else "x86_64"


def _http_get(url: str) -> str | None:
    import httpx

    try:
        response = httpx.get(url, timeout=20, follow_redirects=True)
    except httpx.HTTPError as exc:
        logger.debug("Could not read %s: %s", url, exc)
        return None
    return response.text if response.status_code == 200 else None


def find_wheel_tag(
    torch_version: str,
    driver: tuple[int, int],
    *,
    fetch: Callable[[str], str | None] = _http_get,
    python_tag: str = f"cp{sys.version_info.major}{sys.version_info.minor}",
    platform_name: str = sys.platform,
    machine: str = "",
) -> str | None:
    """The newest CUDA tag with a wheel of *torch_version* for this machine.

    Blocking (one request per tag tried), so call it off the GUI thread.
    """
    base = torch_version.split("+", 1)[0]
    marker = _platform_marker(platform_name, machine)
    for tag in candidate_tags(driver):
        page = fetch(f"{PYTORCH_INDEX}/{tag}/torch/")
        if not page:
            continue
        # The index writes the local-version "+" as %2B in its links.
        wheel = re.compile(
            rf"torch-{re.escape(base)}(?:%2B|\+){tag}-{python_tag}-{python_tag}-[^\"'#<>]*"
            rf"{marker}[^\"'#<>]*\.whl"
        )
        if wheel.search(page):
            return tag
    return None


def install_command(tag: str, torch_version: str, torchvision_version: str | None) -> list[str]:
    """The pip/uv command that swaps in the *tag* build of torch (and torchvision).

    Exact local versions (``torch==2.10.0+cu128``) exist only on PyTorch's
    index, so the CPU wheel cannot be picked again. PyTorch's index goes in
    ``--extra-index-url``: uv tries that before PyPI, and pip pools the two.
    """
    from glider.plugins.installer import installer_command

    requirements = [f"torch=={torch_version.split('+', 1)[0]}+{tag}"]
    if torchvision_version:
        requirements.append(f"torchvision=={torchvision_version.split('+', 1)[0]}+{tag}")
    args = installer_command(requirements[0], requirements[1:])
    return [*args, "--extra-index-url", f"{PYTORCH_INDEX}/{tag}"]


async def install_cuda_torch(
    driver: tuple[int, int],
    *,
    runner=None,
    on_output: Callable[[str], None] | None = None,
    find_tag: Callable[[str, tuple[int, int]], str | None] = find_wheel_tag,
):
    """Find the matching build and install it; returns an ``InstallResult``."""
    import asyncio

    from glider.plugins.installer import (
        InstallResult,
        MalformedEntryError,
        NoInstallerError,
        _default_runner,
    )

    torch = installed_torch()
    if torch is None:
        return InstallResult(ok=False, message="PyTorch is not installed.")
    torch_version = torch[0]
    try:
        torchvision_version: str | None = importlib.metadata.version("torchvision")
    except importlib.metadata.PackageNotFoundError:
        torchvision_version = None

    if on_output:
        on_output("Looking up the matching PyTorch build…")
    tag = await asyncio.to_thread(find_tag, torch_version, driver)
    if tag is None:
        return InstallResult(
            ok=False,
            message=(
                f"PyTorch publishes no CUDA build of torch {torch_version} that a "
                f"CUDA {driver[0]}.{driver[1]} driver can run on this machine. "
                "Updating the NVIDIA driver usually fixes this."
            ),
        )
    try:
        args = install_command(tag, torch_version, torchvision_version)
    except (NoInstallerError, MalformedEntryError) as exc:
        return InstallResult(ok=False, message=str(exc))

    returncode, output = await (runner or _default_runner)(args, on_output)
    if returncode != 0:
        tool = "uv" if args[0] != sys.executable else "pip"
        return InstallResult(
            ok=False, message=f"{tool} exited with code {returncode}.", output=output
        )
    return InstallResult(ok=True, message=f"Installed PyTorch with CUDA ({tag}).", output=output)
