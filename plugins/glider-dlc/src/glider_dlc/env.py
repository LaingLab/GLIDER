"""The DeepLabCut environment GLIDER builds for itself.

DeepLabCut cannot live in GLIDER's environment. It is needed to *read* a
snapshot at all -- a DLC checkpoint is a bare PyTorch ``state_dict``, and
rebuilding the network from one needs DLC's own model classes -- but it brings
torch, timm, albumentations and a long tail with it, about 1.3 GB installed,
which is not something to hand a lab that tracks with YOLO.

The alternative to putting it there is not making the researcher build an
environment by hand. It is building one for them, once, the first time they
select a DeepLabCut model, and keeping it. That is what this module does:
``uv`` creates a private virtualenv under ``~/.glider/envs`` and installs
DeepLabCut into it, and the conversion runs in that interpreter.

A lab that already has a working DeepLabCut environment can point at it with
``GLIDER_DLC_ENV`` and nothing is downloaded at all.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

#: Python for the child environment. Pinned rather than inherited: GLIDER runs
#: on 3.11-3.13 and DeepLabCut does not track that range, so following GLIDER's
#: interpreter would make provisioning succeed or fail depending on which
#: Python the researcher happened to install GLIDER under.
ENV_PYTHON = "3.12"

#: What goes in. onnx and onnxscript are listed because torch does not depend
#: on either, and ``torch.onnx.export`` fails at the end of a long export
#: without them.
ENV_PACKAGES = (
    "deeplabcut>=3.0,<4",
    "onnx>=1.16",
    "onnxscript>=0.1",
    "pyyaml",
)

#: Written into the environment once it is complete. Its absence means a
#: half-built environment -- an interrupted download leaves an interpreter that
#: imports nothing -- so provisioning keys off this rather than off the
#: directory existing.
STAMP_NAME = ".glider_env.json"

#: Roughly what lands on disk, for the sentence shown before the download
#: starts. Measured, not estimated.
INSTALLED_SIZE_GB = 1.3

#: Generous. This is a multi-gigabyte download on whatever connection the lab
#: has, and killing a working install is worse than waiting for a slow one.
PROVISION_TIMEOUT_S = 3600

#: A conversion itself is a graph trace of a network already on disk.
CONVERT_TIMEOUT_S = 900

NO_UV_HINT = (
    "Building the DeepLabCut environment needs `uv`, which is what GLIDER's "
    "own installer uses and is usually already present.\n\n"
    "Install it from https://docs.astral.sh/uv/ and try again, or point "
    "GLIDER at a DeepLabCut environment you already have by setting the "
    "GLIDER_DLC_ENV environment variable to it."
)


class ProvisioningError(RuntimeError):
    """The DeepLabCut environment could not be built.

    Carries a sentence meant for a researcher: it goes on screen verbatim.
    """


def env_dir() -> Path:
    """Where the DeepLabCut environment lives.

    ``GLIDER_DLC_ENV`` overrides it, which is the escape hatch for a lab that
    already has DeepLabCut working somewhere and would rather not have a second
    copy: point it at that virtualenv and nothing is downloaded.
    """
    override = os.environ.get("GLIDER_DLC_ENV")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".glider" / "envs" / "deeplabcut"


def interpreter(env: Path | None = None) -> Path:
    """The Python inside *env*."""
    env = Path(env) if env is not None else env_dir()
    if sys.platform == "win32":
        return env / "Scripts" / "python.exe"
    return env / "bin" / "python"


def _spec() -> dict:
    return {"python": ENV_PYTHON, "packages": list(ENV_PACKAGES)}


def is_provisioned(env: Path | None = None) -> bool:
    """Whether *env* is a complete environment built for the current spec.

    Compares against the recorded spec rather than merely checking that the
    interpreter exists. A plugin upgrade that changes ``ENV_PACKAGES`` has to
    rebuild -- otherwise a fix to the pinned DeepLabCut version would apply to
    new users only, and silently, which is the shape of bug that gets blamed on
    the model.

    A ``GLIDER_DLC_ENV`` the lab manages is exempt: it has no stamp and is not
    ours to rebuild, so an interpreter there is taken at its word.
    """
    env = Path(env) if env is not None else env_dir()
    if not interpreter(env).is_file():
        return False
    if os.environ.get("GLIDER_DLC_ENV"):
        return True
    try:
        return json.loads((env / STAMP_NAME).read_text()) == _spec()
    except (OSError, ValueError):
        return False


def _uv() -> str:
    uv = shutil.which("uv")
    if uv is None:
        raise ProvisioningError(NO_UV_HINT)
    return uv


def _run(args: list[str], *, what: str, timeout: int) -> None:
    logger.info("DeepLabCut environment: %s", what)
    try:
        completed = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise ProvisioningError(
            f"{what} took longer than {timeout // 60} minutes and was stopped."
        ) from exc
    except OSError as exc:
        raise ProvisioningError(f"{what} could not be started: {exc}") from exc

    if completed.returncode != 0:
        # uv's last few lines say what actually went wrong -- an unresolvable
        # pin, no network, no matching Python. The whole log is on the logger
        # for anyone who wants it; the dialog gets the end of it.
        logger.error("DeepLabCut environment: %s failed\n%s", what, completed.stderr)
        tail = "\n".join((completed.stderr or "").strip().splitlines()[-6:])
        raise ProvisioningError(f"{what} failed.\n\n{tail}".strip())


def provision(env: Path | None = None, *, on_progress: Callable[[str], None] | None = None) -> Path:
    """Build the DeepLabCut environment, and return its interpreter.

    Idempotent: an environment already built for the current spec is returned
    untouched. Otherwise a partial one is removed first -- resuming an install
    into a virtualenv whose interpreter is missing does not work, and the
    failure it produces is unreadable.

    Raises:
        ProvisioningError: with a sentence to show the researcher.
    """
    env = Path(env) if env is not None else env_dir()
    if is_provisioned(env):
        return interpreter(env)

    if os.environ.get("GLIDER_DLC_ENV"):
        raise ProvisioningError(
            f"GLIDER_DLC_ENV points at {env}, which has no Python in it. "
            "Point it at a virtualenv with DeepLabCut installed, or unset it "
            "and let GLIDER build its own."
        )

    def say(message: str) -> None:
        logger.info("%s", message)
        if on_progress is not None:
            on_progress(message)

    uv = _uv()
    stamp = env / STAMP_NAME
    if stamp.is_file():
        stamp.unlink()
    if env.exists():
        say(f"Removing the incomplete environment at {env}")
        shutil.rmtree(env, ignore_errors=True)

    env.parent.mkdir(parents=True, exist_ok=True)
    say(f"Creating a Python {ENV_PYTHON} environment at {env}")
    _run(
        [uv, "venv", "--python", ENV_PYTHON, str(env)],
        what=f"Creating the Python {ENV_PYTHON} environment",
        timeout=PROVISION_TIMEOUT_S,
    )

    say(f"Installing DeepLabCut (about {INSTALLED_SIZE_GB} GB, once)")
    _run(
        [uv, "pip", "install", "--python", str(interpreter(env)), *ENV_PACKAGES],
        what="Installing DeepLabCut",
        timeout=PROVISION_TIMEOUT_S,
    )

    stamp.write_text(json.dumps(_spec(), indent=2))
    say("DeepLabCut environment ready")
    return interpreter(env)


def run_converter(
    script: Path,
    model_dir: Path,
    *,
    env: Path | None = None,
    timeout: int = CONVERT_TIMEOUT_S,
) -> str:
    """Run *script* over *model_dir* in the DeepLabCut environment.

    Raises:
        ProvisioningError: if the conversion fails, carrying the child's own
            message -- which is one sentence written for a person, because the
            converter prints only that to stderr.
    """
    python = interpreter(env)
    try:
        completed = subprocess.run(
            [str(python), str(script), str(model_dir)],
            capture_output=True,
            text=True,
            timeout=timeout,
            # torch's exporter prints a check mark on success, which raises
            # UnicodeEncodeError on a Windows console still defaulting to
            # cp1252 -- a conversion that worked, reported as a crash.
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
    except subprocess.TimeoutExpired as exc:
        raise ProvisioningError(
            f"Converting {model_dir.name} took longer than {timeout // 60} "
            "minutes and was stopped."
        ) from exc

    if completed.returncode != 0:
        message = (completed.stderr or "").strip() or "Conversion failed."
        raise ProvisioningError(message)
    return completed.stdout
